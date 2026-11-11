from fractions import Fraction
from ..values import (
    key,
    period,
    integer,
    text,
    canonical,
    parse_json,
    fingerprint,
    fraction,
    rational,
    reject,
)
from .profiles import ProfileBuilder
from .clustering import Clusterer, DISTANCE_METRIC


class LoadModels:
    """Persist frozen load-shape models and classify later days against them.

    Fitting only happens during ``train``; classification afterwards always uses
    the frozen centers, so reruns never move historical labels.  Every stored row
    carries a fingerprint plus reading lineage back to reconstruction intervals.
    """

    def __init__(self, store, reconstructor, catalog):
        self.store = store
        self.catalog = catalog
        self.builder = ProfileBuilder(reconstructor, catalog)
        self.clusterer = Clusterer()

    # ---------------------------------------------------------------- storage
    def _read_model_row(self, model_id):
        row = self.store.one(
            "SELECT * FROM load_models WHERE id=?", (model_id,)
        )
        if row is None:
            reject("missing", "load model not found", model=model_id)
        payload = parse_json(row["payload"])
        if fingerprint(payload) != row["digest"]:
            reject("integrity", "load model payload changed", model=model_id)
        return row, payload

    def _read_assignment_row(self, model_id, date):
        row = self.store.one(
            "SELECT * FROM load_assignments WHERE model=? AND date=?",
            (model_id, date),
        )
        if row is None:
            return None
        payload = parse_json(row["payload"])
        if fingerprint(payload) != row["digest"]:
            reject(
                "integrity",
                "load assignment payload changed",
                model=model_id,
                date=date,
            )
        return row, payload

    def get(self, model_id):
        _, payload = self._read_model_row(model_id)
        return payload

    def list(self, meter=None):
        if meter is None:
            rows = self.store.rows(
                "SELECT id FROM load_models ORDER BY created_at,id"
            )
        else:
            rows = self.store.rows(
                "SELECT id FROM load_models WHERE meter=? ORDER BY created_at,id",
                (meter,),
            )
        return [self.get(row["id"]) for row in rows]

    def verify(self, model_id):
        row, payload = self._read_model_row(model_id)
        for cluster in payload["clusters"]:
            total = sum((fraction(v) for v in cluster["center"]), Fraction(0))
            if total != 1:
                reject("integrity", "frozen center is not normalized")
            if cluster["representative_date"] not in cluster["dates"]:
                reject("integrity", "representative day missing from cluster")
        assignments = self.store.rows(
            "SELECT date FROM load_assignments WHERE model=? ORDER BY date",
            (model_id,),
        )
        for item in assignments:
            stored = self._read_assignment_row(model_id, item["date"])
            if stored is None:
                reject("integrity", "assignment row vanished")
        return {
            "model": model_id,
            "verified": True,
            "digest": row["digest"],
            "assignments": len(assignments),
        }

    # ------------------------------------------------------------------ train
    def train(
        self,
        model_id,
        meter,
        start,
        end,
        seed,
        k=None,
        offset_seconds=0,
        width=900,
        min_coverage="1",
        novelty_threshold="0",
        max_k=6,
        note="",
    ):
        key(model_id)
        start, end = period(start, end)
        integer(seed, 0, 2**63 - 1)
        threshold = fraction(novelty_threshold)
        if threshold < 0:
            reject("input", "novelty threshold cannot be negative")
        note = text(note, 1000, True)
        with self.store.atomic() as con:
            existing = self.store.one(
                "SELECT id FROM load_models WHERE id=?", (model_id,)
            )
            if existing:
                current = self.get(model_id)
                same = (
                    current["meter"] == meter
                    and current["training"]["start"] == start
                    and current["training"]["end"] == end
                    and current["seed"] == seed
                    and current["params"]["width"] == width
                    and current["params"]["offset_seconds"] == offset_seconds
                    and current["params"]["min_coverage"] == min_coverage
                    and current["params"]["novelty_threshold"]
                    == rational(threshold)
                )
                if not same:
                    reject("conflict", "frozen model identity is immutable")
                return {**current, "reused": True}

            profile = self.builder.build(
                meter, start, end, offset_seconds, width, min_coverage
            )
            normalized = [
                self.builder.normalized(day)
                for day in profile["days"]
                if day["eligible"]
            ]
            if not normalized:
                reject("input", "no eligible production days available for training")
            if k is None:
                selection = self.clusterer.select_k(
                    normalized,
                    seed,
                    min_k=min(2, len(normalized)),
                    max_k=min(max_k, len(normalized)),
                )
                chosen_k = selection["best_k"]
            else:
                integer(k, 1, len(normalized))
                selection = None
                chosen_k = k
            fit = self.clusterer.fit(normalized, chosen_k, seed)
            revision_row = self.store.one(
                "SELECT COALESCE(MAX(revision),0) AS revision FROM load_models WHERE meter=?",
                (meter,),
            )
            revision = revision_row["revision"] + 1
            payload = {
                "id": model_id,
                "meter": meter,
                "revision": revision,
                "format_version": 1,
                "seed": seed,
                "k": fit["k"],
                "created_at": end,
                "note": note,
                "params": {
                    "width": width,
                    "cells": profile["cells"],
                    "offset_seconds": offset_seconds,
                    "min_coverage": min_coverage,
                    "novelty_threshold": rational(threshold),
                },
                "training": {
                    "start": start,
                    "end": end,
                    "profile_digest": profile["digest"],
                    "reconstruction_digest": profile["reconstruction_digest"],
                    "dates": fit["dates"],
                    "zero_load_dates": profile["zero_load_dates"],
                    "incomplete_dates": profile["incomplete_dates"],
                },
                "clusters": fit["clusters"],
                "distance_metric": DISTANCE_METRIC,
                "silhouette": fit["silhouette"],
                "inertia": fit["inertia"],
                "iterations": fit["iterations"],
                "k_selection": selection,
            }
            digest = fingerprint(payload)
            con.execute(
                "INSERT INTO load_models VALUES(?,?,?,?,?,?)",
                (model_id, meter, revision, end, canonical(payload), digest),
            )
            # Freeze the training-day assignments so historical labels never move.
            for day in profile["days"]:
                if day["eligible"]:
                    label = fit["assignments"][day["date"]]
                    cluster = next(
                        row for row in fit["clusters"] if row["label"] == label
                    )
                    self._write_assignment(
                        con,
                        model_id,
                        meter,
                        day,
                        {
                            "label": label,
                            "state": "training",
                            "distance": cluster["member_distances"][day["date"]],
                            "threshold": cluster["max_distance"],
                            "candidates": [],
                            "source_digest": next(
                                row["source_digest"]
                                for row in normalized
                                if row["date"] == day["date"]
                            ),
                            "lineage": self._lineage(day),
                        },
                    )
                elif day["zero_load_day"]:
                    self._write_assignment(
                        con,
                        model_id,
                        meter,
                        day,
                        {
                            "label": None,
                            "state": "zero_load",
                            "distance": None,
                            "threshold": rational(threshold),
                            "candidates": [],
                            "source_digest": fingerprint(
                                {"date": day["date"], "total_microkwh": "0/1"}
                            ),
                            "lineage": self._lineage(day),
                        },
                    )
            self.store.event(
                "load_model_trained",
                model_id,
                {
                    "meter": meter,
                    "revision": revision,
                    "digest": digest,
                    "k": fit["k"],
                    "seed": seed,
                },
            )
        return {**self.get(model_id), "reused": False}

    @staticmethod
    def _lineage(day):
        return {
            "reconstruction_intervals": sorted(
                {
                    ref
                    for cell in day["lineage"]
                    for ref in cell["evidence"]
                }
            ),
            "cells": day["lineage"],
        }

    def _write_assignment(self, con, model_id, meter, day, body):
        payload = {
            "model": model_id,
            "meter": meter,
            "date": day["date"],
            "day_start": day["start"],
            "day_end": day["end"],
            "coverage": day["coverage"],
            "total_microkwh": day["total_microkwh"],
            **body,
        }
        digest = fingerprint(payload)
        con.execute(
            "INSERT INTO load_assignments VALUES(?,?,?,?,?,?,?,?,?)",
            (
                model_id,
                meter,
                day["date"],
                payload["label"],
                payload["state"],
                payload["distance"],
                1,
                canonical(payload),
                digest,
            ),
        )
        return payload

    # ------------------------------------------------------------- assign new
    def assign_days(self, model_id, start, end):
        start, end = period(start, end)
        model = self.get(model_id)
        params = model["params"]
        profile = self.builder.build(
            model["meter"],
            start,
            end,
            params["offset_seconds"],
            params["width"],
            params["min_coverage"],
        )
        results = []
        with self.store.atomic() as con:
            for day in profile["days"]:
                stored = self._read_assignment_row(model_id, day["date"])
                if stored is not None:
                    results.append({**stored[1], "reused": True})
                    continue
                if day["zero_load_day"]:
                    payload = self._write_assignment(
                        con,
                        model_id,
                        model["meter"],
                        day,
                        {
                            "label": None,
                            "state": "zero_load",
                            "distance": None,
                            "threshold": params["novelty_threshold"],
                            "candidates": [],
                            "source_digest": fingerprint(
                                {"date": day["date"], "total_microkwh": "0/1"}
                            ),
                            "lineage": self._lineage(day),
                        },
                    )
                elif not day["eligible"]:
                    payload = self._write_assignment(
                        con,
                        model_id,
                        model["meter"],
                        day,
                        {
                            "label": None,
                            "state": "incomplete",
                            "distance": None,
                            "threshold": params["novelty_threshold"],
                            "candidates": [],
                            "source_digest": fingerprint(
                                {
                                    "date": day["date"],
                                    "coverage": day["coverage"],
                                    "zero_load": day["zero_load_day"],
                                }
                            ),
                            "lineage": self._lineage(day),
                        },
                    )
                else:
                    normalized = self.builder.normalized(day)
                    decision = self.clusterer.assign(
                        normalized, model, params["novelty_threshold"]
                    )
                    payload = self._write_assignment(
                        con,
                        model_id,
                        model["meter"],
                        day,
                        {
                            "label": decision["label"],
                            "state": decision["state"],
                            "distance": decision["distance"],
                            "threshold": decision["threshold"],
                            "candidates": decision["candidates"],
                            "source_digest": decision["source_digest"],
                            "lineage": self._lineage(day),
                        },
                    )
                results.append({**payload, "reused": False})
            self.store.event(
                "load_days_classified",
                model_id,
                {
                    "start": start,
                    "end": end,
                    "dates": [row["date"] for row in results],
                },
            )
        return {
            "model": model_id,
            "meter": model["meter"],
            "start": start,
            "end": end,
            "profile_digest": profile["digest"],
            "assignments": results,
        }

    def assignments(self, model_id, state=None):
        self.get(model_id)
        rows = self.store.rows(
            "SELECT date FROM load_assignments WHERE model=? ORDER BY date",
            (model_id,),
        )
        result = [self._read_assignment_row(model_id, row["date"])[1] for row in rows]
        if state is not None:
            result = [row for row in result if row["state"] == state]
        return {"model": model_id, "assignments": result}

    # ------------------------------------------------------------- comparison
    def compare_versions(self, first_id, second_id):
        first = self.get(first_id)
        second = self.get(second_id)
        if first["meter"] != second["meter"]:
            reject("input", "model comparison requires the same meter")
        if first["params"]["cells"] != second["params"]["cells"]:
            reject("input", "model grids differ; shapes are not comparable")
        if (
            first["params"]["offset_seconds"] != second["params"]["offset_seconds"]
            or first["params"]["width"] != second["params"]["width"]
        ):
            reject("input", "model day origins or grid widths differ")
        old_rows = self.store.rows(
            "SELECT date,cluster_label,state FROM load_assignments WHERE model=? ORDER BY date",
            (first_id,),
        )
        new_rows = self.store.rows(
            "SELECT date,cluster_label,state FROM load_assignments WHERE model=? ORDER BY date",
            (second_id,),
        )
        old_map = {row["date"]: row for row in old_rows}
        new_map = {row["date"]: row for row in new_rows}
        common = sorted(set(old_map) & set(new_map))
        moves = []
        old_to_new = {}
        new_to_old = {}
        for date in common:
            old_label = old_map[date]["cluster_label"]
            new_label = new_map[date]["cluster_label"]
            if old_label is None or new_label is None:
                continue
            old_to_new.setdefault(old_label, set()).add(new_label)
            new_to_old.setdefault(new_label, set()).add(old_label)
            if old_label != new_label:
                moves.append(
                    {
                        "date": date,
                        "from": old_label,
                        "to": new_label,
                    }
                )
        splits = [
            {"label": label, "into": sorted(targets)}
            for label, targets in sorted(old_to_new.items())
            if len(targets) > 1
        ]
        merges = [
            {"label": label, "from": sorted(sources)}
            for label, sources in sorted(new_to_old.items())
            if len(sources) > 1
        ]
        only_first = sorted(set(old_map) - set(new_map))
        only_second = sorted(set(new_map) - set(old_map))
        return {
            "meter": first["meter"],
            "first_model": first_id,
            "second_model": second_id,
            "common_dates": common,
            "moves": moves,
            "splits": splits,
            "merges": merges,
            "dates_only_in_first": only_first,
            "dates_only_in_second": only_second,
            "unchanged": len(common) - len(moves),
        }
