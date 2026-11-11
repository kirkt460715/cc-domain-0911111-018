import unittest
from fractions import Fraction
from energyflow.service import EnergyService
from energyflow.values import EnergyError, fraction
from energyflow.load.clustering import euclid, SeededRandom


DAY = 86400
HOUR = 3600
WIDTH = HOUR
CELLS = 24
# Two shapes with identical daily totals but different timing: day shift loads
# the working hours, night shift loads the opposite hours.
DAY_PATTERN = [0] * 6 + [2] * 12 + [0] * 6
NIGHT_PATTERN = [2] * 6 + [0] * 12 + [2] * 6
FLAT_PATTERN = [1] * 24


class LoadTestBase(unittest.TestCase):
    def setUp(self):
        self.s = EnergyService()
        self.addCleanup(self.s.close)
        self.s.catalog.register("m", "Synthetic workshop meter")
        self.s.catalog.install("seg", "m", 0, 0, max_gap=30 * DAY)

    def feed_day(self, day_index, pattern, segment="seg", start_value=None):
        """Insert hourly cumulative counter readings for one production day."""
        counters = getattr(self, "_counter", {})
        state = start_value if start_value is not None else counters.get(segment, 0)
        at = day_index * DAY
        for hour in range(24):
            state += pattern[hour]
            self.s.readings.add(
                dict(
                    id="r:%s:%s:%s" % (segment, day_index, hour),
                    segment=segment,
                    at=at + (hour + 1) * HOUR,
                    value=state,
                )
            )
        counters[segment] = state
        self._counter = counters
        return state

    def feed(self, patterns, segment="seg"):
        self._counter = {segment: 0}
        for index, pattern in enumerate(patterns):
            if pattern is not None:
                self.feed_day(index, pattern, segment)

    def build(self, days, **kwargs):
        end = days * DAY
        options = dict(width=WIDTH, min_coverage="1")
        options.update(kwargs)
        return self.s.load_models.builder.build("m", 0, end, **options)


class ProfileTests(LoadTestBase):
    def test_same_total_different_shapes_are_separated_by_normalization(self):
        self.feed([DAY_PATTERN, NIGHT_PATTERN])
        profile = self.build(2)
        self.assertEqual(profile["eligible_dates"], ["1970-01-01", "1970-01-02"])
        totals = {
            day["date"]: fraction(day["total_microkwh"]) for day in profile["days"]
        }
        # Business totals are kept separately and happen to be equal.
        self.assertEqual(totals["1970-01-01"], totals["1970-01-02"])
        normalized = [self.s.load_models.builder.normalized(day) for day in profile["days"]]
        distance = euclid(
            [fraction(v) for v in normalized[0]["shares"]],
            [fraction(v) for v in normalized[1]["shares"]],
        )
        self.assertGreater(distance, 0)
        for row in normalized:
            self.assertEqual(
                sum((fraction(v) for v in row["shares"]), Fraction(0)), 1
            )

    def test_cross_day_intervals_are_split_to_both_days(self):
        # One single interval spanning 48 hours must be allocated proportionally.
        self.s.readings.add(dict(id="span", segment="seg", at=2 * DAY, value=48))
        profile = self.build(2)
        first, second = profile["days"]
        self.assertTrue(first["complete"] and second["complete"])
        self.assertEqual(
            fraction(first["total_microkwh"]),
            fraction(second["total_microkwh"]),
        )
        self.assertEqual(
            fraction(first["total_microkwh"]) + fraction(second["total_microkwh"]),
            Fraction(48_000_000),
        )
        # Every cell traces back to the same reconstruction interval id.
        refs = {ref for cell in first["lineage"] for ref in cell["evidence"]}
        self.assertEqual(len(refs), 1)

    def test_zero_load_is_distinct_from_missing_measurement(self):
        self.feed([FLAT_PATTERN])
        # Second day: counter flat all day (explicit zero load), third day: no
        # readings at all (missing coverage).
        flat_value = self._counter["seg"]
        self.s.readings.add(
            dict(id="flat-end", segment="seg", at=2 * DAY, value=flat_value)
        )
        profile = self.build(3)
        by_date = {day["date"]: day for day in profile["days"]}
        zero_day = by_date["1970-01-02"]
        missing_day = by_date["1970-01-03"]
        self.assertTrue(zero_day["zero_load_day"])
        self.assertEqual(
            set(zero_day["energy"]), {"0/1"}
        )
        self.assertFalse(missing_day["zero_load_day"])
        self.assertTrue(all(value is None for value in missing_day["energy"]))
        self.assertEqual(profile["zero_load_dates"], ["1970-01-02"])
        self.assertEqual(
            [row["date"] for row in profile["incomplete_dates"]], ["1970-01-03"]
        )
        self.assertNotIn("1970-01-02", profile["eligible_dates"])
        self.assertNotIn("1970-01-03", profile["eligible_dates"])

    def test_partial_day_keeps_nulls_without_zero_padding(self):
        self.feed([FLAT_PATTERN])  # only day one has readings
        profile = self.build(2)
        partial = profile["days"][1]
        self.assertFalse(partial["complete"])
        # The day itself is a full calendar window; the missing cells are gaps.
        self.assertFalse(partial["partial_window"])
        self.assertTrue(all(value is None for value in partial["energy"]))
        self.assertEqual(partial["coverage"], "0/1")
        # A relaxed coverage threshold still never admits an uncovered day.
        relaxed = self.build(2, min_coverage="1/2")
        self.assertFalse(relaxed["days"][1]["eligible"])
        partial_span = self.s.load_models.builder.build("m", 0, DAY + HOUR)
        self.assertEqual(partial_span["days"][0]["coverage"], "1/1")
        self.assertTrue(partial_span["days"][1]["partial_window"])

    def test_offset_shifts_production_day_origin(self):
        self.feed([FLAT_PATTERN, FLAT_PATTERN])
        offset = 8 * HOUR
        profile = self.s.load_models.builder.build(
            "m", 0, 2 * DAY, offset_seconds=offset, width=WIDTH
        )
        first = profile["days"][0]
        # A +08:00 production day covering UTC midnight starts at 16:00 prior day.
        self.assertEqual(first["start"], -16 * HOUR)
        self.assertEqual(first["end"], 8 * HOUR)
        self.assertEqual(first["date"], "1969-12-31")
        # Only 8 of this day's 24 cells fall inside the queried UTC span.
        self.assertEqual(fraction(first["coverage"]), Fraction(8, 24))
        self.assertFalse(first["complete"])
        self.assertTrue(profile["days"][1]["complete"])
        # Negative offset: origin not later than start.
        shifted = self.s.load_models.builder.build(
            "m", 0, 2 * DAY, offset_seconds=-8 * HOUR, width=WIDTH
        )
        self.assertEqual(shifted["days"][0]["start"], -8 * HOUR)

    def test_grid_width_must_divide_day(self):
        with self.assertRaises(EnergyError):
            self.s.load_models.builder.build("m", 0, DAY, width=3700)

    def test_long_interval_crossing_midnight_splits_proportionally(self):
        # One uniform 6-hour interval spanning two production days.
        self.s.readings.add(
            dict(id="r1", segment="seg", at=21 * HOUR, value=0)
        )
        self.s.readings.add(
            dict(id="r2", segment="seg", at=DAY + 3 * HOUR, value=24)
        )
        profile = self.build(2)
        d0, d1 = profile["days"]
        self.assertEqual(d0["energy"][21:], ["4000000/1"] * 3)
        self.assertEqual(d1["energy"][:3], ["4000000/1"] * 3)
        self.assertTrue(all(value is None for value in d1["energy"][3:]))
        self.assertTrue(d0["complete"])
        self.assertFalse(d1["complete"])
        # The same interval id is referenced on both sides of midnight.
        ref0 = {ref for cell in d0["lineage"][21:] for ref in cell["evidence"]}
        ref1 = {ref for cell in d1["lineage"][:3] for ref in cell["evidence"]}
        self.assertEqual(ref0, ref1)

    def test_dispatch_reaches_profile_and_load_actions(self):
        self.feed([FLAT_PATTERN])
        result = self.s.dispatch(
            {
                "action": "profile.build",
                "arguments": {"meter": "m", "start": 0, "end": DAY, "width": HOUR},
            }
        )
        self.assertEqual(result["eligible_dates"], ["1970-01-01"])
        with self.assertRaises(EnergyError):
            self.s.dispatch(
                {
                    "action": "load.train",
                    "arguments": {
                        "model_id": "x",
                        "meter": "m",
                        "start": 0,
                        "end": DAY,
                        "seed": -1,
                    },
                }
            )

    def test_meter_replacement_energy_is_stitched_on_one_curve(self):
        self.feed_day(0, DAY_PATTERN, "seg")
        boundary = DAY
        # The new physical meter starts from zero; its baseline is independent of
        # the retired meter's cumulative register.
        self.s.catalog.replacement(
            "seg", "seg2", boundary, 0, microkwh_per_tick=1_000_000
        )
        self.feed_day(1, NIGHT_PATTERN, "seg2")
        profile = self.build(2)
        day_two = profile["days"][1]
        self.assertTrue(day_two["complete"])
        # Lineage of both segments appears across the two production days; each
        # hourly interval contributes its own reconstruction id.
        all_refs = {
            ref
            for day in profile["days"]
            for cell in day["lineage"]
            for ref in cell["evidence"]
        }
        self.assertEqual(len(all_refs), 48)
        first_segments = {
            ref
            for cell in profile["days"][0]["lineage"]
            for ref in cell["evidence"]
        }
        second_segments = {
            ref for cell in day_two["lineage"] for ref in cell["evidence"]
        }
        self.assertTrue(first_segments and second_segments)
        self.assertFalse(first_segments & second_segments)

    def test_replacement_baseline_seeds_first_interval_without_gap(self):
        # First reading on the new segment lands one hour after installation;
        # the seeded baseline must still produce a complete first cell.
        self.feed_day(0, FLAT_PATTERN, "seg")
        self.s.catalog.replacement("seg", "seg2", DAY, 0)
        self.s.readings.add(
            dict(
                id="seg2-first",
                segment="seg2",
                at=DAY + HOUR,
                value=3,
            )
        )
        profile = self.build(2)
        self.assertTrue(profile["days"][0]["complete"])
        second = profile["days"][1]
        self.assertEqual(fraction(second["coverage"]), Fraction(1, 24))
        self.assertEqual(fraction(second["energy"][0]), Fraction(3_000_000))


class ClusteringTests(LoadTestBase):
    def train_setup(self):
        patterns = [
            DAY_PATTERN, DAY_PATTERN, DAY_PATTERN,
            NIGHT_PATTERN, NIGHT_PATTERN, NIGHT_PATTERN,
        ]
        self.feed(patterns)
        profile = self.build(6)
        normalized = [
            self.s.load_models.builder.normalized(day)
            for day in profile["days"]
            if day["eligible"]
        ]
        return profile, normalized

    def test_fixed_seed_gives_stable_clusters_and_real_representative(self):
        _, normalized = self.train_setup()
        clusterer = self.s.load_models.clusterer
        first = clusterer.fit(normalized, 2, seed=42)
        second = clusterer.fit(normalized, 2, seed=42)
        self.assertEqual(first["assignments"], second["assignments"])
        self.assertEqual(first["clusters"], second["clusters"])
        self.assertEqual({cluster["size"] for cluster in first["clusters"]}, {3})
        for cluster in first["clusters"]:
            self.assertIn(cluster["representative_date"], cluster["dates"])
            self.assertEqual(len(cluster["member_distances"]), cluster["size"])
            self.assertGreaterEqual(
                fraction(cluster["max_distance"]),
                fraction(cluster["mean_distance"]),
            )
        # A different seed may pick different initial centers, but the structure
        # of these well-separated shapes must converge to the same partition.
        other = clusterer.fit(normalized, 2, seed=99)
        self.assertEqual(
            sorted(first["assignments"].values()),
            sorted(other["assignments"].values()),
        )
        self.assertIsNotNone(first["silhouette"])
        self.assertEqual(fraction(first["silhouette"]), 1)

    def test_k_selection_prefers_two_well_separated_shapes(self):
        _, normalized = self.train_setup()
        selection = self.s.load_models.clusterer.select_k(
            normalized, seed=7, min_k=2, max_k=4
        )
        self.assertEqual(selection["best_k"], 2)
        self.assertEqual(len(selection["candidates"]), 3)

    def test_seeded_random_is_deterministic(self):
        left = [SeededRandom(1).next_uint() for _ in range(5)]
        right = [SeededRandom(1).next_uint() for _ in range(5)]
        self.assertEqual(left, right)
        self.assertNotEqual(left, [SeededRandom(2).next_uint() for _ in range(5)])

    def test_single_cluster_has_no_silhouette_and_declared_metric(self):
        self.feed([FLAT_PATTERN, FLAT_PATTERN])
        profile = self.build(2)
        normalized = [
            self.s.load_models.builder.normalized(day)
            for day in profile["days"]
            if day["eligible"]
        ]
        fit = self.s.load_models.clusterer.fit(normalized, 1, seed=3)
        self.assertIsNone(fit["silhouette"])
        self.assertEqual(
            fit["distance_metric"], "squared_euclidean_on_unit_total_shares"
        )

    def test_rejects_unnormalized_input(self):
        self.feed([FLAT_PATTERN])
        profile = self.build(1)
        good = self.s.load_models.builder.normalized(profile["days"][0])
        bad = dict(good)
        doubled = [str(fraction(value) * 2) for value in good["shares"]]
        bad["shares"] = doubled
        with self.assertRaises(EnergyError):
            self.s.load_models.clusterer.fit([bad], 1, seed=1)


class ModelTests(LoadTestBase):
    def train(self, seed=42, k=2, days=6, **kwargs):
        patterns = [
            DAY_PATTERN, DAY_PATTERN, DAY_PATTERN,
            NIGHT_PATTERN, NIGHT_PATTERN, NIGHT_PATTERN,
        ][:days]
        self.feed(patterns)
        return self.s.load_models.train(
            "shape-v1", "m", 0, days * DAY, seed=seed, k=k,
            width=WIDTH, min_coverage="1", novelty_threshold="1/100",
            **kwargs
        )

    def test_train_freezes_representative_shapes_with_quality(self):
        model = self.train()
        self.assertEqual(model["k"], 2)
        self.assertEqual(model["seed"], 42)
        self.assertEqual(len(model["clusters"]), 2)
        self.assertEqual(sum(row["size"] for row in model["clusters"]), 6)
        for cluster in model["clusters"]:
            center_total = sum(
                (fraction(v) for v in cluster["center"]), Fraction(0)
            )
            self.assertEqual(center_total, 1)
            self.assertTrue(cluster["representative_date"])
        self.assertEqual(fraction(model["silhouette"]), 1)
        self.assertTrue(self.s.load_models.verify("shape-v1")["verified"])

    def test_repeated_train_returns_identical_frozen_model(self):
        first = self.train()
        repeated = self.train()
        self.assertTrue(repeated["reused"])
        self.assertEqual(repeated["revision"], first["revision"])
        stored = self.s.load_models.get("shape-v1")
        self.assertEqual(stored["clusters"], first["clusters"])
        # Retraining with changed parameters is refused rather than overwritten.
        with self.assertRaises(EnergyError):
            self.s.load_models.train(
                "shape-v1", "m", 0, 6 * DAY, seed=43, k=2, width=WIDTH
            )

    def test_historical_labels_do_not_move_on_later_runs(self):
        self.train()
        before = self.s.load_models.assignments("shape-v1")["assignments"]
        # Add later days and re-run assignment; old labels remain byte-identical.
        self.feed_day(6, DAY_PATTERN)
        self.feed_day(7, NIGHT_PATTERN)
        report = self.s.load_models.assign_days("shape-v1", 6 * DAY, 8 * DAY)
        self.assertEqual(
            [row["date"] for row in report["assignments"]],
            ["1970-01-07", "1970-01-08"],
        )
        after = self.s.load_models.assignments("shape-v1")["assignments"]
        self.assertEqual(before, after[: len(before)])
        # Replaying the same span is idempotent.
        again = self.s.load_models.assign_days("shape-v1", 6 * DAY, 8 * DAY)
        self.assertTrue(all(row["reused"] for row in again["assignments"]))

    def test_novel_pattern_above_threshold_is_flagged_not_force_fitted(self):
        model = self.train()
        # A flat shape is far from both bimodal centers.
        self.feed_day(6, FLAT_PATTERN)
        report = self.s.load_models.assign_days("shape-v1", 6 * DAY, 7 * DAY)
        decision = report["assignments"][0]
        self.assertEqual(decision["state"], "new_pattern")
        self.assertIsNone(decision["label"])
        self.assertGreater(
            fraction(decision["distance"]), fraction(decision["threshold"])
        )
        # A normal later day still joins an existing cluster.
        self.feed_day(7, DAY_PATTERN)
        follow = self.s.load_models.assign_days("shape-v1", 7 * DAY, 8 * DAY)
        self.assertEqual(follow["assignments"][0]["state"], "assigned")
        self.assertIsNotNone(follow["assignments"][0]["label"])

    def test_zero_and_incomplete_days_are_classified_not_clustered(self):
        self.train(days=4)
        flat_value = self._counter["seg"]
        # Day five: zero load. Day six: nothing at all.
        self.s.readings.add(
            dict(id="flat5", segment="seg", at=5 * DAY, value=flat_value)
        )
        report = self.s.load_models.assign_days("shape-v1", 4 * DAY, 7 * DAY)
        states = {row["date"]: row["state"] for row in report["assignments"]}
        self.assertEqual(states["1970-01-05"], "zero_load")
        self.assertEqual(states["1970-01-06"], "incomplete")
        for row in report["assignments"]:
            self.assertIsNone(row["label"])

    def test_compare_versions_reports_splits_merges_and_moves(self):
        patterns_v1 = [
            DAY_PATTERN, DAY_PATTERN,
            NIGHT_PATTERN, NIGHT_PATTERN,
            FLAT_PATTERN, FLAT_PATTERN,
        ]
        self.feed(patterns_v1)
        self.s.load_models.train(
            "v1", "m", 0, 6 * DAY, seed=42, k=3,
            width=WIDTH, novelty_threshold="1/100",
        )
        # Second version merges everything into two clusters; the flat days move.
        self.s.load_models.train(
            "v2", "m", 0, 6 * DAY, seed=42, k=2,
            width=WIDTH, novelty_threshold="1/100",
        )
        comparison = self.s.load_models.compare_versions("v1", "v2")
        self.assertEqual(len(comparison["common_dates"]), 6)
        self.assertTrue(comparison["merges"])
        self.assertTrue(comparison["moves"])
        merged_labels = {row["label"] for row in comparison["merges"]}
        moved_dates = {row["date"] for row in comparison["moves"]}
        self.assertTrue(moved_dates.issubset(set(comparison["common_dates"])))
        self.assertTrue(merged_labels)
        # Cross-meter comparison is rejected.
        self.s.catalog.register("other", "Other meter")
        self.s.catalog.install("o", "other", 0, 0, max_gap=30 * DAY)
        other_state = 0
        for day_index in range(6):
            for hour in range(24):
                other_state += FLAT_PATTERN[hour]
                self.s.readings.add(
                    dict(
                        id="o:%s:%s" % (day_index, hour),
                        segment="o",
                        at=day_index * DAY + (hour + 1) * HOUR,
                        value=other_state,
                    )
                )
        self.s.load_models.train(
            "v-other", "other", 0, 6 * DAY, seed=42, k=1, width=WIDTH
        )
        with self.assertRaises(EnergyError):
            self.s.load_models.compare_versions("v1", "v-other")

    def test_lineage_reaches_reconstruction_intervals(self):
        model = self.train(days=2)
        stored = self.s.load_models.assignments("shape-v1")["assignments"]
        training_row = next(row for row in stored if row["state"] == "training")
        self.assertTrue(training_row["lineage"]["reconstruction_intervals"])
        self.assertEqual(len(training_row["lineage"]["cells"]), CELLS)
        # Tampering with a stored assignment breaks verification reads.
        self.s.store.connection.execute(
            "UPDATE load_assignments SET digest='bad' WHERE model=? AND date=?",
            ("shape-v1", training_row["date"]),
        )
        with self.assertRaises(EnergyError):
            self.s.load_models.assignments("shape-v1")

    def test_persistence_round_trip_keeps_frozen_model(self):
        import tempfile
        from pathlib import Path

        self.train()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "load.db"
            self.s.store.backup(path)
            with EnergyService(path) as restarted:
                model = restarted.load_models.get("shape-v1")
                self.assertEqual(model["k"], 2)
                self.assertTrue(
                    restarted.load_models.verify("shape-v1")["verified"]
                )
                self.assertTrue(restarted.store.integrity()["verified"])

    def test_dispatch_surfaces_load_actions(self):
        actions = self.s.actions()
        for name in [
            "load.train",
            "load.get",
            "load.assign_days",
            "load.compare_versions",
            "profile.build",
        ]:
            self.assertIn(name, actions)

    def test_integrity_includes_load_tables(self):
        self.train()
        report = self.s.store.integrity()
        self.assertTrue(report["verified"])
        self.assertIn("load_models", report["counts"])
        self.assertIn("load_assignments", report["counts"])

    def test_train_without_eligible_days_is_rejected(self):
        self.s.catalog.register("empty", "No data meter")
        self.s.catalog.install("g", "empty", 0, 0)
        with self.assertRaises(EnergyError):
            self.s.load_models.train(
                "none", "empty", 0, DAY, seed=1, width=WIDTH
            )

    def test_auto_k_adapts_to_a_single_eligible_day(self):
        self.feed([FLAT_PATTERN])
        model = self.s.load_models.train(
            "auto-single", "m", 0, DAY, seed=5, width=WIDTH
        )
        self.assertEqual(model["k"], 1)
        self.assertEqual(model["k_selection"]["best_k"], 1)

    def test_cli_end_to_end_train_and_classify(self):
        import contextlib
        import io
        import json
        import tempfile
        from pathlib import Path
        from energyflow.terminal import main
        from energyflow.values import canonical

        self.train()
        with tempfile.TemporaryDirectory() as folder:
            db = Path(folder) / "cli.db"
            self.s.store.backup(db)
            request = Path(folder) / "request.json"
            request.write_text(
                canonical(
                    {
                        "action": "load.assign_days",
                        "arguments": {
                            "model_id": "shape-v1",
                            "start": 0,
                            "end": 6 * DAY,
                        },
                    }
                )
            )
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                status = main(["--db", str(db), "run", str(request)])
            self.assertEqual(status, 0)
            payload = json.loads(output.getvalue())
            self.assertTrue(payload["ok"])
            self.assertEqual(
                len(payload["result"]["assignments"]), 6
            )
            self.assertTrue(
                all(
                    row["state"] == "training"
                    for row in payload["result"]["assignments"]
                )
            )
