from fractions import Fraction
from datetime import datetime, timezone
from ..values import (
    period,
    integer,
    fraction,
    rational,
    fingerprint,
    reject,
)


class ProfileBuilder:
    """Build fixed-width production-day curves from reconstructed window power.

    A production day starts at ``day_origin`` plus ``offset_seconds`` and always
    spans 86400 seconds.  Reconstruction intervals (including cross-day spans and
    meter replacement segments) are integrated onto the configured grid; a zero
    load window keeps explicit zero values, while missing coverage is recorded as
    ``null`` instead of being silently padded with zeros.  Partial days are never
    backfilled: they are reported with their real coverage fraction.
    """

    def __init__(self, reconstructor, catalog):
        self.reconstructor = reconstructor
        self.catalog = catalog

    @staticmethod
    def settings(meter, start, end, offset_seconds=0, width=900, min_coverage="1"):
        start, end = period(start, end)
        integer(offset_seconds, -86399, 86399)
        integer(width, 1, 86400)
        if 86400 % width:
            reject("input", "grid width must divide the production day evenly")
        coverage = fraction(min_coverage)
        if not 0 < coverage <= 1:
            reject("input", "coverage threshold must lie within (0,1]")
        return {
            "meter": meter,
            "start": start,
            "end": end,
            "offset_seconds": offset_seconds,
            "width": width,
            "cells": 86400 // width,
            "min_coverage": rational(coverage),
        }

    def production_days(self, start, end, offset_seconds):
        """Half-open production-day windows intersecting [start, end)."""
        start, end = period(start, end)
        # Greatest day origin (midnight + offset) not later than start.
        step = (start - offset_seconds) // 86400
        first_midnight = step * 86400 + offset_seconds
        result = []
        cursor = first_midnight
        while cursor < end:
            following = cursor + 86400
            left = max(start, cursor)
            right = min(end, following)
            if left < right:
                label = datetime.fromtimestamp(
                    cursor, timezone.utc
                ).date().isoformat()
                result.append(
                    {
                        "date": label,
                        "start": cursor,
                        "end": following,
                        "window_start": left,
                        "window_end": right,
                    }
                )
            cursor = following
        return result

    @staticmethod
    def integrate_cell(intervals, left, right):
        """Exact energy inside one grid cell plus the evidence interval ids."""
        energy = Fraction(0)
        covered = 0
        evidence = []
        for row in intervals:
            if not row["valid"]:
                continue
            a = max(left, row["start"])
            b = min(right, row["end"])
            if a >= b:
                continue
            energy += fraction(row["microkwh"]) * Fraction(
                b - a, row["seconds"]
            )
            covered += b - a
            evidence.append(row["id"])
        return energy, covered, evidence

    def day_curve(self, meter, day, width, intervals):
        cells = 86400 // width
        values = []
        evidence = []
        covered_seconds = 0
        observed_energy = Fraction(0)
        for index in range(cells):
            left = day["start"] + index * width
            right = left + width
            energy, covered, refs = self.integrate_cell(intervals, left, right)
            complete = covered == width
            if complete:
                covered_seconds += width
                observed_energy += energy
                values.append(rational(energy))
            else:
                values.append(None)
            evidence.append(
                {
                    "index": index,
                    "start": left,
                    "end": right,
                    "covered_seconds": covered,
                    "evidence": refs,
                }
            )
        coverage = Fraction(covered_seconds, 86400)
        all_zero = covered_seconds > 0 and observed_energy == 0
        clipped = day["window_start"] != day["start"] or day["window_end"] != day["end"]
        return {
            "date": day["date"],
            "start": day["start"],
            "end": day["end"],
            "width": width,
            "cells": cells,
            # Exact per-cell energy in microkwh; null means missing measurement.
            "energy": values,
            "total_microkwh": rational(observed_energy),
            "coverage": rational(coverage),
            "covered_seconds": covered_seconds,
            "complete": coverage == 1,
            "partial_window": clipped,
            "zero_load_day": all_zero,
            "lineage": evidence,
        }

    def build(
        self,
        meter,
        start,
        end,
        offset_seconds=0,
        width=900,
        min_coverage="1",
    ):
        config = self.settings(
            meter, start, end, offset_seconds, width, min_coverage
        )
        self.catalog.get(meter)
        # One reconstruction over the whole span keeps cross-day and replacement
        # intervals intact; clipping is performed per cell below.
        report = self.reconstructor.reconstruct(meter, start, end)
        intervals = report["intervals"]
        days = []
        for day in self.production_days(start, end, offset_seconds):
            curve = self.day_curve(meter, day, width, intervals)
            threshold = fraction(config["min_coverage"])
            # Only fully covered non-zero days enter clustering.  Partially
            # covered days stay visible below but are never backfilled or used.
            curve["eligible"] = curve["complete"] and not curve["zero_load_day"]
            curve["meets_coverage_threshold"] = (
                fraction(curve["coverage"]) >= threshold
            )
            days.append(curve)
        eligible = [row for row in days if row["eligible"]]
        zero_days = [row["date"] for row in days if row["zero_load_day"]]
        incomplete = [
            {"date": row["date"], "coverage": row["coverage"]}
            for row in days
            if not row["complete"]
        ]
        result = {
            "meter": meter,
            "start": start,
            "end": end,
            "offset_seconds": offset_seconds,
            "width": width,
            "cells": config["cells"],
            "min_coverage": config["min_coverage"],
            "days": days,
            "eligible_dates": [row["date"] for row in eligible],
            "zero_load_dates": zero_days,
            "incomplete_dates": incomplete,
            "reconstruction_digest": report["digest"],
            "basis": "fixed_grid_integration_of_reconstructed_intervals",
        }
        result["digest"] = fingerprint(
            {
                key: result[key]
                for key in [
                    "meter",
                    "start",
                    "end",
                    "offset_seconds",
                    "width",
                    "cells",
                    "min_coverage",
                    "eligible_dates",
                    "zero_load_dates",
                    "incomplete_dates",
                    "reconstruction_digest",
                ]
            }
        )
        return result

    def normalized(self, curve):
        """Normalize one eligible day by its own total; totals stay untouched."""
        if not curve["eligible"]:
            reject("input", "only eligible non-zero days can be normalized")
        total = fraction(curve["total_microkwh"])
        shares = []
        for value in curve["energy"]:
            if value is None:
                reject(
                    "coverage",
                    "eligible curve contains an uncovered cell",
                    date=curve["date"],
                )
            shares.append(rational(fraction(value) / total))
        total_share = sum((fraction(value) for value in shares), Fraction(0))
        if total_share != 1:
            reject("integrity", "normalized curve does not sum to one")
        return {
            "date": curve["date"],
            "width": curve["width"],
            "shares": shares,
            "total_microkwh": curve["total_microkwh"],
            "coverage": curve["coverage"],
            "source_digest": fingerprint(
                {
                    "date": curve["date"],
                    "energy": curve["energy"],
                    "total_microkwh": curve["total_microkwh"],
                }
            ),
        }
