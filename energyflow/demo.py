from tempfile import TemporaryDirectory
from pathlib import Path
from .service import EnergyService


DAY = 86400
HOUR = 3600
DAY_SHIFT = [0] * 6 + [2] * 12 + [0] * 6
NIGHT_SHIFT = [2] * 6 + [0] * 12 + [2] * 6
FLAT_DAY = [1] * 24


def load_shape_demo(service):
    """Same daily totals, different timing: clustering must separate shapes."""
    service.catalog.register("workshop", "Synthetic workshop")
    service.catalog.install("ws-seg", "workshop", 0, 0, max_gap=30 * DAY)
    patterns = [
        DAY_SHIFT, DAY_SHIFT, DAY_SHIFT,
        NIGHT_SHIFT, NIGHT_SHIFT, NIGHT_SHIFT,
    ]
    counter = 0
    for day, pattern in enumerate(patterns):
        for hour, increment in enumerate(pattern):
            counter += increment
            service.readings.add(
                {
                    "id": "ws-%d-%d" % (day, hour),
                    "segment": "ws-seg",
                    "at": day * DAY + (hour + 1) * HOUR,
                    "value": counter,
                }
            )
    model = service.load_models.train(
        "workshop-shapes",
        "workshop",
        0,
        6 * DAY,
        seed=42,
        k=2,
        width=HOUR,
        novelty_threshold="1/100",
        note="Day versus night load shapes with equal daily totals",
    )
    assert model["k"] == 2
    assert {cluster["size"] for cluster in model["clusters"]} == {3}
    assert service.load_models.verify("workshop-shapes")["verified"]
    # A later flat day is a new pattern, not a forced fit.
    for hour, increment in enumerate(FLAT_DAY):
        counter += increment
        service.readings.add(
            {
                "id": "ws-flat-%d" % hour,
                "segment": "ws-seg",
                "at": 6 * DAY + (hour + 1) * HOUR,
                "value": counter,
            }
        )
    classified = service.load_models.assign_days(
        "workshop-shapes", 6 * DAY, 7 * DAY
    )
    assert classified["assignments"][0]["state"] == "new_pattern"
    # A second frozen version merges into one cluster; comparison explains moves.
    service.load_models.train(
        "workshop-merged",
        "workshop",
        0,
        6 * DAY,
        seed=42,
        k=1,
        width=HOUR,
        novelty_threshold="1/100",
    )
    comparison = service.load_models.compare_versions(
        "workshop-shapes", "workshop-merged"
    )
    assert comparison["merges"]
    return {
        "clusters": model["k"],
        "cluster_sizes": [cluster["size"] for cluster in model["clusters"]],
        "silhouette": model["silhouette"],
        "later_day_state": classified["assignments"][0]["state"],
        "version_merges": comparison["merges"],
    }


def run():
    with TemporaryDirectory() as folder:
        with EnergyService(Path(folder) / "energy.db") as service:
            service.catalog.register("line", "Synthetic production line")
            service.catalog.install("counter", "line", 0, 0, max_gap=7200)
            service.readings.import_batch(
                "shift",
                [
                    {"id": "r1", "segment": "counter", "at": 3600, "value": 10},
                    {"id": "r2", "segment": "counter", "at": 7200, "value": 30},
                ],
            )
            service.production.put(
                "first", "line", 0, 3600, {"productA": 1, "productB": 1}
            )
            service.production.put("second", "line", 3600, 7200, {"productB": 1})
            service.tariffs.put("offpeak", "line", 0, 3600, "0.5")
            service.tariffs.put("peak", "line", 3600, 7200, "1.0")
            service.emissions.put("local", "line", 0, 7200, "500", "synthetic factor")
            preview = service.closing.preview(["line"], 0, 7200, require_factors=True)
            frozen = service.closing.freeze(
                "shift",
                ["line"],
                0,
                7200,
                preview["digest"],
                "controller",
                require_factors=True,
            )
            assert (
                sum(row["microkwh"] for row in frozen["payload"]["energy"]) == 30000000
            )
            assert sum(row["minor"] for row in frozen["payload"]["costs"]) == 2500
            service.readings.correct(
                "r1", 1, 12, "normal", "synthetic calibration correction"
            )
            correction = service.closing.propose_correction(
                "adjust", "shift", "corrected intermediate counter"
            )
            service.closing.approve_correction(
                "adjust", correction["digest"], "controller"
            )
            effective = service.closing.effective("shift")
            assert (
                sum(row["microkwh"] for row in effective["payload"]["energy"])
                == 30000000
            )
            assert sum(row["minor"] for row in effective["payload"]["costs"]) == 2400
            assert service.closing.verify("shift")["verified"]
            assert not service.closing.drift("shift")["changed"]
            target = Path(folder) / "settlement.json"
            service.bundles.export("shift", target, effective=True)
            assert service.bundles.verify(target)["verified"]
            assert service.quality.system()["ok"]
            load_shape = load_shape_demo(service)
            return {
                "status": "ok",
                "energy_microkwh": 30000000,
                "original_cost_minor": 2500,
                "corrected_cost_minor": 2400,
                "frozen_preserved": True,
                "offline": True,
                "load_shape_demo": load_shape,
            }
