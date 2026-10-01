"""audit_margin_greedy.py on a station WITHOUT the on-site battery.

Identical to audit_margin_greedy.py (same day set, policies, metrics) except that the
reference station is built without its 150 kWh battery, so the online controllers face
the same battery-free site the offline oracle LP models. Writes
experiments/results/audit_margin_greedy_nobattery.json.
"""
import json
import os

from chargax import ChargingStation, EVSE, StationBattery, StationSplitter
import experiments.run_experiments as RE


def scarcity_station_no_battery(grid_kw: float = 60.0, n_evses: int = 16) -> ChargingStation:
    """experiments.common.scarcity_station with the battery disabled.

    The baseline policies index into the station battery, so it cannot be deleted
    outright; a battery with zero throughput and negligible capacity can move no
    energy, which makes it equivalent to having none."""
    return ChargingStation(
        max_kw_throughput=grid_kw,
        efficiency=1.0,
        connections=[
            StationSplitter(
                max_kw_throughput=grid_kw * 10,
                efficiency=0.99,
                connections=[
                    EVSE(num_chargers=2, voltage=400, max_current=55, efficiency=0.99)
                    for _ in range(n_evses)
                ]
                + [StationBattery(capacity_kw=1e-6, max_kw_throughput=0.0, efficiency=0.97)],
            )
        ],
    )


RE.scarcity_station = scarcity_station_no_battery   # used by RE._ref_env

import experiments.audit_margin_greedy as AMG  # noqa: E402

# Write to a temporary folder, then rename, so the battery run's
# audit_margin_greedy.json is never opened for writing.
_TMP = os.path.join(AMG.RESULTS, "_nobattery_tmp")
os.makedirs(_TMP, exist_ok=True)
AMG.RESULTS = _TMP


def _finish():
    src = os.path.join(_TMP, "audit_margin_greedy.json")
    out = json.load(open(src))
    out["station"] = "reference site WITHOUT on-site battery"
    dst = os.path.join(os.path.dirname(_TMP), "audit_margin_greedy_nobattery.json")
    json.dump(out, open(dst, "w"), indent=2)
    os.remove(src); os.rmdir(_TMP)
    print("wrote", dst)


if __name__ == "__main__":
    AMG.main()
    _finish()
