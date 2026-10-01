"""The reference station with its on-site battery disabled (shared by the no-battery audits)."""
from chargax import ChargingStation, EVSE, StationBattery, StationSplitter


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
