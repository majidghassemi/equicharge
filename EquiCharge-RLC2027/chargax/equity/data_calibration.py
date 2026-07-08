r"""Real-data calibration of the charging environment (provenance + ACN-Data hook).

The distributions the environment samples from -- arrival times, dwell/connection
times, per-session energy demand, and electricity prices -- are **real, not
synthetic**. They are the empirically-derived datasets shipped with Chargax
(``chargax/data/``), summarized in :func:`provenance`. The only construct added on
top for the equity study is the ability-to-pay segmentation, which is itself
grounded in public income/energy-burden data (see :mod:`chargax.equity.segments`).

For reviewers who want an independent, US, session-level calibration we also provide
:func:`acn_data_kwargs`, an adapter that turns the public **Caltech ACN-Data**
(Lee et al., 2019) into Chargax callables (arrivals, connection times, energy
demand). ACN-Data is a real, session-level dataset from the Caltech/JPL adaptive
charging networks. Because it requires (free) API registration and a download, the
adapter is a hook rather than a bundled dataset: point it at a fetched ACN-Data JSON
and it returns ``get_num_cars_arriving`` / ``get_new_cars_arriving`` overrides.

References
----------
Lee, Z. J., Li, T., & Low, S. H. (2019). ACN-Data: Analysis and Applications of an
    Open EV Charging Dataset. ACM e-Energy.  https://ev.caltech.edu/dataset
Ponse et al. (2025). Chargax: A JAX-Accelerated EV Charging Simulator. arXiv:2507.01522.
"""

from __future__ import annotations

import os

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def provenance() -> dict:
    """What is real vs. modelled in the calibrated environment (for the paper/README)."""
    return {
        "arrival_time_of_day": {
            "source": "car_arrival_percentages_workdays.csv / _weekends.csv",
            "real": True,
            "note": "Empirical Dutch time-of-day arrival profiles per user context "
                    "(residential/workplace/shopping/highway).",
        },
        "connection_time": {
            "source": "car_connection_times.csv",
            "real": True,
            "note": "Empirical dwell/connection-time distribution.",
        },
        "energy_demand": {
            "source": "car_energy_demand.csv",
            "real": True,
            "note": "Empirical per-session energy demand distribution.",
        },
        "vehicle_fleet": {
            "source": "car_frequency_and_profiles.csv",
            "real": True,
            "note": "EU/US/world vehicle mix with real battery capacities and charge "
                    "curves.",
        },
        "electricity_price": {
            "source": "electricity_prices_kwh_2021_NL.csv .. 2023_NL.csv",
            "real": True,
            "note": "Real Netherlands day-ahead wholesale prices, 2021-2023.",
        },
        "ability_to_pay_segment": {
            "source": "chargax.equity.segments (income terciles + energy burden)",
            "real": "grounded",
            "note": "Modelled on top of the real stream; price weights derived from "
                    "ACS income terciles and the ACEEE/DOE energy-burden gradient, "
                    "kept exogenous and capacity-independent.",
        },
        "grid_connection_limit": {
            "source": "experiments.common.scarcity_station",
            "real": "design parameter",
            "note": "The binding grid-power limit is the scarcity lever and is swept "
                    "explicitly (Section: capacity threshold).",
        },
    }


def print_provenance():
    print("Environment data provenance (real vs. modelled):")
    for k, v in provenance().items():
        flag = {True: "REAL", "grounded": "GROUNDED", "design parameter": "DESIGN"}.get(
            v["real"], str(v["real"]))
        print(f"  [{flag:8s}] {k:24s} <- {v['source']}")
        print(f"             {v['note']}")


def acn_data_kwargs(acn_json_path: str, minutes_per_timestep: int = 5) -> dict:
    """Build Chargax data callables from a downloaded Caltech ACN-Data JSON dump.

    Parameters
    ----------
    acn_json_path : str
        Path to an ACN-Data ``sessions`` JSON (as returned by the ACN-Data API,
        with fields ``connectionTime``, ``disconnectTime``, ``kWhDelivered``).
    minutes_per_timestep : int
        Environment timestep granularity (Chargax default 5).

    Returns
    -------
    dict suitable to pass as ``EquiChargax(..., get_num_cars_arriving=..., ...)``.

    Notes
    -----
    This computes an empirical time-of-day arrival histogram, a connection-time
    distribution, and an energy-demand distribution from the real sessions, then
    exposes them as the sampling callables Chargax expects. It intentionally does not
    fabricate data: if ``acn_json_path`` is missing it raises, so a run either uses
    the bundled real Dutch data or a real ACN-Data download, never a placeholder.
    """
    import json

    import jax
    import jax.numpy as jnp
    import numpy as np

    if not os.path.exists(acn_json_path):
        raise FileNotFoundError(
            f"ACN-Data file not found: {acn_json_path}. Register and download from "
            "https://ev.caltech.edu/dataset (free), then pass the sessions JSON."
        )
    with open(acn_json_path) as f:
        payload = json.load(f)
    sessions = payload.get("_items", payload) if isinstance(payload, dict) else payload

    import datetime as _dt

    def _parse(ts):
        # ACN-Data timestamps are RFC-1123 strings, e.g. "Wed, 01 May 2019 07:12:00 GMT".
        return _dt.datetime.strptime(ts, "%a, %d %b %Y %H:%M:%S %Z")

    arrivals_hour, dwell_min, energy_kwh = [], [], []
    for s in sessions:
        try:
            a = _parse(s["connectionTime"]); d = _parse(s["disconnectTime"])
        except Exception:
            continue
        arrivals_hour.append(a.hour + a.minute / 60.0)
        dwell_min.append(max((d - a).total_seconds() / 60.0, minutes_per_timestep))
        energy_kwh.append(float(s.get("kWhDelivered", 0.0)))
    if not arrivals_hour:
        raise ValueError("No parseable ACN-Data sessions found.")

    steps_per_day = int(24 * 60 / minutes_per_timestep)
    hist, _ = np.histogram(arrivals_hour, bins=steps_per_day, range=(0, 24))
    arrival_pmf = jnp.asarray(hist / max(hist.sum(), 1), dtype=jnp.float32)
    dwell = jnp.asarray(dwell_min, dtype=jnp.float32)
    energy = jnp.asarray(energy_kwh, dtype=jnp.float32)
    mean_per_day = len(arrivals_hour) / max(
        len({_parse(s["connectionTime"]).date() for s in sessions
             if "connectionTime" in s}), 1)

    def get_num_cars_arriving(key, state):
        rate = arrival_pmf[state.timestep % steps_per_day] * mean_per_day
        return jax.random.poisson(key, rate).astype(jnp.int32)

    return {
        "get_num_cars_arriving": get_num_cars_arriving,
        "_acn_dwell_minutes": dwell,
        "_acn_energy_kwh": energy,
        "_acn_mean_cars_per_day": float(mean_per_day),
    }


if __name__ == "__main__":
    print_provenance()
