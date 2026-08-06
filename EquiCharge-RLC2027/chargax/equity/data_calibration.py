r"""Real-data calibration of the charging environment (provenance + ACN-Data hook).

The distributions the environment samples from -- arrival times, dwell/connection
times, per-session energy demand, and electricity prices -- are **real, not
synthetic**. They are the empirically-derived datasets shipped with Chargax
(``chargax/data/``), summarized in :func:`provenance`. The only construct added on
top for the equity study is the ability-to-pay segmentation, which is itself
grounded in public income/energy-burden data (see :mod:`chargax.equity.segments`).

For reviewers who want an independent, US, session-level calibration we also provide
:func:`acn_scenario`, an adapter that turns the public **Caltech ACN-Data**
(Lee et al., 2019) into Chargax callables (arrivals, connection times, energy
demand). ACN-Data is a real, session-level dataset from the Caltech/JPL adaptive
charging networks. Because it needs a download, the adapter is a hook rather than a
bundled dataset: point it at a fetched ACN-Data JSON and it returns
``get_num_cars_arriving`` / ``get_new_cars_arriving``.

These must be passed to the environment as **top-level fields**. Chargax reads only
``car_profile`` / ``user_profile`` / ``average_cars_per_day`` / ``grid_price_dataset``
out of ``default_data_kwargs``, so callables placed in that dict are silently dropped
and ``__post_init__`` rebuilds the bundled Dutch loaders instead -- a config labelled
as an ACN calibration would then contain no ACN data at all.
:func:`experiments.common.acn_env_or_none` wires them up and asserts they landed.

References
----------
Lee, Z. J., Li, T., & Low, S. H. (2019). ACN-Data: Analysis and Applications of an
    Open EV Charging Dataset. ACM e-Energy.  https://ev.caltech.edu/dataset
Ponse et al. (2025). Chargax: A JAX-Accelerated EV Charging Simulator. arXiv:2507.01522.
"""

from __future__ import annotations

import os
from functools import lru_cache as _lru_cache

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


#: Default calibration window for the Caltech site. ACN-Data continues past the
#: March 2020 campus closure, where session volume collapses to near zero. Averaging
#: arrivals over the closure would silently turn a power-scarce site into an
#: unconstrained one, so the default window ends before it. Pass ``date_range=None``
#: to use every session in the dump.
ACN_PRECOVID_WINDOW = ("2018-04-25", "2020-03-01")


@_lru_cache(maxsize=4)
def _parse_acn_sessions(acn_json_path, minutes_per_timestep=5, date_range=None,
                        energy_field="requested_then_delivered"):
    """Parse an ACN-Data dump into paired per-session records.

    Returns ``(records, meta)`` where ``records`` is a list of
    ``(arrival_hour, is_workday, dwell_minutes, energy_kwh, date)`` and ``meta``
    carries the provenance counts the audit reports.
    """
    import datetime as _dt
    import json

    if not os.path.exists(acn_json_path):
        raise FileNotFoundError(
            f"ACN-Data file not found: {acn_json_path}. Download from "
            "https://ev.caltech.edu/dataset (free), then pass the sessions JSON."
        )
    with open(acn_json_path) as f:
        payload = json.load(f)
    sessions = payload.get("_items", payload) if isinstance(payload, dict) else payload

    def _parse(ts):
        # ACN-Data timestamps are RFC-1123 strings, e.g. "Wed, 01 May 2019 07:12:00 GMT".
        return _dt.datetime.strptime(ts, "%a, %d %b %Y %H:%M:%S %Z")

    lo = hi = None
    if date_range is not None:
        lo = _dt.date.fromisoformat(date_range[0])
        hi = _dt.date.fromisoformat(date_range[1])

    records = []
    n_total = len(sessions)
    n_unparsed = n_out_of_window = n_nonpositive = 0
    n_requested = n_delivered_fallback = 0
    for s in sessions:
        try:
            a = _parse(s["connectionTime"])
            d = _parse(s["disconnectTime"])
        except Exception:
            n_unparsed += 1
            continue
        if lo is not None and not (lo <= a.date() < hi):
            n_out_of_window += 1
            continue
        dwell = (d - a).total_seconds() / 60.0
        # Prefer the driver's own stated demand: the satisfaction metric is delivered
        # vs. *each driver's own target*, and kWhDelivered is an outcome of the
        # (possibly power-limited) session, so using it as the target would bake the
        # incumbent controller's rationing into the demand distribution.
        kwh = None
        ui = s.get("userInputs")
        if ui:
            try:
                kwh = float(ui[0]["kWhRequested"])
                n_requested += 1
            except (KeyError, TypeError, ValueError, IndexError):
                kwh = None
        if kwh is None:
            try:
                kwh = float(s.get("kWhDelivered", 0.0))
                n_delivered_fallback += 1
            except (TypeError, ValueError):
                kwh = 0.0
        if dwell <= 0 or kwh <= 0:
            n_nonpositive += 1
            continue
        records.append((a.hour + a.minute / 60.0,
                        a.weekday() < 5,
                        max(dwell, float(minutes_per_timestep)),
                        kwh,
                        a.date()))
    if not records:
        raise ValueError(
            f"No usable ACN-Data sessions in {acn_json_path} "
            f"(parsed {n_total}, unparsed {n_unparsed}, out of window "
            f"{n_out_of_window}, non-positive dwell/energy {n_nonpositive}).")

    dates = {r[4] for r in records}
    meta = {
        "sessions_in_dump": n_total,
        "sessions_used": len(records),
        "sessions_unparsed": n_unparsed,
        "sessions_out_of_window": n_out_of_window,
        "sessions_nonpositive": n_nonpositive,
        "energy_from_kWhRequested": n_requested,
        "energy_from_kWhDelivered_fallback": n_delivered_fallback,
        "date_range_used": [str(min(dates)), str(max(dates))],
        "distinct_days": len(dates),
        "distinct_workdays": len({d for d in dates if d.weekday() < 5}),
        "distinct_weekend_days": len({d for d in dates if d.weekday() >= 5}),
        "energy_field_policy": energy_field,
    }
    return records, meta


def acn_scenario(station, acn_json_path, *, minutes_per_timestep: int = 5,
                 car_profile: str = "us", seed: int = 0,
                 date_range=ACN_PRECOVID_WINDOW, charge_sensitive_p: float = 0.1):
    """Build real Chargax arrival + session callables from a Caltech ACN-Data dump.

    Mirrors :func:`chargax._default_data_loaders.build_default_scenario`, but every
    demand-side distribution comes from the ACN sessions rather than the bundled
    Dutch CSVs:

    * **arrivals** -- an empirical per-timestep arrival rate, computed separately for
      workdays and weekends (a campus site has a strong weekday pattern) by dividing
      the binned session counts by the number of distinct observed days of that type,
      then pre-sampled as Poisson draws exactly as the default loader does;
    * **dwell time** -- the real ``disconnectTime - connectionTime``;
    * **energy demand** -- the driver's stated ``kWhRequested`` where present.

    Dwell and energy are sampled **as a pair from the same session**, preserving the
    real within-session correlation that the default loader (which samples the two
    independently from separate CSVs) cannot represent.

    The vehicle fleet still comes from Chargax's US fleet mix: ACN-Data records
    sessions, not vehicle models, so battery capacities and charge curves are not
    identifiable from it. That is the one modelled component here and it is reported
    as such in the returned provenance.

    Returns
    -------
    (get_num_cars_arriving, get_new_cars_arriving, provenance)
        The two callables are passed to ``EquiChargax(...)`` as **top-level fields**,
        not via ``default_data_kwargs`` -- Chargax only reads ``car_profile`` /
        ``user_profile`` / ``average_cars_per_day`` out of that dict, so callables
        placed there are silently ignored and the env falls back to the Dutch data.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np

    from chargax._default_data_loaders import _load_car_profiles

    records, _meta = _parse_acn_sessions(
        acn_json_path, minutes_per_timestep=minutes_per_timestep, date_range=date_range)
    meta = dict(_meta)  # the parse is lru_cached; never mutate the shared dict

    steps_per_day = int(24 * 60 / minutes_per_timestep)
    hours = np.array([r[0] for r in records])
    is_wd = np.array([r[1] for r in records])
    dwell_min = np.array([r[2] for r in records], dtype=np.float32)
    energy_kwh = np.array([r[3] for r in records], dtype=np.float32)

    # Per-timestep expected arrivals = binned counts / number of observed days of
    # that type. This is a rate per day, so it needs no separate cars-per-day scale.
    def _rate(mask, n_days):
        hist, _ = np.histogram(hours[mask], bins=steps_per_day, range=(0, 24))
        return hist / max(n_days, 1)

    wd_rate = _rate(is_wd, meta["distinct_workdays"])
    we_rate = _rate(~is_wd, meta["distinct_weekend_days"])
    meta["mean_sessions_per_workday"] = float(wd_rate.sum())
    meta["mean_sessions_per_weekend_day"] = float(we_rate.sum())
    # A dump covering no days of one type yields an all-zero rate for it, and every
    # simulated day of that type then has no arrivals at all. That is the honest
    # reading of the data, but it silently shrinks the usable day set, so say so.
    for lbl, n in (("workday", meta["distinct_workdays"]),
                   ("weekend day", meta["distinct_weekend_days"])):
        if n == 0:
            meta.setdefault("warnings", []).append(
                f"no {lbl}s observed in this window; simulated {lbl}s will have zero "
                f"arrivals and be filtered as degenerate")

    N_PRESAMPLE = 1000
    key = jax.random.PRNGKey(seed)
    k_wd, k_we, k1, k2, k4, k5 = jax.random.split(key, 6)

    weekday_data = jax.random.poisson(
        k_wd, lam=jnp.asarray(wd_rate), shape=(N_PRESAMPLE, steps_per_day))
    weekend_data = jax.random.poisson(
        k_we, lam=jnp.asarray(we_rate), shape=(N_PRESAMPLE, steps_per_day))
    stacked = (weekday_data, weekend_data)

    def get_num_cars_arriving(key, state):
        arrival_means = jax.lax.select(
            state.is_workday, stacked[0][:, state.timestep], stacked[1][:, state.timestep])
        randint = jax.random.randint(key, (), 0, arrival_means.shape[0])
        return arrival_means[randint]

    # --- session callable: ACN dwell/energy pairs on a US vehicle fleet ---
    car_data = jnp.array(_load_car_profiles(car_profile))
    car_frequencies, car_profiles = car_data[:, 0], car_data[:, 1:]
    num_chargers = station.num_chargers

    car_indices = jax.random.categorical(
        k1, car_frequencies, shape=(N_PRESAMPLE, num_chargers))
    car_profiles_sampled = car_profiles[car_indices]

    # One index per slot -> dwell and energy come from the SAME real session.
    sess_idx = jax.random.randint(k2, (N_PRESAMPLE, num_chargers), 0, len(records))
    connection_times_sampled = jnp.asarray(dwell_min)[sess_idx]
    energy_demands_sampled = jnp.asarray(energy_kwh)[sess_idx]

    car_desired_battery_percentage = jax.random.uniform(
        k4, (N_PRESAMPLE, num_chargers), minval=0.8, maxval=0.95)
    car_desired_kw = car_profiles_sampled[..., 1] * car_desired_battery_percentage
    car_battery_now_kw = jnp.clip(
        car_desired_kw - energy_demands_sampled,
        0.03 * car_profiles_sampled[..., 1], car_profiles_sampled[..., 1])
    charge_sensitive = jax.random.bernoulli(
        k5, charge_sensitive_p, shape=(N_PRESAMPLE, num_chargers))

    presampled_flat_evse = station.evses_flat.replace(
        car_ac_absolute_max_charge_rate_kw=car_profiles_sampled[..., 2],
        car_ac_optimal_charge_threshold=car_profiles_sampled[..., 0],
        car_dc_absolute_max_charge_rate_kw=car_profiles_sampled[..., 3],
        car_dc_optimal_charge_threshold=car_profiles_sampled[..., 0],
        car_battery_capacity_kw=car_profiles_sampled[..., 1],
        car_time_till_leave=connection_times_sampled.astype(int),
        car_battery_now_kw=car_battery_now_kw,
        car_desired_battery_percentage=car_desired_battery_percentage,
        charge_sensitive=charge_sensitive,
        car_arrival_battery_kw=car_battery_now_kw,
    )

    def get_new_cars_arriving(key, state):
        num_samples = presampled_flat_evse.car_battery_capacity_kw.shape[0]
        n_chargers = presampled_flat_evse.car_battery_capacity_kw.shape[1]
        random_indices = jax.random.randint(key, (n_chargers,), 0, num_samples)
        charger_indices = jnp.arange(n_chargers)

        def _sample(x):
            if hasattr(x, "ndim") and x.ndim == 2:
                return x[random_indices, charger_indices]
            return x

        return jax.tree.map(_sample, presampled_flat_evse)

    provenance = {
        "source": "Caltech ACN-Data (Lee et al. 2019), site=caltech",
        # Basename only: this provenance is written into committed result JSONs, and the
        # paper is under double-blind review, so a local absolute path would carry the
        # author's username into a reviewer-visible artifact.
        "path": os.path.basename(acn_json_path),
        "real_from_acn": ["arrival_time_of_day (workday/weekend separately)",
                          "arrival_rate_per_day", "connection_time", "energy_demand"],
        "modelled": ["vehicle_fleet (Chargax US mix; ACN-Data records sessions, not "
                     "vehicle models)",
                     "ability_to_pay_segment (chargax.equity.segments)"],
        "dwell_energy_paired_from_same_session": True,
        "median_dwell_minutes": float(np.median(dwell_min)),
        "median_energy_kwh": float(np.median(energy_kwh)),
        **meta,
    }
    return get_num_cars_arriving, get_new_cars_arriving, provenance


if __name__ == "__main__":
    print_provenance()
