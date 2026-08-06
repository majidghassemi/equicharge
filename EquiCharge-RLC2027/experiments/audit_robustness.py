r"""Robustness of the disparate-impact result across genuinely different station configs.

The entire box-1 finding (value-weighted allocation systematically starves the budget
tier) has so far been established at a single operating point: 16 chargers, 30 kW,
residential arrivals, Chargax defaults. A reviewer will ask whether it is a property of
that one configuration. This reruns the box-1 oracle audit at several genuinely different
sites -- varying charger count, grid size, user/demand profile, arrival rate, and vehicle
fleet -- and reports whether the systematic disadvantage persists.

If budget-is-worst on most days holds across configs, the claim graduates from "true at
this operating point" to structural. RL-free, CPU-only (LP + short rollouts).

A2 also asks whether the *tariff-spread lever* (box 2) survives across configs: at every
site we recompute the profit-optimal tier gap under the status-quo tiered tariff, an
income-neutral (flat) tariff, and a budget-only subsidy to parity, and check that the
lever behaves the same way everywhere it bites (income-neutral removes the gap; a
budget-only subsidy does not, it relocates the shortfall). For an independent, US,
session-level calibration we optionally add a fifth config driven by the public Caltech
ACN-Data (Lee et al. 2019); set ``EQUICHARGE_ACN_JSON`` to a downloaded sessions dump to
include it, otherwise it is skipped with a note (the adapter never fabricates data).
"""

from __future__ import annotations

import json
import os

import jax
import numpy as np

from chargax.equity import oracle as O
from chargax.equity import segments as SEG
from chargax.equity import tariffs as T
from experiments.common import acn_env_or_none, make_env, scarcity_station

RESULTS = os.path.join(os.path.dirname(__file__), "results")
GROUP_KW = dict(n_groups=3, group_probs=SEG.GROUP_PROBS, price_by_group=SEG.PRICE_BY_GROUP)
NAMES = ("budget", "mid", "premium")
N_DAYS = 32

# Genuinely different sites: charger count, grid kW, and the data-generating process
# (user profile, arrivals per day, vehicle fleet). Grids chosen to stay power-scarce.
CONFIGS = [
    ("reference_16ch_30kW_residential_eu",
     dict(n_evses=8, grid_kw=30.0),
     dict(car_profile="eu", user_profile="residential", average_cars_per_day=30, grid_price_dataset="2023_NL")),
    ("workplace_24ch_55kW_us",
     dict(n_evses=12, grid_kw=55.0),
     dict(car_profile="us", user_profile="workplace", average_cars_per_day=45, grid_price_dataset="2023_NL")),
    ("shopping_8ch_16kW_world",
     dict(n_evses=4, grid_kw=16.0),
     dict(car_profile="world", user_profile="shopping", average_cars_per_day=20, grid_price_dataset="2023_NL")),
    ("highway_20ch_45kW_eu_highrate",
     dict(n_evses=10, grid_kw=45.0),
     dict(car_profile="eu", user_profile="highway", average_cars_per_day=60, grid_price_dataset="2023_NL")),
]


def _stats(x):
    x = np.asarray(x, float)
    return dict(median=float(np.median(x)), iqr=float(np.subtract(*np.percentile(x, [75, 25]))),
                min=float(x.min()), max=float(x.max()))


def _solve(cust, env, objective, tariff=None):
    """Solve the oracle LP on an already-extracted stream (avoids re-rolling out)."""
    price = tuple(env.price_by_group) if tariff is None else tuple(tariff)
    return O.solve_oracle_lp(
        cust, grid_limit_kw=float(env.station.max_kw_throughput),
        minutes_per_timestep=env.minutes_per_timestep,
        horizon_steps=env.max_episode_steps, objective=objective,
        price_by_group=price, n_groups=int(env.n_groups))


def audit_config(env, key):
    """Audit one site. Uses the SAME estimator as the reference-site mechanism run
    (``audit_mechanism.py``): the disparate impact is max-min of the DAY-AVERAGED per-tier
    satisfaction (a systematic disadvantage), so every gap in the paper is one quantity.
    ``budget_worst_fraction`` and the revenue price of fairness stay per-day, since they are
    fractions/ratios over days and are the scarcity-boundary evidence A2 is actually for."""
    from collections import Counter
    base = tuple(env.price_by_group)
    worst, pof, dev, ncust = [], [], [], []
    # Per-day tier-mean vectors, day-averaged after the loop (SAME days for every tariff).
    sq_gms, flat_gms, sub_gms = [], [], []
    subsidy_to_parity = max(base[1] - base[0], 0.0)
    ok = 0
    for i in range(N_DAYS):
        k = jax.random.fold_in(key, i)
        try:
            cust = O.extract_arrival_stream(env, k)
            if not cust:
                continue
            pf = _solve(cust, env, "profit")
            mm = _solve(cust, env, "maximin")
            ut = _solve(cust, env, "utilitarian")
        except Exception:
            continue
        # Skip degenerate days and days where any tier had NO arrivals, so worst-tier
        # attribution reflects starvation rather than absence. A tier that is present but
        # gets 0 satisfaction (genuinely starved) is kept -- that is real disparate impact.
        if pf.get("degenerate") or min(pf.get("group_counts", [0])) == 0:
            continue
        ok += 1
        sq_gms.append(pf["group_means"])
        worst.append(int(np.argmin(pf["group_means"])))  # per-day worst (for the fraction)
        pof.append((pf["revenue_value"] - mm["revenue_value"]) / (pf["revenue_value"] + 1e-9))
        dev.append(max(abs(pf["delivered_kwh"] - mm["delivered_kwh"]),
                       abs(pf["delivered_kwh"] - ut["delivered_kwh"])))
        ncust.append(pf.get("n_customers", 0))
        # box-2: same realized day, different tariff.
        flat_gms.append(_solve(cust, env, "profit", tariff=T.flat(base))["group_means"])
        sub_gms.append(_solve(cust, env, "profit",
                              tariff=T.low_income_subsidy(subsidy_to_parity, base))["group_means"])

    def _davg(gms):
        """Day-averaged tier means -> (gap, worst-tier-index)."""
        if not gms:
            return 0.0, 0
        a = np.mean(np.array(gms), axis=0)
        return float(a.max() - a.min()), int(np.argmin(a))

    sq_gap, sq_worst = _davg(sq_gms)
    flat_gap, flat_worst = _davg(flat_gms)
    sub_gap, sub_worst = _davg(sub_gms)
    wc = Counter(worst)
    return {
        "lp_success": ok, "n_days": N_DAYS,
        "mean_customers_per_day": float(np.mean(ncust)) if ncust else 0.0,
        "profit_optimal_disparity_dayavg": round(sq_gap, 4),
        "worst_tier_days": {NAMES[t]: wc.get(t, 0) for t in range(3)},
        "budget_worst_fraction": wc.get(0, 0) / max(ok, 1),
        "revenue_pof_pct": _stats([p * 100 for p in pof]),
        "delivered_energy_max_dev_kwh": _stats(dev),
        # box-2 tariff lever at this config (day-averaged, matching the reference run):
        "tariff_lever": {
            "status_quo_disparity": round(sq_gap, 4),
            "income_neutral_disparity": round(flat_gap, 4),
            "income_neutral_worst_tier": NAMES[flat_worst],
            "budget_subsidy_parity_disparity": round(sub_gap, 4),
            "budget_subsidy_parity_worst_tier": NAMES[sub_worst],
            "subsidy_to_parity": subsidy_to_parity,
        },
    }


def _env_for(layout, data):
    station = scarcity_station(grid_kw=layout["grid_kw"], n_evses=layout["n_evses"])
    return make_env(station=station, num_discretization_levels=4, allow_discharging=False,
                    alpha=0.0, lam=1.0, outer="rawlsian", data_kwargs=data, **GROUP_KW)


def _acn_config():
    """Optional US session-level config from a downloaded Caltech ACN-Data dump.

    Returns (name, env, provenance) or None if EQUICHARGE_ACN_JSON is unset, so the
    sweep runs unchanged without it and never fabricates data. A dump that is present
    but unusable raises: quietly falling back to the bundled Dutch data would report
    an "ACN-Data" row that contains none.
    """
    got = acn_env_or_none(grid_kw=30.0, n_evses=8, num_disc=4)
    if got is None:
        return None
    env, prov = got
    return ("acn_data_caltech_8ch_30kW", env, prov)


def main():
    key = jax.random.PRNGKey(7)
    out = {"n_days": N_DAYS, "configs": {}}
    envs = [(name, _env_for(layout, data), layout) for name, layout, data in CONFIGS]
    acn = _acn_config()
    if acn:
        envs.append((acn[0], acn[1], acn[2]))
        out["acn_provenance"] = acn[2]

    for name, env, layout in envs:
        r = audit_config(env, key)
        out["configs"][name] = {**r, "layout": layout}
        tl = r["tariff_lever"]
        print(f"[{name}] cust/day={r['mean_customers_per_day']:.0f} "
              f"disparity(dayavg)={r['profit_optimal_disparity_dayavg']:.3f} "
              f"budget-worst={r['worst_tier_days']['budget']}/{r['lp_success']} "
              f"({100*r['budget_worst_fraction']:.0f}%) "
              f"PoF med={r['revenue_pof_pct']['median']:.1f}% | "
              f"tariff: statusquo={tl['status_quo_disparity']:.3f} "
              f"income-neutral={tl['income_neutral_disparity']:.3f} "
              f"budget-subsidy={tl['budget_subsidy_parity_disparity']:.3f}")
        json.dump(out, open(os.path.join(RESULTS, "audit_robustness.json"), "w"), indent=2)

    # Verdict 1: is budget systematically worst across ALL configs? (box 1)
    fracs = {n: c["budget_worst_fraction"] for n, c in out["configs"].items()}
    structural = all(f >= 0.75 for f in fracs.values())
    # Verdict 2: does the tariff-spread lever behave the same way wherever the harm is
    # PRICE-DRIVEN? The harm exists only where power binds (positive revenue PoF). At those
    # sites the claim is (a) an income-neutral tariff removes a large share of the gap (the
    # price-driven part; the residual is scarcity-driven and only capacity removes it), and
    # (b) it beats a budget-only subsidy, which relocates rather than removes the shortfall.
    # Where power does not bind the gap is small and the lever is moot -- we mark it so
    # rather than count it as a pass or a failure.
    def _reduction(a, b):
        return (a - b) / a if a > 1e-9 else 0.0
    lever = {}
    for n, c in out["configs"].items():
        tl = c["tariff_lever"]
        sq = tl["status_quo_disparity"]
        pof = c["revenue_pof_pct"].get("median", 0.0)
        neutral_red = _reduction(sq, tl["income_neutral_disparity"])
        subsidy_red = _reduction(sq, tl["budget_subsidy_parity_disparity"])
        power_bound = pof > 1.0  # a material revenue price of fairness => rationing by tier pays
        lever[n] = {
            "power_bound": bool(power_bound),
            "status_quo_gap": sq,
            "income_neutral_reduction_pct": round(100 * neutral_red, 1),
            "budget_subsidy_reduction_pct": round(100 * subsidy_red, 1),
            "neutral_beats_subsidy": bool(neutral_red > subsidy_red + 0.05),
        }
    # At every power-bound site: income-neutral removes a large share (>40%) of the gap AND
    # beats the budget-only subsidy. That is the consistent lever behavior we claim.
    bound_sites = [n for n, v in lever.items() if v["power_bound"]]
    lever_consistent = all(
        lever[n]["income_neutral_reduction_pct"] >= 40.0 and lever[n]["neutral_beats_subsidy"]
        for n in bound_sites) if bound_sites else False
    out["verdict"] = {
        "budget_worst_fraction_by_config": fracs,
        "disparate_impact_structural": bool(structural),
        "tariff_lever_by_config": lever,
        "tariff_lever_consistent_at_power_bound_sites": bool(lever_consistent),
        "interpretation": ("Budget worst-served on >=75% of days at EVERY config -> the "
                           "disparate impact is STRUCTURAL, not an artifact of one operating "
                           "point." if structural else
                           "Budget-worst is SCARCITY-SPECIFIC: it holds at power-bound sites "
                           "and fades where power is not binding; the revenue price of "
                           "fairness tracks it. Report the scoped claim."),
        "tariff_interpretation": ("At every power-bound site the income-neutral tariff removes "
                                  "the price-driven share of the gap (>=40%) and beats a "
                                  "budget-only subsidy, which relocates the shortfall; the "
                                  "residual is scarcity-driven and only capacity removes it."
                                  if lever_consistent else
                                  "The tariff lever is not uniform across power-bound sites; "
                                  "report the exception."),
    }
    json.dump(out, open(os.path.join(RESULTS, "audit_robustness.json"), "w"), indent=2)
    print("\n=== ROBUSTNESS VERDICT ===")
    for n, f in fracs.items():
        lv = lever[n]
        tag = ("power-bound: income-neutral -%.0f%% vs subsidy -%.0f%%"
               % (lv["income_neutral_reduction_pct"], lv["budget_subsidy_reduction_pct"])
               ) if lv["power_bound"] else "power not binding (gap small, lever moot)"
        print(f"  {n}: budget worst {100*f:.0f}% of days | {tag}")
    print("disparate impact structural (budget worst >=75% at every config):", structural)
    print("tariff spread lever consistent at power-bound sites:", lever_consistent)


if __name__ == "__main__":
    main()
