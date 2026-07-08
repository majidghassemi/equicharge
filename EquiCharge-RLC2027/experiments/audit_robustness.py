r"""Robustness of the disparate-impact result across genuinely different station configs.

The entire box-1 finding (value-weighted allocation systematically starves the budget
tier) has so far been established at a single operating point: 16 chargers, 30 kW,
residential arrivals, Chargax defaults. A reviewer will ask whether it is a property of
that one configuration. This reruns the box-1 oracle audit at several genuinely different
sites -- varying charger count, grid size, user/demand profile, arrival rate, and vehicle
fleet -- and reports whether the systematic disadvantage persists.

If budget-is-worst on most days holds across configs, the claim graduates from "true at
this operating point" to structural. RL-free, CPU-only (LP + short rollouts).
"""

from __future__ import annotations

import json
import os

import jax
import numpy as np

from chargax.equity import oracle as O
from chargax.equity import segments as SEG
from experiments.common import make_env, scarcity_station

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


def audit_config(n_evses, grid_kw, data_kwargs, key):
    station = scarcity_station(grid_kw=grid_kw, n_evses=n_evses)
    env = make_env(station=station, num_discretization_levels=4, allow_discharging=False,
                   alpha=0.0, lam=1.0, outer="rawlsian", data_kwargs=data_kwargs, **GROUP_KW)
    disp, worst, pof, dev, ncust = [], [], [], [], []
    ok = 0
    for i in range(N_DAYS):
        k = jax.random.fold_in(key, i)
        try:
            pf = O.oracle_welfare(env, k, objective="profit")
            mm = O.oracle_welfare(env, k, objective="maximin")
            ut = O.oracle_welfare(env, k, objective="utilitarian")
        except Exception:
            continue
        # Skip degenerate days and days where any tier had NO arrivals, so worst-tier
        # attribution reflects starvation rather than absence. A tier that is present but
        # gets 0 satisfaction (genuinely starved) is kept -- that is real disparate impact.
        if pf.get("degenerate") or min(pf.get("group_counts", [0])) == 0:
            continue
        ok += 1
        gm = pf["group_means"]
        disp.append(max(gm) - min(gm))
        worst.append(int(np.argmin(gm)))
        pof.append((pf["revenue_value"] - mm["revenue_value"]) / (pf["revenue_value"] + 1e-9))
        dev.append(max(abs(pf["delivered_kwh"] - mm["delivered_kwh"]),
                       abs(pf["delivered_kwh"] - ut["delivered_kwh"])))
        ncust.append(pf.get("n_customers", 0))
    from collections import Counter
    wc = Counter(worst)
    return {
        "lp_success": ok, "n_days": N_DAYS,
        "mean_customers_per_day": float(np.mean(ncust)) if ncust else 0.0,
        "profit_optimal_disparity": _stats(disp),
        "worst_tier_days": {NAMES[t]: wc.get(t, 0) for t in range(3)},
        "budget_worst_fraction": wc.get(0, 0) / max(ok, 1),
        "revenue_pof_pct": _stats([p * 100 for p in pof]),
        "delivered_energy_max_dev_kwh": _stats(dev),
    }


def main():
    key = jax.random.PRNGKey(7)
    out = {"n_days": N_DAYS, "configs": {}}
    for name, layout, data in CONFIGS:
        r = audit_config(layout["n_evses"], layout["grid_kw"], data, key)
        out["configs"][name] = {**r, "layout": layout}
        print(f"[{name}] cust/day={r['mean_customers_per_day']:.0f} "
              f"disparity med={r['profit_optimal_disparity']['median']:.3f} "
              f"budget-worst={r['worst_tier_days']['budget']}/{r['lp_success']} "
              f"({100*r['budget_worst_fraction']:.0f}%) "
              f"PoF med={r['revenue_pof_pct']['median']:.1f}% "
              f"energy-dev={r['delivered_energy_max_dev_kwh']['max']:.2f}")
        json.dump(out, open(os.path.join(RESULTS, "audit_robustness.json"), "w"), indent=2)

    # Verdict: is budget systematically worst across ALL configs?
    fracs = {n: c["budget_worst_fraction"] for n, c in out["configs"].items()}
    structural = all(f >= 0.75 for f in fracs.values())
    out["verdict"] = {
        "budget_worst_fraction_by_config": fracs,
        "disparate_impact_structural": bool(structural),
        "interpretation": ("Budget worst-served on >=75% of days at EVERY config -> the "
                           "disparate impact is STRUCTURAL, not an artifact of one operating "
                           "point." if structural else
                           "Budget-worst does NOT hold at every config -> the result is "
                           "operating-point dependent; report the dependence."),
    }
    json.dump(out, open(os.path.join(RESULTS, "audit_robustness.json"), "w"), indent=2)
    print("\n=== ROBUSTNESS VERDICT ===")
    for n, f in fracs.items():
        print(f"  {n}: budget worst {100*f:.0f}% of days")
    print("disparate impact structural (budget worst >=75% at every config):", structural)


if __name__ == "__main__":
    main()
