r"""Adversarial seed-level (per-day) audit of the offline pillars (boxes 1 and 2).

These are the venue-carrying contributions and have had less scrutiny than the RL. This
script runs the oracle and the tariff analysis over many realized days and reports the
per-day DISTRIBUTIONS, not just the averages, and actively looks for days that break the
claims:

Box 1 (disparate-impact characterization):
  - Is the profit-optimal tier gap robust across days, or driven by a few outliers?
  - Is the BUDGET tier the systematically worst-served one, or does it flip across days?
  - Is delivered energy really identical across profit/utilitarian/maximin every day
    (the "same total energy, redistributed" claim), or only on average?
  - Is the revenue price of fairness stable per day?
Box 2 (tariff spread):
  - Is flat-tariff disparity always ~0.04 (income-neutral removes the systematic gap)?
  - Does a budget-only subsidy always shift the worst tier to mid?
  - Does a premium cap always leave budget worst-served?
  - LP success on every day (no infeasibility / solver failure).
"""

from __future__ import annotations

import json
import os

import jax
import numpy as np

from chargax.equity import oracle as O
from chargax.equity import tariffs as T
from chargax.equity import segments as SEG
from experiments.run_experiments import _ref_env

RESULTS = os.path.join(os.path.dirname(__file__), "results")
N_DAYS = 32
NAMES = ("budget", "mid", "premium")


def _stats(x):
    x = np.asarray(x, float)
    return {"median": float(np.median(x)), "iqr": float(np.subtract(*np.percentile(x, [75, 25]))),
            "min": float(x.min()), "max": float(x.max()), "mean": float(x.mean())}


def main():
    env = _ref_env(30.0, 8, 4)
    key = jax.random.PRNGKey(7)
    rows = {"profit": [], "utilitarian": [], "maximin": []}
    worst_tier_profit = []
    delivered = {"profit": [], "utilitarian": [], "maximin": []}
    revenue = {"profit": [], "maximin": []}
    lp_ok = 0
    # Box 1: per-day oracle.
    for i in range(N_DAYS):
        k = jax.random.fold_in(key, i)
        try:
            r = {o: O.oracle_welfare(env, k, objective=o) for o in ("profit", "utilitarian", "maximin")}
            lp_ok += 1
        except Exception as e:
            print(f"  day {i}: LP FAILED: {e}")
            continue
        for o in rows:
            gm = r[o]["group_means"]
            rows[o].append(max(gm) - min(gm))
            delivered[o].append(r[o]["delivered_kwh"])
        revenue["profit"].append(r["profit"]["revenue_value"])
        revenue["maximin"].append(r["maximin"]["revenue_value"])
        worst_tier_profit.append(int(np.argmin(r["profit"]["group_means"])))

    pof = [(revenue["profit"][j] - revenue["maximin"][j]) / (revenue["profit"][j] + 1e-9)
           for j in range(len(revenue["profit"]))]
    # Delivered-energy equality across objectives, per day (max abs deviation).
    dev = [max(abs(delivered["profit"][j] - delivered["utilitarian"][j]),
               abs(delivered["profit"][j] - delivered["maximin"][j]))
           for j in range(len(delivered["profit"]))]
    from collections import Counter
    wt = Counter(worst_tier_profit)

    audit = {"n_days": N_DAYS, "lp_success": lp_ok,
             "box1": {
                 "profit_optimal_disparity": _stats(rows["profit"]),
                 "maximin_disparity": _stats(rows["maximin"]),
                 "revenue_price_of_fairness_pct": _stats([p * 100 for p in pof]),
                 "worst_tier_under_profit_optimal": {NAMES[t]: wt.get(t, 0) for t in range(3)},
                 "delivered_energy_max_dev_across_objectives_kwh": _stats(dev),
             }}

    print("=== BOX 1 (oracle, %d days) ===" % N_DAYS)
    print("LP success:", lp_ok, "/", N_DAYS)
    print("profit-optimal disparity:", {k: round(v, 3) for k, v in audit["box1"]["profit_optimal_disparity"].items()})
    print("worst tier under profit-optimal (count of days):", audit["box1"]["worst_tier_under_profit_optimal"])
    print("delivered-energy max deviation across objectives (kWh):",
          {k: round(v, 2) for k, v in audit["box1"]["delivered_energy_max_dev_across_objectives_kwh"].items()})
    print("revenue PoF %:", {k: round(v, 1) for k, v in audit["box1"]["revenue_price_of_fairness_pct"].items()})

    # Box 2: per-day tariff interventions.
    base = SEG.PRICE_BY_GROUP
    flat_disp, flat_worst = [], []
    sub_disp, sub_worst = [], []   # budget subsidy to parity (weight 1.0)
    cap_disp, cap_worst = [], []   # premium cap to 1.0
    for i in range(N_DAYS):
        k = jax.random.fold_in(key, i)
        for w, dl, wl in [(T.flat(base), flat_disp, flat_worst),
                          (T.low_income_subsidy(0.4, base), sub_disp, sub_worst),
                          (T.cap_premium(1.0, base), cap_disp, cap_worst)]:
            r = O.oracle_welfare(env, k, objective="profit", tariff=w)
            gm = r["group_means"]
            dl.append(max(gm) - min(gm)); wl.append(NAMES[int(np.argmin(gm))])
    from collections import Counter as C
    audit["box2"] = {
        "flat_disparity": _stats(flat_disp),
        "flat_worst_tier": dict(C(flat_worst)),
        "budget_subsidy_parity_disparity": _stats(sub_disp),
        "budget_subsidy_parity_worst_tier": dict(C(sub_worst)),
        "premium_cap_1.0_disparity": _stats(cap_disp),
        "premium_cap_1.0_worst_tier": dict(C(cap_worst)),
    }
    print("\n=== BOX 2 (tariff, %d days) ===" % N_DAYS)
    print("flat disparity:", {k: round(v, 3) for k, v in audit["box2"]["flat_disparity"].items()},
          "worst tier:", audit["box2"]["flat_worst_tier"])
    print("budget-subsidy-to-parity disparity:", {k: round(v, 3) for k, v in audit["box2"]["budget_subsidy_parity_disparity"].items()},
          "worst tier:", audit["box2"]["budget_subsidy_parity_worst_tier"])
    print("premium-cap-1.0 disparity:", {k: round(v, 3) for k, v in audit["box2"]["premium_cap_1.0_disparity"].items()},
          "worst tier:", audit["box2"]["premium_cap_1.0_worst_tier"])

    json.dump(audit, open(os.path.join(RESULTS, "audit_offline.json"), "w"), indent=2)
    print("\nwrote", os.path.join(RESULTS, "audit_offline.json"))


if __name__ == "__main__":
    main()
