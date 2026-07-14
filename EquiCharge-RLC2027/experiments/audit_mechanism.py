r"""The tariff mechanism, established on ONE fixed, identically-filtered day set.

This run pins box-1 (the oracle objectives), the elasticity-invariance summary, and the
box-2 tariff levers to a single realized day set with a single filtering rule and a single
estimator, so every gap in the paper is the *same quantity computed the same way* (fixes the
0.537/0.556/0.558 and 0.280/0.374 inconsistencies across earlier analyses).

It then tests the real mechanism. The hypothesis is NOT "the harm scales with the margin
spread." It is:

    Under a linear revenue objective and scarcity, the profit optimum routes power up the
    willingness-to-pay ordering. The budget tier is systematically starved exactly when it is
    the STRICTLY lowest rank, invariant to how much lower. The disparate impact moves only
    when the budget tier stops being uniquely lowest.

The decisive evidence is an asymmetry that spread-magnitude cannot explain. We compare two
interventions that both compress the spread:

  * premium-capped-to-mid (0.6, 1.0, 1.0): spread 0.4, budget STILL uniquely lowest.
  * budget-tied-to-mid   (1.0, 1.0, 1.5): spread 0.5, budget NO LONGER uniquely lowest.

If the harm tracked spread magnitude, the smaller-spread config (premium cap, 0.4) would have
the smaller gap. If the harm tracks rank position, the premium cap keeps the large gap and the
budget-parity config (larger spread, 0.5) is the one that drops. The latter is the prediction.

Estimator: the paper's disparate-impact measure = max-min of the DAY-AVERAGED per-tier
satisfaction (systematic disadvantage), not the average of each day's own gap (which is
per-day noise). Under this estimator a facially neutral (flat) tariff reads ~0 because the
worst tier roams day to day, which is the correct behavior. RL-free, CPU-only.
"""

from __future__ import annotations

import json
import os

import jax
import numpy as np

from chargax.equity import oracle as O
from chargax.equity import segments as SEG
from experiments.run_experiments import _ref_env

RESULTS = os.path.join(os.path.dirname(__file__), "results")
NAMES = ("budget", "mid", "premium")
N_DAYS = 32
STATUS_QUO = tuple(SEG.PRICE_BY_GROUP)  # (0.6, 1.0, 1.5)

# Margin configs, chosen to separate rank position from spread magnitude.
CONFIGS = [
    ("status_quo_distinct",   (0.6, 1.0, 1.5)),  # three distinct ranks, budget uniquely lowest
    ("premium_capped_to_mid", (0.6, 1.0, 1.0)),  # spread 0.4, budget STILL uniquely lowest
    ("budget_tied_to_mid",    (1.0, 1.0, 1.5)),  # spread 0.5, budget NO LONGER uniquely lowest
    ("fully_flat",            (1.0, 1.0, 1.0)),   # no ranks -> scarcity floor
]


def _budget_uniquely_lowest(m):
    return bool(m[0] < min(m[1], m[2]))


def main():
    env = _ref_env(30.0, 8, 4)
    key = jax.random.PRNGKey(7)
    grid_kw = float(env.station.max_kw_throughput)
    mpt = env.minutes_per_timestep
    horizon = env.max_episode_steps
    ng = int(env.n_groups)

    def solve(cust, objective, margins):
        return O.solve_oracle_lp(cust, grid_kw, mpt, horizon, objective=objective,
                                 price_by_group=tuple(margins), n_groups=ng)

    # --- Build the ONE canonical, filtered day set (identical rule everywhere). ---
    streams = []
    for i in range(N_DAYS):
        k = jax.random.fold_in(key, i)
        cust = O.extract_arrival_stream(env, k)
        if not cust:
            continue
        pf = solve(cust, "profit", STATUS_QUO)
        if pf.get("degenerate") or min(pf.get("group_counts", [0])) == 0:
            continue
        streams.append(cust)
    D = len(streams)

    def day_averaged(margins, objective="profit"):
        """Day-averaged per-tier means and the disparate-impact gap (max-min of the average)."""
        gms = np.array([solve(c, objective, margins)["group_means"] for c in streams])
        avg = gms.mean(axis=0)
        return avg, float(avg.max() - avg.min()), int(np.argmin(avg))

    # --- Mechanism: four margin configs on the SAME day set, SAME estimator. ---
    mech = []
    for name, m in CONFIGS:
        avg, gap, worst = day_averaged(m)
        mech.append({
            "config": name, "margins": list(m), "spread": round(max(m) - min(m), 3),
            "budget_uniquely_lowest": _budget_uniquely_lowest(m),
            "tier_means": [round(x, 3) for x in avg], "gap": round(gap, 3),
            "worst_tier": NAMES[worst],
        })

    # --- Box-1 objectives on the SAME day set (canonical table numbers). ---
    box1 = {}
    for obj in ("profit", "utilitarian", "maximin", "egalitarian_equal"):
        avg, gap, worst = day_averaged(STATUS_QUO, objective=obj)
        # revenue (value-weighted delivered energy) under the status-quo margins, day-averaged.
        rev = np.mean([solve(c, obj, STATUS_QUO)["revenue_value"] for c in streams])
        box1[obj] = {"tier_means": [round(x, 3) for x in avg], "gap": round(gap, 3),
                     "worst_tier": NAMES[worst], "revenue": round(float(rev), 1)}
    # Price of fairness (revenue), per-day median, on the same day set.
    pof = []
    for c in streams:
        pr = solve(c, "profit", STATUS_QUO)["revenue_value"]
        mr = solve(c, "maximin", STATUS_QUO)["revenue_value"]
        pof.append(100.0 * (pr - mr) / (pr + 1e-9))
    pof_median = float(np.median(pof))

    # --- Elasticity invariance on the SAME day set + estimator (for the structural reframe). ---
    inv = []
    for eta in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        # Round to the 1-decimal precision the paper reports margins at, so the eta=0.6 row
        # is EXACTLY the committed reference (0.6,1.0,1.5) and its gap equals the box-1
        # status-quo gap, rather than differing by LP tie-breaking on 1.5 vs 1.499.
        m = SEG.derive_price_multipliers(elasticity=eta, round_to=1)
        avg, gap, worst = day_averaged(m)
        inv.append({"eta": eta, "margins": [round(x, 3) for x in m], "gap": round(gap, 3),
                    "budget_uniquely_lowest": _budget_uniquely_lowest(m),
                    "worst_tier": NAMES[worst]})

    out = {
        "day_set": {"requested": N_DAYS, "used_after_filter": D, "seed": 7,
                    "env": "ref 8ch/30kW, 4 disc", "estimator": "max-min of day-averaged tier means"},
        "status_quo_gap": mech[0]["gap"],
        "subsidy_to_parity_gap": mech[2]["gap"],
        "box1": box1, "revenue_price_of_fairness_pct_median": round(pof_median, 1),
        "mechanism": mech,
        "elasticity_invariance": inv,
        "verdict": None,
    }
    # Verdict: does rank beat spread? premium-cap (smaller spread) keeps a larger gap than
    # budget-parity (larger spread) -> spread magnitude does NOT predict the gap; rank does.
    cap = next(x for x in mech if x["config"] == "premium_capped_to_mid")
    par = next(x for x in mech if x["config"] == "budget_tied_to_mid")
    rank_beats_spread = (cap["spread"] < par["spread"]) and (cap["gap"] > par["gap"])
    out["verdict"] = {
        "premium_cap_spread": cap["spread"], "premium_cap_gap": cap["gap"],
        "budget_parity_spread": par["spread"], "budget_parity_gap": par["gap"],
        "rank_position_beats_spread_magnitude": bool(rank_beats_spread),
        "interpretation": (
            "CONFIRMED: the premium cap has the SMALLER spread yet the LARGER gap, so the harm "
            "cannot be a function of spread magnitude. It tracks rank position: the budget tier "
            "is starved exactly while it is uniquely lowest, and is relieved only when it stops "
            "being uniquely lowest (budget-parity or full flattening)."
            if rank_beats_spread else
            "NOT confirmed on this day set; inspect the per-config gaps before rewriting box 2."),
    }
    json.dump(out, open(os.path.join(RESULTS, "audit_mechanism.json"), "w"), indent=2)

    print(f"=== CANONICAL DAY SET: {D}/{N_DAYS} days (seed 7), day-averaged estimator ===\n")
    print("BOX 1 (status-quo margins 0.6/1.0/1.5):")
    for obj, r in box1.items():
        print(f"  {obj:18s} tiers={r['tier_means']} gap={r['gap']} worst={r['worst_tier']} rev={r['revenue']}")
    print(f"  revenue price of fairness (median): {pof_median:.1f}%\n")
    print("MECHANISM (rank position vs spread magnitude):")
    print(f"  {'config':22s} {'margins':17s} {'spread':>6} {'gap':>6} {'uniq-low':>8} worst")
    for r in mech:
        print(f"  {r['config']:22s} {str(r['margins']):17s} {r['spread']:>6} {r['gap']:>6} "
              f"{str(r['budget_uniquely_lowest']):>8} {r['worst_tier']}")
    print("\nELASTICITY INVARIANCE (same day set/estimator):")
    for r in inv:
        print(f"  eta={r['eta']:.1f} margins={r['margins']} gap={r['gap']} "
              f"uniq-low={r['budget_uniquely_lowest']} worst={r['worst_tier']}")
    v = out["verdict"]
    print(f"\nVERDICT: premium-cap spread {v['premium_cap_spread']} gap {v['premium_cap_gap']} "
          f"vs budget-parity spread {v['budget_parity_spread']} gap {v['budget_parity_gap']}")
    print("rank position beats spread magnitude:", v["rank_position_beats_spread_magnitude"])
    print("\nwrote", os.path.join(RESULTS, "audit_mechanism.json"))


if __name__ == "__main__":
    main()
