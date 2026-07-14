r"""Are the two levers actually orthogonal? Put one number on the complementary-levers claim.

The paper now says pricing and capacity address two distinct harms: income-neutral pricing removes
the systematic BETWEEN-TIER disparate impact, grid capacity removes the WITHIN-POPULATION scarcity
inequality. That is a cleaner story than the residual it replaced, but it asserts a separation the
shopping result shows can be violated (the two harms coexist). So we check the cross-effects
directly, on ONE fixed stream set, before letting the separation stand as the policy headline:

  * pricing cross-effect: at fixed capacity (30 kW), does flattening prices (status-quo -> flat)
    move WITHIN-population inequality, or only the between-tier gap?
  * capacity cross-effect: at fixed pricing (status-quo), does raising capacity (30 -> 90 kW)
    move the BETWEEN-tier gap, or only within-population inequality?

If both cross-effects are ~0 the levers are orthogonal and we can say so with a number. If a lever
moves the other harm too, the separation is only approximate and the paper must say approximate.

Measures (both on the profit-optimal oracle allocation, day-averaged / pooled over one day set):
  between_tier_gap   = max-min of the day-averaged per-tier mean satisfaction (the disparate impact)
  within_tier_gini   = mean over tiers of the Gini of per-customer satisfaction WITHIN that tier
                       (inequality that is NOT between-tier -- the within-population component)
  overall_gini       = Gini of the pooled per-customer satisfaction (for context; mixes both)

The stream (which cars arrive/are admitted) depends on charger count, not grid power, so the SAME
realized streams are reused and only the LP's grid limit / margins change -- a clean isolation.
"""

from __future__ import annotations

import json
import os

import jax
import numpy as np

from chargax.equity import oracle as O
from chargax.equity import metrics as M
from chargax.equity import tariffs as T
from experiments.run_experiments import _ref_env

RESULTS = os.path.join(os.path.dirname(__file__), "results")
NAMES = ("budget", "mid", "premium")
N_DAYS = 32
STATUS_QUO = (0.6, 1.0, 1.5)
FLAT = (1.0, 1.0, 1.0)


def main():
    env = _ref_env(30.0, 8, 4)
    key = jax.random.PRNGKey(7)
    mpt, hor, ng = env.minutes_per_timestep, env.max_episode_steps, int(env.n_groups)

    # One fixed, filtered stream set (same rule as the mechanism run).
    streams = []
    for i in range(N_DAYS):
        k = jax.random.fold_in(key, i)
        cust = O.extract_arrival_stream(env, k)
        if not cust:
            continue
        pf = O.solve_oracle_lp(cust, 30.0, mpt, hor, objective="profit",
                               price_by_group=STATUS_QUO, n_groups=ng)
        if pf.get("degenerate") or min(pf.get("group_counts", [0])) == 0:
            continue
        streams.append(cust)
    D = len(streams)

    def measures(margins, grid_kw):
        gms, sat_all, grp_all = [], [], []
        for cust in streams:
            r = O.solve_oracle_lp(cust, grid_kw, mpt, hor, objective="profit",
                                  price_by_group=tuple(margins), n_groups=ng)
            gms.append(r["group_means"])
            sat_all.append(np.asarray(r["satisfaction"], float))
            grp_all.append(np.asarray(r["groups"], int))
        avg = np.mean(np.array(gms), axis=0)
        between = float(avg.max() - avg.min())
        sat = np.concatenate(sat_all); grp = np.concatenate(grp_all)
        within = float(np.mean([M.gini(sat[grp == g]) for g in range(ng) if np.any(grp == g)]))
        overall = float(M.gini(sat))
        return {"between_tier_gap": round(between, 3),
                "within_tier_gini": round(within, 3),
                "overall_gini": round(overall, 3)}

    cells = {
        "statusquo_30kW": measures(STATUS_QUO, 30.0),
        "flat_30kW":      measures(FLAT, 30.0),
        "statusquo_90kW": measures(STATUS_QUO, 90.0),
        "flat_90kW":      measures(FLAT, 90.0),
    }

    def d(a, b, kmetric):
        return round(cells[a][kmetric] - cells[b][kmetric], 3)

    # Pricing lever at fixed 30 kW: status-quo -> flat.
    pricing = {
        "d_between_tier_gap": d("statusquo_30kW", "flat_30kW", "between_tier_gap"),
        "d_within_tier_gini": d("statusquo_30kW", "flat_30kW", "within_tier_gini"),
    }
    # Capacity lever at fixed status-quo pricing: 30 -> 90 kW.
    capacity = {
        "d_between_tier_gap": d("statusquo_30kW", "statusquo_90kW", "between_tier_gap"),
        "d_within_tier_gini": d("statusquo_30kW", "statusquo_90kW", "within_tier_gini"),
    }
    # Orthogonality: a lever is "clean" on the OTHER harm if its cross-effect is small
    # relative to its main effect.
    def clean(main, cross):
        return abs(cross) <= 0.2 * max(abs(main), 1e-9)
    pricing_clean = clean(pricing["d_between_tier_gap"], pricing["d_within_tier_gini"])
    capacity_clean = clean(capacity["d_within_tier_gini"], capacity["d_between_tier_gap"])

    out = {
        "day_set": {"used": D, "requested": N_DAYS, "seed": 7},
        "cells": cells,
        "pricing_lever_30kW_sq_to_flat": pricing,
        "capacity_lever_sq_30_to_90kW": capacity,
        "pricing_orthogonal_to_within": bool(pricing_clean),
        "capacity_orthogonal_to_between": bool(capacity_clean),
        "orthogonal": bool(pricing_clean and capacity_clean),
        "interpretation": None,
    }
    out["interpretation"] = (
        "Levers are ORTHOGONAL: pricing moves only the between-tier gap and capacity moves only "
        "within-population inequality; the two-harms separation is exact."
        if out["orthogonal"] else
        "Levers are NOT fully orthogonal; the separation is APPROXIMATE. Report the cross-effects: "
        "state exactly which lever also moves the other harm and by how much.")

    json.dump(out, open(os.path.join(RESULTS, "audit_levers.json"), "w"), indent=2)

    print(f"=== LEVER ORTHOGONALITY ({D}/{N_DAYS} days, reference site) ===\n")
    print(f"{'cell':16s} {'between-tier':>12} {'within-gini':>12} {'overall-gini':>12}")
    for name, m in cells.items():
        print(f"{name:16s} {m['between_tier_gap']:>12} {m['within_tier_gini']:>12} {m['overall_gini']:>12}")
    print(f"\nPRICING lever (30kW, status-quo->flat): "
          f"d(between)={pricing['d_between_tier_gap']:+.3f}  d(within)={pricing['d_within_tier_gini']:+.3f}")
    print(f"CAPACITY lever (status-quo, 30->90kW):  "
          f"d(between)={capacity['d_between_tier_gap']:+.3f}  d(within)={capacity['d_within_tier_gini']:+.3f}")
    print(f"\npricing orthogonal to within-pop: {pricing_clean} | "
          f"capacity orthogonal to between-tier: {capacity_clean}")
    print("=>", out["interpretation"])
    print("\nwrote", os.path.join(RESULTS, "audit_levers.json"))


if __name__ == "__main__":
    main()
