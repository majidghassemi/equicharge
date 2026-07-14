r"""Grid-capacity sweep: how much connection size removes the harm, and the threshold.

The paper recommends sizing the grid connection above the scarcity range. That number was
previously asserted from a stale run; this recomputes it fresh on the offline oracle, on the
same reference site and the same fixed day set as every other table, so the threshold is
regenerated rather than inherited.

For each grid limit we report two inequalities that answer two different questions with two
different instruments:
  between_tier_gap  = the between-tier disparate impact under the profit-optimal OFFLINE oracle
                      (day-averaged max-min per-tier satisfaction), the harm the paper audits;
  maxcharge_gini    = the within-population inequality under the DEPLOYED max-charge policy
                      (Gini of the inclusive per-driver satisfaction, rejected counted at zero),
                      i.e. the inequality a driver actually experiences on the ground.

As capacity rises the scarcity that forces rationing goes away and both fall toward an abundant
floor. We report the smallest connection at which each metric has closed 90% of its distance to
that floor (the "within 10% of floor" threshold). The oracle stream is grid-independent (admission
depends on charger count, not grid power), so one stream set is reused and only the LP grid limit
changes; the max-charge Gini is a fresh policy rollout at a station built for each grid. CPU-only.
"""

from __future__ import annotations

import json
import os

import jax
import numpy as np

from chargax.equity import oracle as O
from chargax.equity import metrics as M
from chargax.equity.baselines import max_charge_policy
from experiments.common import evaluate_policy
from experiments.run_experiments import _ref_env

RESULTS = os.path.join(os.path.dirname(__file__), "results")
N_DAYS = 32
STATUS_QUO = (0.6, 1.0, 1.5)
GRIDS = [16, 20, 24, 30, 40, 50, 60, 70, 80, 100, 120, 200, 600]  # kW; 600 = abundant floor


def main():
    env = _ref_env(30.0, 8, 4)
    key = jax.random.PRNGKey(7)
    mpt, hor, ng = env.minutes_per_timestep, env.max_episode_steps, int(env.n_groups)

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

    rows = []
    for gkw in GRIDS:
        # Oracle between-tier gap on the fixed stream set (change only the LP grid limit).
        gms = []
        for cust in streams:
            r = O.solve_oracle_lp(cust, float(gkw), mpt, hor, objective="profit",
                                  price_by_group=STATUS_QUO, n_groups=ng)
            gms.append(r["group_means"])
        avg = np.mean(np.array(gms), axis=0)
        # Deployed max-charge policy Gini: a fresh rollout at a station sized for this grid.
        env_g = _ref_env(float(gkw), 8, 4)
        mc = evaluate_policy(env_g, max_charge_policy, jax.random.PRNGKey(7), n_episodes=32)
        rows.append({"grid_kw": gkw,
                     "between_tier_gap": round(float(avg.max() - avg.min()), 4),
                     "maxcharge_gini": round(float(mc["gini"]), 4)})
        print(f"  {gkw:>4} kW: between-tier gap {rows[-1]['between_tier_gap']:.3f}  "
              f"max-charge Gini {rows[-1]['maxcharge_gini']:.3f}")

    def threshold(metric):
        vals = [r[metric] for r in rows]
        floor, peak = vals[-1], max(vals)
        target = floor + 0.10 * (peak - floor)  # closed 90% of the distance to the floor
        for r in rows:
            if r[metric] <= target:
                return r["grid_kw"], round(target, 4), round(floor, 4), round(peak, 4)
        return GRIDS[-1], round(target, 4), round(floor, 4), round(peak, 4)

    g_gap, t_gap, f_gap, p_gap = threshold("between_tier_gap")
    g_gini, t_gini, f_gini, p_gini = threshold("maxcharge_gini")

    out = {
        "day_set": {"oracle_days_used": D, "requested": N_DAYS, "seed": 7,
                    "site": "reference 16ch", "maxcharge_episodes": 32},
        "rows": rows,
        "threshold_between_tier_kw": g_gap,
        "threshold_maxcharge_gini_kw": g_gini,
        "between_tier_floor": f_gap, "between_tier_peak": p_gap,
        "maxcharge_gini_floor": f_gini, "maxcharge_gini_peak": p_gini,
        "interpretation": (
            f"The max-charge policy Gini reaches within 10% of its abundant floor by ~{g_gini} kW "
            f"and the profit-optimal between-tier gap by ~{g_gap} kW for this 16-charger site. "
            f"Above that range scarcity no longer drives the disadvantage."),
    }
    json.dump(out, open(os.path.join(RESULTS, "audit_capacity.json"), "w"), indent=2)

    print(f"\n=== CAPACITY SWEEP (oracle {D}/{N_DAYS} days + max-charge 32 eps, reference site) ===")
    print(f"threshold (within 10% of floor): max-charge Gini ~{g_gini} kW, "
          f"between-tier gap ~{g_gap} kW")
    print("wrote", os.path.join(RESULTS, "audit_capacity.json"))


if __name__ == "__main__":
    main()
