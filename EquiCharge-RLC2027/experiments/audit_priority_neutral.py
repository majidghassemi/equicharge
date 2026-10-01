"""Cost of priority-neutral (tier-blind) allocation under today's tiered prices.

Priority neutrality keeps the tiered prices but forbids allocating power by tier. The
tier-blind benchmark is the utilitarian allocation, which maximizes total satisfaction and
never reads a session's tier. This script solves, on the canonical 256-day set used by
audit_mechanism.py, the revenue optimum and the utilitarian allocation under the status-quo
margins, and reports the revenue cost of neutrality the way the paper reports the price of
fairness: per day, (R_rev - R_util) / R_rev, summarized by the median over days.

Writes experiments/results/audit_priority_neutral.json.
"""
from __future__ import annotations

import json
import os

import jax
import numpy as np

from chargax.equity import oracle as O
from experiments.run_experiments import _ref_env

N_DAYS = int(os.environ.get("N_DAYS", "256"))
STATUS_QUO = (0.6, 1.0, 1.5)
RESULTS = os.path.join(os.path.dirname(__file__), "results")


def main():
    env = _ref_env(30.0, 8, 4)
    key = jax.random.PRNGKey(7)
    grid_kw = float(env.station.max_kw_throughput)
    mpt, horizon, ng = env.minutes_per_timestep, env.max_episode_steps, int(env.n_groups)

    def solve(cust, objective):
        return O.solve_oracle_lp(cust, grid_kw, mpt, horizon, objective=objective,
                                 price_by_group=STATUS_QUO, n_groups=ng)

    rows = []
    for i in range(N_DAYS):
        cust = O.extract_arrival_stream(env, jax.random.fold_in(key, i))
        if not cust:
            continue
        rev = solve(cust, "profit")
        if rev.get("degenerate") or min(rev.get("group_counts", [0])) == 0:
            continue
        rows.append({o: solve(cust, o) for o in ("utilitarian", "maximin")} | {"profit": rev})
    D = len(rows)

    def summary(obj):
        gm = np.array([r[obj]["group_means"] for r in rows]).mean(axis=0)
        return dict(tier_means=[round(float(x), 4) for x in gm],
                    gamma=round(float(gm.max() - gm.min()), 4),
                    revenue_mean=round(float(np.mean([r[obj]["revenue_value"] for r in rows])), 2),
                    delivered_kwh_mean=round(float(np.mean([r[obj]["delivered_kwh"] for r in rows])), 2))

    rev = np.array([r["profit"]["revenue_value"] for r in rows])
    out = {"n_days": D, "margins": list(STATUS_QUO), "objectives": {}}
    for obj in ("profit", "utilitarian", "maximin"):
        out["objectives"][obj] = summary(obj)
    for obj, name in (("utilitarian", "price_of_neutrality_pct"), ("maximin", "price_of_fairness_pct")):
        alt = np.array([r[obj]["revenue_value"] for r in rows])
        ok = rev > 1e-9
        pct = 100 * (rev[ok] - alt[ok]) / rev[ok]
        out[name] = dict(median=round(float(np.median(pct)), 2),
                         iqr=round(float(np.percentile(pct, 75) - np.percentile(pct, 25)), 2),
                         ratio_of_means=round(float(100 * (rev.mean() - alt.mean()) / rev.mean()), 2))
    path = os.path.join(RESULTS, "audit_priority_neutral.json")
    json.dump(out, open(path, "w"), indent=1)
    print(json.dumps(out, indent=1))
    print("wrote", path)


if __name__ == "__main__":
    main()
