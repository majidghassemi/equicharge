r"""Survivorship check + n=10 robustness for the surviving RL result.

The 3M gate showed egal_l050 (lambda=0.5) beating max_charge on served-satisfaction,
but served-satisfaction is survivorship-biased (computed over ADMITTED drivers only).
If the learned policy wins by rejecting more marginal drivers, the advantage is an
artifact that the paper's own inclusive utility (rejected = 0) condemns. This script
redoes the head-to-head on the honest axes:

  profit (up)  |  INCLUSIVE mean satisfaction (up)  |  Gini over the inclusive
  population (down)  |  rejection rate (down / comparable)  |  inclusive welfare (up)

against the non-learned heuristics (the clean baseline; NOT our own lambda=0 run, which
is confounded by training dynamics). 10 seeds, full metrics, checkpoints saved.

Writes results/check.json.
"""

from __future__ import annotations

import json
import os
import time

import jax
import numpy as np

from chargax.equity import baselines as B
from chargax.equity import segments as SEG
from experiments.common import (evaluate_policy, make_env, make_trained_policy,
                                scarcity_station, train_ppo)

RESULTS = os.path.join(os.path.dirname(__file__), "results")
GROUP_KW = dict(n_groups=3, group_probs=SEG.GROUP_PROBS, price_by_group=SEG.PRICE_BY_GROUP)
CFG = dict(alpha=0.0, lam=0.5, outer="rawlsian")   # egal_l050, the surviving result
SEEDS = list(range(10))
BUDGET = 3_000_000
ENT = 0.01
# Honest head-to-head axes vs a baseline: (key, higher_is_better).
AXES = [("profit_mean", True), ("inclusive_mean_sat", True),
        ("gini", False), ("rejection_rate", False)]


def _full(m):
    """Keep the full inclusive metric dict (drop array fields)."""
    return {k: (float(v) if hasattr(v, "__float__") else v)
            for k, v in m.items() if not hasattr(v, "shape")}


def main():
    station = scarcity_station(grid_kw=30.0, n_evses=8)
    key = jax.random.PRNGKey(7)
    renv = make_env(station=station, num_discretization_levels=4, allow_discharging=False,
                    alpha=0.0, lam=0.5, outer="rawlsian", **GROUP_KW)
    out = {"budget": BUDGET, "ent_coef": ENT, "n_seeds": len(SEEDS), "config": "egal_l050",
           "baselines": {}, "seeds": []}
    for name in ("max_charge", "least_laxity", "proportional_fair", "saffe", "random"):
        out["baselines"][name] = _full(evaluate_policy(renv, B.POLICIES[name], key, n_episodes=24))
    mc = out["baselines"]["max_charge"]
    print(f"[max_charge] profit={mc['profit_mean']:.0f} incl_sat={mc['inclusive_mean_sat']:.3f} "
          f"gini={mc['gini']:.3f} reject={mc['rejection_rate']:.3f} welfare={mc['welfare_mean']:.3f}")

    ckpt = os.path.join(RESULTS, "checkpoints"); os.makedirs(ckpt, exist_ok=True)
    for seed in SEEDS:
        env = make_env(station=station, num_discretization_levels=4, allow_discharging=False,
                       **GROUP_KW, **CFG)
        t0 = time.time()
        agent = train_ppo(env, total_timesteps=BUDGET, num_envs=32, num_steps=288,
                          seed=seed, ent_coef=ENT, anneal_ent_coef=True)
        m = _full(evaluate_policy(env, make_trained_policy(agent), jax.random.PRNGKey(1000 + seed),
                                  n_episodes=24))
        m["seed"] = seed; m["sec"] = round(time.time() - t0, 1)
        out["seeds"].append(m)
        try:
            import equinox as eqx
            eqx.tree_serialise_leaves(os.path.join(ckpt, f"check_egal_l050_seed{seed}.eqx"), agent)
        except Exception as e:
            print(f"  (ckpt skip: {e})")
        print(f"[egal_l050 s{seed}] profit={m['profit_mean']:.0f} incl_sat={m['inclusive_mean_sat']:.3f} "
              f"gini={m['gini']:.3f} reject={m['rejection_rate']:.3f} welfare={m['welfare_mean']:.3f} "
              f"({m['sec']}s)")
        json.dump(out, open(os.path.join(RESULTS, "check.json"), "w"), indent=2)

    # Analysis: per-seed dominance vs max_charge on the honest axes; win rates; medians.
    seeds = out["seeds"]
    def med(k): return float(np.median([s[k] for s in seeds]))
    def wins(k, hib): return sum((s[k] > mc[k]) == hib for s in seeds)
    per_axis = {k: {"win_rate": wins(k, hib) / len(seeds), "median": med(k),
                    "maxcharge": mc[k]} for k, hib in AXES}
    dom4 = sum(all((s[k] > mc[k]) == hib for k, hib in AXES) for s in seeds)  # all 4 axes
    dom3 = sum(sum((s[k] > mc[k]) == hib for k, hib in AXES) >= 3 for s in seeds)
    rej_med = med("rejection_rate")
    # Survivorship verdict: inclusive service win AND rejection not materially worse.
    survives = (med("inclusive_mean_sat") > mc["inclusive_mean_sat"] and
                rej_med <= mc["rejection_rate"] + 0.02)
    out["analysis"] = {
        "per_axis_vs_maxcharge": per_axis,
        "dominate_all4_seeds": dom4, "dominate_ge3of4_seeds": dom3, "n_seeds": len(seeds),
        "median_rejection": rej_med, "maxcharge_rejection": mc["rejection_rate"],
        "median_inclusive_welfare": med("welfare_mean"), "maxcharge_inclusive_welfare": mc["welfare_mean"],
        "median_inclusive_sat": med("inclusive_mean_sat"), "maxcharge_inclusive_sat": mc["inclusive_mean_sat"],
        "SURVIVES_survivorship_check": bool(survives),
    }
    json.dump(out, open(os.path.join(RESULTS, "check.json"), "w"), indent=2)
    print("\n=== SURVIVORSHIP CHECK (vs max_charge, inclusive) ===")
    for k, hib in AXES:
        print(f"  {k}: median {per_axis[k]['median']:.3f} vs mc {mc[k]:.3f}  win_rate {per_axis[k]['win_rate']:.0%}")
    print(f"  inclusive welfare: median {med('welfare_mean'):.3f} vs mc {mc['welfare_mean']:.3f}")
    print(f"  dominate all 4 axes: {dom4}/{len(seeds)} seeds; >=3 of 4: {dom3}/{len(seeds)}")
    print(f"  rejection: {rej_med:.3f} vs max_charge {mc['rejection_rate']:.3f}")
    print(f"  SURVIVES (inclusive service win AND rejection not materially worse): {survives}")


if __name__ == "__main__":
    main()
