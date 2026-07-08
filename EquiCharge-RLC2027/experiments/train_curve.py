r"""Learning-curve diagnostic: does the learned policy improve with training budget?

At the committed 250k-timestep budget the learned policies are dominated by the
`random` baseline. This script trains the two headline configs (profit and
egalitarian) at increasing timestep budgets and evaluates each, so we can state
whether the policies are still improving (undertrained) or have plateaued below the
heuristics. Writes ``results/learning_curve.json`` and prints the random reference.
"""

from __future__ import annotations

import json
import os
import time

import jax

from chargax.equity import baselines as B
from chargax.equity import segments as SEG
from experiments.common import (evaluate_policy, make_env, make_trained_policy,
                                scarcity_station, train_ppo)

RESULTS = os.path.join(os.path.dirname(__file__), "results")
GROUP_KW = dict(n_groups=3, group_probs=SEG.GROUP_PROBS, price_by_group=SEG.PRICE_BY_GROUP)
BUDGETS = [250_000, 1_000_000, 3_000_000]
CONFIGS = {"profit_ppo": dict(alpha=0.0, lam=0.0, outer="rawlsian"),
           "egal_l100": dict(alpha=0.0, lam=1.0, outer="rawlsian")}
SEEDS = [0]  # single-seed directional diagnostic


def main():
    station = scarcity_station(grid_kw=30.0, n_evses=8)
    key = jax.random.PRNGKey(7)
    # Random reference (the bar the learned policies must clear).
    renv = make_env(station=station, num_discretization_levels=4, allow_discharging=False,
                    alpha=0.0, lam=1.0, outer="rawlsian", **GROUP_KW)
    rnd = evaluate_policy(renv, B.POLICIES["random"], key, n_episodes=24)
    mc = evaluate_policy(renv, B.POLICIES["max_charge"], key, n_episodes=24)
    out = {"reference": {"random": {"profit": rnd["profit_mean"], "gini": rnd["gini"],
                                    "sat": rnd["served_mean_sat"]},
                         "max_charge": {"profit": mc["profit_mean"], "gini": mc["gini"],
                                        "sat": mc["served_mean_sat"]}},
           "curves": {}}
    for name, cfg in CONFIGS.items():
        out["curves"][name] = []
        for budget in BUDGETS:
            for seed in SEEDS:
                env = make_env(station=station, num_discretization_levels=4,
                               allow_discharging=False, **GROUP_KW, **cfg)
                t0 = time.time()
                agent = train_ppo(env, total_timesteps=budget, num_envs=32,
                                  num_steps=288, seed=seed)
                pol = make_trained_policy(agent)
                m = evaluate_policy(env, pol, jax.random.PRNGKey(1000 + seed), n_episodes=24)
                row = {"budget": budget, "seed": seed, "profit": m["profit_mean"],
                       "gini": m["gini"], "sat": m["served_mean_sat"],
                       "worst10": m["worst10_sat"], "seconds": round(time.time() - t0, 1)}
                out["curves"][name].append(row)
                print(f"[{name}] budget={budget:>9} profit={m['profit_mean']:.1f} "
                      f"gini={m['gini']:.3f} sat={m['served_mean_sat']:.3f} "
                      f"(random profit={rnd['profit_mean']:.1f}) ({row['seconds']}s)")
                json.dump(out, open(os.path.join(RESULTS, "learning_curve.json"), "w"), indent=2)
    print("done ->", os.path.join(RESULTS, "learning_curve.json"))


if __name__ == "__main__":
    main()
