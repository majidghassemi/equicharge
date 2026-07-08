r"""Ablation: is the RL efficiency gain the welfare CONTENT, or generic dense shaping?

The mechanism claim hedges that we do not isolate whether the welfare content or any
dense potential-based signal drives PPO's efficiency gain. This closes the hedge. We
retrain at lambda=0.5 with the welfare potential replaced by a CONTENT-FREE dense
potential (cumulative delivered energy / grid utilization -- no tier, no per-driver
satisfaction information), matched in magnitude to the welfare potential, 10 seeds, and
compare on the same inclusive metrics to the welfare-shaped controller (check.json).

Verdict logic:
- If the efficiency gain (profit, inclusive sat, rejection) SURVIVES with the
  content-free potential, the gain is generic dense shaping -> the hedge becomes a
  finding (welfare content is not what helps online).
- If it VANISHES, the welfare content matters after all, and the controller is more
  interesting than "reward shaping".

Writes results/ablation.json.
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
SEEDS = list(range(10))
BUDGET = 3_000_000
ENT = 0.01


def _full(m):
    return {k: (float(v) if hasattr(v, "__float__") else v)
            for k, v in m.items() if not hasattr(v, "shape")}


def calibrate_throughput_scale(station, key, n_days=8):
    """Set throughput_scale so the content-free potential's terminal magnitude matches
    the welfare potential's, on the max_charge reference policy (matched magnitude)."""
    env = make_env(station=station, num_discretization_levels=4, allow_discharging=False,
                   alpha=0.0, lam=0.5, outer="rawlsian", **GROUP_KW)
    pol = B.POLICIES["max_charge"]

    def terminal_potentials(k):
        obs, state = env.reset_env(k)
        def step(carry, _):
            kk, st = carry
            kk, ka, ks = jax.random.split(kk, 3)
            a = pol(ka, env, st, None)
            _, ns = env.step_env(ks, st, a)
            return (kk, ns), None
        (_, final), _ = jax.lax.scan(step, (k, state), None, length=env.max_episode_steps)
        wpot = env._welfare_potential(final)
        tpot = env._throughput_potential(final)  # scale=1 here
        return wpot, tpot

    ws, ts = [], []
    for i in range(n_days):
        w, t = jax.jit(terminal_potentials)(jax.random.fold_in(key, i))
        ws.append(float(w)); ts.append(float(t))
    scale = float(np.mean(ws) / (np.mean(ts) + 1e-9))
    print(f"[calibration] welfare terminal={np.mean(ws):.3f} throughput terminal(scale1)="
          f"{np.mean(ts):.3f} -> throughput_scale={scale:.3f}")
    return scale


def main():
    station = scarcity_station(grid_kw=30.0, n_evses=8)
    key = jax.random.PRNGKey(7)
    scale = calibrate_throughput_scale(station, key)

    # Reference: the welfare-shaped result and max_charge (from check.json if present).
    out = {"budget": BUDGET, "ent_coef": ENT, "n_seeds": len(SEEDS),
           "throughput_scale": scale, "control": "throughput (content-free)", "seeds": []}
    renv = make_env(station=station, num_discretization_levels=4, allow_discharging=False,
                    alpha=0.0, lam=0.5, outer="rawlsian", **GROUP_KW)
    out["baselines"] = {"max_charge": _full(evaluate_policy(renv, B.POLICIES["max_charge"], key, 24))}
    try:
        chk = json.load(open(os.path.join(RESULTS, "check.json")))
        out["welfare_reference"] = {k: float(np.median([s[k] for s in chk["seeds"]]))
                                    for k in ("profit_mean", "inclusive_mean_sat", "gini",
                                              "rejection_rate", "welfare_mean")}
    except Exception:
        out["welfare_reference"] = None

    ckpt = os.path.join(RESULTS, "checkpoints"); os.makedirs(ckpt, exist_ok=True)
    for seed in SEEDS:
        env = make_env(station=station, num_discretization_levels=4, allow_discharging=False,
                       alpha=0.0, lam=0.5, outer="rawlsian",
                       shaping_mode="throughput", throughput_scale=scale, **GROUP_KW)
        t0 = time.time()
        agent = train_ppo(env, total_timesteps=BUDGET, num_envs=32, num_steps=288,
                          seed=seed, ent_coef=ENT, anneal_ent_coef=True)
        m = _full(evaluate_policy(env, make_trained_policy(agent), jax.random.PRNGKey(1000 + seed), 24))
        m["seed"] = seed; m["sec"] = round(time.time() - t0, 1)
        out["seeds"].append(m)
        try:
            import equinox as eqx
            eqx.tree_serialise_leaves(os.path.join(ckpt, f"ablation_throughput_seed{seed}.eqx"), agent)
        except Exception:
            pass
        print(f"[throughput s{seed}] profit={m['profit_mean']:.0f} incl_sat={m['inclusive_mean_sat']:.3f} "
              f"gini={m['gini']:.3f} reject={m['rejection_rate']:.3f} welfare={m['welfare_mean']:.3f} ({m['sec']}s)")
        json.dump(out, open(os.path.join(RESULTS, "ablation.json"), "w"), indent=2)

    # Verdict: does the efficiency gain survive with the content-free potential?
    S = out["seeds"]; mc = out["baselines"]["max_charge"]; wr = out.get("welfare_reference") or {}
    def med(k): return float(np.median([s[k] for s in S]))
    def winrate(k, hib): return sum((s[k] > mc[k]) == hib for s in S) / len(S)
    axes = [("profit_mean", True), ("inclusive_mean_sat", True), ("gini", False), ("rejection_rate", False)]
    survives = (med("inclusive_mean_sat") > mc["inclusive_mean_sat"] and
                med("profit_mean") > mc["profit_mean"] and
                med("rejection_rate") <= mc["rejection_rate"] + 0.02)
    out["analysis"] = {
        "throughput_median": {k: med(k) for k, _ in axes} | {"welfare_mean": med("welfare_mean")},
        "welfare_median": wr, "maxcharge": {k: mc[k] for k, _ in axes} | {"welfare_mean": mc["welfare_mean"]},
        "throughput_winrate_vs_maxcharge": {k: winrate(k, hib) for k, hib in axes},
        "efficiency_gain_survives_content_free_shaping": bool(survives),
        "interpretation": ("SURVIVES -> gain is generic dense shaping (welfare content not required); "
                           "hedge becomes a finding." if survives else
                           "VANISHES -> welfare content matters; controller is more than shaping."),
    }
    json.dump(out, open(os.path.join(RESULTS, "ablation.json"), "w"), indent=2)
    print("\n=== ABLATION VERDICT ===")
    print(f"content-free (throughput) median: profit {med('profit_mean'):.0f}, incl_sat "
          f"{med('inclusive_mean_sat'):.3f}, reject {med('rejection_rate'):.3f}, welfare {med('welfare_mean'):.3f}")
    if wr:
        print(f"welfare-shaped median:            profit {wr['profit_mean']:.0f}, incl_sat "
              f"{wr['inclusive_mean_sat']:.3f}, reject {wr['rejection_rate']:.3f}, welfare {wr['welfare_mean']:.3f}")
    print(f"max_charge:                       profit {mc['profit_mean']:.0f}, incl_sat "
          f"{mc['inclusive_mean_sat']:.3f}, reject {mc['rejection_rate']:.3f}, welfare {mc['welfare_mean']:.3f}")
    print(f"efficiency gain survives content-free shaping: {survives}")
    print(out["analysis"]["interpretation"])


if __name__ == "__main__":
    main()
