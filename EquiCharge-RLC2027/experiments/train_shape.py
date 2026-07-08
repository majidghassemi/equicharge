r"""Third condition: does SHAPE or CONTENT drive the RL gain?

The first ablation confounded two things: it replaced the welfare potential with a
potential that was BOTH monotone (no saturation) AND content-free (no tier/satisfaction).
So it can only show that a magnitude-matched *monotone* potential fails; it cannot
attribute the gain to welfare content rather than to the concave/saturating shape.

This condition isolates them. It uses a SATURATING but CONTENT-FREE potential:
per-connector delivered energy as a fraction of a CONSTANT reference, clipped to [0,1],
so it saturates exactly like satisfaction but references no per-driver desired target and
no tier. Magnitude-matched, lambda=0.5, 10 seeds.

Verdict:
- If it RECOVERS the gain (~ welfare on efficiency), the SHAPE (concave saturation) does
  the work, and the mechanism is "concave saturating shaping is a good scheduling prior
  under deadlines" -- a cleaner, more general claim than "welfare content".
- If it DOES NOT recover it (~ the monotone control, below max_charge), then the
  satisfaction/tier content matters and the stronger claim is earned.

Writes results/shape.json.
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


def calibrate(station, key, n_days=8):
    """Match the saturating control's terminal magnitude to the welfare potential's."""
    env = make_env(station=station, num_discretization_levels=4, allow_discharging=False,
                   alpha=0.0, lam=0.5, outer="rawlsian", shaping_mode="sat_throughput",
                   throughput_scale=1.0, **GROUP_KW)
    pol = B.POLICIES["max_charge"]

    def terminals(k):
        obs, state = env.reset_env(k)
        def step(carry, _):
            kk, st = carry
            kk, ka, ks = jax.random.split(kk, 3)
            a = pol(ka, env, st, None)
            _, ns = env.step_env(ks, st, a)
            return (kk, ns), None
        (_, final), _ = jax.lax.scan(step, (k, state), None, length=env.max_episode_steps)
        return env._welfare_potential(final), env._saturating_potential(final)

    ws, ss = [], []
    for i in range(n_days):
        w, s = jax.jit(terminals)(jax.random.fold_in(key, i))
        ws.append(float(w)); ss.append(float(s))
    scale = float(np.mean(ws) / (np.mean(ss) + 1e-9))
    print(f"[calibration] welfare terminal={np.mean(ws):.3f} sat terminal(scale1)={np.mean(ss):.3f}"
          f" -> throughput_scale={scale:.4f}")
    return scale


def main():
    station = scarcity_station(grid_kw=30.0, n_evses=8)
    key = jax.random.PRNGKey(7)
    scale = calibrate(station, key)
    out = {"budget": BUDGET, "ent_coef": ENT, "n_seeds": len(SEEDS), "throughput_scale": scale,
           "control": "sat_throughput (saturating, content-free)", "seeds": []}
    renv = make_env(station=station, num_discretization_levels=4, allow_discharging=False,
                    alpha=0.0, lam=0.5, outer="rawlsian", **GROUP_KW)
    out["baselines"] = {"max_charge": _full(evaluate_policy(renv, B.POLICIES["max_charge"], key, 24))}
    # References from prior runs.
    for name, f in [("welfare", "check.json"), ("monotone_throughput", "ablation.json")]:
        try:
            d = json.load(open(os.path.join(RESULTS, f)))
            out[name + "_reference"] = {k: float(np.median([s[k] for s in d["seeds"]]))
                                        for k in ("profit_mean", "inclusive_mean_sat", "gini",
                                                  "rejection_rate", "welfare_mean")}
        except Exception:
            out[name + "_reference"] = None

    ckpt = os.path.join(RESULTS, "checkpoints"); os.makedirs(ckpt, exist_ok=True)
    for seed in SEEDS:
        env = make_env(station=station, num_discretization_levels=4, allow_discharging=False,
                       alpha=0.0, lam=0.5, outer="rawlsian",
                       shaping_mode="sat_throughput", throughput_scale=scale, **GROUP_KW)
        t0 = time.time()
        agent = train_ppo(env, total_timesteps=BUDGET, num_envs=32, num_steps=288,
                          seed=seed, ent_coef=ENT, anneal_ent_coef=True)
        m = _full(evaluate_policy(env, make_trained_policy(agent), jax.random.PRNGKey(1000 + seed), 24))
        m["seed"] = seed; m["sec"] = round(time.time() - t0, 1)
        out["seeds"].append(m)
        try:
            import equinox as eqx
            eqx.tree_serialise_leaves(os.path.join(ckpt, f"shape_sat_seed{seed}.eqx"), agent)
        except Exception:
            pass
        print(f"[sat s{seed}] profit={m['profit_mean']:.0f} incl_sat={m['inclusive_mean_sat']:.3f} "
              f"gini={m['gini']:.3f} reject={m['rejection_rate']:.3f} welfare={m['welfare_mean']:.3f} ({m['sec']}s)")
        json.dump(out, open(os.path.join(RESULTS, "shape.json"), "w"), indent=2)

    S = out["seeds"]; mc = out["baselines"]["max_charge"]
    def med(k): return float(np.median([s[k] for s in S]))
    # Does the SATURATING content-free control recover the gain (beat max_charge on profit
    # AND inclusive sat, i.e. behave like welfare rather than like the monotone control)?
    recovers = (med("profit_mean") > mc["profit_mean"] and med("inclusive_mean_sat") > mc["inclusive_mean_sat"])
    wf = out.get("welfare_reference") or {}
    out["analysis"] = {
        "sat_median": {k: med(k) for k in ("profit_mean", "inclusive_mean_sat", "gini",
                                           "rejection_rate", "welfare_mean")},
        "welfare_reference": wf, "monotone_reference": out.get("monotone_throughput_reference"),
        "maxcharge": {k: mc[k] for k in ("profit_mean", "inclusive_mean_sat", "gini",
                                         "rejection_rate", "welfare_mean")},
        "saturating_content_free_recovers_gain": bool(recovers),
        "interpretation": ("RECOVERS -> the concave/saturating SHAPE drives the gain, not "
                           "welfare content; mechanism = 'saturating shaping is a good "
                           "scheduling prior'." if recovers else
                           "DOES NOT recover -> satisfaction/tier CONTENT matters beyond the "
                           "shape; the stronger 'welfare content' claim is earned."),
    }
    json.dump(out, open(os.path.join(RESULTS, "shape.json"), "w"), indent=2)
    print("\n=== SHAPE-vs-CONTENT VERDICT ===")
    print(f"saturating content-free median: profit {med('profit_mean'):.0f}, incl_sat "
          f"{med('inclusive_mean_sat'):.3f}, reject {med('rejection_rate'):.3f}, welfare {med('welfare_mean'):.3f}")
    if wf:
        print(f"welfare-shaped median:          profit {wf['profit_mean']:.0f}, incl_sat "
              f"{wf['inclusive_mean_sat']:.3f}, reject {wf['rejection_rate']:.3f}, welfare {wf['welfare_mean']:.3f}")
    mono = out.get("monotone_throughput_reference")
    if mono:
        print(f"monotone content-free median:   profit {mono['profit_mean']:.0f}, incl_sat "
              f"{mono['inclusive_mean_sat']:.3f}, reject {mono['rejection_rate']:.3f}, welfare {mono['welfare_mean']:.3f}")
    print(f"max_charge:                     profit {mc['profit_mean']:.0f}, incl_sat "
          f"{mc['inclusive_mean_sat']:.3f}, reject {mc['rejection_rate']:.3f}, welfare {mc['welfare_mean']:.3f}")
    print("=>", out["analysis"]["interpretation"])


if __name__ == "__main__":
    main()
