r"""Main experiment runner for "The Price of a Fair Charge".

Produces the data for the paper. The trustworthy, RL-independent core (the offline
oracle characterization, the tariff-design and grid-capacity policy findings, and
the scarcity contrast) is cheap and runs on CPU; the learned-policy benchmark is a
multi-seed PPO sweep (GPU-friendly, ``--quick`` for a smoke run).

Sections
--------
1. **Offline oracle** -- the clairvoyant LP characterization of the
   efficiency-equity tradeoff (profit / utilitarian / true segment-maximin /
   strict-equalization), with the **revenue-based** price of fairness. RL-free.
2. **Rate-design finding** -- a low-income-subsidy sweep on the profit-optimal
   allocation, reporting the subsidy that closes the equity gap (Sec. tariffs).
3. **Grid-capacity threshold** -- sweep the grid connection and report the capacity
   at which scarcity-induced inequality vanishes (infrastructure planning).
4. **Baselines + learned policies** -- one unified table (max-charge, proportional-
   fair, least-laxity, SAFFE, random, and the PPO welfare sweep) on matched metrics
   with multi-seed dispersion for the learned policies.
5. **Endogeneity audit** -- quantifies the reject-to-look-fair loophole the utility
   design closes (Sec. 3.2).
6. **Scarcity contrast** -- inequity opens under a scarce grid, closes when abundant.

Run ``python -m experiments.run_experiments --quick`` for a fast smoke run, or
``python -m experiments.run_experiments --oracle-only`` for just the RL-free core.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import jax
import numpy as np

from chargax.equity import baselines as B
from chargax.equity import oracle as O
from chargax.equity import segments as SEG
from chargax.equity import tariffs as T
from experiments.common import (
    aggregate_seeds,
    evaluate_policy,
    make_env,
    make_trained_policy,
    scarcity_station,
    train_ppo,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# V2G discharging is disabled for the core experiments: it roughly doubles the
# (factored) action space and introduces a discharge "trap" that destabilizes
# learning. It can be re-enabled to study the richer-dynamics regime.
ALLOW_DISCHARGING = False

# Ability-to-pay segmentation, grounded in income terciles + energy burden
# (chargax.equity.segments). The price heterogeneity is what creates the genuine
# efficiency-equity tension: a profit maximizer is incentivized to favour the
# high-paying ("premium") segment under scarcity.
N_GROUPS = 3
GROUP_PROBS = SEG.GROUP_PROBS
PRICE_BY_GROUP = SEG.PRICE_BY_GROUP  # affordability-grounded (budget/mid/premium)
GROUP_KW = dict(n_groups=N_GROUPS, group_probs=GROUP_PROBS, price_by_group=PRICE_BY_GROUP)

# Metrics carried through the multi-seed aggregation.
_SEED_KEYS = ["profit_mean", "worst10_sat", "min_sat", "gini", "atkinson",
              "served_mean_sat", "inclusive_mean_sat", "rejection_rate",
              "welfare_mean", "group_disparity"]


def _save(results: dict, name: str = "results.json"):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, name), "w") as f:
        json.dump(results, f, indent=2)


def _ref_env(grid_kw, n_evses, num_disc, **extra):
    station = scarcity_station(grid_kw=grid_kw, n_evses=n_evses)
    return make_env(station=station, num_discretization_levels=num_disc,
                    allow_discharging=ALLOW_DISCHARGING, alpha=0.0, lam=1.0,
                    outer="rawlsian", **GROUP_KW, **extra)


# ============================================================ 1. offline oracle
def eval_oracle(env, key, n_days: int) -> dict:
    """Clairvoyant LP characterization. Reports per-segment satisfaction, the
    revenue each allocation earns, and the price of fairness in *revenue* and in
    *mean satisfaction* (the corrected, non-leveling maximin)."""
    objs = ("profit", "utilitarian", "maximin", "egalitarian_equal")
    agg = {o: {"mean": [], "min": [], "disp": [], "ws": [], "rev": [], "e": [],
               "groups": []} for o in objs}
    for i in range(n_days):
        k = jax.random.fold_in(key, i)
        for o in objs:
            r = O.oracle_welfare(env, k, objective=o)
            agg[o]["mean"].append(r["mean_satisfaction"])
            agg[o]["min"].append(r["min_satisfaction"])
            agg[o]["disp"].append(r["group_disparity"])
            agg[o]["ws"].append(r["worst_segment_mean"])
            agg[o]["rev"].append(r["revenue_value"])
            agg[o]["e"].append(r["delivered_kwh"])
            agg[o]["groups"].append(r["group_means"])
    out = {"label": "oracle (offline)"}
    for o in objs:
        # Report disparity and worst-segment AS FUNCTIONS of the day-averaged group
        # means, so the table is internally consistent: a reader who computes
        # min / (max-min) of the reported group_means row gets exactly these numbers.
        # (This differs from averaging each day's own worst segment, which by the
        # min-of-means inequality is always smaller and would read as an error.)
        gmeans = np.mean(np.array(agg[o]["groups"]), axis=0)
        out[o] = {
            "mean_satisfaction": float(np.mean(agg[o]["mean"])),
            "min_satisfaction": float(np.mean(agg[o]["min"])),
            "worst_segment_mean": float(np.min(gmeans)),          # = min of group_means
            "group_disparity": float(np.max(gmeans) - np.min(gmeans)),  # = max-min of group_means
            "revenue_value": float(np.mean(agg[o]["rev"])),
            "delivered_kwh": float(np.mean(agg[o]["e"])),
            "group_means": gmeans.tolist(),
            "worst_segment_index": int(np.argmin(gmeans)),
        }
    pf, mm = out["profit"], out["maximin"]
    out["price_of_fairness_revenue"] = float(
        (pf["revenue_value"] - mm["revenue_value"]) / (pf["revenue_value"] + 1e-9))
    out["price_of_fairness_meansat"] = float(
        (pf["mean_satisfaction"] - mm["mean_satisfaction"]) / (pf["mean_satisfaction"] + 1e-9))
    # Back-compat keys used by the plots.
    out["utilitarian_mean_sat_ceiling"] = out["utilitarian"]["mean_satisfaction"]
    out["maximin_min_sat_ceiling"] = out["maximin"]["worst_segment_mean"]
    print(f"  [oracle] profit: groups={[round(x,2) for x in pf['group_means']]} "
          f"disp={pf['group_disparity']:.2f} rev={pf['revenue_value']:.0f}")
    print(f"  [oracle] maximin: groups={[round(x,2) for x in mm['group_means']]} "
          f"disp={mm['group_disparity']:.2f} rev={mm['revenue_value']:.0f} "
          f"| PoF(revenue)={out['price_of_fairness_revenue']*100:.1f}% "
          f"PoF(meanSat)={out['price_of_fairness_meansat']*100:.1f}%")
    return out


# ============================================= 1b. robustness of the "cheap equity" finding
def eval_robustness(n_evses, grid_kw, num_disc, key, n_days) -> dict:
    """How does the price of fairness scale with the price gradient, and what does the
    premium segment actually give up? The mean-satisfaction rise under equalization is
    a mechanical consequence of concave, saturating charging utility (premium sits near
    the 1.0 ceiling, so energy moved off it costs little satisfaction while it buys a
    large gain for the budget segment on the steep part of the curve). We therefore
    report the finding honestly as a *redistribution the premium segment pays for*, and
    check that it is not an artifact of one price gradient by sweeping the premium
    multiplier."""
    station = scarcity_station(grid_kw=grid_kw, n_evses=n_evses)
    rows = []
    for premium in (1.2, 1.5, 2.0, 3.0):
        env = make_env(station=station, num_discretization_levels=num_disc,
                       allow_discharging=ALLOW_DISCHARGING, alpha=0.0, lam=1.0,
                       outer="rawlsian", n_groups=3, group_probs=GROUP_PROBS,
                       price_by_group=(0.6, 1.0, premium))
        pf = _avg_oracle(env, key, "profit", n_days)
        mm = _avg_oracle(env, key, "maximin", n_days)
        rows.append({
            "premium_weight": premium,
            "profit_premium_sat": pf["group_means"][2],
            "maximin_premium_sat": mm["group_means"][2],
            "premium_sat_loss": pf["group_means"][2] - mm["group_means"][2],
            "profit_budget_sat": pf["group_means"][0],
            "maximin_budget_sat": mm["group_means"][0],
            "budget_sat_gain": mm["group_means"][0] - pf["group_means"][0],
            "mean_sat_profit": pf["mean"], "mean_sat_maximin": mm["mean"],
            "pof_revenue_pct": 100.0 * (pf["rev"] - mm["rev"]) / (pf["rev"] + 1e-9),
        })
        print(f"  [robustness premium={premium}] premium {pf['group_means'][2]:.2f}->"
              f"{mm['group_means'][2]:.2f} (loses {rows[-1]['premium_sat_loss']:.2f}), "
              f"budget +{rows[-1]['budget_sat_gain']:.2f}, PoF_rev={rows[-1]['pof_revenue_pct']:.0f}%")
    return {"rows": rows}


def _avg_oracle(env, key, obj, n_days):
    rs = [O.oracle_welfare(env, jax.random.fold_in(key, i), objective=obj)
          for i in range(n_days)]
    gm = np.mean([r["group_means"] for r in rs], axis=0)
    return {"group_means": gm.tolist(),
            "mean": float(np.mean([r["mean_satisfaction"] for r in rs])),
            "rev": float(np.mean([r["revenue_value"] for r in rs]))}


# ==================================================== 2. rate-design (tariff) sweep
def eval_tariffs(env, key, n_days: int) -> dict:
    """Low-income-subsidy sweep + the flat-tariff identity, as actionable rate
    design. Reports the subsidy weight at which the profit-optimal allocation stops
    starving the budget segment (disparity -> 0)."""
    subsidies = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]  # added to the budget weight
    sweep = T.subsidy_sweep(env, key, subsidies, n_days=n_days)
    # Cap-the-top (premium price cap / compression) as the deployable alternative.
    caps = [1.5, 1.3, 1.1, 1.0, 0.8, 0.6]
    sweep["premium_cap"] = T.premium_cap_sweep(env, key, caps, n_days=n_days)
    best_cap = min(sweep["premium_cap"]["rows"], key=lambda r: r["disparity"])
    # The best top-cap ABOVE the budget margin (0.6) -- i.e. not full flattening.
    partial_caps = [r for r in sweep["premium_cap"]["rows"] if r["cap"] > 0.6 + 1e-9]
    sweep["best_partial_cap_disparity"] = float(min(r["disparity"] for r in partial_caps))
    print(f"  [tariff] value-weighted disp={sweep['value_weighted_disparity']:.3f} -> "
          f"income-neutral disp={sweep['flat_tariff_disparity']:.3f} "
          f"({sweep['price_driven_reduction_pct']:.0f}% removed). "
          f"Best budget-only subsidy leaves {sweep['best_budget_only_disparity']:.3f} "
          f"(shifts burden to mid); best partial premium-cap leaves "
          f"{sweep['best_partial_cap_disparity']:.3f} (budget still starved). "
          f"Only equal margins close it.")
    return sweep


# ==================================================== 3. grid-capacity threshold
def capacity_sweep(n_evses, num_disc, n_eval, key, oracle_days, grids=None) -> dict:
    """Sweep the grid connection and trace inequality vs capacity, for the profit-blind
    max-charge policy (Gini) and the clairvoyant profit optimum (segment disparity).

    The oracle disparity is computed over the SAME ``oracle_days`` and the SAME key as
    the main oracle (Section 1) and from the day-averaged group means, so at the shared
    30 kW grid this row equals the main-oracle number exactly. Reports the threshold
    under BOTH metrics (Gini and oracle disparity), which differ, so neither reads as
    cherry-picked."""
    grids = grids or [20.0, 25.0, 30.0, 40.0, 50.0, 70.0, 100.0, 150.0, 300.0, 600.0]
    rows = []
    for grid in grids:
        env = _ref_env(grid, n_evses, num_disc)
        m = evaluate_policy(env, B.POLICIES["max_charge"], key, n_episodes=n_eval)
        gm = np.mean([O.oracle_welfare(env, jax.random.fold_in(key, i),
                                       objective="profit")["group_means"]
                      for i in range(oracle_days)], axis=0)
        disp = float(np.max(gm) - np.min(gm))
        rows.append({"grid_kw": grid, "gini": m["gini"],
                     "served_mean_sat": m["served_mean_sat"],
                     "worst10_sat": m["worst10_sat"], "profit_mean": m["profit_mean"],
                     "oracle_profit_disparity": disp})
        print(f"  [capacity {int(grid)}kW] gini={m['gini']:.3f} "
              f"mean_sat={m['served_mean_sat']:.3f} oracle_disp={disp:.3f}")
    abundant_gini = rows[-1]["gini"]
    abundant_disp = rows[-1]["oracle_profit_disparity"]
    thr_gini = next((r["grid_kw"] for r in rows if r["gini"] <= abundant_gini * 1.10), None)
    thr_disp = next((r["grid_kw"] for r in rows
                     if r["oracle_profit_disparity"] <= abundant_disp * 1.10), None)
    return {"rows": rows, "abundant_gini": abundant_gini, "abundant_disparity": abundant_disp,
            "threshold_kw_gini": thr_gini, "threshold_kw_oracle": thr_disp,
            "threshold_kw": thr_gini}  # back-compat


# ==================================================== 4. baselines + learned
def eval_baselines(n_evses, grid_kw, num_disc, n_eval, key) -> dict:
    env = _ref_env(grid_kw, n_evses, num_disc)
    out = {}
    for name, pol in B.POLICIES.items():
        m = evaluate_policy(env, pol, key, n_episodes=n_eval)
        m["label"] = name
        m["kind"] = "baseline"
        out[name] = m
        print(f"  [baseline {name}] profit={m['profit_mean']:.0f} "
              f"gini={m['gini']:.3f} worst10={m['worst10_sat']:.3f} "
              f"disp={m.get('group_disparity', float('nan')):.3f}")
    return out


def train_and_eval(cfg: dict, label: str, *, timesteps: int, seeds, n_eval: int,
                   num_envs: int, n_evses: int, grid_kw: float, num_disc: int) -> dict:
    """Train PPO under one welfare configuration across seeds and aggregate."""
    station = scarcity_station(grid_kw=grid_kw, n_evses=n_evses)
    per_seed = []
    for seed in seeds:
        env = make_env(station=station, num_discretization_levels=num_disc,
                       allow_discharging=ALLOW_DISCHARGING, **GROUP_KW, **cfg)
        t0 = time.time()
        agent = train_ppo(env, total_timesteps=timesteps, num_envs=num_envs,
                          num_steps=288, seed=seed)
        policy = make_trained_policy(agent)
        m = evaluate_policy(env, policy, jax.random.PRNGKey(1000 + seed), n_episodes=n_eval)
        m["train_seconds"] = round(time.time() - t0, 1)
        per_seed.append(m)
        print(f"  [{label}] seed={seed} profit={m['profit_mean']:.0f} "
              f"gini={m['gini']:.3f} worst10={m['worst10_sat']:.3f} "
              f"disp={m.get('group_disparity', float('nan')):.3f} ({m['train_seconds']}s)")
    out = aggregate_seeds(per_seed, _SEED_KEYS)
    out.update({"label": label, "kind": "learned", "config": dict(cfg)})
    return out


# ==================================================== 5. endogeneity audit
def endogeneity_audit(n_evses, grid_kw, num_disc, n_eval, key) -> dict:
    """Quantify the reject-to-look-fair loophole the utility design closes.

    The welfare is defined over the *true arrival stream* (rejected customers enter
    at utility 0). Ablating ``count_rejections`` measures how much a policy could
    inflate its apparent welfare by shedding customers it cannot serve well -- the
    gaming that endogeneity (policy-controlled admission) would otherwise allow.
    Because departure/admission are policy-influenced, this gap is the concrete
    payoff of modelling the population as endogenous rather than fixed."""
    station = scarcity_station(grid_kw=grid_kw, n_evses=n_evses)
    out = {}
    for name in ("max_charge", "least_laxity"):
        pol = B.POLICIES[name]
        env_on = make_env(station=station, num_discretization_levels=num_disc,
                          allow_discharging=ALLOW_DISCHARGING, alpha=0.0, lam=1.0,
                          outer="rawlsian", count_rejections=True, **GROUP_KW)
        env_off = make_env(station=station, num_discretization_levels=num_disc,
                           allow_discharging=ALLOW_DISCHARGING, alpha=0.0, lam=1.0,
                           outer="rawlsian", count_rejections=False, **GROUP_KW)
        m_on = evaluate_policy(env_on, pol, key, n_episodes=n_eval)
        m_off = evaluate_policy(env_off, pol, key, n_episodes=n_eval)
        gap = m_off["welfare_mean"] - m_on["welfare_mean"]
        out[name] = {
            "welfare_inclusive": m_on["welfare_mean"],
            "welfare_served_only": m_off["welfare_mean"],
            "gaming_gap": float(gap),
            "rejection_rate": m_on["rejection_rate"],
        }
        print(f"  [endogeneity {name}] welfare served-only={m_off['welfare_mean']:.3f} "
              f"inclusive={m_on['welfare_mean']:.3f} gaming_gap={gap:.3f} "
              f"(reject={m_on['rejection_rate']:.2f})")
    return out


# ==================================================== 6. scarcity contrast
def scarcity_contrast(n_evses, num_disc, n_eval, key, scarce_grid_kw=30.0) -> dict:
    out = {}
    for tag, grid in [(f"scarce_{int(scarce_grid_kw)}kW", scarce_grid_kw),
                      ("abundant_600kW", 600.0)]:
        env = _ref_env(grid, n_evses, num_disc)
        m = evaluate_policy(env, B.POLICIES["max_charge"], key, n_episodes=n_eval)
        out[tag] = {"grid_kw": grid, "served_mean_sat": m["served_mean_sat"],
                    "gini": m["gini"], "worst10_sat": m["worst10_sat"],
                    "profit_mean": m["profit_mean"]}
        print(f"  [scarcity {tag}] mean_sat={m['served_mean_sat']:.3f} "
              f"gini={m['gini']:.3f}")
    return out


# =========================================================================== main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="fast smoke run")
    p.add_argument("--oracle-only", action="store_true",
                   help="skip PPO training; run only the RL-free core (oracle, tariff, "
                        "capacity, endogeneity, scarcity)")
    p.add_argument("--timesteps", type=int, default=1_500_000)
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--n-eval", type=int, default=32)
    p.add_argument("--num-envs", type=int, default=32)
    p.add_argument("--n-evses", type=int, default=8)  # 16 chargers
    p.add_argument("--grid-kw", type=float, default=30.0)
    p.add_argument("--num-disc", type=int, default=4)
    p.add_argument("--oracle-days", type=int, default=16)
    args = p.parse_args()

    if args.quick:
        (args.timesteps, args.seeds, args.n_eval, args.num_envs, args.oracle_days) = (
            40_000, 3, 12, 8, 8)

    seeds = list(range(args.seeds))
    common = dict(timesteps=args.timesteps, seeds=seeds, n_eval=args.n_eval,
                  num_envs=args.num_envs, n_evses=args.n_evses, grid_kw=args.grid_kw,
                  num_disc=args.num_disc)
    key = jax.random.PRNGKey(7)
    ref = _ref_env(args.grid_kw, args.n_evses, args.num_disc)

    results = {"meta": {**vars(args), "price_by_group": list(PRICE_BY_GROUP),
                        "profit_scale": float(ref.profit_scale)},
               "trained": {}, "baselines": {}, "oracle": {}, "robustness": {},
               "tariffs": {}, "capacity": {}, "endogeneity": {}, "scarcity": {}}

    print("=== 1. Offline oracle (RL-free characterization) ===")
    results["oracle"] = eval_oracle(ref, key, args.oracle_days)
    _save(results)

    print("=== 1b. Robustness of the cheap-equity finding (price-gradient sweep) ===")
    results["robustness"] = eval_robustness(args.n_evses, args.grid_kw, args.num_disc,
                                            key, args.oracle_days)
    _save(results)

    print("=== 2. Rate-design finding (low-income subsidy sweep) ===")
    results["tariffs"] = eval_tariffs(ref, key, args.oracle_days)
    _save(results)

    print("=== 3. Grid-capacity threshold ===")
    results["capacity"] = capacity_sweep(args.n_evses, args.num_disc, args.n_eval, key,
                                         args.oracle_days)
    _save(results)

    print("=== 4a. Non-learned baselines ===")
    results["baselines"] = eval_baselines(args.n_evses, args.grid_kw, args.num_disc,
                                          args.n_eval, key)
    _save(results)

    print("=== 5. Endogeneity audit ===")
    results["endogeneity"] = endogeneity_audit(args.n_evses, args.grid_kw,
                                               args.num_disc, args.n_eval, key)
    _save(results)

    print("=== 6. Scarcity contrast ===")
    results["scarcity"] = scarcity_contrast(args.n_evses, args.num_disc, args.n_eval,
                                            key, scarce_grid_kw=args.grid_kw)
    _save(results)

    if not args.oracle_only:
        sweep = [
            ("profit_ppo", dict(alpha=0.0, lam=0.0, outer="rawlsian")),
            ("egal_l025", dict(alpha=0.0, lam=0.25, outer="rawlsian")),
            ("egal_l050", dict(alpha=0.0, lam=0.50, outer="rawlsian")),
            ("egal_l075", dict(alpha=0.0, lam=0.75, outer="rawlsian")),
            ("egal_l100", dict(alpha=0.0, lam=1.00, outer="rawlsian")),
            ("util_l050", dict(alpha=0.0, lam=0.50, outer="utilitarian")),
            ("util_l100", dict(alpha=0.0, lam=1.00, outer="utilitarian")),
        ]
        if args.quick:
            sweep = [sweep[0], sweep[4], sweep[6]]
        print("=== 4b. Learned policy suite (multi-seed) ===")
        for label, cfg in sweep:
            results["trained"][label] = train_and_eval(cfg, label, **common)
            _save(results)

    print(f"Saved results to {RESULTS_DIR}/results.json")


if __name__ == "__main__":
    main()
