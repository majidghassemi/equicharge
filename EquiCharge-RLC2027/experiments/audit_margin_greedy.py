r"""The disparate impact realized ONLINE, by a policy an operator could deploy today.

The offline oracle says what a revenue-maximizing allocation *would* do with perfect
foreknowledge of the day. A reviewer can reasonably ask whether that ceiling is an
artifact of clairvoyance, so this script runs the causal, online counterpart,
:func:`chargax.equity.baselines.margin_greedy_policy`, on the *same 32 realized days*
as ``audit_offline.py`` and reports the same quantities the paper reports for the
oracle:

* **profit per day**, so the reader can see that ranking by margin is not a strawman,
  a greedy operator really does earn more this way than by charging everyone flat out;
* **Gamma**, the systematic between-tier gap, measured with the estimator the rest of
  the paper uses, max-minus-min of the DAY-AVERAGED per-tier mean satisfaction;
* **rotation entropy**, the normalized entropy of the per-day worst-tier distribution,
  which separates a systematic between-tier harm (one tier is always worst, entropy
  near 0) from a rotating one that day-averaging would manufacture;
* the **budget-worst share**, the fraction of days on which the budget tier is the
  worst-served one; and
* the **30 / 45 / 60 kW scarcity sweep**, which tests online whether the harm is
  specific to power-scarce sites the way it is offline.

Per-tier satisfaction is *inclusive*, rejected customers enter their tier's mean at
zero, so a policy cannot look fair by turning needy drivers away. The offline
profit-optimal oracle is solved on the same days and reported alongside, so the
online and offline harms can be read as a paired comparison rather than two
separately-scaled numbers.

Writes ``experiments/results/audit_margin_greedy.json``.
"""

from __future__ import annotations

import json
import os
from collections import Counter

import jax
import numpy as np

from chargax.equity import baselines as B
from chargax.equity import oracle as O
from experiments.common import _rollout
from experiments.run_experiments import _ref_env

RESULTS = os.path.join(os.path.dirname(__file__), "results")
#: Days in the realized day set. ``EQUICHARGE_N_DAYS`` overrides it so the whole
#: suite can be re-run at a larger day count without editing every script.
N_DAYS = int(os.environ.get("EQUICHARGE_N_DAYS") or 32)
KEY = jax.random.PRNGKey(7)          # the day set audit_offline.py uses
SWEEP_KW = (30.0, 45.0, 60.0)
NAMES = ("budget", "mid", "premium")


def _stats(x):
    x = np.asarray(x, float)
    return {"median": float(np.median(x)), "iqr": float(np.subtract(*np.percentile(x, [75, 25]))),
            "min": float(x.min()), "max": float(x.max()), "mean": float(x.mean())}


def _rollout_fn(env, policy):
    """A jitted one-day rollout, compiled once per (env, policy).

    ``_rollout`` builds a ``lax.scan`` every time it is called, so calling it once per
    day compiles a fresh executable per day and the compilation cache grows until the
    process runs out of memory. Compiling once and reusing is both the fix and much
    faster.
    """
    return jax.jit(lambda k: _rollout(env, policy, k))


def _day(env, rollout, key) -> dict:
    """One realized day under a compiled rollout. Profit and inclusive tier means."""
    welf, prof, util, mask, grp, rej = (np.array(a) for a in rollout(key))
    served_sat, served_grp = util[mask > 0], grp[mask > 0]
    rejected = rej.reshape(-1, int(env.n_groups)).sum(axis=0)
    means = []
    for g in range(int(env.n_groups)):
        s = served_sat[served_grp == g]
        denom = len(s) + rejected[g]
        means.append(float(s.sum() / denom) if denom > 0 else np.nan)
    return {"profit": float(prof[-1]), "group_means": means,
            "n_served": int(len(served_sat)), "n_rejected": float(rejected.sum())}


def _rotation(group_means: np.ndarray) -> dict:
    """Gamma, the worst-tier distribution and its normalized entropy, over a day set.

    Gamma is max-minus-min of the DAY-AVERAGED tier means (the paper's estimator).
    The entropy is 0 when one tier is worst on every day (a systematic between-tier
    harm) and 1 when the worst tier is uniform over the three tiers (a rotating one).
    """
    n_tiers = group_means.shape[1]
    worst = [int(np.argmin(g)) for g in group_means]
    counts = Counter(worst)
    n = len(worst)
    ps = np.array([counts[t] / n for t in range(n_tiers) if counts.get(t, 0) > 0])
    entropy = float(-(ps * np.log(ps)).sum() / np.log(n_tiers)) if len(ps) > 1 else 0.0
    day_avg = group_means.mean(axis=0)
    per_day = group_means.max(axis=1) - group_means.min(axis=1)
    return {
        "gamma": float(day_avg.max() - day_avg.min()),
        "day_averaged_group_means": [float(v) for v in day_avg],
        "per_day_gap": _stats(per_day),
        "worst_tier_counts": {NAMES[t]: counts.get(t, 0) for t in range(n_tiers)},
        "budget_worst_share": float(counts.get(0, 0) / n),
        "worst_tier_entropy": entropy,
        "reading": ("systematic between-tier (one tier is worst on most days)"
                    if max(counts.values()) / n >= 0.75 else
                    "NON-SYSTEMATIC / rotating (day-averaging would hide the per-day harm)"),
    }


def _run(env, policy, n_days: int, oracle: bool = False) -> dict:
    profits, means, skipped = [], [], 0
    oracle_means = []
    rollout = _rollout_fn(env, policy)
    for i in range(n_days):
        k = jax.random.fold_in(KEY, i)
        d = _day(env, rollout, k)
        if not np.all(np.isfinite(d["group_means"])):
            skipped += 1          # a tier had no arrivals at all; it has no mean
            continue
        profits.append(d["profit"])
        means.append(d["group_means"])
        if oracle:
            r = O.oracle_welfare(env, k, objective="profit")
            if not r.get("degenerate") and min(r.get("group_counts", [0])) > 0:
                oracle_means.append(r["group_means"])
    out = {"n_days": n_days, "days_used": len(means), "days_skipped_no_tier": skipped,
           "profit_per_day": _stats(profits)}
    out.update(_rotation(np.array(means)))
    if oracle and oracle_means:
        out["offline_profit_optimal"] = _rotation(np.array(oracle_means))
        out["offline_profit_optimal"]["days_used"] = len(oracle_means)
    return out


def main():
    out = {"n_days": N_DAYS, "policy": "margin_greedy", "day_key": 7,
           "estimator": "max-min of day-averaged inclusive per-tier mean satisfaction"}

    print("=== ONLINE margin-greedy at the reference site (30 kW, 8 EVSEs), %d days ===" % N_DAYS)
    env = _ref_env(30.0, 8, 4)
    ref = _run(env, B.margin_greedy_policy, N_DAYS, oracle=True)
    out["reference"] = ref
    print("  profit/day:", {k: round(v, 1) for k, v in ref["profit_per_day"].items()})
    print("  tier means:", [round(v, 3) for v in ref["day_averaged_group_means"]],
          " Gamma=%.3f" % ref["gamma"])
    print("  worst tier:", ref["worst_tier_counts"],
          " budget-worst share=%.0f%%" % (100 * ref["budget_worst_share"]),
          " entropy=%.2f" % ref["worst_tier_entropy"])
    print("  ->", ref["reading"])
    if "offline_profit_optimal" in ref:
        o = ref["offline_profit_optimal"]
        print("  offline profit-optimal oracle on the same days: Gamma=%.3f "
              "budget-worst=%.0f%% entropy=%.2f"
              % (o["gamma"], 100 * o["budget_worst_share"], o["worst_tier_entropy"]))

    # Contrast policies on the identical day set, so the profit and the gap are paired.
    print("\n=== the same day set under the other online baselines ===")
    contrast = {}
    for name in ("max_charge", "proportional_fair", "least_laxity", "saffe"):
        r = _run(env, B.POLICIES[name], N_DAYS)
        contrast[name] = r
        print("  %-18s profit/day=%6.1f  Gamma=%.3f  budget-worst=%3.0f%%  entropy=%.2f"
              % (name, r["profit_per_day"]["mean"], r["gamma"],
                 100 * r["budget_worst_share"], r["worst_tier_entropy"]))
    out["other_online_baselines"] = contrast
    out["margin_greedy_profit_premium_pct"] = {
        n: 100.0 * (ref["profit_per_day"]["mean"] - c["profit_per_day"]["mean"])
        / (abs(c["profit_per_day"]["mean"]) + 1e-9)
        for n, c in contrast.items()
    }

    # Scarcity sweep: the harm should fade as the grid connection stops binding.
    print("\n=== scarcity sweep (grid kW), online margin-greedy ===")
    sweep = []
    for kw in SWEEP_KW:
        e = _ref_env(kw, 8, 4)
        r = _run(e, B.margin_greedy_policy, N_DAYS)
        r["grid_kw"] = kw
        sweep.append(r)
        print("  %4.0f kW: profit/day=%6.1f  Gamma=%.3f  budget-worst=%3.0f%%  entropy=%.2f  "
              "tier means=%s" % (kw, r["profit_per_day"]["mean"], r["gamma"],
                                 100 * r["budget_worst_share"], r["worst_tier_entropy"],
                                 [round(v, 3) for v in r["day_averaged_group_means"]]))
    out["scarcity_sweep"] = sweep
    out["scarcity_sweep_gamma"] = {str(int(r["grid_kw"])): r["gamma"] for r in sweep}

    path = os.path.join(RESULTS, "audit_margin_greedy.json")
    json.dump(out, open(path, "w"), indent=2)
    print("\nwrote", path)


if __name__ == "__main__":
    main()
