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

Uncertainty: every reported quantity is a functional of an average over a finite set of
realized days, so we attach a **day-level nonparametric bootstrap** (resample days with
replacement, recompute the day-average, then the functional). The gap is max-min of a mean
vector, a nonlinear functional whose sampling error is not the mean's standard error, so a
bootstrap rather than a closed-form interval is the right instrument. All quantities are
resampled with the SAME day indices, which keeps them paired: that is what lets us put an
interval on the decisive premium-cap-minus-budget-parity *difference* rather than only on
the two gaps separately.
"""

from __future__ import annotations

import json
import os

import jax
import numpy as np

from chargax.equity import oracle as O
from chargax.equity import segments as SEG
from experiments.common import acn_env_or_none
from experiments.run_experiments import _ref_env

RESULTS = os.path.join(os.path.dirname(__file__), "results")
NAMES = ("budget", "mid", "premium")
N_DAYS = 32
STATUS_QUO = tuple(SEG.PRICE_BY_GROUP)  # (0.6, 1.0, 1.5)
N_BOOT = 10000
BOOT_SEED = 20260806

# Margin configs, chosen to separate rank position from spread magnitude.
CONFIGS = [
    ("status_quo_distinct",   (0.6, 1.0, 1.5)),  # three distinct ranks, budget uniquely lowest
    ("premium_capped_to_mid", (0.6, 1.0, 1.0)),  # spread 0.4, budget STILL uniquely lowest
    ("budget_tied_to_mid",    (1.0, 1.0, 1.5)),  # spread 0.5, budget NO LONGER uniquely lowest
    ("fully_flat",            (1.0, 1.0, 1.0)),   # no ranks -> scarcity floor
]


def _budget_uniquely_lowest(m):
    return bool(m[0] < min(m[1], m[2]))


def _ci(samples, lo=2.5, hi=97.5, nd=3):
    """Percentile interval of a bootstrap sample, rounded for the paper."""
    return [round(float(np.percentile(samples, lo)), nd),
            round(float(np.percentile(samples, hi)), nd)]


def main(use_acn: bool = False):
    if use_acn:
        got = acn_env_or_none(30.0, 8, 4)
        if got is None:
            raise SystemExit(
                "--acn requires EQUICHARGE_ACN_JSON to point at a Caltech ACN-Data "
                "sessions JSON (https://ev.caltech.edu/dataset).")
        env, acn_prov = got
    else:
        env, acn_prov = _ref_env(30.0, 8, 4), None
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

    # One LP solve per (day, margins, objective), reused by the point estimate and by
    # every bootstrap replicate -- the bootstrap resamples cached day rows, never re-solves.
    _cache = {}

    def per_day(margins, objective="profit"):
        """(D,3) per-day tier means and (D,) per-day revenue for this configuration."""
        ck = (tuple(margins), objective)
        if ck not in _cache:
            res = [solve(c, objective, margins) for c in streams]
            _cache[ck] = (np.array([r["group_means"] for r in res], dtype=float),
                          np.array([r["revenue_value"] for r in res], dtype=float))
        return _cache[ck]

    # Shared bootstrap day indices: every quantity below is resampled on the SAME days,
    # so gaps across configs stay paired and their differences get honest intervals.
    rng = np.random.default_rng(BOOT_SEED)
    boot_idx = rng.integers(0, D, size=(N_BOOT, D))

    def boot_gaps(margins, objective="profit"):
        """(N_BOOT,) bootstrap distribution of the day-averaged max-min tier gap."""
        gms, _ = per_day(margins, objective)
        avg = gms[boot_idx].mean(axis=1)          # (N_BOOT, 3)
        return avg.max(axis=1) - avg.min(axis=1)

    def boot_tiers(margins, objective="profit"):
        """(N_BOOT, 3) bootstrap distribution of the day-averaged per-tier means."""
        gms, _ = per_day(margins, objective)
        return gms[boot_idx].mean(axis=1)

    def day_averaged(margins, objective="profit"):
        """Day-averaged per-tier means and the disparate-impact gap (max-min of the average)."""
        gms, _ = per_day(margins, objective)
        avg = gms.mean(axis=0)
        return avg, float(avg.max() - avg.min()), int(np.argmin(avg))

    # --- Mechanism: four margin configs on the SAME day set, SAME estimator. ---
    mech = []
    for name, m in CONFIGS:
        avg, gap, worst = day_averaged(m)
        bt, bg = boot_tiers(m), boot_gaps(m)
        mech.append({
            "config": name, "margins": list(m), "spread": round(max(m) - min(m), 3),
            "budget_uniquely_lowest": _budget_uniquely_lowest(m),
            "tier_means": [round(x, 3) for x in avg],
            "tier_means_ci95": [_ci(bt[:, g]) for g in range(3)],
            "gap": round(gap, 3), "gap_ci95": _ci(bg),
            "worst_tier": NAMES[worst],
            # How stable is the worst-tier attribution itself under resampling?
            "worst_tier_boot_share": {
                NAMES[g]: round(float((bt.argmin(axis=1) == g).mean()), 3)
                for g in range(3)},
        })

    # --- Box-1 objectives on the SAME day set (canonical table numbers). ---
    box1 = {}
    for obj in ("profit", "utilitarian", "maximin", "egalitarian_equal"):
        avg, gap, worst = day_averaged(STATUS_QUO, objective=obj)
        _, rev_days = per_day(STATUS_QUO, obj)
        bt, bg = boot_tiers(STATUS_QUO, obj), boot_gaps(STATUS_QUO, obj)
        box1[obj] = {"tier_means": [round(x, 3) for x in avg],
                     "tier_means_ci95": [_ci(bt[:, g]) for g in range(3)],
                     "gap": round(gap, 3), "gap_ci95": _ci(bg),
                     "worst_tier": NAMES[worst],
                     "revenue": round(float(rev_days.mean()), 1),
                     "revenue_ci95": _ci(rev_days[boot_idx].mean(axis=1), nd=1)}
    # Price of fairness (revenue), per-day median, on the same day set.
    _, rev_profit = per_day(STATUS_QUO, "profit")
    _, rev_maximin = per_day(STATUS_QUO, "maximin")
    pof_days = 100.0 * (rev_profit - rev_maximin) / (rev_profit + 1e-9)
    pof = list(pof_days)
    pof_median = float(np.median(pof_days))
    pof_median_ci = _ci(np.median(pof_days[boot_idx], axis=1), nd=1)

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
                    "env": ("ACN-Data caltech 8ch/30kW, 4 disc" if use_acn
                            else "ref 8ch/30kW, 4 disc"),
                    "estimator": "max-min of day-averaged tier means",
                    "uncertainty": (f"day-level nonparametric bootstrap, B={N_BOOT}, "
                                    f"seed={BOOT_SEED}, 95% percentile intervals, "
                                    f"shared day indices across all configs")},
        "status_quo_gap": mech[0]["gap"],
        "subsidy_to_parity_gap": mech[2]["gap"],
        "box1": box1, "revenue_price_of_fairness_pct_median": round(pof_median, 1),
        "revenue_price_of_fairness_pct_median_ci95": pof_median_ci,
        "mechanism": mech,
        "elasticity_invariance": inv,
        "verdict": None,
    }
    if acn_prov is not None:
        out["acn_provenance"] = acn_prov
    # Verdict: does rank beat spread? premium-cap (smaller spread) keeps a larger gap than
    # budget-parity (larger spread) -> spread magnitude does NOT predict the gap; rank does.
    cap = next(x for x in mech if x["config"] == "premium_capped_to_mid")
    par = next(x for x in mech if x["config"] == "budget_tied_to_mid")
    rank_beats_spread = (cap["spread"] < par["spread"]) and (cap["gap"] > par["gap"])
    # The claim is a DIFFERENCE between two gaps on the same days, so bootstrap the
    # difference directly (paired) rather than eyeballing whether two CIs overlap --
    # non-overlap is sufficient for a difference but not necessary.
    cap_m = dict(CONFIGS)["premium_capped_to_mid"]
    par_m = dict(CONFIGS)["budget_tied_to_mid"]
    diff = boot_gaps(cap_m) - boot_gaps(par_m)
    out["verdict"] = {
        "premium_cap_spread": cap["spread"], "premium_cap_gap": cap["gap"],
        "premium_cap_gap_ci95": cap["gap_ci95"],
        "budget_parity_spread": par["spread"], "budget_parity_gap": par["gap"],
        "budget_parity_gap_ci95": par["gap_ci95"],
        "gap_difference_cap_minus_parity": round(float(cap["gap"] - par["gap"]), 3),
        "gap_difference_ci95": _ci(diff),
        "gap_difference_bootstrap_p_positive": round(float((diff > 0).mean()), 4),
        "rank_position_beats_spread_magnitude": bool(rank_beats_spread),
        "interpretation": (
            "CONFIRMED: the premium cap has the SMALLER spread yet the LARGER gap, so the harm "
            "cannot be a function of spread magnitude. It tracks rank position: the budget tier "
            "is starved exactly while it is uniquely lowest, and is relieved only when it stops "
            "being uniquely lowest (budget-parity or full flattening)."
            if rank_beats_spread else
            "NOT confirmed on this day set; inspect the per-config gaps before rewriting box 2."),
    }
    fname = "audit_mechanism_acn.json" if use_acn else "audit_mechanism.json"
    json.dump(out, open(os.path.join(RESULTS, fname), "w"), indent=2)

    def _fmt(pt, ci):
        return f"{pt:.3f} [{ci[0]:.3f},{ci[1]:.3f}]"

    print(f"=== CANONICAL DAY SET: {D}/{N_DAYS} days (seed 7), day-averaged estimator ===")
    print(f"    env: {out['day_set']['env']}")
    print(f"    95% CIs: day-level bootstrap, B={N_BOOT}\n")
    print("BOX 1 (status-quo margins 0.6/1.0/1.5):")
    for obj, r in box1.items():
        tiers = " ".join(_fmt(m, c) for m, c in zip(r["tier_means"], r["tier_means_ci95"]))
        print(f"  {obj:18s} {tiers}  gap={_fmt(r['gap'], r['gap_ci95'])} "
              f"worst={r['worst_tier']} rev={r['revenue']} {r['revenue_ci95']}")
    print(f"  revenue price of fairness (median): {pof_median:.1f}% "
          f"[{pof_median_ci[0]:.1f},{pof_median_ci[1]:.1f}]\n")
    print("MECHANISM (rank position vs spread magnitude):")
    print(f"  {'config':22s} {'margins':17s} {'spread':>6} {'gap [95% CI]':>22} {'uniq-low':>8} worst")
    for r in mech:
        print(f"  {r['config']:22s} {str(r['margins']):17s} {r['spread']:>6} "
              f"{_fmt(r['gap'], r['gap_ci95']):>22} "
              f"{str(r['budget_uniquely_lowest']):>8} {r['worst_tier']}")
    print("\nELASTICITY INVARIANCE (same day set/estimator):")
    for r in inv:
        print(f"  eta={r['eta']:.1f} margins={r['margins']} gap={r['gap']} "
              f"uniq-low={r['budget_uniquely_lowest']} worst={r['worst_tier']}")
    v = out["verdict"]
    print(f"\nVERDICT: premium-cap spread {v['premium_cap_spread']} gap "
          f"{v['premium_cap_gap']} {v['premium_cap_gap_ci95']} "
          f"vs budget-parity spread {v['budget_parity_spread']} gap "
          f"{v['budget_parity_gap']} {v['budget_parity_gap_ci95']}")
    print(f"  paired gap difference (cap - parity): "
          f"{v['gap_difference_cap_minus_parity']} {v['gap_difference_ci95']}, "
          f"P(diff>0)={v['gap_difference_bootstrap_p_positive']}")
    print("rank position beats spread magnitude:", v["rank_position_beats_spread_magnitude"])
    print("\nwrote", os.path.join(RESULTS, fname))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Tariff mechanism audit with bootstrap CIs.")
    ap.add_argument("--acn", action="store_true",
                    help="run on the Caltech ACN-Data (US session-level) calibration "
                         "instead of the bundled reference site; requires "
                         "EQUICHARGE_ACN_JSON. Writes audit_mechanism_acn.json.")
    main(use_acn=ap.parse_args().acn)
