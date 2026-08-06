r"""Stress-test the estimator switch, before committing the scarcity boundary as the thesis.

We changed the disparate-impact measure from the per-day-median gap to max-minus-min of the
DAY-AVERAGED per-tier satisfaction, at the same time as we sharpened the claim. This run checks,
on fixed day sets, whether the day-averaging reveals structure or manufactures it. Three checks:

CHECK 1 (rotation). At every site, is the per-day WORST tier stable (systematic between-tier
harm) or rotating (day-level harm that day-averaging washes out)? We report, per site, the
per-day gap magnitude AND the per-day worst-tier distribution AND the day-averaged gap. A site
with a large per-day gap but a rotating worst tier has real inequality that is NOT a systematic
between-tier disparate impact -- it must be described as non-systematic, not absent.

CHECK 2 (the flat-tariff floor). Under a flat tariff the profit optimum has no price ordering to
exploit. Is the residual day-averaged between-tier gap a genuine scarcity floor, or finite-sample
noise that shrinks toward zero as days increase? We trace the flat-tariff gap at 32/64/128/256
days at the reference site, against the status-quo gap for scale. If it shrinks toward zero, the
honest statement is that flattening removes the systematic between-tier harm essentially entirely,
and the residual harm flattening does NOT remove is within-population inequality (the capacity
lever's job), a distinct harm.

CHECK 3 (the 0.564 vs 0.567 crack). status-quo margins (0.6,1.0,1.5) and the eta=0.6 derived
margins (0.600,1.0,1.499) are the same configuration but gave 0.567 and 0.564. We confirm the
cause (LP tie-breaking on the infinitesimal premium difference) and that rounding the derivation
to the committed reference removes it.
"""

from __future__ import annotations

import json
import os
from collections import Counter

import jax
import numpy as np

from chargax.equity import oracle as O
from chargax.equity import segments as SEG
from chargax.equity import tariffs as T
from experiments.audit_robustness import CONFIGS, _env_for
from experiments.common import acn_env_or_none
from experiments.run_experiments import _ref_env

RESULTS = os.path.join(os.path.dirname(__file__), "results")
NAMES = ("budget", "mid", "premium")


def _streams(env, key, n_days):
    grid = float(env.station.max_kw_throughput)
    mpt, hor, ng = env.minutes_per_timestep, env.max_episode_steps, int(env.n_groups)
    out = []
    for i in range(n_days):
        k = jax.random.fold_in(key, i)
        cust = O.extract_arrival_stream(env, k)
        if not cust:
            continue
        pf = O.solve_oracle_lp(cust, grid, mpt, hor, objective="profit",
                               price_by_group=tuple(env.price_by_group), n_groups=ng)
        if pf.get("degenerate") or min(pf.get("group_counts", [0])) == 0:
            continue
        out.append((cust, grid, mpt, hor, ng))
    return out


def _gaps(streams, margins):
    """Return (per-day gaps list, per-day worst-tier list, day-averaged gap)."""
    gms = []
    for cust, grid, mpt, hor, ng in streams:
        r = O.solve_oracle_lp(cust, grid, mpt, hor, objective="profit",
                              price_by_group=tuple(margins), n_groups=ng)
        gms.append(r["group_means"])
    gms = np.array(gms)
    perday = gms.max(axis=1) - gms.min(axis=1)
    worst = [int(np.argmin(g)) for g in gms]
    avg = gms.mean(axis=0)
    return perday, worst, float(avg.max() - avg.min())


def main(use_acn: bool = False):
    out = {}
    acn = None
    if use_acn:
        got = acn_env_or_none(30.0, 8, 4)
        if got is None:
            raise SystemExit(
                "--acn requires EQUICHARGE_ACN_JSON to point at a Caltech ACN-Data "
                "sessions JSON (https://ev.caltech.edu/dataset).")
        acn = got
        out["acn_provenance"] = acn[1]

    # ---- CHECK 1: rotation at every site (status-quo margins) ----
    print("=== CHECK 1: is off-boundary ~0 real (stable) or averaging (rotating)? ===")
    check1 = {}
    sites = [(n, _env_for(l, d)) for n, l, d in CONFIGS]
    if acn is not None:
        sites.append(("acn_data_caltech_8ch_30kW", acn[0]))
    for name, env in sites:
        streams = _streams(env, jax.random.PRNGKey(7), 32)
        base = tuple(env.price_by_group)
        perday, worst, davg = _gaps(streams, base)
        wc = Counter(worst)
        n = len(worst)
        maxfrac = max(wc.values()) / max(n, 1)
        dom = NAMES[max(wc, key=wc.get)] if wc else "-"
        # Normalized entropy of the worst-tier distribution: 0 = one tier always worst
        # (systematic), 1 = uniform across the three tiers (fully rotating).
        ps = np.array([wc.get(t, 0) / n for t in range(3) if wc.get(t, 0) > 0])
        ent = float(-(ps * np.log(ps)).sum() / np.log(3)) if len(ps) > 1 else 0.0
        check1[name] = {
            "days": n, "per_day_gap_median": round(float(np.median(perday)), 3),
            "per_day_gap_max": round(float(perday.max()), 3), "day_averaged_gap": round(davg, 3),
            "worst_tier_counts": {NAMES[t]: wc.get(t, 0) for t in range(3)},
            "dominant_worst_tier": dom, "dominant_fraction": round(maxfrac, 2),
            "worst_tier_entropy": round(ent, 2),
            "reading": ("systematic between-tier (stable worst tier)" if maxfrac >= 0.75
                        else "NON-SYSTEMATIC / rotating (day-averaging hides per-day harm)"),
        }
        print(f"  {name:34s} perday-med={check1[name]['per_day_gap_median']:.3f} "
              f"perday-max={check1[name]['per_day_gap_max']:.3f} dayavg={davg:.3f} "
              f"worst={dom}({maxfrac:.0%}) entropy={ent:.2f} -> {check1[name]['reading']}")
    out["check1_rotation"] = check1

    # ---- CHECK 2: flat-tariff floor vs number of days (reference site) ----
    print("\n=== CHECK 2: is the flat-tariff floor a real scarcity residual or sampling noise? ===")
    # On --acn, checks 2 and 3 move to the ACN-calibrated site, so the estimator is
    # validated on the same US session data the re-run reports.
    ref = acn[0] if acn is not None else _ref_env(30.0, 8, 4)
    base = tuple(ref.price_by_group)
    floor = {}
    for nd in (32, 64, 128, 256):
        streams = _streams(ref, jax.random.PRNGKey(7), nd)
        sq_perday, _, sq_davg = _gaps(streams, base)
        fl_perday, fl_worst, fl_davg = _gaps(streams, T.flat(base))
        fwc = Counter(fl_worst)
        floor[nd] = {
            "days_used": len(streams),
            "status_quo_day_averaged": round(sq_davg, 3),
            "flat_day_averaged": round(fl_davg, 4),
            "flat_per_day_median": round(float(np.median(fl_perday)), 3),
            "flat_worst_tier_counts": {NAMES[t]: fwc.get(t, 0) for t in range(3)},
        }
        print(f"  n={nd:3d} (used {len(streams)}): status-quo dayavg={sq_davg:.3f} | "
              f"FLAT dayavg={fl_davg:.4f} flat perday-med={np.median(fl_perday):.3f} "
              f"flat worst={dict(floor[nd]['flat_worst_tier_counts'])}")
    out["check2_flat_floor"] = floor

    # ---- CHECK 3: the 0.564 vs 0.567 crack ----
    print("\n=== CHECK 3: the 0.564 vs 0.567 crack (same config, two values) ===")
    streams = _streams(ref, jax.random.PRNGKey(7), 32)
    _, _, g_committed = _gaps(streams, (0.6, 1.0, 1.5))
    m_raw = SEG.derive_price_multipliers(elasticity=0.6, round_to=None)
    _, _, g_raw = _gaps(streams, m_raw)
    m_round = SEG.derive_price_multipliers(elasticity=0.6, round_to=1)
    _, _, g_round = _gaps(streams, m_round)
    out["check3_crack"] = {
        "committed_(0.6,1.0,1.5)": round(g_committed, 4),
        "eta0.6_raw": {"margins": [round(x, 4) for x in m_raw], "gap": round(g_raw, 4)},
        "eta0.6_rounded": {"margins": [round(x, 3) for x in m_round], "gap": round(g_round, 4)},
        "cause": "LP tie-breaking on the infinitesimal premium difference (1.5 vs 1.499)",
        "fix": "use round_to=1 in the invariance sweep so eta=0.6 == committed reference",
        "fixed": bool(round(g_round, 4) == round(g_committed, 4)),
    }
    print(f"  committed (0.6,1.0,1.5): gap={g_committed:.4f}")
    print(f"  eta0.6 raw {[round(x,4) for x in m_raw]}: gap={g_raw:.4f}")
    print(f"  eta0.6 rounded {[round(x,3) for x in m_round]}: gap={g_round:.4f}  "
          f"-> matches committed: {round(g_round,4)==round(g_committed,4)}")

    fname = "audit_estimator_check_acn.json" if use_acn else "audit_estimator_check.json"
    json.dump(out, open(os.path.join(RESULTS, fname), "w"), indent=2)
    print("\nwrote", os.path.join(RESULTS, fname))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Estimator stress-test.")
    ap.add_argument("--acn", action="store_true",
                    help="add the Caltech ACN-Data (US) site to check 1 and run checks 2 "
                         "and 3 on it; requires EQUICHARGE_ACN_JSON.")
    main(use_acn=ap.parse_args().acn)
