r"""How much of the reported tier gap is the revenue optimum, and how much is the solver?

Every disparity number in this paper is read off *one* allocation, the vertex HiGHS
happens to return for the revenue-maximizing LP. That is a real hazard for an audit.
A linear program's optimum is generically a whole face, not a point, and different
vertices of that face can distribute the same revenue-optimal energy very differently
across the tiers. If the reported gap were an artifact of the solver's pivoting rule,
the headline would evaporate under a different solver, and a reader is entitled to
ask.

This script removes the question by characterizing the face and optimizing over it.
The revenue optimum is pinned by the chain of equalities (see ``verify_theory.py``),

    E(V_l) = g_hat(V_l)      for every top set V_l,

and because the revenue objective telescopes over the chain,

    revenue = sum_l (p_l - p_{l+1}) E(V_l),      p_{L+1} = 0,

a feasible allocation is revenue-optimal **exactly when** it satisfies all of those
equalities. So appending them to the base LP as constraint rows carves out the optimal
face itself, no more and no less. We then re-solve on that face to get, for each tier,
the largest and smallest mean satisfaction any revenue-optimal allocation could give
it, and the largest and smallest tier gap Gamma any revenue-optimal allocation could
exhibit.

Both ends of the Gamma range are exact linear programs, not samples. Gamma is a max
over ordered tier pairs of ``mean_g - mean_h``, so its maximum over the face is the
largest of those pairwise maxima, and its minimum is an epigraph LP, minimize ``t``
subject to ``t >= mean_g - mean_h`` for every pair. What comes out is a *bound*. If
the lower end of the range is still large, the harm is a property of revenue
maximization and no choice of solver, tie-break, or vertex can make it go away. If the
lower end is near zero, the honest reading is that revenue maximization *permits* a
fair allocation and the reported gap is a tie-break, which is a finding in its own
right and one an operator could act on directly.

The same is computed under each audited tariff, so the rank-ordering mechanism can be
read as a range rather than as a paired difference of two point estimates.

Writes ``experiments/results/audit_optimal_face.json``.
"""

from __future__ import annotations

import json
import os

import jax
import numpy as np

from chargax.equity import oracle as O
from chargax.equity import tariffs as T
from experiments.verify_theory import (_base, build_face, realized_days,
                                       solve_on_face)

RESULTS = os.path.join(os.path.dirname(__file__), "results")
#: Days in the realized day set. ``EQUICHARGE_N_DAYS`` overrides it so the whole
#: suite can be re-run at a larger day count without editing every script.
N_DAYS = int(os.environ.get("EQUICHARGE_N_DAYS") or 32)
NAMES = ("budget", "mid", "premium")


def tariff_menu(base_margins) -> list:
    """The audited tariffs, named as the paper names them."""
    return [
        ("value_weighted", tuple(base_margins)),
        ("premium_cap_1.0", T.cap_premium(1.0, base_margins)),
        ("budget_subsidy_0.4_parity", T.low_income_subsidy(0.4, base_margins)),
        ("budget_subsidy_0.9_overshoot", T.low_income_subsidy(0.9, base_margins)),
        ("flat_income_neutral", T.flat(base_margins)),
    ]


def _tier_mean_coeffs(cust, base, n_groups) -> list:
    """Per-tier mean satisfaction as a linear functional of the LP variables.

    ``mean_g = (1/N_g) sum_{i in tier g} (1/desired_i) sum_t x[i,t]``, so each tier's
    mean is linear and can be maximized or minimized directly.
    """
    members = {g: [] for g in range(n_groups)}
    for i in base["cust_vars"]:
        members[cust[i].group].append(i)
    coeffs = []
    for g in range(n_groups):
        c = np.zeros(base["n_xy"])
        idxs = members[g]
        if idxs:
            for i in idxs:
                w = 1.0 / (len(idxs) * cust[i].desired_kwh)
                for v in base["cust_vars"][i]:
                    c[v] = w
        coeffs.append(c)
    return coeffs


def face_ranges(cust, base, margins, n_groups) -> dict:
    """Per-tier mean ranges and the exact Gamma range over the revenue-optimal face."""
    face = build_face(cust, base, margins)
    coeffs = _tier_mean_coeffs(cust, base, n_groups)
    present = [g for g in range(n_groups) if coeffs[g].any()]

    tier_min, tier_max = {}, {}
    for g in present:
        tier_max[g] = float(-solve_on_face(-coeffs[g], face, base).fun)
        tier_min[g] = float(solve_on_face(coeffs[g], face, base).fun)

    # Gamma_max: the largest pairwise gap any revenue-optimal allocation can show.
    # The matching minima are collected in the same pass, they are what bounds the
    # day-averaged estimator from below.
    pair_max, pair_min = {}, {}
    for g in present:
        for h in present:
            if g == h:
                continue
            diff = coeffs[g] - coeffs[h]
            pair_max[(g, h)] = float(-solve_on_face(-diff, face, base).fun)
            pair_min[(g, h)] = float(solve_on_face(diff, face, base).fun)
    gamma_max = max(pair_max.values()) if pair_max else 0.0

    # Gamma_min: epigraph LP, minimize t subject to mean_g - mean_h - t <= 0.
    rows = list(face["rows"]); cols = list(face["cols"])
    data = list(face["data"]); b_ub = list(face["b_ub"])
    r = face["n_rows"]
    t_col = base["n_xy"]
    for g in present:
        for h in present:
            if g == h:
                continue
            diff = coeffs[g] - coeffs[h]
            for v in np.nonzero(diff)[0]:
                rows.append(r); cols.append(int(v)); data.append(float(diff[v]))
            rows.append(r); cols.append(t_col); data.append(-1.0)
            b_ub.append(0.0); r += 1
    epi = dict(rows=rows, cols=cols, data=data, b_ub=b_ub, n_rows=r)
    c = np.zeros(base["n_xy"] + 1); c[t_col] = 1.0
    gamma_min = float(solve_on_face(c, epi, base, extra_col=True).fun)

    # The vertex the plain revenue LP hands back, i.e. the number the paper reports.
    r_lp = O.solve_oracle_lp(
        cust, grid_limit_kw=base["_grid"], minutes_per_timestep=base["_mpt"],
        horizon_steps=base["_horizon"], objective="profit",
        price_by_group=tuple(margins), n_groups=n_groups,
    )
    reported = r_lp["group_means"]
    return {
        "tier_min": [tier_min.get(g, float("nan")) for g in range(n_groups)],
        "tier_max": [tier_max.get(g, float("nan")) for g in range(n_groups)],
        "reported": [float(v) for v in reported],
        "gamma_reported": float(max(reported) - min(reported)),
        "gamma_min": max(0.0, gamma_min),
        "gamma_max": gamma_max,
        "pairwise_max": {f"{NAMES[g]}-{NAMES[h]}": v for (g, h), v in pair_max.items()},
        "pairwise_min": {f"{NAMES[g]}-{NAMES[h]}": v for (g, h), v in pair_min.items()},
        "g_hat": face["g_hat"],
    }


def run_tariff(days, margins, n_groups) -> dict:
    per_day = []
    for cust, grid, mpt, horizon in days:
        base = _base(cust, grid, mpt, horizon)
        base["_grid"], base["_mpt"], base["_horizon"] = grid, mpt, horizon
        per_day.append(face_ranges(cust, base, margins, n_groups))

    tier_min = np.array([d["tier_min"] for d in per_day])
    tier_max = np.array([d["tier_max"] for d in per_day])
    reported = np.array([d["reported"] for d in per_day])
    g_rep = np.array([d["gamma_reported"] for d in per_day])
    g_lo = np.array([d["gamma_min"] for d in per_day])
    g_hi = np.array([d["gamma_max"] for d in per_day])

    # The paper's estimator is max-min of the DAY-AVERAGED tier means. Days are
    # independent, so the face is a product across days and each tier's day-averaged
    # mean ranges over the average of its per-day range, exactly.
    avg_lo, avg_hi = tier_min.mean(axis=0), tier_max.mean(axis=0)

    # Upper end of the day-averaged Gamma: pick, for one ordered pair, the per-day
    # allocation that maximizes that pair's gap. Achievable, hence exact.
    pairs = sorted({k for d in per_day for k in d["pairwise_max"]})
    pair_avg_max = {p: float(np.mean([d["pairwise_max"][p] for d in per_day])) for p in pairs}
    pair_avg_min = {p: float(np.mean([d["pairwise_min"][p] for d in per_day])) for p in pairs}
    dayavg_gamma_max = max(pair_avg_max.values()) if pair_avg_max else 0.0
    # Lower end: any allocation shows at least the best pair's average gap, so the
    # largest of the pairwise averaged minima is a certified floor. It is a bound
    # rather than an exact minimum, and is labelled as such.
    dayavg_gamma_floor = max([0.0] + [v for v in pair_avg_min.values()])

    return {
        "margins": [float(w) for w in margins],
        "days": len(per_day),
        "day_averaged": {
            "reported_tier_means": reported.mean(axis=0).tolist(),
            "tier_mean_min_over_face": avg_lo.tolist(),
            "tier_mean_max_over_face": avg_hi.tolist(),
            "gamma_reported": float(reported.mean(axis=0).max() - reported.mean(axis=0).min()),
            "gamma_max_over_face": dayavg_gamma_max,
            "gamma_certified_floor": dayavg_gamma_floor,
            "pairwise_gap_max_over_face": pair_avg_max,
            "pairwise_gap_min_over_face": pair_avg_min,
        },
        "per_day": {
            "gamma_reported": {"median": float(np.median(g_rep)), "mean": float(g_rep.mean())},
            "gamma_min": {"median": float(np.median(g_lo)), "mean": float(g_lo.mean()),
                          "max": float(g_lo.max())},
            "gamma_max": {"median": float(np.median(g_hi)), "mean": float(g_hi.mean()),
                          "min": float(g_hi.min())},
            "reported_is_inside_range": bool(np.all(g_rep <= g_hi + 1e-6)
                                             and np.all(g_rep >= g_lo - 1e-6)),
            "days_where_face_is_a_point": int(np.sum(g_hi - g_lo <= 1e-6)),
        },
    }


def main():
    # Imported here, not at module scope, so the helpers above can be used by the
    # test suite without pulling in the training stack.
    from experiments.run_experiments import _ref_env

    env = _ref_env(30.0, 8, 4)
    n_groups = int(env.n_groups)
    base_margins = tuple(float(w) for w in env.price_by_group)
    print("reference site 30 kW / 8 EVSEs, tier margins", base_margins)
    days = realized_days(env, N_DAYS)
    print("realized days usable: %d / %d\n" % (len(days), N_DAYS))

    out = {"n_days": N_DAYS, "days_used": len(days), "day_key": 7,
           "base_margins": list(base_margins), "tariffs": {}}
    for name, margins in tariff_menu(base_margins):
        res = run_tariff(days, margins, n_groups)
        out["tariffs"][name] = res
        da = res["day_averaged"]
        print("=== %s  margins=%s ===" % (name, tuple(round(w, 2) for w in margins)))
        print("  tier means reported : %s" % [round(v, 3) for v in da["reported_tier_means"]])
        print("  tier means over face: %s .. %s"
              % ([round(v, 3) for v in da["tier_mean_min_over_face"]],
                 [round(v, 3) for v in da["tier_mean_max_over_face"]]))
        print("  Gamma reported = %.3f,  over the face Gamma in [%.3f, %.3f]"
              % (da["gamma_reported"], da["gamma_certified_floor"], da["gamma_max_over_face"]))
        print("  per-day Gamma range (median): [%.3f, %.3f];  face is a single point on %d/%d days"
              % (res["per_day"]["gamma_min"]["median"], res["per_day"]["gamma_max"]["median"],
                 res["per_day"]["days_where_face_is_a_point"], res["days"]))
        if not res["per_day"]["reported_is_inside_range"]:
            print("  WARNING: the reported gap fell outside the computed face range on some day")
        print()

    # The headline, restated as a bound rather than as a paired difference. The
    # quantity that carries the claim is the FLOOR, the gap revenue maximization
    # forces on every optimal allocation. A ceiling says what an adversarial operator
    # could do, which is a different and weaker statement.
    vw = out["tariffs"]["value_weighted"]["day_averaged"]
    flat = out["tariffs"]["flat_income_neutral"]["day_averaged"]
    out["headline"] = {
        "value_weighted_gamma_floor": vw["gamma_certified_floor"],
        "value_weighted_gamma_ceiling": vw["gamma_max_over_face"],
        "flat_gamma_floor": flat["gamma_certified_floor"],
        "flat_gamma_ceiling": flat["gamma_max_over_face"],
        # What the audit can assert without appealing to a tie-break: under the
        # status quo every revenue-optimal allocation is at least this unequal,
        # under income-neutral pricing none of them has to be.
        "forced_gap_removed_by_income_neutrality": bool(
            flat["gamma_certified_floor"] < vw["gamma_certified_floor"]),
        "forced_gap_reduction": float(vw["gamma_certified_floor"]
                                      - flat["gamma_certified_floor"]),
        # The conservative reading, worst case on one side against best case on the
        # other. It fails whenever the income-neutral optimum is a large face, which
        # is exactly what a flat tariff produces, so a False here is not a refutation
        # of the claim, it is the reason the claim has to be stated as "no longer
        # forced" rather than "cannot occur".
        "separation_survives_worst_against_best": bool(
            vw["gamma_certified_floor"] > flat["gamma_max_over_face"]),
    }
    print("=== the headline as a bound ===")
    print("  status quo:      EVERY revenue-optimal allocation has Gamma >= %.3f"
          % vw["gamma_certified_floor"])
    print("  income-neutral:  revenue optimality forces only Gamma >= %.3f "
          "(it permits up to %.3f, it does not require it)"
          % (flat["gamma_certified_floor"], flat["gamma_max_over_face"]))
    print("  -> income neutrality removes %.3f of FORCED tier gap. What it cannot do is "
          "forbid an unequal\n     allocation, it only stops revenue from paying for one."
          % out["headline"]["forced_gap_reduction"])

    path = os.path.join(RESULTS, "audit_optimal_face.json")
    json.dump(out, open(path, "w"), indent=2)
    print("\nwrote", path)


if __name__ == "__main__":
    main()
