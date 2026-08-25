r"""Numerically verify the structural results the audit rests on, day by day.

The paper's characterization of the revenue optimum is a *chain of equalities*. Order
the tiers by margin and group tiers that share a margin into classes
$\mathcal{C}_1,\dots,\mathcal{C}_L$ with $v_1<\dots<v_L$, and let
$\mathcal{V}_l=\mathcal{C}_l\cup\dots\cup\mathcal{C}_L$ be the nested **top sets**. For
a set of sessions $S$ write $\hat g(S)$ for the largest total energy the site could
deliver to $S$ alone, that is, the same LP relaxation the oracle uses (per-step grid
energy, per-session power cap, per-session demand) with the objective restricted to $S$
and everyone else zeroed out.

Theorem (Chain characterization) says a feasible allocation is revenue-optimal **if and
only if** it saturates the chain, $E(\mathcal{V}_l)=\hat g(\mathcal{V}_l)$ for every
$l$; that consequently every revenue optimum delivers the same aggregate energy to each
margin class, $E(\mathcal{C}_l)=\hat g(\mathcal{V}_l)-\hat g(\mathcal{V}_{l+1})$; and
that the revenue-optimal *set* depends on the tariff only through the partition of tiers
into margin classes and their order, never through the margin magnitudes.

The Levers corollary then reads five consequences off that structure, and the paper
says each is verified numerically. This script is that verification. Under
$m_1<m_2<m_3$ the budget tier is the residual claimant, with aggregate exactly
$\hat g(\mathcal{I})-\hat g(\mathcal{I}_2\cup\mathcal{I}_3)$, and:

  (i)   capping the premium margin to the middle merges the top classes and leaves that
        aggregate **exactly** unchanged;
  (ii)  lifting the budget margin to parity merges the bottom classes, the pooled pair
        becomes the residual claimant with aggregate
        $\hat g(\mathcal{I})-\hat g(\mathcal{I}_3)$, and the division *within* the pool
        is unconstrained on the optimal face, so the shortfall is relocated inside the
        pool rather than removed;
  (iii) equal margins collapse the chain to $E(\mathcal{I})=\hat g(\mathcal{I})$, so no
        tier is a residual claimant and any remaining inequality is selection among
        optima;
  (iv)  the revenue difference against any fixed alternative is linear in the margins,
        so the price of fairness moves with the magnitudes while the optimal face does
        not;
  (v)   if scarcity does not bind, every objective serves every session fully and both
        the tier gap and the price of fairness are zero.

Every clause is checked on the realized day set, not argued. Clause (ii)'s
"unconstrained within the pool" is checked by optimizing over the face itself, which is
the same machinery ``audit_optimal_face.py`` uses and which lives here so both scripts
share one definition of the face.

Writes ``experiments/results/verify_theory.json``. Exits non-zero if any check fails.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter

import jax
import numpy as np

from chargax.equity import oracle as O
from chargax.equity import tariffs as T

RESULTS = os.path.join(os.path.dirname(__file__), "results")
#: Days in the realized day set. ``EQUICHARGE_N_DAYS`` overrides it so the whole
#: suite can be re-run at a larger day count without editing every script.
N_DAYS = int(os.environ.get("EQUICHARGE_N_DAYS") or 32)
KEY = jax.random.PRNGKey(7)          # the day set audit_offline.py uses
NAMES = ("budget", "mid", "premium")
CAPS = (1.5, 1.25, 1.0, 0.8)         # premium-margin caps to re-solve under
SUBSIDIES = (0.0, 0.2, 0.4, 0.6, 0.9)  # added to the budget tier's margin
#: Margin vectors with the SAME strict ordering, used for the magnitude-invariance
#: half of the theorem. These are the elasticity sweep's margins.
SAME_ORDER_MARGINS = ((0.9, 1.0, 1.1), (0.8, 1.0, 1.2), (0.7, 1.0, 1.4),
                      (0.6, 1.0, 1.5), (0.5, 1.0, 1.8), (0.4, 1.0, 2.0))
ABUNDANT_KW = 600.0                  # capacity at which scarcity cannot bind


# --------------------------------------------------------------------- the chain
def top_sets(margins) -> list:
    """The margin classes and nested top sets of a margin vector.

    Returns ``[(margin_l, C_l, V_l), ...]`` in strictly descending margin. Tiers that
    share a margin share a class, so a cap that compresses two tiers onto the same price
    merges them instead of ranking one above the other on a floating-point tie.
    """
    margins = [float(w) for w in margins]
    out, acc = [], []
    for level in sorted(set(margins), reverse=True):
        cls = tuple(g for g, w in enumerate(margins) if w == level)
        acc = acc + list(cls)
        out.append((level, cls, tuple(sorted(acc))))
    return out


def _base(cust, grid_kw, mpt, horizon):
    """The shared LP skeleton, ``_build_base_lp`` plus the bound list."""
    (var_index, ub, cust_vars, rows, cols, data, b_ub, n_rows, n_xy) = O._build_base_lp(
        cust, grid_kw, mpt, horizon
    )
    return dict(var_index=var_index, cust_vars=cust_vars, rows=rows, cols=cols,
                data=data, b_ub=b_ub, n_rows=n_rows, n_xy=n_xy,
                bounds=[(0.0, u) for u in ub])


def _tier_selector(cust, base, tiers):
    """The LP columns belonging to the sessions in ``tiers``."""
    tiers = set(tiers)
    return [v for i, vs in base["cust_vars"].items() if cust[i].group in tiers for v in vs]


def g_hat(cust, base, tiers) -> float:
    """Largest total energy deliverable to the tiers in ``tiers``, alone."""
    cols = _tier_selector(cust, base, tiers)
    if not cols:
        return 0.0
    c = np.zeros(base["n_xy"])
    c[cols] = -1.0
    res = O._solve(c, base["rows"], base["cols"], base["data"], base["b_ub"],
                   base["bounds"], base["n_xy"], base["n_rows"])
    return float(-res.fun)


def revenue_optimum(cust, base, margins):
    """Solve the revenue LP. Returns ``(x, revenue, delivered energy per tier)``."""
    c = np.zeros(base["n_xy"])
    for i, vs in base["cust_vars"].items():
        for v in vs:
            c[v] = -float(margins[cust[i].group])
    res = O._solve(c, base["rows"], base["cols"], base["data"], base["b_ub"],
                   base["bounds"], base["n_xy"], base["n_rows"])
    per_tier = np.zeros(len(margins))
    for i, vs in base["cust_vars"].items():
        per_tier[cust[i].group] += sum(res.x[v] for v in vs)
    return res.x, float(-res.fun), per_tier


def build_face(cust, base, margins, rel_tol: float = 1e-6) -> dict:
    """Append the chain equalities to the base LP, carving out the optimal face.

    By the chain characterization a feasible allocation is revenue-optimal exactly when
    it satisfies these, so the appended rows describe the optimal face itself, no more
    and no less. Each equality is written as the two inequalities the solver interface
    takes, with a tolerance scaled to the day's energy so floating-point slack in
    ``g_hat`` cannot make the face spuriously empty.
    """
    rows, cols = list(base["rows"]), list(base["cols"])
    data, b_ub = list(base["data"]), list(base["b_ub"])
    r = base["n_rows"]
    chain = top_sets(margins)
    bounds_info = []
    for _level, cls, V in chain:
        bound = g_hat(cust, base, V)
        tol = rel_tol * max(1.0, bound)
        members = _tier_selector(cust, base, V)
        for sign, rhs in ((1.0, bound + tol), (-1.0, -(bound - tol))):
            for v in members:
                rows.append(r); cols.append(v); data.append(sign)
            b_ub.append(rhs); r += 1
        bounds_info.append({"margin": _level, "class": [NAMES[g] for g in cls],
                            "top_set": [NAMES[g] for g in V], "g_hat_V": bound})
    return dict(rows=rows, cols=cols, data=data, b_ub=b_ub, n_rows=r,
                chain=chain, g_hat=bounds_info)


def solve_on_face(c_obj, face, base, extra_col: bool = False):
    """Minimize ``c_obj`` over the revenue-optimal face."""
    n_var = base["n_xy"] + (1 if extra_col else 0)
    bounds = base["bounds"] + ([(0.0, 1.0)] if extra_col else [])
    return O._solve(c_obj, face["rows"], face["cols"], face["data"], face["b_ub"],
                    bounds, n_var, face["n_rows"])


def class_energy_range(cust, base, face, tiers) -> tuple:
    """``(min, max)`` energy the tiers in ``tiers`` can receive over the optimal face."""
    cols = _tier_selector(cust, base, tiers)
    if not cols:
        return 0.0, 0.0
    c = np.zeros(base["n_xy"]); c[cols] = 1.0
    lo = float(solve_on_face(c, face, base).fun)
    hi = float(-solve_on_face(-c, face, base).fun)
    return lo, hi


# ------------------------------------------------------------------- the day set
def realized_days(env, n_days: int) -> list:
    """The realized arrival streams, skipping days a tier does not appear on."""
    grid = float(env.station.max_kw_throughput)
    mpt, horizon, n_groups = env.minutes_per_timestep, env.max_episode_steps, int(env.n_groups)
    days = []
    for i in range(n_days):
        cust = O.extract_arrival_stream(env, jax.random.fold_in(KEY, i))
        cust = [c for c in cust if c.desired_kwh > 1e-3]
        if not cust:
            continue
        if len({c.group for c in cust}) < n_groups:
            continue      # a tier with no sessions has no aggregate to compare
        days.append((cust, grid, mpt, horizon))
    return days


def _worst_tier_under(days, margins) -> Counter:
    """Which tier the revenue optimum serves worst, counted over the day set."""
    counts = Counter()
    for cust, grid, mpt, horizon in days:
        r = O.solve_oracle_lp(cust, grid, mpt, horizon, objective="profit",
                              price_by_group=tuple(margins), n_groups=len(margins))
        counts[int(np.argmin(r["group_means"]))] += 1
    return counts


# ----------------------------------------------------------------------- checks
class Checks:
    """Accumulates pass/fail verdicts so one run reports every failure, not the first."""

    def __init__(self):
        self.failures = []

    def ok(self, name: str, condition: bool, **detail):
        if not condition:
            self.failures.append({"check": name, **detail})
        return bool(condition)

    def close(self, name: str, a: float, b: float, scale: float = 1.0, **detail) -> bool:
        tol = 1e-6 * max(1.0, abs(scale))
        return self.ok(name, abs(a - b) <= tol, observed=a, expected=b, tol=tol, **detail)


def check_chain(days, margins, chk: Checks, label: str) -> dict:
    """The chain identity, the class-aggregate identity, and the "only if" direction."""
    worst_chain, worst_class, worst_rev = 0.0, 0.0, 0.0
    example = None
    for d, (cust, grid, mpt, horizon) in enumerate(days):
        base = _base(cust, grid, mpt, horizon)
        chain = top_sets(margins)
        _x, revenue, per_tier = revenue_optimum(cust, base, margins)
        bounds = [g_hat(cust, base, V) for _lv, _cls, V in chain]
        scale = max([1.0] + bounds)

        rows = []
        for l, (level, cls, V) in enumerate(chain):
            delivered = float(per_tier[list(V)].sum())
            worst_chain = max(worst_chain, abs(delivered - bounds[l]))
            chk.close(f"chain[{label}]", delivered, bounds[l], scale, day=d,
                      top_set=[NAMES[g] for g in V])
            # E(C_l) = g_hat(V_l) - g_hat(V_{l+1}), the class-aggregate identity. The
            # paper indexes the chain by ascending margin, so V_{l+1} is the SMALLER
            # top set; ``chain`` here runs descending, where that is the entry before.
            inner = bounds[l - 1] if l > 0 else 0.0
            cls_e = float(per_tier[list(cls)].sum())
            worst_class = max(worst_class, abs(cls_e - (bounds[l] - inner)))
            chk.close(f"class_aggregate[{label}]", cls_e, bounds[l] - inner, scale, day=d,
                      margin_class=[NAMES[g] for g in cls])
            rows.append({"margin": level, "class": [NAMES[g] for g in cls],
                         "top_set": [NAMES[g] for g in V], "E_V": delivered,
                         "g_hat_V": bounds[l], "E_C": cls_e,
                         "abs_dev_kwh": abs(delivered - bounds[l])})

        # The "only if" direction, read as a revenue identity. Writing the objective as
        # a telescoping sum over the chain gives revenue = sum_l (v_l - v_{l+1}) E(V_l),
        # so saturating the chain forces the optimal revenue and nothing else can reach
        # it. Checking it also catches a chain assembled in the wrong order.
        levels = [lv for lv, _c, _V in chain] + [0.0]
        predicted = sum((levels[l] - levels[l + 1]) * bounds[l] for l in range(len(chain)))
        worst_rev = max(worst_rev, abs(revenue - predicted))
        chk.close(f"revenue_identity[{label}]", revenue, predicted, max(1.0, revenue), day=d)
        if example is None:
            example = {"chain": rows, "revenue": revenue, "revenue_from_chain": predicted}
    return {"max_chain_dev_kwh": worst_chain, "max_class_aggregate_dev_kwh": worst_class,
            "max_revenue_dev": worst_rev, "example_day": example}


def tier_aggregates(days, margins) -> np.ndarray:
    """Per-day, per-tier delivered energy at the revenue optimum."""
    out = []
    for cust, grid, mpt, horizon in days:
        base = _base(cust, grid, mpt, horizon)
        _x, _rev, per_tier = revenue_optimum(cust, base, margins)
        out.append(per_tier)
    return np.array(out)


def main() -> int:
    # Imported here, not at module scope, so the helpers above can be used by the test
    # suite and by audit_optimal_face without pulling in the training stack.
    from experiments.run_experiments import _ref_env

    env = _ref_env(30.0, 8, 4)
    base_margins = tuple(float(w) for w in env.price_by_group)
    n_groups = len(base_margins)
    print("reference site 30 kW / 8 EVSEs, tier margins", base_margins)
    days = realized_days(env, N_DAYS)
    print("realized days usable: %d / %d\n" % (len(days), N_DAYS))
    if not days:
        print("no usable days")
        return 1

    chk = Checks()
    out = {"n_days": N_DAYS, "days_used": len(days), "day_key": 7,
           "base_margins": list(base_margins)}

    # --- Theorem: the chain, the class aggregates, and the revenue identity ---------
    print("=== Theorem (Chain characterization) ===")
    out["chain"] = check_chain(days, base_margins, chk, "value_weighted")
    print("  max |E(V_l) - g_hat(V_l)|          : %.3e kWh" % out["chain"]["max_chain_dev_kwh"])
    print("  max |E(C_l) - (g_hat_l - g_hat_l+1)|: %.3e kWh"
          % out["chain"]["max_class_aggregate_dev_kwh"])
    print("  max revenue identity deviation      : %.3e" % out["chain"]["max_revenue_dev"])
    print("  example day, the chain:")
    for row in out["chain"]["example_day"]["chain"]:
        print("    v=%.3f  C={%-22s} V={%-24s}  E(V)=%8.2f  g_hat(V)=%8.2f  E(C)=%8.2f"
              % (row["margin"], ",".join(row["class"]), ",".join(row["top_set"]),
                 row["E_V"], row["g_hat_V"], row["E_C"]))

    # --- Theorem: the optimal set ignores margin MAGNITUDES, only the order ---------
    print("\n=== Theorem (magnitudes do not move the optimal set) ===")
    ref_agg = tier_aggregates(days, base_margins)
    mag = []
    for margins in SAME_ORDER_MARGINS:
        agg = tier_aggregates(days, margins)
        dev = float(np.abs(agg - ref_agg).max())
        chk.ok("magnitude_invariance", dev <= 1e-6 * max(1.0, float(ref_agg.max())),
               margins=list(margins), max_abs_dev_kwh=dev)
        mag.append({"margins": list(margins), "max_abs_dev_kwh": dev})
        print("  margins=%-18s max |per-tier energy - reference| = %.3e kWh"
              % (str(tuple(margins)), dev))
    out["magnitude_invariance"] = mag

    base_worst = _worst_tier_under(days, base_margins)
    out["status_quo_worst_tier"] = {NAMES[t]: base_worst.get(t, 0) for t in range(n_groups)}
    print("  status-quo worst-served tier:", out["status_quo_worst_tier"])

    # --- Corollary, the base claim: budget is the residual claimant -----------------
    print("\n=== Corollary (Levers), base claim: budget is the residual claimant ===")
    residual_dev, budget_agg_ref = 0.0, []
    for d, (cust, grid, mpt, horizon) in enumerate(days):
        base = _base(cust, grid, mpt, horizon)
        _x, _rev, per_tier = revenue_optimum(cust, base, base_margins)
        expected = g_hat(cust, base, (0, 1, 2)) - g_hat(cust, base, (1, 2))
        residual_dev = max(residual_dev, abs(per_tier[0] - expected))
        chk.close("residual_claimant", float(per_tier[0]), expected,
                  max(1.0, expected), day=d)
        budget_agg_ref.append(float(per_tier[0]))
    out["residual_claimant"] = {"max_abs_dev_kwh": residual_dev,
                                "budget_energy_kwh_mean": float(np.mean(budget_agg_ref))}
    print("  max |E(budget) - (g_hat(I) - g_hat(mid,premium))| = %.3e kWh" % residual_dev)

    # --- (i) a premium cap leaves budget's aggregate EXACTLY unchanged --------------
    print("\n=== Corollary (i): a premium cap merges the top classes and does not move budget ===")
    cap_rows = []
    for cap in CAPS:
        margins = T.cap_premium(float(cap), base_margins)
        agg = tier_aggregates(days, margins)
        dev = float(np.abs(agg[:, 0] - np.array(budget_agg_ref)).max())
        counts = _worst_tier_under(days, margins)
        lowest = [NAMES[g] for g, w in enumerate(margins) if w == min(margins)]
        dominant = NAMES[max(counts, key=counts.get)]
        share = counts[max(counts, key=counts.get)] / len(days)
        # The corollary's content is the exact invariance of budget's aggregate while
        # budget remains uniquely lowest. Below that, budget's class has merged and the
        # claim no longer applies.
        applies = lowest == ["budget"]
        if applies:
            chk.ok("corollary_i_budget_energy_unchanged",
                   dev <= 1e-6 * max(1.0, float(np.max(agg))), cap=float(cap),
                   max_abs_dev_kwh=dev)
            chk.ok("corollary_i_worst_tier_unmoved", dominant == "budget",
                   cap=float(cap), observed_worst=dominant)
        cap_rows.append({"cap": float(cap), "margins": list(margins),
                         "lowest_margin_tiers": lowest, "claim_applies": applies,
                         "budget_energy_max_abs_dev_kwh": dev,
                         "observed_worst": dominant, "observed_worst_share": share,
                         "worst_tier_counts": {NAMES[t]: counts.get(t, 0) for t in range(n_groups)}})
        print("  cap=%.2f margins=%-18s budget energy dev=%.3e kWh  worst=%s (%.0f%%)%s"
              % (cap, str(tuple(round(w, 2) for w in margins)), dev, dominant,
                 100 * share, "" if applies else "   [budget no longer uniquely lowest]"))
    out["corollary_i_premium_cap"] = cap_rows

    # --- (ii) a budget subsidy pools the bottom pair and relocates the shortfall -----
    print("\n=== Corollary (ii): a budget subsidy pools the bottom pair ===")
    sub_rows = []
    for s in SUBSIDIES:
        margins = T.low_income_subsidy(float(s), base_margins)
        classes = top_sets(margins)
        pooled = classes[-1][1]                      # the lowest margin class
        pool_dev, split_range = 0.0, 0.0
        for d, (cust, grid, mpt, horizon) in enumerate(days):
            base = _base(cust, grid, mpt, horizon)
            _x, _rev, per_tier = revenue_optimum(cust, base, margins)
            above = tuple(g for g in range(n_groups) if g not in pooled)
            expected = g_hat(cust, base, tuple(range(n_groups))) - (
                g_hat(cust, base, above) if above else 0.0)
            pool_dev = max(pool_dev, abs(float(per_tier[list(pooled)].sum()) - expected))
            chk.close("corollary_ii_pool_is_residual_claimant",
                      float(per_tier[list(pooled)].sum()), expected, max(1.0, expected),
                      subsidy=float(s), day=d)
            if len(pooled) > 1 and d == 0:
                # The division inside the pool is unconstrained on the optimal face.
                face = build_face(cust, base, margins)
                lo, hi = class_energy_range(cust, base, face, (pooled[0],))
                split_range = hi - lo
        counts = _worst_tier_under(days, margins)
        dominant = NAMES[max(counts, key=counts.get)]
        share = counts[max(counts, key=counts.get)] / len(days)
        if len(pooled) > 1:
            chk.ok("corollary_ii_split_unconstrained", split_range > 1e-6,
                   subsidy=float(s), split_range_kwh=split_range)
        sub_rows.append({"subsidy": float(s), "margins": list(margins),
                         "pooled_bottom_class": [NAMES[g] for g in pooled],
                         "pool_energy_max_abs_dev_kwh": pool_dev,
                         "within_pool_energy_range_kwh_day0": split_range,
                         "observed_worst": dominant, "observed_worst_share": share,
                         "worst_tier_counts": {NAMES[t]: counts.get(t, 0) for t in range(n_groups)}})
        print("  subsidy=%.2f margins=%-18s pool={%s} residual dev=%.2e  "
              "within-pool range=%.1f kWh  worst=%s (%.0f%%)"
              % (s, str(tuple(round(w, 2) for w in margins)),
                 ",".join(NAMES[g] for g in pooled), pool_dev, split_range,
                 dominant, 100 * share))
    out["corollary_ii_budget_subsidy"] = sub_rows
    pooled_levels = [r for r in sub_rows if len(r["pooled_bottom_class"]) > 1]
    out["harm_relocated_within_pool"] = bool(pooled_levels)
    if pooled_levels:
        print("  -> the shortfall is relocated inside the pooled bottom pair, not removed;"
              " which member is worst is a tie-break")

    # --- (iii) equal margins collapse the chain to a single link --------------------
    print("\n=== Corollary (iii): income-neutral pricing collapses the chain ===")
    flat = T.flat(base_margins)
    flat_chain = top_sets(flat)
    chk.ok("corollary_iii_single_link", len(flat_chain) == 1, n_links=len(flat_chain))
    tot_dev, no_claimant = 0.0, True
    for d, (cust, grid, mpt, horizon) in enumerate(days):
        base = _base(cust, grid, mpt, horizon)
        _x, _rev, per_tier = revenue_optimum(cust, base, flat)
        total = g_hat(cust, base, tuple(range(n_groups)))
        tot_dev = max(tot_dev, abs(per_tier.sum() - total))
        chk.close("corollary_iii_total_energy", float(per_tier.sum()), total,
                  max(1.0, total), day=d)
        if d == 0:
            face = build_face(cust, base, flat)
            # No tier is a residual claimant: every tier's aggregate is free to move.
            for g in range(n_groups):
                lo, hi = class_energy_range(cust, base, face, (g,))
                no_claimant &= (hi - lo) > 1e-6
    chk.ok("corollary_iii_no_residual_claimant", no_claimant)
    out["corollary_iii_flat"] = {"chain_links": len(flat_chain),
                                 "total_energy_max_abs_dev_kwh": tot_dev,
                                 "every_tier_free_on_face_day0": bool(no_claimant)}
    print("  chain links under equal margins: %d, total-energy dev %.2e kWh, "
          "every tier free on the face: %s" % (len(flat_chain), tot_dev, no_claimant))

    # --- (iv) magnitudes move the price of fairness, not the face -------------------
    print("\n=== Corollary (iv): magnitudes move the price of fairness, not the face ===")
    pof = []
    for margins in (base_margins,) + SAME_ORDER_MARGINS:
        rev_opt, rev_mm = [], []
        for cust, grid, mpt, horizon in days:
            p = O.solve_oracle_lp(cust, grid, mpt, horizon, objective="profit",
                                  price_by_group=tuple(margins), n_groups=n_groups)
            m = O.solve_oracle_lp(cust, grid, mpt, horizon, objective="maximin",
                                  price_by_group=tuple(margins), n_groups=n_groups)
            rev_opt.append(p["revenue_value"]); rev_mm.append(m["revenue_value"])
        rev_opt, rev_mm = np.array(rev_opt), np.array(rev_mm)
        value = float(np.median((rev_opt - rev_mm) / (rev_opt + 1e-9)) * 100)
        pof.append({"margins": list(margins), "price_of_fairness_pct": value})
        print("  margins=%-18s price of fairness = %5.2f%%" % (str(tuple(margins)), value))
    spread = max(p["price_of_fairness_pct"] for p in pof) - min(
        p["price_of_fairness_pct"] for p in pof)
    chk.ok("corollary_iv_pof_moves_with_magnitude", spread > 1e-3, spread_pct=spread)
    out["corollary_iv_price_of_fairness"] = {"rows": pof, "spread_pct": spread}
    print("  -> the face is invariant (above) while the price of fairness spans %.2f points"
          % spread)

    # --- (v) without binding scarcity the objective stops mattering ------------------
    print("\n=== Corollary (v): without binding scarcity the harm is zero ===")
    ample_env = _ref_env(ABUNDANT_KW, 8, 4)
    ample_days = realized_days(ample_env, min(N_DAYS, 32))
    # "Serves every session fully" has to be read against each session's OWN ceiling.
    # A driver who plugs in for one five-minute step cannot take a full charge however
    # large the grid connection is, and that shortfall is not rationing. What the clause
    # asserts is that scarcity stops mediating, so every session receives its
    # individually feasible maximum and the choice of objective no longer decides
    # anything. Both halves are checked directly.
    worst_shortfall, worst_objective_gap, pof_max = 0.0, 0.0, 0.0
    for d, (cust, grid, mpt, horizon) in enumerate(ample_days):
        base = _base(cust, grid, mpt, horizon)
        dt_h = mpt / 60.0
        own_cap = np.array([
            min(c.desired_kwh,
                max(min(c.arrival_step + c.window_steps, horizon) - c.arrival_step, 0)
                * c.pmax_kw * dt_h)
            for c in cust])

        def per_session(margins_or_none):
            """Delivered energy per session under revenue, or under utilitarian."""
            c_obj = np.zeros(base["n_xy"])
            for i, vs in base["cust_vars"].items():
                w = (float(base_margins[cust[i].group]) if margins_or_none
                     else 1.0 / cust[i].desired_kwh)
                for v in vs:
                    c_obj[v] = -w
            res = O._solve(c_obj, base["rows"], base["cols"], base["data"], base["b_ub"],
                           base["bounds"], base["n_xy"], base["n_rows"])
            return np.array([sum(res.x[v] for v in base["cust_vars"].get(i, []))
                             for i in range(len(cust))])

        rev_e, util_e = per_session(True), per_session(False)
        scale = max(1.0, float(own_cap.sum()))
        shortfall = float(np.abs(rev_e - own_cap).max())
        objective_gap = float(np.abs(rev_e - util_e).max())
        worst_shortfall = max(worst_shortfall, shortfall)
        worst_objective_gap = max(worst_objective_gap, objective_gap)
        chk.ok("corollary_v_every_session_at_its_own_ceiling",
               shortfall <= 1e-6 * scale, day=d, max_abs_dev_kwh=shortfall)
        chk.ok("corollary_v_objective_does_not_matter",
               objective_gap <= 1e-6 * scale, day=d, max_abs_dev_kwh=objective_gap)

        p = O.solve_oracle_lp(cust, grid, mpt, horizon, objective="profit",
                              price_by_group=base_margins, n_groups=n_groups)
        m = O.solve_oracle_lp(cust, grid, mpt, horizon, objective="maximin",
                              price_by_group=base_margins, n_groups=n_groups)
        pof = (p["revenue_value"] - m["revenue_value"]) / (p["revenue_value"] + 1e-9)
        pof_max = max(pof_max, abs(float(pof)))
        chk.ok("corollary_v_price_of_fairness_zero", abs(pof) <= 1e-6, day=d, pof=float(pof))

    out["corollary_v_abundant"] = {
        "grid_kw": ABUNDANT_KW, "days": len(ample_days),
        "max_shortfall_vs_own_ceiling_kwh": worst_shortfall,
        "max_revenue_minus_utilitarian_kwh": worst_objective_gap,
        "max_abs_price_of_fairness": pof_max,
    }
    print("  at %.0f kW: every session is at its own ceiling to %.2e kWh, revenue and"
          % (ABUNDANT_KW, worst_shortfall))
    print("  utilitarian deliver the same energy to %.2e kWh, price of fairness %.2e"
          % (worst_objective_gap, pof_max))
    print("  (residual tier differences at this capacity come from dwell windows and per-car")
    print("   power caps, not from rationing, which is exactly what the clause asserts)")

    out["failures"] = chk.failures
    out["all_checks_passed"] = not chk.failures
    path = os.path.join(RESULTS, "verify_theory.json")
    json.dump(out, open(path, "w"), indent=2)
    print("\nwrote", path)
    if chk.failures:
        by_check = Counter(f["check"] for f in chk.failures)
        print("FAILED: %d check(s) did not hold: %s"
              % (len(chk.failures), dict(by_check)))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
