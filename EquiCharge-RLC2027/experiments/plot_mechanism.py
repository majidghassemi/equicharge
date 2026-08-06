r"""Regenerate the two FAccT paper figures from the CANONICAL run (audit_mechanism.json).

The paper's oracle and tariff figures were previously built from results.json (the RL-era
full study) and encoded the superseded "spread" framing. These two figures now come from the
one fixed day set / day-averaged estimator used by every table in the paper, so figure and
table cannot disagree, and rate_design.png is redesigned around the rank mechanism:

  oracle_tradeoff.png -- per-tier satisfaction by objective + the revenue equity costs.
  rate_design.png     -- the tier gap under four tariffs, showing it tracks RANK POSITION
                         (is the budget tier uniquely lowest?), not the margin spread: the
                         premium cap has a SMALLER spread yet a LARGER gap than the subsidy.

Reuses the shared, colorblind-safe house style (experiments/plot_style.py).
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments import plot_style as S

RESULTS = os.path.join(os.path.dirname(__file__), "results")
FIGDIR = os.path.join(RESULTS, "figures")
os.makedirs(FIGDIR, exist_ok=True)
S.apply_style()

# Distinctive teal + clay system (validated with the dataviz palette checker: the tier
# ramp passes the ordinal checks; clay vs teal passes CVD dE 29.9). Teal = tiers / relief,
# clay = cost / harm, so both figures read as one system and it is not the default blue/orange.
TIER = {"budget": "#66BDB2", "mid": "#2E9284", "premium": "#124F49"}  # ordinal ramp, light->dark
TIER_ORDER = ["budget", "mid", "premium"]
HARM = "#BF5A38"    # clay: the cost / the budget tier still lowest-ranked
RELIEF = "#12A08A"  # teal: relief / the budget tier rescued from lowest rank


# bar()/barh() take ecolor and capsize directly; everything else goes via error_kw.
ERRBAR = dict(ecolor="#4d4d4d", capsize=2.5,
              error_kw=dict(elinewidth=1.1, capthick=1.1, zorder=5))


def _load(name):
    with open(os.path.join(RESULTS, name)) as f:
        return json.load(f)


def _load_optional(name):
    try:
        return _load(name)
    except FileNotFoundError:
        return None


def _asym(points, cis):
    """Percentile CIs are asymmetric, so matplotlib needs [[below...],[above...]].
    Clipped at 0 because a resampled bound can land a hair past the point estimate."""
    lo = [max(p - c[0], 0.0) for p, c in zip(points, cis)]
    hi = [max(c[1] - p, 0.0) for p, c in zip(points, cis)]
    return np.array([lo, hi])


def plot_oracle(M):
    """Minimal: per-tier satisfaction by objective (left) + operator revenue (right).
    Numbers (gap 0.57, PoF ~9%) live in the caption/text, not on the figure."""
    box1 = M["box1"]
    objs = [("profit", "profit-opt."), ("utilitarian", "utilitarian"),
            ("maximin", "tier maximin"), ("egalitarian_equal", "strict equal.")]
    xl = [l for _, l in objs]
    x = np.arange(len(objs)); w = 0.26
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.0),
                                  gridspec_kw={"width_ratios": [2.1, 1]})
    for j, t in enumerate(TIER_ORDER):
        vals = [box1[k]["tier_means"][j] for k, _ in objs]
        cis = [box1[k].get("tier_means_ci95", [None] * 3)[j] for k, _ in objs]
        yerr = _asym(vals, cis) if all(c for c in cis) else None
        ax.bar(x + (j - 1) * w, vals, w, color=TIER[t], edgecolor="white",
               linewidth=0.8, zorder=3, label=t, yerr=yerr, **(ERRBAR if yerr is not None else {}))
    S.clean(ax)
    ax.set_xticks(x); ax.set_xticklabels(xl, rotation=18, ha="right")
    ax.set_ylabel("satisfaction"); ax.set_ylim(0, 1.06)
    ax.legend(loc="upper right", fontsize=9.5, frameon=False, handlelength=1.0,
              labelspacing=0.3, borderaxespad=0.1)
    revs = [box1[k]["revenue"] for k, _ in objs]
    rcis = [box1[k].get("revenue_ci95") for k, _ in objs]
    rerr = _asym(revs, rcis) if all(rcis) else None
    bars = ax2.bar(x, revs, 0.62, color=S.MUTED, edgecolor="white", linewidth=0.8, zorder=3,
                   yerr=rerr, **(ERRBAR if rerr is not None else {}))
    bars[0].set_color(HARM)  # profit optimum earns the premium -- the cost of fairness
    S.clean(ax2)
    ax2.set_xticks(x); ax2.set_xticklabels(xl, rotation=18, ha="right")
    ax2.set_ylabel("revenue (€/day)"); ax2.set_ylim(0, max(revs) * 1.18)
    S.savefig(fig, os.path.join(FIGDIR, "oracle_tradeoff.png"))


def plot_rate_design(M):
    by = {r["config"]: r for r in M["mechanism"]}
    # worst -> best, matching Table (tariff): status quo, premium cap, budget subsidy, flat.
    order = [
        ("tiered\n(status quo)", "status_quo_distinct"),
        ("cap top tier\n(to middle)", "premium_capped_to_mid"),
        ("subsidize bottom\n(to parity)", "budget_tied_to_mid"),
        ("income-neutral\n(flat)", "fully_flat"),
    ]
    labels = [l for l, _ in order]
    gaps = [by[c]["gap"] for _, c in order]
    uniq = [by[c]["budget_uniquely_lowest"] for _, c in order]
    # Color IS the encoding: is the budget tier still the strictly lowest rank (harm) or not?
    # The spread values and the rank-vs-spread argument move to the caption/text.
    colors = [HARM if u else RELIEF for u in uniq]
    floor = by["fully_flat"]["gap"]

    cis = [by[c].get("gap_ci95") for _, c in order]
    xerr = _asym(gaps, cis) if all(cis) else None

    y = np.arange(len(order))[::-1]
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    ax.barh(y, gaps, 0.62, color=colors, edgecolor="white", linewidth=0.8, zorder=3,
            xerr=xerr, **(ERRBAR if xerr is not None else {}))
    # Label past the CI whisker, not the bar end, so the two never collide.
    tips = [c[1] for c in cis] if xerr is not None else gaps
    for yi, g, t in zip(y, gaps, tips):
        ax.annotate("%.2f" % g, (t + 0.014, yi), va="center", ha="left",
                    fontsize=10.5, color=S.INK)
    ax.axvline(floor, ls=(0, (4, 2)), color=S.MUTED, lw=1.5, zorder=2)
    S.clean(ax, ygrid=False)
    ax.grid(axis="x", which="major"); ax.grid(axis="y", visible=False)
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlabel("profit-optimal tier gap")
    ax.set_xlim(0, max(tips) * 1.20)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=HARM, label="budget still lowest"),
                       Patch(color=RELIEF, label="budget rescued")],
              loc="lower right", fontsize=9.5, frameon=False, handlelength=1.0,
              labelspacing=0.3)
    S.savefig(fig, os.path.join(FIGDIR, "rate_design.png"))


def plot_grounding(M):
    """The gap is flat across the WTP elasticity -> structural, not a calibration artifact.
    The one number (PoF 3->13%) stays in the table; the figure is just the flat line + cliff."""
    inv = sorted(M["elasticity_invariance"], key=lambda d: d["eta"])
    etas = [d["eta"] for d in inv]
    gaps = [d["gap"] for d in inv]
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    ax.plot(etas, gaps, "-", color=HARM, lw=2.6, zorder=3)
    # eta>0: harm persists (clay); eta=0: no ordering, harm gone (teal).
    ax.plot(etas[1:], gaps[1:], "o", color=HARM, ms=8, zorder=4)
    ax.plot([0], [gaps[0]], "o", color=RELIEF, ms=10, zorder=5)
    ax.annotate("no ordering\n(flat tariff)", (0, gaps[0]), (0.12, gaps[0] + 0.10),
                fontsize=9.5, color=RELIEF, va="bottom",
                arrowprops=dict(arrowstyle="->", color=RELIEF, lw=1.1))
    S.clean(ax)
    ax.set_xlabel(r"willingness-to-pay elasticity  $\eta$")
    ax.set_ylabel("tier gap (day-averaged)")
    ax.set_ylim(0, 0.66); ax.set_xlim(-0.03, 1.03)
    S.savefig(fig, os.path.join(FIGDIR, "grounding_invariance.png"))


def plot_scarcity_boundary(rob, chk):
    """Two measures of the gap per site: systematic (day-averaged, clay) vs typical per-day
    (gray). Equal -> systematic disparate impact; per-day >> systematic -> rotating (within-pop).

    Sites are listed power-bound first. The ACN-Data site is included whenever both inputs
    carry it, so the figure cannot show four sites while the text describes five."""
    order = [("reference", "reference_16ch_30kW_residential_eu"),
             ("ACN-Data", "acn_data_caltech_8ch_30kW"),
             ("highway", "highway_20ch_45kW_eu_highrate"),
             ("workplace", "workplace_24ch_55kW_us"),
             ("shopping", "shopping_8ch_16kW_world")]
    order = [(l, c) for l, c in order
             if c in rob["configs"] and c in chk["check1_rotation"]]
    labels = [l for l, _ in order]
    systm = [rob["configs"][c]["profit_optimal_disparity_dayavg"] for _, c in order]
    perday = [chk["check1_rotation"][c]["per_day_gap_median"] for _, c in order]
    x = np.arange(len(order)); w = 0.38
    fig, ax = plt.subplots(figsize=(6.4 if len(order) > 4 else 5.6, 3.0))
    ax.bar(x - w / 2, systm, w, color=HARM, edgecolor="white", linewidth=0.8, zorder=3,
           label="systematic (day-averaged)")
    ax.bar(x + w / 2, perday, w, color=S.MUTED, edgecolor="white", linewidth=0.8, zorder=3,
           label="typical per-day")
    S.clean(ax)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("tier gap"); ax.set_ylim(0, max(perday + systm) * 1.18)
    ax.legend(loc="upper right", fontsize=9.5, frameon=False, handlelength=1.0, labelspacing=0.3)
    S.savefig(fig, os.path.join(FIGDIR, "scarcity_boundary.png"))


def plot_levers(lev):
    """Reduction in each harm under each lever: pricing is targeted (between-tier only),
    capacity hits both -> the levers are not orthogonal."""
    pr = lev["pricing_lever_30kW_sq_to_flat"]
    ca = lev["capacity_lever_sq_30_to_90kW"]
    groups = ["pricing\n(status quo -> flat)", "capacity\n(30 -> 90 kW)"]
    between = [pr["d_between_tier_gap"], ca["d_between_tier_gap"]]
    within = [pr["d_within_tier_gini"], ca["d_within_tier_gini"]]
    x = np.arange(len(groups)); w = 0.38
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    ax.bar(x - w / 2, between, w, color=HARM, edgecolor="white", linewidth=0.8, zorder=3,
           label="between-tier gap")
    ax.bar(x + w / 2, within, w, color=RELIEF, edgecolor="white", linewidth=0.8, zorder=3,
           label="within-tier Gini (oracle)")
    S.clean(ax)
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylabel("reduction in harm"); ax.set_ylim(0, max(between + within) * 1.18)
    ax.legend(loc="upper center", fontsize=9.5, frameon=False, handlelength=1.0, labelspacing=0.3)
    S.savefig(fig, os.path.join(FIGDIR, "lever_cross_effects.png"))


def plot_capacity(cap):
    """Both inequalities fall with grid capacity and flatten by the threshold (dashed). The
    between-tier gap (oracle) goes to zero; the max-charge Gini plateaus at a nonzero floor
    set by charger-count rejections, which grid power does not add."""
    rows = [r for r in cap["rows"] if r["grid_kw"] <= 120]
    grids = [r["grid_kw"] for r in rows]
    gap = [r["between_tier_gap"] for r in rows]
    gini = [r["maxcharge_gini"] for r in rows]
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    thr = sorted({cap["threshold_between_tier_kw"], cap["threshold_maxcharge_gini_kw"]})
    for t in thr:
        ax.axvline(t, ls=(0, (4, 2)), color=S.MUTED, lw=1.5, zorder=0)
    ax.plot(grids, gap, "-o", color=HARM, ms=7, lw=2.6, zorder=3,
            label="between-tier gap (oracle)")
    ax.plot(grids, gini, "-o", color=RELIEF, ms=7, lw=2.6, zorder=3,
            label="within-population Gini (max-charge)")
    S.clean(ax)
    ax.set_xlabel("grid connection (kW)"); ax.set_ylabel("inequality")
    ax.set_ylim(0, max(gap + gini) * 1.08)
    ax.legend(loc="upper right", fontsize=9.5, frameon=False, handlelength=1.4, labelspacing=0.3)
    S.savefig(fig, os.path.join(FIGDIR, "capacity_threshold.png"))


def main():
    M = _load("audit_mechanism.json")
    plot_oracle(M)
    plot_rate_design(M)
    plot_grounding(M)
    # The --acn estimator run repeats check 1 on the SAME bundled envs and adds the ACN
    # site, so it is a strict superset; prefer it so the figure matches the 5-site sweep.
    chk = _load_optional("audit_estimator_check_acn.json") or _load("audit_estimator_check.json")
    plot_scarcity_boundary(_load("audit_robustness.json"), chk)
    plot_levers(_load("audit_levers.json"))
    plot_capacity(_load("audit_capacity.json"))


if __name__ == "__main__":
    main()
