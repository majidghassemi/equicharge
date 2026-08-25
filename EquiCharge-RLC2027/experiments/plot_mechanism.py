r"""Regenerate the FAccT paper figures from the CANONICAL run (audit_mechanism.json).

Every figure here is built at the exact width it occupies on the printed page (see
``experiments/plot_style.py``), so the ``.tex`` includes it at scale 1.0 and the type sizes
in this file are the type sizes the reader gets at 100% zoom. That constraint is what drives
the layout choices below: legends become direct labels where a legend would cost plot width,
long category names are set on two lines rather than rotated, and each panel carries one
comparison. Figures come from the one fixed day set / day-averaged estimator used by every
table in the paper, so figure and table cannot disagree.

  oracle_tradeoff.png      -- per-tier satisfaction by objective + the revenue equity cost.
                              Two panels, so this one is a full-width figure* at \textwidth.
  rate_design.png          -- the tier gap under four tariffs, showing it tracks RANK POSITION
                              (is the budget tier uniquely lowest?), not the margin spread.
  grounding_invariance.png -- the gap is flat in the WTP elasticity, and only the cliff at 0.
  scarcity_boundary.png    -- systematic vs typical per-day gap, per site.
  lever_cross_effects.png  -- the pricing and capacity levers are not orthogonal.
  capacity_threshold.png   -- both inequalities against grid connection, and their floors.
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from experiments import plot_style as S
from experiments.plot_style import COL, FULL, TIER, TIER_ORDER, HARM, RELIEF

RESULTS = os.path.join(os.path.dirname(__file__), "results")
FIGDIR = os.path.join(RESULTS, "figures")
os.makedirs(FIGDIR, exist_ok=True)
S.apply_style()

# Width of the slot each single-panel figure will occupy on the page. Set with --across N
# (N figures composed side by side across \textwidth) or --width INCHES. Everything below
# is drawn at this width, so composing the figures into a grid never rescales their type.
W = S.COL
NARROW = False


def _h(ratio, lo=1.80, hi=2.60):
    """Panel height from the slot width, floored so a short-but-wide cell stays readable."""
    return min(max(W * ratio, lo), hi)


def pick(wide, narrow):
    """Choose label/tick detail for the current slot width."""
    return narrow if NARROW else wide

# bar()/barh() take ecolor and capsize directly; everything else goes via error_kw.
# Sized for the 1:1 canvas: a 2.5pt cap reads at 2.5pt on the page.
ERRBAR = dict(ecolor="#3f3f3f", capsize=1.6,
              error_kw=dict(elinewidth=0.8, capthick=0.8, zorder=5))


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
    """Per-tier satisfaction by objective (left) + operator revenue (right).

    The only two-panel figure in the paper, so it is the only one drawn at \\textwidth and
    placed in a figure*. Category names are set on two lines instead of rotated: rotated
    labels cost vertical space and are markedly harder to read at 7.5pt.
    Numbers (gap 0.57, PoF ~9%) live in the caption, not on the figure."""
    box1 = M["box1"]
    # Single-line names, and the right panel is given enough width to seat the longest of
    # them: breaking "utilitarian" mid-word to fit was worse to read than widening the panel.
    objs = [("profit", "revenue-optimal"), ("utilitarian", "utilitarian"),
            ("maximin", "tier maximin"), ("egalitarian_equal", "strict equality")]
    xl = [l for _, l in objs]
    x = np.arange(len(objs)); w = 0.26
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(FULL, 2.40),
                                  gridspec_kw={"width_ratios": [1.5, 1]})
    for j, t in enumerate(TIER_ORDER):
        vals = [box1[k]["tier_means"][j] for k, _ in objs]
        cis = [box1[k].get("tier_means_ci95", [None] * 3)[j] for k, _ in objs]
        yerr = _asym(vals, cis) if all(c for c in cis) else None
        ax.bar(x + (j - 1) * w, vals, w, color=TIER[t], edgecolor="white",
               linewidth=0.5, zorder=3, label=t, yerr=yerr,
               **(ERRBAR if yerr is not None else {}))
    S.clean(ax)
    ax.set_xticks(x); ax.set_xticklabels(xl)
    ax.set_ylabel("satisfaction"); ax.set_ylim(0, 1.10)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # Horizontal legend on one line above the bars, so it never overlaps the tall premium bar.
    ax.legend(loc="upper left", ncol=3, columnspacing=0.9, handlelength=0.9,
              borderaxespad=0.0, bbox_to_anchor=(0.0, 1.02))

    revs = [box1[k]["revenue"] for k, _ in objs]
    rcis = [box1[k].get("revenue_ci95") for k, _ in objs]
    rerr = _asym(revs, rcis) if all(rcis) else None
    bars = ax2.bar(x, revs, 0.60, color=S.MUTED, edgecolor="white", linewidth=0.5, zorder=3,
                   yerr=rerr, **(ERRBAR if rerr is not None else {}))
    bars[0].set_color(HARM)  # profit optimum earns the premium -- the cost of fairness
    S.clean(ax2)
    ax2.set_xticks(x); ax2.set_xticklabels(xl)
    ax2.set_ylabel("revenue (€/day)"); ax2.set_ylim(0, max(revs) * 1.22)
    S.label_line(ax2, x[0], revs[0], "revenue-optimal", HARM, dy=9, ha="center", va="bottom")
    S.savefig(fig, os.path.join(FIGDIR, "oracle_tradeoff.png"))


def plot_rate_design(M):
    by = {r["config"]: r for r in M["mechanism"]}
    # worst -> best, matching Table (tariff): status quo, premium cap, budget subsidy, flat.
    order = [
        (pick("tiered\n(status quo)", "tiered"), "status_quo_distinct"),
        (pick("cap top tier\n(to middle)", "cap top"), "premium_capped_to_mid"),
        (pick("subsidize bottom\n(to parity)", "subsidize"), "budget_tied_to_mid"),
        (pick("income-neutral\n(flat)", "flat"), "fully_flat"),
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
    fig, ax = plt.subplots(figsize=(W, _h(0.70)))
    ax.barh(y, gaps, 0.58, color=colors, edgecolor="white", linewidth=0.5, zorder=3,
            xerr=xerr, **(ERRBAR if xerr is not None else {}))
    # Label past the CI whisker, not the bar end, so the two never collide.
    tips = [c[1] for c in cis] if xerr is not None else gaps
    for yi, g, t in zip(y, gaps, tips):
        ax.annotate("%.2f" % g, (t + 0.012, yi), va="center", ha="left",
                    fontsize=S.FS_ANNOT, color=S.INK)
    ax.axvline(floor, ls=(0, (3, 2)), color=S.MUTED, lw=0.9, zorder=2)
    S.clean(ax, ygrid=False)
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlabel("revenue-optimal tier gap")
    ax.set_xlim(0, max(tips) * 1.22)
    ax.set_xticks(pick([0, 0.1, 0.2, 0.3, 0.4, 0.5], [0, 0.25, 0.5]))
    ax.legend(handles=[Patch(color=HARM, label=pick("budget still lowest", "still lowest")),
                       Patch(color=RELIEF, label=pick("budget rescued", "rescued"))],
              loc="lower right", borderaxespad=0.3)
    S.savefig(fig, os.path.join(FIGDIR, "rate_design.png"))


def plot_grounding(M):
    """The gap is flat across the WTP elasticity -> structural, not a calibration artifact.
    The one number (PoF 3->13%) stays in the table; the figure is just the flat line + cliff."""
    inv = sorted(M["elasticity_invariance"], key=lambda d: d["eta"])
    etas = [d["eta"] for d in inv]
    gaps = [d["gap"] for d in inv]
    fig, ax = plt.subplots(figsize=(W, _h(0.61)))
    ax.plot(etas, gaps, "-", color=HARM, lw=1.7, zorder=3)
    # eta>0: harm persists (clay); eta=0: no ordering, harm gone (teal).
    ax.plot(etas[1:], gaps[1:], "o", color=HARM, ms=4.2, zorder=4)
    ax.plot([0], [gaps[0]], "o", color=RELIEF, ms=5.5, zorder=5)
    ax.annotate(pick("no ordering\n(flat tariff)", "no\nordering"), (0, gaps[0]), (0.10, gaps[0] + 0.11),
                fontsize=S.FS_ANNOT, color=RELIEF, va="bottom",
                arrowprops=dict(arrowstyle="->", color=RELIEF, lw=0.8,
                                shrinkA=1.0, shrinkB=2.5))
    S.clean(ax)
    ax.set_xlabel(r"willingness-to-pay elasticity $\eta$")
    ax.set_ylabel("tier gap")
    ax.set_ylim(0, 0.68); ax.set_xlim(-0.03, 1.03)
    ax.set_yticks(pick([0, 0.2, 0.4, 0.6], [0, 0.3, 0.6]))
    ax.set_xticks(pick([0, 0.2, 0.4, 0.6, 0.8, 1.0], [0, 0.5, 1.0]))
    S.savefig(fig, os.path.join(FIGDIR, "grounding_invariance.png"))


def plot_scarcity_boundary(rob, chk, acn_chk=None, acn_mech=None):
    """Two measures of the gap per site: systematic (day-averaged, clay) vs typical per-day
    (gray). Equal -> systematic disparate impact; per-day >> systematic -> rotating (within-pop).

    Horizontal bars, so five site names set at full size with no rotation and no collision in
    a 3.3in column. Sites are listed power-bound first.

    Both bars for a given site must come from the SAME run. The ACN-Data configuration needs
    the Caltech dataset and is skipped when it is absent, so its files can be a full run
    behind the rest; its row is therefore taken from the ACN files explicitly and its day
    count is reported, rather than letting a stale superset file supply every site's bars.
    That substitution is what once put this figure's numbers a run behind the sites table."""
    order = [("reference", "reference_16ch_30kW_residential_eu"),
             ("ACN-Data", "acn_data_caltech_8ch_30kW"),
             ("highway", "highway_20ch_45kW_eu_highrate"),
             ("workplace", "workplace_24ch_55kW_us"),
             ("shopping", "shopping_8ch_16kW_world")]
    ACN = "acn_data_caltech_8ch_30kW"
    rows, day_counts = [], {}
    for label, cfg in order:
        if cfg == ACN:
            # Only from the ACN files, and only when both halves are there.
            if acn_chk and ACN in acn_chk.get("check1_rotation", {}) and acn_mech:
                rows.append((label, acn_mech["box1"]["profit"]["gap"],
                             acn_chk["check1_rotation"][ACN]["per_day_gap_median"]))
                day_counts[label] = acn_chk["check1_rotation"][ACN].get("days")
            continue
        if cfg in rob["configs"] and cfg in chk["check1_rotation"]:
            rows.append((label, rob["configs"][cfg]["profit_optimal_disparity_dayavg"],
                         chk["check1_rotation"][cfg]["per_day_gap_median"]))
            day_counts[label] = rob["configs"][cfg].get("n_days")
    if len(set(day_counts.values())) > 1:
        print("  [scarcity_boundary] NOTE: sites come from runs of different length, %s. "
              "The shorter ones are hatched in the figure; re-run with EQUICHARGE_ACN_JSON "
              "set to put every site on the same horizon." % day_counts)
    labels = [r[0] for r in rows]
    systm = [r[1] for r in rows]
    perday = [r[2] for r in rows]
    y = np.arange(len(order))[::-1]; h = 0.36
    # A site drawn from a shorter run is hatched, so the mixed horizon is visible in the
    # chart itself. The caption discloses it too, but the chart is what gets scanned.
    short = [d is not None and d < max(c for c in day_counts.values() if c) for d in
             (day_counts.get(l) for l in labels)]
    fig, ax = plt.subplots(figsize=(W, _h(0.70)))
    hatch = ["//" if sh else None for sh in short]
    b_sys = ax.barh(y + h / 2, systm, h, color=HARM, edgecolor="white", linewidth=0.5, zorder=3,
                    label=pick("systematic (day-averaged)", "systematic"))
    b_day = ax.barh(y - h / 2, perday, h, color=S.MUTED, edgecolor="white", linewidth=0.5,
                    zorder=3, label=pick("typical per-day", "per-day"))
    for bars in (b_sys, b_day):
        for bar, hh in zip(bars, hatch):
            if hh:
                bar.set_hatch(hh)
                bar.set_edgecolor("white")
    legend_extra = []
    if any(short):
        n_short = max(c for l, c in day_counts.items() if short[labels.index(l)])
        n_full = max(c for c in day_counts.values() if c)
        # A proxy patch, not an empty bar: an empty bar container takes the next colour
        # from the cycle and drops the hatch, so the swatch comes out a solid off-palette
        # block that reads as a third data series.
        legend_extra.append(Patch(facecolor="white", edgecolor=S.INK, hatch="///",
                                  linewidth=0.5,
                                  label=pick("hatched: %d days, not %d" % (n_short, n_full),
                                             "%d days" % n_short)))
    S.clean(ax, ygrid=False)
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlabel("tier gap")
    ax.set_xlim(0, max(perday + systm) * 1.30)
    ax.set_xticks(pick([0, 0.1, 0.2, 0.3, 0.4, 0.5], [0, 0.25, 0.5]))
    # The two short bottom rows leave the lower right empty, so the legend costs no data space.
    handles, lbls = ax.get_legend_handles_labels()
    ax.legend(handles + legend_extra, lbls + [p.get_label() for p in legend_extra],
              loc="lower right", borderaxespad=0.3)
    S.savefig(fig, os.path.join(FIGDIR, "scarcity_boundary.png"))


def plot_levers(lev):
    """Reduction in each harm under each lever: pricing is targeted (between-tier only),
    capacity hits both -> the levers are not orthogonal."""
    pr = lev["pricing_lever_30kW_sq_to_flat"]
    ca = lev["capacity_lever_sq_30_to_90kW"]
    groups = pick(["pricing\n(status quo $\\rightarrow$ flat)", "capacity\n(30 $\\rightarrow$ 90 kW)"],
                  ["pricing", "capacity"])
    between = [pr["d_between_tier_gap"], ca["d_between_tier_gap"]]
    within = [pr["d_within_tier_gini"], ca["d_within_tier_gini"]]
    x = np.arange(len(groups)); w = 0.30
    fig, ax = plt.subplots(figsize=(W, _h(0.64)))
    b1 = ax.bar(x - w / 2, between, w, color=HARM, edgecolor="white", linewidth=0.5, zorder=3,
                label=pick("between-tier gap", "between-tier"))
    b2 = ax.bar(x + w / 2, within, w, color=RELIEF, edgecolor="white", linewidth=0.5, zorder=3,
                label=pick("within-population Gini (oracle)", "within-population"))
    S.label_bars(ax, list(b1) + list(b2), dy=0.008)
    S.clean(ax)
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylabel("reduction in harm")
    ax.set_ylim(0, max(between + within) * 1.32)
    ax.set_yticks(pick([0, 0.2, 0.4, 0.6], [0, 0.3, 0.6]))
    ax.legend(loc="upper center", ncol=1, borderaxespad=0.0, bbox_to_anchor=(0.5, 1.03))
    S.savefig(fig, os.path.join(FIGDIR, "lever_cross_effects.png"))


def plot_capacity(cap):
    """Both inequalities fall with grid capacity and flatten by the threshold (dashed). The
    between-tier gap (oracle) goes to zero; the max-charge Gini plateaus at a nonzero floor
    set by charger-count rejections, which grid power does not add.

    The two series are labelled directly on the curves. A two-entry legend with names this
    long would need roughly half the column width, and direct labels also tie each name to
    the shape it explains."""
    rows = [r for r in cap["rows"] if r["grid_kw"] <= 120]
    grids = [r["grid_kw"] for r in rows]
    gap = [r["between_tier_gap"] for r in rows]
    gini = [r["maxcharge_gini"] for r in rows]
    fig, ax = plt.subplots(figsize=(W, _h(0.64)))
    thr = sorted({cap["threshold_between_tier_kw"], cap["threshold_maxcharge_gini_kw"]})
    for t in thr:
        ax.axvline(t, ls=(0, (3, 2)), color=S.MUTED, lw=0.9, zorder=0)
    ax.plot(grids, gap, "-o", color=HARM, ms=3.6, lw=1.7, zorder=3)
    ax.plot(grids, gini, "-o", color=RELIEF, ms=3.6, lw=1.7, zorder=3)
    S.clean(ax)
    ax.set_xlabel("grid connection (kW)"); ax.set_ylabel("inequality")
    ax.set_ylim(0, max(gap + gini) * 1.16); ax.set_xlim(10, 126)
    ax.set_yticks(pick([0, 0.2, 0.4, 0.6], [0, 0.3, 0.6]))
    ax.set_xticks(pick([20, 40, 60, 80, 100, 120], [20, 60, 120]))
    # Direct labels, placed in the empty wedge each curve opens up.
    cx, cy = pick((34, 0.60), (68, 0.09))
    S.label_line(ax, cx, cy, pick("between-tier gap\n(oracle)", "between-\ntier gap"), HARM,
                 ha="left", va="bottom")
    S.label_line(ax, 66, gini[-1], pick("within-population Gini\n(max-charge)", "within-pop.\nGini"),
                 RELIEF, dy=5, ha="left", va="bottom")
    S.label_line(ax, thr[-1], ax.get_ylim()[1], "%g kW" % thr[-1], S.MUTED,
                 dx=-2.5, dy=-1.5, ha="right", va="top")
    S.savefig(fig, os.path.join(FIGDIR, "capacity_threshold.png"))


# The .tex reads from paper/figures/, which had drifted a full experiment run behind
# results/figures/ (missing the bootstrap error bars and the ACN-Data site). Copying on
# every regeneration means the paper cannot silently ship a stale figure again.
PAPERDIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "paper", "figures")


def sync_to_paper():
    import shutil
    if not os.path.isdir(PAPERDIR):
        return
    for name in sorted(os.listdir(FIGDIR)):
        if name.endswith(".png"):
            shutil.copy2(os.path.join(FIGDIR, name), os.path.join(PAPERDIR, name))
            print("synced", os.path.join("paper/figures", name))


def main(argv=None):
    """--across N composes N figures side by side across \\textwidth; --width INCHES sets the
    slot directly; --outdir writes elsewhere (and then skips the paper/figures sync)."""
    import argparse
    global W, NARROW, FIGDIR
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--across", type=int, metavar="N",
                   help="N single-panel figures composed side by side across \\textwidth")
    g.add_argument("--width", type=float, metavar="INCHES",
                   help="slot width directly, for a grid this script cannot infer")
    ap.add_argument("--outdir", help="write here instead of experiments/results/figures")
    a = ap.parse_args(argv)

    if a.width is not None:
        W = a.width
    elif a.across is not None:
        W = S.slot(a.across)
    NARROW = W < S.NARROW
    if a.outdir:
        FIGDIR = a.outdir
        os.makedirs(FIGDIR, exist_ok=True)
    print("slot width %.2f in%s -> %s" % (W, " (narrow)" if NARROW else "", FIGDIR))

    M = _load("audit_mechanism.json")
    plot_oracle(M)
    plot_rate_design(M)
    plot_grounding(M)
    # The --acn estimator run repeats check 1 on the SAME bundled envs and adds the ACN
    # site, so it is a strict superset; prefer it so the figure matches the 5-site sweep.
    # The bundled-data estimator run is authoritative for the bundled sites. The ACN files
    # are consulted only for the ACN row, because that configuration is skipped whenever
    # EQUICHARGE_ACN_JSON is unset and its files then lag the rest of the suite.
    chk = _load("audit_estimator_check.json")
    acn_chk = _load_optional("audit_estimator_check_acn.json")
    acn_mech = _load_optional("audit_mechanism_acn.json")
    plot_scarcity_boundary(_load("audit_robustness.json"), chk, acn_chk, acn_mech)
    plot_levers(_load("audit_levers.json"))
    plot_capacity(_load("audit_capacity.json"))
    if not a.outdir:
        sync_to_paper()


if __name__ == "__main__":
    main()
