r"""Publication figures for "The Price of a Fair Charge" (top-ML-conference style).

Legibility-first (thick trends, large fonts, high DPI so they survive downscaling) and
a validated colorblind-safe palette (see experiments/plot_style.py). Figures:

  oracle_tradeoff.png   -- per-tier satisfaction by objective (disparity = within-group
                           spread) + the revenue that equity costs.
  rate_design.png       -- disparity vs the margin SPREAD: the disparate impact is a
                           function of the spread; only spread -> 0 (income-neutral) closes it.
  capacity_threshold.png-- inequality vs grid capacity, with the threshold range.
  robustness_sites.png  -- the disparate impact tracks scarcity across station types.
  scarcity_contrast.png -- scarce vs abundant grid under a profit-blind policy.
  learning_curve.png    -- learned policies vs the baselines they must clear (appendix).
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
FIGDIR = os.path.join(RESULTS, "figures")   # all generated figures live here
os.makedirs(FIGDIR, exist_ok=True)
S.apply_style()


def _load(name):
    p = os.path.join(RESULTS, name)
    return json.load(open(p)) if os.path.exists(p) else None


# ------------------------------------------------------------------ 1. oracle
def plot_oracle():
    R = _load("results.json")
    o = R.get("oracle", {})
    if not isinstance(o.get("profit"), dict):
        return
    objs = [("profit", "profit-opt."), ("utilitarian", "utilitarian"),
            ("maximin", "tier maximin"), ("egalitarian_equal", "strict equal.")]
    xl = [l for _, l in objs]
    x = np.arange(len(objs)); w = 0.26
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.3),
                                  gridspec_kw={"width_ratios": [2.15, 1]})
    for j, t in enumerate(S.TIER_ORDER):
        vals = [o[k]["group_means"][j] for k, _ in objs]
        ax.bar(x + (j - 1) * w, vals, w, color=S.TIER[t], edgecolor="white",
               linewidth=0.8, zorder=3)
    # Direct-label the three tiers once, on the profit-optimal group (no legend box).
    gm = o["profit"]["group_means"]
    for j, t in enumerate(S.TIER_ORDER):
        ax.annotate(t, (x[0] + (j - 1) * w, gm[j] + 0.015), ha="center", va="bottom",
                    fontsize=10, color=S.INK)
    S.clean(ax)
    ax.set_xticks(x); ax.set_xticklabels(xl, rotation=18, ha="right")
    ax.set_ylabel("satisfaction"); ax.set_ylim(0, 1.06)
    ax.set_title("Who gets charged, by objective", fontsize=12.5)
    # Gap annotation to the LEFT of the profit-optimal group (well clear of the bars).
    gx = x[0] - w - 0.42
    ax.annotate("", (gx, gm[0]), (gx, gm[2]),
                arrowprops=dict(arrowstyle="<->", color=S.INK, lw=1.3))
    ax.text(gx - 0.08, (gm[0] + gm[2]) / 2, "gap\n%.2f" % (gm[2] - gm[0]),
            ha="right", va="center", fontsize=10.5, color=S.INK)
    ax.set_xlim(gx - 0.55, None)
    # Revenue panel (single series, neutral) -- where the price of fairness is paid.
    revs = [o[k]["revenue_value"] for k, _ in objs]
    bars = ax2.bar(x, revs, 0.62, color=S.MUTED, edgecolor="white", linewidth=0.8, zorder=3)
    bars[0].set_color(S.VERMILLION)  # profit optimum earns the premium
    S.clean(ax2)
    ax2.set_xticks(x); ax2.set_xticklabels(xl, rotation=18, ha="right")
    ax2.set_ylabel("operator revenue"); ax2.set_ylim(0, max(revs) * 1.2)
    ax2.set_title("The cost is revenue", fontsize=12.5)
    pof = 100 * (revs[0] - revs[2]) / revs[0]
    ax2.annotate("-%.0f%%" % pof, (x[2], revs[2] + 6), (x[2], revs[0]),
                 ha="center", fontsize=11.5, color=S.INK,
                 arrowprops=dict(arrowstyle="->", color=S.INK, lw=1.2))
    S.savefig(fig, os.path.join(FIGDIR, "oracle_tradeoff.png"))


# ------------------------------------------------------------- 2. rate design
def _spread(weights):
    return max(weights) - min(weights)


def plot_rate_design():
    R = _load("results.json")
    t = R.get("tariffs", {})
    if not t.get("rows"):
        return
    caps = t.get("premium_cap", {}).get("rows", [])
    best_cap = min((r["disparity"] for r in caps if r["cap"] > 0.6 + 1e-9), default=0.50)
    floor = t.get("flat_tariff_disparity", 0.04)
    # Ordered worst -> best; the two partial fixes fail, only income-neutral hits the floor.
    items = [("tiered\n(status quo)", t.get("value_weighted_disparity", 0.54), S.VERMILLION),
             ("cap top tier\nalone", best_cap, S.MUTED),
             ("subsidize bottom\ntier alone", t.get("best_budget_only_disparity", 0.28), S.MUTED),
             ("income-neutral\n(equal margins)", floor, S.BLUE)]
    y = np.arange(len(items))[::-1]  # first item on top
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    bars = ax.barh(y, [v for _, v, _ in items], 0.62,
                   color=[c for _, _, c in items], edgecolor="white", linewidth=0.8, zorder=3)
    for b, (_, v, _) in zip(bars, items):
        ax.annotate("%.2f" % v, (v + 0.012, b.get_y() + b.get_height() / 2),
                    va="center", ha="left", fontsize=11, color=S.INK)
    ax.axvline(floor, ls=(0, (4, 2)), color=S.MUTED, lw=1.6, zorder=2)
    S.clean(ax, ygrid=False)
    ax.grid(axis="x", which="major"); ax.grid(axis="y", visible=False)
    ax.set_yticks(y); ax.set_yticklabels([l for l, _, _ in items])
    ax.set_xlabel("profit-optimal tier gap  (disparate impact)")
    ax.set_xlim(0, max(v for _, v, _ in items) * 1.18)
    ax.set_title("Only equal margins close the gap", fontsize=12.5)
    S.savefig(fig, os.path.join(FIGDIR, "rate_design.png"))


# ------------------------------------------------------- 3. capacity threshold
def plot_capacity():
    R = _load("results.json")
    c = R.get("capacity", {})
    if not c.get("rows"):
        return
    rows = c["rows"]
    g = np.array([r["grid_kw"] for r in rows])
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.plot(g, [r["gini"] for r in rows], "-o", color=S.BLUE,
            label="Gini (max-charge)", zorder=4)
    ax.plot(g, [r["oracle_profit_disparity"] for r in rows], "-s", color=S.VERMILLION,
            label="oracle tier gap", zorder=3)
    tg, to = c.get("threshold_kw_gini"), c.get("threshold_kw_oracle")
    if tg and to:
        ax.axvspan(tg, to, color=S.GRID, alpha=0.9, zorder=1)
        ax.text(np.sqrt(tg * to), ax.get_ylim()[1] * 0.92, "threshold\n%g-%g kW" % (tg, to),
                ha="center", va="top", fontsize=10, color=S.MUTED)
    S.clean(ax)
    ax.set_xscale("log")
    import matplotlib.ticker as mtick
    ticks = [20, 50, 100, 300, 600]
    ax.set_xticks(ticks); ax.set_xticklabels([str(v) for v in ticks])
    ax.xaxis.set_minor_formatter(mtick.NullFormatter())
    ax.set_xlabel("grid connection (kW, log scale)")
    ax.set_ylabel("inequality")
    ax.set_ylim(0, None)
    ax.set_title("Capacity removes scarcity-driven inequality", fontsize=12.5)
    ax.legend(loc="upper right")
    S.savefig(fig, os.path.join(FIGDIR, "capacity_threshold.png"))


# ------------------------------------------------------- 4. robustness (sites)
def plot_robustness():
    d = _load("audit_robustness.json")
    if not d or not d.get("configs"):
        return
    short = {"reference_16ch_30kW_residential_eu": "residential\n(scarce)",
             "highway_20ch_45kW_eu_highrate": "highway",
             "shopping_8ch_16kW_world": "shopping",
             "workplace_24ch_55kW_us": "workplace"}
    xs, ys, labs = [], [], []
    for n, c in d["configs"].items():
        xs.append(c["revenue_pof_pct"]["median"])
        ys.append(c["profit_optimal_disparity"]["median"])
        labs.append(short.get(n, n))
    xs, ys = np.array(xs), np.array(ys)
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    # trend line (disparity rises with scarcity, measured by revenue PoF)
    if len(xs) >= 2:
        b, a = np.polyfit(xs, ys, 1)
        xr = np.linspace(xs.min() - 0.5, xs.max() + 1, 50)
        ax.plot(xr, a + b * xr, "-", color=S.MUTED, lw=1.8, zorder=2)
    ax.scatter(xs, ys, s=150, color=S.BLUE, edgecolor="white", linewidth=1.4, zorder=4)
    # Place labels with per-point offsets so none sit on the axis edge or overlap.
    off = {"residential\n(scarce)": (0, -0.055, "center", "top"),
           "highway": (0, 0.03, "center", "bottom"),
           "shopping": (0.35, 0.02, "left", "bottom"),
           "workplace": (0.35, -0.02, "left", "top")}
    for xi, yi, li in zip(xs, ys, labs):
        dx, dy, ha, va = off.get(li, (0.3, 0.02, "left", "bottom"))
        ax.annotate(li, (xi + dx, yi + dy), ha=ha, va=va, fontsize=10.5, color=S.INK)
    S.clean(ax)
    ax.set_xlabel("revenue price of fairness (%)  =  scarcity")
    ax.set_ylabel("profit-optimal tier gap")
    ax.set_xlim(-1.2, xs.max() + 2.2); ax.set_ylim(-0.03, ys.max() * 1.22)
    ax.set_title("Disparate impact tracks power scarcity", fontsize=12.5)
    S.savefig(fig, os.path.join(FIGDIR, "robustness_sites.png"))


# ------------------------------------------------------- 5. scarcity contrast
def plot_scarcity():
    R = _load("results.json")
    sc = R.get("scarcity", {})
    tags = [t for t in ["scarce_30kW", "abundant_600kW"] if t in sc]
    if len(tags) < 2:
        return
    x = np.arange(len(tags)); w = 0.34
    fig, ax = plt.subplots(figsize=(4.0, 3.3))
    b1 = ax.bar(x - w / 2, [sc[t]["served_mean_sat"] for t in tags], w, color=S.BLUE,
                label="mean satisfaction", edgecolor="white", linewidth=0.8, zorder=3)
    b2 = ax.bar(x + w / 2, [sc[t]["gini"] for t in tags], w, color=S.VERMILLION,
                label="Gini (inequality)", edgecolor="white", linewidth=0.8, zorder=3)
    S.label_bars(ax, list(b1) + list(b2))
    S.clean(ax)
    ax.set_xticks(x); ax.set_xticklabels(["%g kW\n(scarce)" % sc[tags[0]]["grid_kw"],
                                          "%g kW\n(abundant)" % sc[tags[1]]["grid_kw"]])
    ax.set_ylim(0, 1.0); ax.set_ylabel("value")
    ax.set_title("Scarcity drives inequality", fontsize=12.5)
    ax.legend(loc="upper center", ncol=1)
    S.savefig(fig, os.path.join(FIGDIR, "scarcity_contrast.png"))


# ------------------------------------------------------------ 6. learning curve
def plot_learning_curve():
    c = _load("learning_curve.json")
    if not c:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    styles = {"profit_ppo": ("profit RL", S.BLUE, "-o"),
              "egal_l100": ("egalitarian RL", S.VERMILLION, "-s")}
    for ax, key, ylab, ttl in [(axes[0], "profit", "profit (EUR/day)", "Profit vs budget"),
                               (axes[1], "gini", "Gini (lower fairer)", "Inequality vs budget")]:
        for name, rows in c["curves"].items():
            lab, col, mk = styles.get(name, (name, S.GREEN, "-^"))
            b = [r["budget"] / 1e6 for r in rows]
            ax.plot(b, [r[key] for r in rows], mk, color=col, label=lab, zorder=4)
        ax.axhline(c["reference"]["random"][key], ls=(0, (4, 2)), color=S.REF, lw=1.8,
                   label="random", zorder=2)
        ax.axhline(c["reference"]["max_charge"][key], ls=(0, (1, 1.5)), color=S.INK, lw=1.8,
                   label="max-charge", zorder=2)
        S.clean(ax)
        ax.set_xlabel("training budget (M steps)"); ax.set_ylabel(ylab)
        ax.set_title(ttl, fontsize=12.5)
    axes[0].legend(loc="lower right", fontsize=10)
    S.savefig(fig, os.path.join(FIGDIR, "learning_curve.png"))


def main():
    plot_oracle()
    plot_rate_design()
    plot_capacity()
    plot_robustness()
    plot_scarcity()
    plot_learning_curve()


if __name__ == "__main__":
    main()
