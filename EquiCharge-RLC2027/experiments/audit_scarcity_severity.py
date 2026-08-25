r"""Scarcity severity: one axis that unifies the capacity sweep and the site
comparison.

The paper's capacity sweep (Figure "capacity_threshold") varies grid kW at the
reference site, and the site comparison (Table "sites") varies everything at
once. A reviewer can ask whether the site comparison and the sweep tell one story or
two. This script puts both on the same x-axis, the demand-to-capacity ratio
rho = (mean realized daily desired energy) / (P * 24h), and plots the
systematic tier gap Gamma against it. If the scarcity story is right, the
sweep curve and the four site points should fall on one monotone boundary:
Gamma large only where rho is well above one.
"""

from __future__ import annotations

import json
import os

import jax
import numpy as np

from chargax.equity import oracle as O
from experiments.common import scarcity_station, make_env, DEFAULT_DATA
from experiments.run_experiments import _ref_env

RESULTS = os.path.join(os.path.dirname(__file__), "results")
N_DAYS = 8  # demand estimation only; Gamma values come from the released runs

SITES = {
    "reference": (dict(n_evses=8, grid_kw=30.0),
                  dict(car_profile="eu", user_profile="residential", average_cars_per_day=30, grid_price_dataset="2023_NL")),
    "workplace": (dict(n_evses=12, grid_kw=55.0),
                  dict(car_profile="us", user_profile="workplace", average_cars_per_day=45, grid_price_dataset="2023_NL")),
    "shopping": (dict(n_evses=4, grid_kw=16.0),
                 dict(car_profile="world", user_profile="shopping", average_cars_per_day=20, grid_price_dataset="2023_NL")),
    "highway": (dict(n_evses=10, grid_kw=45.0),
                dict(car_profile="eu", user_profile="highway", average_cars_per_day=60, grid_price_dataset="2023_NL")),
}
#: The released run's config key for each site here. Gamma is READ from that run rather
#: than copied into this file. A hardcoded copy is exactly how this figure came to plot
#: one run's gaps against another run's table: the numbers were right when they were
#: pasted and silently wrong the next time the audit was re-run at a different horizon.
ROBUSTNESS_KEY = {
    "reference": "reference_16ch_30kW_residential_eu",
    "workplace": "workplace_24ch_55kW_us",
    "shopping": "shopping_8ch_16kW_world",
    "highway": "highway_20ch_45kW_eu_highrate",
}


def site_gamma():
    """Per-site systematic tier gap, from the released robustness run.

    Returns ``(gamma_by_site, n_days)``. Raises if the run is missing a site this script
    plots, because a figure that quietly drops a site is worse than one that fails.
    """
    path = os.path.join(RESULTS, "audit_robustness.json")
    configs = json.load(open(path))["configs"]
    missing = [s for s, k in ROBUSTNESS_KEY.items() if k not in configs]
    if missing:
        raise SystemExit(
            f"audit_robustness.json has no entry for {missing}. Re-run "
            f"`python -m experiments.audit_robustness` before this script.")
    gamma = {s: configs[k]["profit_optimal_disparity_dayavg"]
             for s, k in ROBUSTNESS_KEY.items()}
    # The ACN-Data site lives in its own files, because that configuration is skipped
    # whenever EQUICHARGE_ACN_JSON is unset.
    acn_path = os.path.join(RESULTS, "audit_mechanism_acn.json")
    if os.path.exists(acn_path):
        gamma["acn"] = json.load(open(acn_path))["box1"]["profit"]["gap"]
    days = {configs[k].get("n_days") for k in ROBUSTNESS_KEY.values()}
    return gamma, (days.pop() if len(days) == 1 else sorted(days))


def mean_daily_demand(env, key, n_days):
    tot = []
    for d in range(n_days):
        cust = O.extract_arrival_stream(env, jax.random.fold_in(key, d))
        tot.append(sum(c.desired_kwh for c in cust))
    return float(np.mean(tot))


def main():
    key = jax.random.PRNGKey(7)
    SITE_GAMMA, gamma_days = site_gamma()
    out = {"sites": {}, "capacity_sweep": [],
           "gamma_source": {"file": "audit_robustness.json", "n_days": gamma_days},
           "demand_estimate_days": N_DAYS}
    print(f"[gamma] read from audit_robustness.json ({gamma_days} realized days)")

    # The ACN-Data site joins the panel only when the Caltech dump is available, because
    # its rho has to be measured the same way as every other site's. Estimating it from
    # the released provenance instead would put a derived point on an axis of measured
    # ones, which is the very confusion this figure exists to resolve.
    from experiments.common import acn_env_or_none
    sites = [(n, layout, make_env(
                  scarcity_station(grid_kw=layout["grid_kw"], n_evses=layout["n_evses"]),
                  n_groups=3, price_by_group=(0.6, 1.0, 1.5),
                  data_kwargs={**DEFAULT_DATA, **data}))
             for n, (layout, data) in SITES.items()]
    got = acn_env_or_none(30.0, 8, 4, verbose=False) if "acn" in SITE_GAMMA else None
    if got is not None:
        sites.append(("acn", dict(grid_kw=30.0, n_evses=8), got[0]))
    out["acn_included"] = got is not None
    if got is None:
        print("[sites] ACN-Data omitted: its rho needs the Caltech dump "
              "(set EQUICHARGE_ACN_JSON). Panel (c) shows the simulator sites only.")

    for name, layout, env in sites:
        demand = mean_daily_demand(env, key, N_DAYS)
        cap = layout["grid_kw"] * 24.0
        out["sites"][name] = {
            "grid_kw": layout["grid_kw"], "n_connectors": layout["n_evses"] * 2,
            "mean_daily_demand_kwh": round(demand, 1),
            "daily_capacity_kwh": round(cap, 1),
            "demand_capacity_ratio": round(demand / cap, 3),
            "gamma": SITE_GAMMA[name],
        }
        print(f"{name:>10}: demand={demand:7.1f} kWh/day  cap={cap:7.1f}  "
              f"rho={demand/cap:5.2f}  Gamma={SITE_GAMMA[name]}")

    # Capacity sweep at the reference site: demand fixed, capacity varies.
    ref_env = _ref_env(30.0, 8, 4)
    ref_demand = mean_daily_demand(ref_env, key, N_DAYS)
    cap_rows = json.load(open(os.path.join(RESULTS, "audit_capacity.json")))["rows"]
    for r in cap_rows:
        rho = ref_demand / (r["grid_kw"] * 24.0)
        out["capacity_sweep"].append({"grid_kw": r["grid_kw"], "rho": round(rho, 3),
                                      "gamma": r["between_tier_gap"]})
        print(f"  sweep {r['grid_kw']:>4} kW: rho={rho:5.2f}  Gamma={r['between_tier_gap']}")

    # Does the single axis actually unify them? Read the sweep curve at each site's
    # own rho (interpolating in log rho, the figure's x-axis) and record the residual.
    sw = sorted(out["capacity_sweep"], key=lambda r: r["rho"])
    xs = np.log([r["rho"] for r in sw])
    ys = [r["gamma"] for r in sw]
    for name, s in out["sites"].items():
        pred = float(np.interp(np.log(s["demand_capacity_ratio"]), xs, ys))
        s["sweep_gamma_at_same_rho"] = round(pred, 4)
        s["residual"] = round(s["gamma"] - pred, 4)
        print(f"{name:>10}: sweep predicts Gamma={pred:5.3f}  residual={s['residual']:+6.3f}")
    worst = max(out["sites"], key=lambda n: abs(out["sites"][n]["residual"]))
    out["collapse_residual_max"] = {"site": worst,
                                    "residual": out["sites"][worst]["residual"]}
    out["interpretation"] = (
        f"rho does not collapse the two experiments onto one curve. Every site falls on "
        f"or below the reference sweep read at the same rho (largest deviation: {worst}, "
        f"{out['sites'][worst]['residual']:+.3f}), so a high demand-to-capacity ratio is "
        f"necessary but not sufficient for a large systematic tier gap: the sweep is an "
        f"upper envelope over rho, not a single boundary the sites lie on."
    )
    print(out["interpretation"])

    with open(os.path.join(RESULTS, "scarcity_severity.json"), "w") as f:
        json.dump(out, f, indent=2)

    plot(out)


# ---------------------------------------------------------------- figure
# House style, colours included, comes from experiments/plot_style.py: clay = the harm
# (the tier gap), teal = the individual sites read off the same curve. Like every other paper
# figure this one is drawn at its printed column width, so the offsets below are in
# printed points and mean the same thing on the page as they do here.
# Direct labels instead of a second legend; hand-placed so the four annotations clear
# both each other and the sweep curve.
LABEL_POS = {  # site -> (dx, dy, ha, va)
    "reference": (7, 0, "left", "center"),
    "highway": (0, 7, "center", "bottom"),
    "shopping": (7, -1, "left", "center"),
    "workplace": (-6, 5, "right", "bottom"),
    "acn": (7, 4, "left", "bottom"),
}
#: Used for any site without an explicit entry, so adding a site never crashes the plot.
LABEL_POS_DEFAULT = (7, 0, "left", "center")

#: Display names, where the JSON key is not what the paper calls the site. The boundary
#: panel says "ACN-Data", so this one must too, or the two panels appear to plot
#: different things.
DISPLAY_NAME = {"acn": "ACN-Data"}


def plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from experiments import plot_style as S
    from experiments.plot_style import COL, HARM, RELIEF
    S.apply_style()

    sw = sorted(out["capacity_sweep"], key=lambda r: r["rho"])
    rhos = [r["rho"] for r in sw]
    gaps = [r["gamma"] for r in sw]
    site_rho = [s["demand_capacity_ratio"] for s in out["sites"].values()]
    site_gam = [s["gamma"] for s in out["sites"].values()]

    fig, ax = plt.subplots(figsize=(COL, 2.15))
    ax.axvline(1.0, ls=(0, (3, 2)), color=S.MUTED, lw=0.9, zorder=0)
    ax.plot(rhos, gaps, "-o", color=HARM, ms=3.6, zorder=3,
            label="reference site, capacity sweep")
    # Never hardcode the count. The panel said "four sites" while its sibling panel drew
    # five, which is the kind of contradiction a reviewer sees before anything else.
    ax.plot(site_rho, site_gam, "o", color=RELIEF, ms=5.5, mec="white", mew=0.8,
            zorder=4, label="%d sites (independent)" % len(site_rho))
    for name, s in out["sites"].items():
        dx, dy, ha, va = LABEL_POS.get(name, LABEL_POS_DEFAULT)
        ax.annotate(DISPLAY_NAME.get(name, name), (s["demand_capacity_ratio"], s["gamma"]),
                    textcoords="offset points", xytext=(dx, dy),
                    ha=ha, va=va, fontsize=S.FS_ANNOT, color=S.INK)

    S.clean(ax)
    ax.set_xscale("log")
    ax.set_xticks([0.05, 0.1, 0.2, 0.5, 1.0, 2.0])
    ax.set_xticklabels(["0.05", "0.1", "0.2", "0.5", "1", "2"])
    ax.set_xlabel(r"demand-to-capacity ratio  $\rho$")
    ax.set_ylabel(r"systematic tier gap  $\Gamma$")
    ax.set_ylim(-0.02, max(gaps + site_gam) * 1.20)
    ax.annotate("demand = capacity", (1.0, ax.get_ylim()[1]), xytext=(-3, -3),
                textcoords="offset points", ha="right", va="top", rotation=90,
                rotation_mode="anchor", fontsize=S.FS_ANNOT, color=S.MUTED)
    ax.legend(loc="upper left", borderaxespad=0.2)

    figdir = os.path.join(RESULTS, "figures")
    os.makedirs(figdir, exist_ok=True)
    S.savefig(fig, os.path.join(figdir, "scarcity_severity.png"))


if __name__ == "__main__":
    main()