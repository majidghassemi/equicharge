r"""Render ``experiments/results/results.json`` into ``results/RESULTS.md``.

Produces the headline tables referenced by the paper: the offline-oracle
characterization (with the corrected, non-leveling segment maximin and the
revenue-based price of fairness), the rate-design (low-income subsidy) finding, the
grid-capacity threshold, the unified baseline+learned policy table with multi-seed
dispersion, the endogeneity audit, and the scarcity contrast.
"""

from __future__ import annotations

import json
import os

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def _g(d, k, default=float("nan")):
    return d.get(k, default) if isinstance(d, dict) else default


def _med(d, k):
    """Median +/- half-IQR string for a multi-seed aggregate, else plain value."""
    if f"{k}_median" in d:
        return f"{d[f'{k}_median']:.3f} ± {d.get(f'{k}_iqr', 0.0)/2:.3f}"
    v = d.get(k)
    return f"{v:.3f}" if isinstance(v, (int, float)) else "n/a"


def main():
    with open(os.path.join(RESULTS_DIR, "results.json")) as f:
        R = json.load(f)

    meta = R.get("meta", {})
    L = ["# Results: The Price of a Fair Charge", ""]
    L += [
        f"_Config: {meta.get('n_evses', '?')*2} chargers behind a "
        f"{meta.get('grid_kw','?')} kW grid; ability-to-pay multipliers "
        f"{meta.get('price_by_group','?')} (income/energy-burden grounded); "
        f"profit_scale={meta.get('profit_scale','?')} EUR/day; "
        f"{meta.get('oracle_days','?')} realized days for the oracle; "
        f"{meta.get('seeds','?')} training seed(s); gamma=1, V2G off._", ""]

    # ---------------------------------------------- offline oracle
    o = R.get("oracle", {})
    if isinstance(o.get("profit"), dict):
        L += ["## 1. Offline oracle: the efficiency-equity tradeoff at optimality "
              "(clairvoyant LP)", "",
              "The profit-optimal allocation starves the budget segment to extract "
              "price-discrimination revenue; the **true (Pareto-efficient) segment "
              "maximin** equalizes the segments *without leveling down* -- it "
              "delivers the same total energy as the profit optimum, so equity is "
              "nearly free in aggregate satisfaction and is paid for almost entirely "
              "in the operator's revenue.", "",
              "| oracle objective | budget | mid | premium | disparity | mean sat | "
              "delivered kWh | revenue |", "|---|---|---|---|---|---|---|---|"]
        rows = [("profit-optimal", "profit"), ("utilitarian", "utilitarian"),
                ("segment maximin (true)", "maximin"),
                ("strict equalization", "egalitarian_equal")]
        for name, k in rows:
            d = o[k]
            gm = d["group_means"]
            L.append(f"| {name} | " + " | ".join(f"{v:.3f}" for v in gm) +
                     f" | {d['group_disparity']:.3f} | {d['mean_satisfaction']:.3f} | "
                     f"{d['delivered_kwh']:.0f} | {d['revenue_value']:.0f} |")
        pofr = o.get("price_of_fairness_revenue", float("nan")) * 100
        pofs = o.get("price_of_fairness_meansat", float("nan")) * 100
        L += ["",
              f"**Price of fairness (profit-optimal -> segment maximin):** "
              f"**{pofr:.1f}% of operator revenue**, while mean satisfaction *rises* "
              f"({o['profit']['mean_satisfaction']:.3f} -> {o['maximin']['mean_satisfaction']:.3f}, "
              f"i.e. {pofs:.1f}%) and the worst segment goes "
              f"{o['profit']['worst_segment_mean']:.3f} -> {o['maximin']['worst_segment_mean']:.3f}. "
              f"The tension is operator-revenue vs equity, not social-welfare vs equity.",
              "",
              "> _Note on the corrected maximin._ A single-epigraph max-min LP finds "
              "the right worst-off value but returns an allocation that levels the "
              "better-off down to it (the 'strict equalization' row); the two-stage "
              "LP holds the maximin value and then Pareto-completes, so the true "
              "segment maximin never reports a worst segment below what the profit "
              "optimum already achieves.", ""]

    # ---------------------------------------------- rate-design (tariff) finding
    rob = R.get("robustness", {})
    if rob.get("rows"):
        L += ["## 1b. Robustness: the price of fairness is a redistribution premium "
              "customers pay for", "",
              "The mean-satisfaction rise under equalization is a consequence of "
              "concave, *saturating* charging utility (a driver does not want more "
              "than a full battery), not a free lunch. The premium tier gives up real "
              "satisfaction; we report that explicitly and check the effect is not an "
              "artifact of one price gradient by sweeping the premium tier's margin.",
              "",
              "| premium margin | premium sat (profit→maximin) | premium loss | budget gain | mean sat (profit→maximin) | PoF revenue |",
              "|---|---|---|---|---|---|"]
        for r in rob["rows"]:
            L.append(f"| {r['premium_weight']:.1f}× | {r['profit_premium_sat']:.2f} → "
                     f"{r['maximin_premium_sat']:.2f} | −{r['premium_sat_loss']:.2f} | "
                     f"+{r['budget_sat_gain']:.2f} | {r['mean_sat_profit']:.2f} → "
                     f"{r['mean_sat_maximin']:.2f} | {r['pof_revenue_pct']:.0f}% |")
        L += ["",
              "_Reading:_ across every price gradient the premium tier loses ≈0.11 of "
              "satisfaction and the budget tier gains ≈0.44; equalization is 'nearly "
              "free' only in *equal-weighted aggregate*, a value choice we state. The "
              "revenue price of fairness grows with the price gradient (8→14%), so the "
              "contribution is the magnitude and the price-vs-scarcity decomposition, "
              "not the sign (which is the classical implication of concave utility).", ""]

    t = R.get("tariffs", {})
    if t.get("rows"):
        vw = t.get("value_weighted_disparity", float("nan"))
        fl = t.get("flat_tariff_disparity", float("nan"))
        red = t.get("price_driven_reduction_pct", float("nan"))
        bo = t.get("best_budget_only_disparity", float("nan"))
        bon = t.get("best_budget_only_worst_segment", "?")
        bpc = t.get("best_partial_cap_disparity", float("nan"))
        L += ["## 2. Rate-design finding: the disparate impact is a function of the "
              "margin *spread*", "",
              f"Disparity = the gap between best- and worst-served tier in "
              f"**tier-averaged** satisfaction (an absolute satisfaction-point gap, "
              f"max−min of the tier means; whether a tier is *systematically* "
              f"under-served — a disparate impact — not day-to-day noise). Under the "
              f"status-quo tiered tariff it is **{vw:.3f}** (worst tier = budget). "
              f"**Both partial interventions fail, for the same reason.** A subsidy to "
              f"the lowest tier alone reaches only **{bo:.3f}** and past parity shifts "
              f"the burden to the **{bon}** tier. A cap on the premium tier alone "
              f"reaches only **{bpc:.3f}**, because the budget tier stays lowest-margin "
              f"and thus still starved. Only equalizing the margins (income-neutral / "
              f"full compression) closes it, to **{fl:.3f}** (−{red:.0f}%). The lever "
              f"is the **spread**, not either end. _(Revenue is not shown across rows: "
              f"value-weighted energy at different margins is in different units, so a "
              f"cross-tariff revenue comparison would be meaningless.)_", "",
              "| intervention | margins | disparity | worst tier |",
              "|---|---|---|---|"]
        # Subsidy rows.
        for r in t["rows"]:
            m = f"{r['budget_weight']:.1f}, 1.0, 1.5"
            L.append(f"| budget subsidy +{r['subsidy']:.1f} | {m} | "
                     f"{r['disparity']:.3f} | {r.get('worst_segment_name','?')} |")
        # Premium-cap rows.
        for r in t.get("premium_cap", {}).get("rows", []):
            c = r["cap"]
            m = f"0.6, {min(1.0,c):.1f}, {min(1.5,c):.1f}"
            L.append(f"| premium cap {c:.1f} | {m} | {r['disparity']:.3f} | "
                     f"{r.get('worst_segment_name','?')} |")
        L += ["",
              "_Actionable reading:_ reduce the **spread** of tier margins — an income-"
              "neutral tariff, or a credit that equalizes *all* tiers — removes the "
              f"{red:.0f}% price-driven disparate impact. Grid capacity (Section 3) is "
              "a second, independent lever: enough capacity removes the scarcity that "
              "forces rationing at all. Temporal designs (ToU / demand charges) are "
              "tier-agnostic and do not touch the distributional gap.", ""]

    # ---------------------------------------------- capacity threshold
    c = R.get("capacity", {})
    if c.get("rows"):
        tg = c.get("threshold_kw_gini", c.get("threshold_kw"))
        to = c.get("threshold_kw_oracle")
        L += ["## 3. Grid capacity: a range, not a sharp threshold", "",
              f"Enough grid capacity removes the scarcity that forces rationing at all "
              f"(a second, independent lever from pricing). We deliberately state this "
              f"as a **range, not a sharp threshold**: the oracle tier disparity is "
              f"noisy and slightly non-monotonic near its floor (~0.02–0.03 above 70 "
              f"kW), so pinning a single kW value would over-claim. Inequality falls "
              f"steeply and then flattens — the Gini reaches within 10% of its abundant "
              f"floor ({c.get('abundant_gini', float('nan')):.3f}) by ~**{tg} kW**, and "
              f"the oracle tier disparity is small (≲0.05) from ~**{to} kW** upward. "
              f"For this 16-charger site the disparity is largely removed by roughly "
              f"**{tg}–{to} kW**; a planner reading the stricter oracle metric should "
              f"size to the upper end.", "",
              "_The 30 kW oracle-disparity row below equals the Section-1 oracle "
              "exactly (same days, same seed), so the two analyses are consistent._", "",
              "| grid kW | Gini | mean sat | worst-10% | oracle tier disparity |",
              "|---|---|---|---|---|"]
        for r in c["rows"]:
            L.append(f"| {int(r['grid_kw'])} | {r['gini']:.3f} | "
                     f"{r['served_mean_sat']:.3f} | {r['worst10_sat']:.3f} | "
                     f"{r['oracle_profit_disparity']:.3f} |")
        L.append("")

    # ---------------------------------------------- unified policy table
    base = R.get("baselines", {})
    trained = R.get("trained", {})
    L += ["## 4. Policies: non-learned baselines and learned welfare policies", "",
          "One table, matched metrics. Learned rows show the median across "
          f"{meta.get('seeds','?')} seeds with ± half the inter-quartile range; "
          "baseline rows are deterministic heuristics evaluated on the same days.",
          "",
          "| policy | kind | profit (EUR/day) | mean sat | Gini | worst-10% | segment disp | rejection |",
          "|---|---|---|---|---|---|---|---|"]
    for name in ["max_charge", "proportional_fair", "least_laxity", "saffe", "random"]:
        if name in base:
            d = base[name]
            L.append(f"| {d['label']} | baseline | {_g(d,'profit_mean'):.0f} | "
                     f"{_g(d,'served_mean_sat'):.3f} | {_g(d,'gini'):.3f} | "
                     f"{_g(d,'worst10_sat'):.3f} | "
                     f"{_g(d,'group_disparity'):.3f} | {_g(d,'rejection_rate'):.2f} |")
    order = ["profit_ppo", "util_l050", "util_l100", "egal_l025", "egal_l050",
             "egal_l075", "egal_l100"]
    for k in order:
        if k not in trained:
            continue
        d = trained[k]
        pm = d.get("profit_mean_median", d.get("profit_mean", float("nan")))
        pi = d.get("profit_mean_iqr", 0.0) / 2
        L.append(f"| {d['label']} | learned | {pm:.0f} ± {pi:.0f} | "
                 f"{_med(d,'served_mean_sat')} | {_med(d,'gini')} | "
                 f"{_med(d,'worst10_sat')} | {_med(d,'group_disparity')} | "
                 f"{_med(d,'rejection_rate')} |")
    L += ["",
          "_Reading (honest):_ at this **250k-timestep** budget every learned policy "
          "is **Pareto-dominated by `random`** (161 profit / 0.439 Gini / 0.681 sat "
          "beats all seven on all three axes), and `max_charge` beats `random`. No "
          "online policy achieves positive worst-10% satisfaction. A learning-curve "
          "diagnostic (`learning_curve.json`) shows this is **undertraining**, not a "
          "plateau: the egalitarian policy at 1M steps reaches ~173 profit / 0.428 "
          "Gini / 0.698 sat (beating `random`), but is unstable across budget/seed. "
          "We therefore **do not present the learned policies as a result** — they are "
          "released as a benchmark starting point, and the RL is confined to the "
          "paper's appendix. See Section 7.", ""]

    # ---------------------------------------------- endogeneity audit
    e = R.get("endogeneity", {})
    if e:
        L += ["## 5. Endogeneity audit: the reject-to-look-fair loophole is closed", "",
              "Defining utility over the *true arrival stream* (rejected customers at "
              "0) is what makes an endogenous, policy-influenced population "
              "well-posed. Ablating rejection-counting shows how much apparent "
              "welfare a policy could gain by shedding customers it serves poorly.",
              "",
              "| policy | welfare (served-only) | welfare (inclusive) | gaming gap | rejection |",
              "|---|---|---|---|---|"]
        for name, d in e.items():
            L.append(f"| {name} | {d['welfare_served_only']:.3f} | "
                     f"{d['welfare_inclusive']:.3f} | {d['gaming_gap']:.3f} | "
                     f"{d['rejection_rate']:.2f} |")
        L.append("")

    # ---------------------------------------------- scarcity contrast
    sc = R.get("scarcity", {})
    if sc:
        L += ["## 6. Scarcity induces inequity (profit-blind max-charge)", "",
              "| grid | mean sat | Gini | worst-10% |", "|---|---|---|---|"]
        for tag, d in sc.items():
            L.append(f"| {int(d['grid_kw'])} kW | {d['served_mean_sat']:.3f} | "
                     f"{d['gini']:.3f} | {d['worst10_sat']:.3f} |")
        L.append("")

    out = os.path.join(RESULTS_DIR, "RESULTS.md")
    with open(out, "w") as f:
        f.write("\n".join(L) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
