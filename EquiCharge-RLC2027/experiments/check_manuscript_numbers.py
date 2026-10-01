"""Check that the numbers printed in the FAccT manuscript match the released results.

Every value below is recomputed from the JSON files in experiments/results/ (the 256-day
run) and looked up verbatim in the .tex file. A mismatch means the paper and the results
have drifted apart again.

Usage (from EquiCharge-RLC2027/):
    python check_manuscript_numbers.py path/to/facct27_14pages.tex
"""
import json
import math
import os
import re
import sys

TEX = sys.argv[1]
RES = sys.argv[2] if len(sys.argv) > 2 else "experiments/results"


def load(name):
    return json.load(open(os.path.join(RES, name)))


def norm(t):
    """Drop comment lines and collapse whitespace so line breaks do not matter."""
    t = "\n".join(l for l in t.split("\n") if not l.lstrip().startswith("%"))
    return re.sub(r"\s+", " ", t)


tex = norm(open(TEX).read())
off = load("audit_offline.json")
mech = load("audit_mechanism.json")
gro = load("audit_grounding.json")
rob = load("audit_robustness.json")
est = load("audit_estimator_check.json")
lev = load("audit_levers.json")
sev = load("scarcity_severity.json")
mg = load("audit_margin_greedy.json")
res = load("results.json")


def f2(x):
    return "%.2f" % x


def pct(x):
    return "%d\\%%" % round(100 * x)


def H(counts):
    n = sum(counts.values())
    return -sum(c / n * math.log(c / n) for c in counts.values() if c) / math.log(3)


checks = []  # (description, string that must appear in the manuscript)

# --- day count -------------------------------------------------------------------------
assert off["n_days"] == 256 and mech["day_set"]["used_after_filter"] == 256
checks.append(("day count", "two hundred fifty-six realized days"))

# --- Table 1 (elasticity sweep) ----------------------------------------------------------
gap_by_eta = {r["eta"]: r["gap"] for r in mech["elasticity_invariance"]}
for eta, label in [(0.2, "elasticity $0.2$ (weak spread)"), (0.6, "elasticity $0.6$ (reference)"),
                   (1.0, "elasticity $1.0$ (strong spread)")]:
    g = gro["elasticity_sweep"]["%.1f" % eta]
    row = "%s & %.3f & %s & %.1f\\%% \\\\" % (label, gap_by_eta[eta], pct(g["budget_worst_fraction"]),
                                            g["revenue_pof_pct"]["median"])
    checks.append(("Table 1 eta=%.1f" % eta, row))
g0 = gro["elasticity_sweep"]["0.0"]
checks.append(("Table 1 eta=0", "elasticity $0$ (no spread) & %.3f & %s & 0\\%% \\\\"
               % (gap_by_eta[0.0], pct(g0["budget_worst_fraction"]))))

# --- Table 2 (oracle) ----------------------------------------------------------------------
names = [("profit", "revenue optimal"), ("utilitarian", "utilitarian"),
         ("maximin", "tier maximin (true)"), ("egalitarian_equal", "strict equalization")]
for key, label in names:
    o = res["oracle"][key]
    gm = o["group_means"]
    row = "%s & %s & %s & %s & %s & %d \\\\" % (label, f2(gm[0]), f2(gm[1]), f2(gm[2]),
                                               f2(o["group_disparity"]), round(o["revenue_value"]))
    checks.append(("Table 2 " + key, row))

# --- price of fairness: one definition, one value ---------------------------------------
pof = off["box1"]["revenue_price_of_fairness_pct"]["median"]
assert round(pof, 1) == 10.5
checks.append(("PoF definition", "computed for each day and reported as the median over days"))
checks.append(("PoF in text", "a median near ten and a half percent"))

# --- Table 3 (tariff mechanism) -----------------------------------------------------------
tm = {r["config"]: r for r in mech["mechanism"]}
for cfg, label, margins, spread in [
        ("status_quo_distinct", "tiered (status quo)", "0.6, 1.0, 1.5", "0.9"),
        ("premium_capped_to_mid", "premium cap to middle", "0.6, 1.0, 1.0", "0.4"),
        ("budget_tied_to_mid", "budget subsidy to parity", "1.0, 1.0, 1.5", "0.5"),
        ("fully_flat", "income neutral (equal)", "all equal", "0.0")]:
    checks.append(("Table 3 " + cfg, "%s & %s & %s & %s &" % (label, margins, spread, f2(tm[cfg]["gap"]))))
checks.append(("mechanism text cap", "barely moves, from $%s$ to $%s$" % (f2(tm["status_quo_distinct"]["gap"]),
                                                                          f2(tm["premium_capped_to_mid"]["gap"]))))
checks.append(("mechanism text subsidy", "the gap drops to $%s$" % f2(tm["budget_tied_to_mid"]["gap"])))

# --- Table 4 (sites) -----------------------------------------------------------------------
sites = [("reference_16ch_30kW_residential_eu", "reference (16 ch, 30 kW)", "%.2f"),
         ("highway_20ch_45kW_eu_highrate", "highway (20 ch, 45 kW)", "%.2f"),
         ("workplace_24ch_55kW_us", "workplace (24 ch, 55 kW)", "%.3f"),
         ("shopping_8ch_16kW_world", "shopping (8 ch, 16 kW)", "%.2f")]
for cfg, label, gfmt in sites:
    c = rob["configs"][cfg]
    pd_gap = est["check1_rotation"][cfg]["per_day_gap_median"]
    p = c["revenue_pof_pct"]["median"]
    pof_cell = "0\\%" if round(p, 1) == 0 else "%.1f\\%%" % p
    gam = gfmt % c["profit_optimal_disparity_dayavg"]
    row = "%s & %s & %s" % (label, pct(c["budget_worst_fraction"]), gam)
    checks.append(("Table 4 %s (worst, Gamma)" % label, row))
    checks.append(("Table 4 %s (per-day, H, PoF)" % label,
                   "%s & %s & %s \\\\" % (f2(pd_gap), f2(H(c["worst_tier_days"])), pof_cell)))

# --- levers --------------------------------------------------------------------------------
pl, cl = lev["pricing_lever_30kW_sq_to_flat"], lev["capacity_lever_sq_30_to_90kW"]
checks.append(("capacity lever", "between-tier gap, by about $%s$, and the within-population inequality, by about $%s$"
               % (f2(cl["d_between_tier_gap"]), f2(cl["d_within_tier_gini"]))))
checks.append(("pricing lever", "between-tier gap, about $%s$, while barely moving the within-population inequality, about $%s$"
               % (f2(pl["d_between_tier_gap"]), f2(pl["d_within_tier_gini"]))))

# --- severity axis ------------------------------------------------------------------------
sw = {r["grid_kw"]: r for r in sev["capacity_sweep"]}
checks.append(("severity sweep start", "collapses from $%s$ to noise" % f2(sw[40]["gamma"])))
checks.append(("severity shopping/highway", "with $\\Gamma\\approx %s$ and highway at $\\rho\\approx %.1f$ with $\\Gamma\\approx %s$"
               % (f2(sev["sites"]["shopping"]["gamma"]), sev["sites"]["highway"]["demand_capacity_ratio"],
                  f2(sev["sites"]["highway"]["gamma"]))))
assert "acn" in sev["sites"]  # the released file includes ACN; the figure in this version omits it

# --- margin-greedy (Appendix B), battery-free station ------------------------------
mgn = load("audit_margin_greedy_nobattery.json")
r = mgn["reference"]
checks.append(("greedy Gamma and H", "$\\Gamma\\approx %s$ and rotation entropy $H\\approx %s$"
               % (f2(r["gamma"]), f2(r["worst_tier_entropy"]))))
others = mgn["other_online_baselines"]
gmin, gmax = min(v["gamma"] for v in others.values()), max(v["gamma"] for v in others.values())
checks.append(("non-rationing heuristics", "$\\Gamma\\approx %.3f$ to $%.3f$" % (gmin, gmax)))
assert round(min(v["worst_tier_entropy"] for v in others.values()), 2) == 1.00
assert round(r["budget_worst_share"] * 100) == 85
checks.append(("greedy budget-worst share", "worst-served on eighty-five percent of days"))
checks.append(("greedy offline gap, same days", "of the offline optimum's gap on the same days ($%s$)"
               % f2(r["offline_profit_optimal"]["gamma"])))
sweep = {s_["grid_kw"]: s_ for s_ in mgn["scarcity_sweep"]}
checks.append(("greedy sweep", "gap falls from about $%s$ at thirty kilowatts" % f2(sweep[30.0]["gamma"])))
checks.append(("greedy sweep 45 kW", "to about $%s$ by forty-five kilowatts" % f2(sweep[45.0]["gamma"])))
checks.append(("greedy sweep 60 kW", "to about $%s$ by sixty" % f2(sweep[60.0]["gamma"])))
checks.append(("greedy profit", "about %d euros per day against about %d for max-charge"
               % (round(r["profit_per_day"]["mean"]), round(others["max_charge"]["profit_per_day"]["mean"]))))
checks.append(("greedy with battery", "same controller still reproduces the harm, with $\\Gamma\\approx %s$"
               % f2(mg["reference"]["gamma"])))

# --- battery sweep (Section 6 and appendix table) -----------------------------------
bs = {r_["capacity_kwh"]: r_ for r_ in load("audit_battery_sweep.json")["rows"]}
assert load("audit_battery_sweep.json")["n_days"] == 256
checks.append(("storage paragraph", "from $%s$ without storage to $%s$ at 100 kWh and $%s$ at 150 kWh"
               % (f2(bs[0.0]["gamma"]), f2(bs[100.0]["gamma"]), f2(bs[150.0]["gamma"]))))
for cap, r_ in bs.items():
    checks.append(("battery table %d kWh" % cap, "%d & %.2f & %.2f & %.2f & %d\\%% & %d \\\\"
                   % (cap, r_["gamma"], r_["tier_means"][0], r_["tier_means"][2],
                      round(100 * r_["budget_worst_share"]), round(r_["delivered_kwh_mean"]))))
assert load("audit_battery.json")["chain_identity_with_battery"]["max_abs_dev_kwh"] < 1e-9

# --- tier-to-income propagation (Appendix D) ------------------------------------------
gam = rob["configs"]["reference_16ch_30kW_residential_eu"]["profit_optimal_disparity_dayavg"]
checks.append(("propagation", "an income-level gap of $%s$ at $\\kappa=0.8$, $%s$ at $\\kappa=0.5$, and $%s$"
               % (f2(0.8 * gam), f2(0.5 * gam), f2(gam / 3))))

# --- priority neutrality (Option B) -------------------------------------------------
pn = load("audit_priority_neutral.json")
assert pn["n_days"] == 256
u = pn["objectives"]["utilitarian"]; pr = pn["objectives"]["profit"]
checks.append(("neutrality: budget tier", "lifts the budget tier from %s to %s"
               % (f2(pr["tier_means"][0]), f2(u["tier_means"][0]))))
checks.append(("neutrality: median cost", "median cost of %.1f percent of revenue"
               % pn["price_of_neutrality_pct"]["median"]))
flat = pr["revenue_mean"] / pr["delivered_kwh_mean"]   # Theorem 5.4: same energy at any flat optimum
assert round(pn["price_of_neutrality_pct"]["median"]) == 10
checks.append(("neutrality cost in Sec. 7", "priority neutrality is a private revenue transfer near ten percent"))
checks.append(("revenue-neutral flat margin", "from 0.6 to about %.2f" % flat))
assert abs(pr["delivered_kwh_mean"] - u["delivered_kwh_mean"]) < 1e-6

# --- ACN-Data fifth site ------------------------------------------------------------
acn = rob["configs"]["acn_data_caltech_8ch_30kW"]
acn_est = load("audit_estimator_check_acn.json")["check1_rotation"]["acn_data_caltech_8ch_30kW"]
assert acn["lp_success"] == 255 and acn_est["days"] == 255
checks.append(("Table sites ACN row", "ACN-Data (16 ch, 30 kW) & %s & %s & %s & %s & %.1f\\%% \\\\"
               % (pct(acn["budget_worst_fraction"]), f2(acn["profit_optimal_disparity_dayavg"]),
                  f2(acn_est["per_day_gap_median"]), f2(H(acn["worst_tier_days"])),
                  acn["revenue_pof_pct"]["median"])))
checks.append(("sites text ACN gap", "$0.25$ on the Caltech ACN-Data sessions"))
checks.append(("severity ACN", "at $\\rho\\approx %s$ with $\\Gamma\\approx %s$ against the curve's $%s$"
               % (f2(sev["sites"]["acn"]["demand_capacity_ratio"]), f2(sev["sites"]["acn"]["gamma"]),
                  f2(sev["sites"]["acn"]["sweep_gamma_at_same_rho"]))))
prov = acn["layout"]
assert prov["sessions_used"] == 27584 and prov["date_range_used"][0].startswith("2018-04")
checks.append(("ACN sessions", "27,584 sessions at the Caltech garage between April 2018 and February 2020"))
# subsidy leaves about half the gap or more at every power-bound site
for cfg in ("reference_16ch_30kW_residential_eu", "highway_20ch_45kW_eu_highrate", "acn_data_caltech_8ch_30kW"):
    lv = rob["configs"][cfg]["tariff_lever"]
    assert lv["budget_subsidy_parity_disparity"] / lv["status_quo_disparity"] > 0.45, cfg
    assert lv["income_neutral_disparity"] / lv["status_quo_disparity"] < 0.10, cfg

failed = 0
for desc, needle in checks:
    ok = norm(needle) in tex
    failed += not ok
    print("%-4s %s" % ("ok" if ok else "FAIL", desc + ("" if ok else "   expected: " + needle)))
print("\n%d checks, %d failed" % (len(checks), failed))
sys.exit(1 if failed else 0)
