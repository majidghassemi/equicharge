"""Battery sensitivity of the offline audit.

The reference station carries a 150 kWh / 40 kW on-site battery that starts each day at
25% charge (experiments/common.py, chargax/_station_layout.py), and the simulator adds
battery discharge to the power available at the grid root. The original oracle LP drops
the battery. This script re-solves the core offline audit on the SAME 256-day set used by
audit_mechanism.py (PRNGKey(7), fold_in(i), same filter), twice: without the battery
(must reproduce the released numbers) and with a lossless battery (a true relaxation).

It also checks the chain identity of Theorem 5.3 with the battery in the LP: at the
revenue optimum, E(V_l) must equal g_hat(V_l) for every top set V_l.

Writes experiments/results/audit_battery.json.
"""
from __future__ import annotations

import json
import os
import sys
import time

import jax
import numpy as np

from chargax.equity import oracle as O
from experiments.run_experiments import _ref_env

N_DAYS = int(os.environ.get("N_DAYS", "256"))
STATUS_QUO = (0.6, 1.0, 1.5)
TARIFFS = {
    "status_quo": STATUS_QUO,
    "premium_cap_to_mid": (0.6, 1.0, 1.0),
    "budget_subsidy_to_parity": (1.0, 1.0, 1.5),
    "income_neutral": (1.0, 1.0, 1.0),
}
BATTERY = (150.0, 40.0, 150.0 * 0.25)  # capacity kWh, max kW, initial charge kWh
RESULTS = os.path.join(os.path.dirname(__file__), "results")


def entropy(counts):
    n = sum(counts)
    p = [c / n for c in counts if c]
    return float(-sum(x * np.log(x) for x in p) / np.log(3))


def main():
    env = _ref_env(30.0, 8, 4)
    key = jax.random.PRNGKey(7)
    grid_kw = float(env.station.max_kw_throughput)
    mpt, horizon, ng = env.minutes_per_timestep, env.max_episode_steps, int(env.n_groups)

    def solve(cust, objective, margins, battery):
        return O.solve_oracle_lp(cust, grid_kw, mpt, horizon, objective=objective,
                                 price_by_group=tuple(margins), n_groups=ng, battery=battery)

    # The canonical filtered day set, built exactly as in audit_mechanism.py.
    t0 = time.time()
    streams = []
    for i in range(N_DAYS):
        cust = O.extract_arrival_stream(env, jax.random.fold_in(key, i))
        if not cust:
            continue
        pf = solve(cust, "profit", STATUS_QUO, None)
        if pf.get("degenerate") or min(pf.get("group_counts", [0])) == 0:
            continue
        streams.append(cust)
    print("day set: %d days (%.0fs)" % (len(streams), time.time() - t0), flush=True)

    out = {"n_days": len(streams), "battery": dict(capacity_kwh=BATTERY[0], max_kw=BATTERY[1],
                                                   soc0_kwh=BATTERY[2], losses="none (relaxation)"),
           "runs": {}}
    for label, batt in (("no_battery", None), ("battery", BATTERY)):
        t0 = time.time()
        run = {"tariffs": {}, "objectives": {}}
        for name, m in TARIFFS.items():
            res = [solve(c, "profit", m, batt) for c in streams]
            gms = np.array([r["group_means"] for r in res])
            avg = gms.mean(axis=0)
            worst = np.argmin(gms, axis=1)
            counts = [int(np.sum(worst == g)) for g in range(3)]
            run["tariffs"][name] = dict(
                margins=list(m), tier_means=[round(float(x), 4) for x in avg],
                gamma=round(float(avg.max() - avg.min()), 4),
                worst_tier=["budget", "mid", "premium"][int(np.argmin(avg))],
                budget_worst_share=round(counts[0] / len(res), 4),
                worst_tier_entropy=round(entropy(counts), 4),
                delivered_kwh_mean=round(float(np.mean([r["delivered_kwh"] for r in res])), 2))
        per_obj = {}
        for obj in ("profit", "utilitarian", "maximin", "egalitarian_equal"):
            res = [solve(c, obj, STATUS_QUO, batt) for c in streams]
            per_obj[obj] = res
            gms = np.array([r["group_means"] for r in res]); avg = gms.mean(axis=0)
            run["objectives"][obj] = dict(
                tier_means=[round(float(x), 4) for x in avg],
                gamma=round(float(avg.max() - avg.min()), 4),
                mean_satisfaction=round(float(np.mean([r["mean_satisfaction"] for r in res])), 4),
                revenue_mean=round(float(np.mean([r["revenue_value"] for r in res])), 2),
                delivered_kwh_mean=round(float(np.mean([r["delivered_kwh"] for r in res])), 2))
        rev = np.array([r["revenue_value"] for r in per_obj["profit"]])
        rmm = np.array([r["revenue_value"] for r in per_obj["maximin"]])
        ok = rev > 1e-9
        run["price_of_fairness_pct_median"] = round(float(np.median(100 * (rev[ok] - rmm[ok]) / rev[ok])), 2)
        e_dev = max(abs(a["delivered_kwh"] - b["delivered_kwh"])
                    for a, b in zip(per_obj["profit"], per_obj["maximin"]))
        run["total_energy_max_dev_profit_vs_maximin_kwh"] = float(e_dev)
        out["runs"][label] = run
        print("%s done (%.0fs): gamma status quo %.4f" % (label, time.time() - t0,
              run["tariffs"]["status_quo"]["gamma"]), flush=True)

    # Chain identity with the battery: E(V_l) at the revenue optimum vs g_hat(V_l).
    t0 = time.time()
    tops = {"premium": (0.0, 0.0, 1.0), "mid+premium": (0.0, 1.0, 1.0), "all": (1.0, 1.0, 1.0)}
    max_dev = 0.0
    for c in streams:
        r = solve(c, "profit", STATUS_QUO, BATTERY)
        # delivered per customer is needed; recompute it from group shares is not enough,
        # so solve with a weight vector and read revenue_value as E(V) directly.
        for name, w in tops.items():
            ghat = solve(c, "profit", w, BATTERY)["revenue_value"]
            ev = sum(wg * dg for wg, dg in zip(w, _group_energy(r, c)))
            max_dev = max(max_dev, abs(ghat - ev))
    out["chain_identity_with_battery"] = dict(days_checked=len(streams),
                                              max_abs_dev_kwh=float(max_dev))
    print("chain check done (%.0fs): max dev %.2e kWh" % (time.time() - t0, max_dev), flush=True)

    path = os.path.join(RESULTS, "audit_battery.json")
    json.dump(out, open(path, "w"), indent=1)
    print("wrote", path)


def _group_energy(res, cust):
    """Per-tier delivered energy at an optimum, from per-customer satisfaction."""
    e = np.zeros(3)
    for s, g, c in zip(res["satisfaction"], res["groups"], [c for c in cust if c.desired_kwh > 1e-3]):
        e[int(g)] += float(s) * c.desired_kwh
    return e




def sweep():
    """Status-quo gap as the battery grows, at fixed 40 kW throughput and 25% start."""
    env = _ref_env(30.0, 8, 4)
    key = jax.random.PRNGKey(7)
    grid_kw = float(env.station.max_kw_throughput)
    mpt, horizon, ng = env.minutes_per_timestep, env.max_episode_steps, int(env.n_groups)
    streams = []
    for i in range(N_DAYS):
        cust = O.extract_arrival_stream(env, jax.random.fold_in(key, i))
        if not cust:
            continue
        pf = O.solve_oracle_lp(cust, grid_kw, mpt, horizon, "profit", STATUS_QUO, ng)
        if pf.get("degenerate") or min(pf.get("group_counts", [0])) == 0:
            continue
        streams.append(cust)
    rows = []
    for cap in (0.0, 10.0, 25.0, 50.0, 75.0, 100.0, 150.0):
        batt = None if cap == 0 else (cap, 40.0, 0.25 * cap)
        res = [O.solve_oracle_lp(c, grid_kw, mpt, horizon, "profit", STATUS_QUO, ng, battery=batt)
               for c in streams]
        gms = np.array([r["group_means"] for r in res]); avg = gms.mean(axis=0)
        worst = np.argmin(gms, axis=1)
        rows.append(dict(capacity_kwh=cap, gamma=round(float(avg.max() - avg.min()), 4),
                         tier_means=[round(float(x), 4) for x in avg],
                         budget_worst_share=round(float(np.mean(worst == 0)), 4),
                         delivered_kwh_mean=round(float(np.mean([r["delivered_kwh"] for r in res])), 2)))
        print("cap %5.0f kWh: gamma %.4f, delivered %.1f kWh" % (cap, rows[-1]["gamma"],
              rows[-1]["delivered_kwh_mean"]), flush=True)
    path = os.path.join(RESULTS, "audit_battery_sweep.json")
    json.dump(dict(n_days=len(streams), max_kw=40.0, soc0_fraction=0.25, rows=rows),
              open(path, "w"), indent=1)
    print("wrote", path)


if __name__ == "__main__":
    sweep() if "--sweep" in sys.argv else main()
