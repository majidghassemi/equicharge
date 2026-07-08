r"""Clairvoyant offline welfare oracle (an upper bound / "ceiling").

Given a *realized* day -- the actual stream of admitted customers with their
arrival times, dwell windows, desired energy and power caps -- this module solves
a linear program for the best achievable welfare *with full foreknowledge of the
arrivals*. It provides the ceiling against which online policies (learned or
heuristic) are measured: the **online--offline gap** is the price of not knowing
the future.

The LP is a relaxation of the true dynamics (it keeps the binding grid-power limit,
per-car power caps and per-car energy demand, but drops the nonlinear charge curve,
battery round-trip and per-subtree limits), so its optimum is a valid *upper bound*
on the welfare any causal policy could achieve on that day.

Objectives (see :func:`solve_oracle_lp`)::

    "profit"          value-weighted delivered energy (what an unconstrained profit
                      maximizer does; reveals the inequity profit induces).
    "utilitarian"     maximize mean satisfaction (the alpha=0 efficiency ceiling).
    "maximin"         TRUE, Pareto-efficient segment maximin: maximize the worst-off
                      *segment mean* satisfaction, then (stage two) maximize total
                      satisfaction subject to holding that worst-off level. This is
                      the Rawlsian (egalitarian) ceiling and, crucially, it does
                      *not* level the better-off segments down -- so its worst
                      segment is guaranteed to be >= the worst segment of any other
                      feasible allocation, including the profit optimum.
    "maximin_individual"  same idea at the *individual* customer level (max the
                      worst individual, then Pareto-complete).
    "egalitarian_equal"   the STRICT-EQUALITY objective: force all segment means to
                      a common level and maximize it. Unlike a true maximin this
                      *levels down* -- reporting it alongside "maximin" makes the
                      difference explicit (see Sec. "Two-stage maximin" below).

Why two stages (the soundness fix)
----------------------------------
A single-objective "maximize the epigraph variable z with z <= u_i for all i" LP
finds the correct maximin *value*, but the allocation it returns is a arbitrary
vertex that typically pins every non-critical customer at z as well -- i.e. it
*levels everyone down* to the worst-off level. That produces a perfectly-equalized
row at a value *below* what is provably achievable (the profit-optimal allocation
already delivers its worst segment a higher mean), which is the tell-tale signature
of a strict-equality constraint masquerading as maximin. The fix is a second stage
that, holding the maximin value fixed, maximizes total delivered satisfaction --
returning a *Pareto-efficient* egalitarian allocation that does not drag the
better-off below what costs the worst-off nothing.

Requires SciPy (HiGHS LP solver).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from chargax.equity import welfare as W


@dataclass
class Customer:
    arrival_step: int
    window_steps: int
    desired_kwh: float
    pmax_kw: float
    group: int


def extract_arrival_stream(env, key, policy=None, max_steps: int | None = None) -> list:
    """Roll out one day and record every *admitted* customer's arrival event.

    Uses the environment's arrival log (emitted in ``info``) collected via a fast
    jitted scan, so back-to-back charger reuse is captured exactly (no
    transition-detection undercounting). The control ``policy`` only affects which
    cars are admitted (slot occupancy); allocation quality is decided by the
    oracle, so any reasonable reference policy works (default: max-charge).
    """
    from chargax.equity.baselines import max_charge_policy

    policy = policy or max_charge_policy
    max_steps = max_steps or env.max_episode_steps

    def rollout(key):
        obs, state = env.reset_env(key)

        def step(carry, _):
            k, st, ob = carry
            k, ka, ks = jax.random.split(k, 3)
            a = policy(ka, env, st, ob)
            ts, ns = env.step_env(ks, st, a)
            log = (
                ts.info["arrival_mask"],
                ts.info["arrival_desired_kwh"],
                ts.info["arrival_window_steps"],
                ts.info["arrival_pmax_kw"],
                ts.info["arrival_group"],
            )
            return (k, ns, ts.observation), log

        _, logs = jax.lax.scan(step, (key, state, obs), None, length=max_steps)
        return logs

    masks, desired, window, pmax, group = jax.jit(rollout)(key)
    masks = np.array(masks)
    desired = np.array(desired)
    window = np.array(window)
    pmax = np.array(pmax)
    group = np.array(group)

    customers: list[Customer] = []
    steps, chargers = masks.shape
    for t in range(steps):
        for c in np.where(masks[t] > 0)[0]:
            d = float(desired[t, c])
            if d < 1e-3:
                continue
            customers.append(
                Customer(
                    arrival_step=t,
                    window_steps=max(int(round(window[t, c])), 1),
                    desired_kwh=d,
                    pmax_kw=float(pmax[t, c]),
                    group=int(group[t, c]),
                )
            )
    return customers


# ---------------------------------------------------------------------------
# LP assembly helpers
# ---------------------------------------------------------------------------
def _build_base_lp(cust, grid_limit_kw, minutes_per_timestep, horizon_steps):
    """Assemble the variable map and the two structural constraint blocks shared
    by every objective: per-step grid energy and per-customer energy demand.

    Returns ``(var_index, ub, cust_vars, rows, cols, data, b_ub, n_rows)`` where the
    constraint matrix is in COO form so callers can append objective-specific rows.
    """
    dt_h = minutes_per_timestep / 60.0
    e_grid = grid_limit_kw * dt_h  # max energy delivered per step (kWh)

    # Variable index map: one var per (customer, step-in-window).
    var_index = {}
    ub = []  # per-variable upper bound (power cap * dt)
    for i, c in enumerate(cust):
        end = min(c.arrival_step + c.window_steps, horizon_steps)
        for t in range(c.arrival_step, end):
            var_index[(i, t)] = len(ub)
            ub.append(c.pmax_kw * dt_h)
    n_xy = len(ub)

    cust_vars: dict[int, list] = {}
    step_vars: dict[int, list] = {}
    for (i, t), v in var_index.items():
        cust_vars.setdefault(i, []).append(v)
        step_vars.setdefault(t, []).append(v)

    rows, cols, data, b_ub = [], [], [], []
    r = 0
    # (1) per-step grid energy: sum_i x[i,t] <= e_grid
    for t, vs in step_vars.items():
        for v in vs:
            rows.append(r); cols.append(v); data.append(1.0)
        b_ub.append(e_grid)
        r += 1
    # (2) per-customer energy cap: sum_t x[i,t] <= desired_i  (=> satisfaction <= 1)
    for i, vs in cust_vars.items():
        for v in vs:
            rows.append(r); cols.append(v); data.append(1.0)
        b_ub.append(cust[i].desired_kwh)
        r += 1
    return var_index, ub, cust_vars, rows, cols, data, b_ub, r, n_xy


def _solve(c_obj, rows, cols, data, b_ub, bounds, n_var, n_rows):
    from scipy.optimize import linprog
    from scipy.sparse import csr_matrix

    A_ub = csr_matrix((data, (rows, cols)), shape=(n_rows, n_var))
    res = linprog(c_obj, A_ub=A_ub, b_ub=np.array(b_ub), bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"Oracle LP failed: {res.message}")
    return res


def _sat_from_x(x, cust_vars, cust):
    """Return (satisfaction, delivered_kwh) per customer."""
    n = len(cust)
    sat = np.zeros(n)
    delivered = np.zeros(n)
    for i, vs in cust_vars.items():
        d = sum(x[v] for v in vs)
        delivered[i] = d
        sat[i] = min(1.0, d / cust[i].desired_kwh)
    return sat, delivered


def _group_index(cust, n_groups):
    """Return, per group, the list of customer indices."""
    members: dict[int, list] = {g: [] for g in range(n_groups)}
    for i, c in enumerate(cust):
        members.setdefault(c.group, []).append(i)
    return members


def solve_oracle_lp(
    customers: list,
    grid_limit_kw: float,
    minutes_per_timestep: int,
    horizon_steps: int,
    objective: str = "utilitarian",
    price_by_group: tuple = (),
    n_groups: int | None = None,
) -> dict:
    """Solve the clairvoyant allocation LP. Returns per-customer satisfactions.

    See the module docstring for the objective semantics. The maximin variants use
    a two-stage (lexicographic) LP so the returned allocation is *Pareto efficient*
    and does not level the better-off down to the worst-off level.
    """
    cust = [c for c in customers if c.desired_kwh > 1e-3]
    n = len(cust)
    if n == 0:
        # Degenerate day (no qualifying demand). Return the FULL schema (all keys _package
        # produces) so callers never hit a KeyError; group_means has one entry per tier.
        ng = n_groups if n_groups else 1
        return {
            "satisfaction": np.array([]), "groups": np.array([]),
            "mean_satisfaction": 0.0, "min_satisfaction": 0.0, "worst_segment_mean": 0.0,
            "group_means": [0.0] * ng, "group_disparity": 0.0, "worst_segment_index": 0,
            "delivered_kwh": 0.0, "revenue_value": 0.0, "welfare": 0.0,
            "objective": objective, "degenerate": True,
        }

    (var_index, ub, cust_vars, rows0, cols0, data0, b_ub0, n_rows0, n_xy) = _build_base_lp(
        cust, grid_limit_kw, minutes_per_timestep, horizon_steps
    )
    if n_xy == 0:
        return _package(np.zeros(n), np.zeros(n), cust, objective, price_by_group)

    if n_groups is None:
        n_groups = int(max(c.group for c in cust)) + 1

    # ---- simple single-stage objectives -----------------------------------
    if objective in ("utilitarian", "profit"):
        bounds = [(0.0, u) for u in ub]
        c_obj = np.zeros(n_xy)
        if objective == "profit":  # maximize value-weighted delivered energy
            for (i, t), v in var_index.items():
                mult = price_by_group[cust[i].group] if price_by_group else 1.0
                c_obj[v] = -mult
        else:  # utilitarian: maximize mean satisfaction
            for (i, t), v in var_index.items():
                c_obj[v] = -1.0 / cust[i].desired_kwh / n
        res = _solve(c_obj, rows0, cols0, data0, b_ub0, bounds, n_xy, n_rows0)
        sat, delivered = _sat_from_x(res.x, cust_vars, cust)
        return _package(sat, delivered, cust, objective, price_by_group)

    # ---- two-stage maximin / equalization ---------------------------------
    # An epigraph variable z is appended as the last column. Stage 1 maximizes z
    # subject to z <= (level of each unit: segment mean or individual sat). Stage 2
    # fixes z >= z* and maximizes total satisfaction (Pareto completion).
    members = _group_index(cust, n_groups)
    nonempty = [g for g in range(n_groups) if members[g]]

    def epigraph_rows(kind, r_start):
        """Rows encoding z <= level_u for each unit u (segment mean or individual)."""
        rows, cols, data, b = [], [], [], []
        r = r_start
        z = n_xy
        if kind == "segment":
            for g in nonempty:
                idxs = members[g]
                Ng = len(idxs)
                # z - (1/Ng) sum_{i in g} (1/desired_i) sum_t x[i,t] <= 0
                rows.append(r); cols.append(z); data.append(1.0)
                for i in idxs:
                    for v in cust_vars[i]:
                        rows.append(r); cols.append(v); data.append(-1.0 / (Ng * cust[i].desired_kwh))
                b.append(0.0); r += 1
        else:  # individual
            for i in cust_vars:
                rows.append(r); cols.append(z); data.append(1.0)
                for v in cust_vars[i]:
                    rows.append(r); cols.append(v); data.append(-1.0 / cust[i].desired_kwh)
                b.append(0.0); r += 1
        return rows, cols, data, b, r

    kind = "individual" if objective == "maximin_individual" else "segment"
    n_var = n_xy + 1
    bounds = [(0.0, u) for u in ub] + [(0.0, 1.0)]  # z in [0,1]

    if objective == "egalitarian_equal":
        # Strict equality across segment means: force every segment mean == z and
        # maximize z (this is the "levels down" objective we contrast against).
        rows = list(rows0); cols = list(cols0); data = list(data0); b_ub = list(b_ub0)
        r = n_rows0
        z = n_xy
        for g in nonempty:
            idxs = members[g]; Ng = len(idxs)
            # segment_mean_g - z <= 0  AND  z - segment_mean_g <= 0  (=> equality)
            for sign in (1.0, -1.0):
                rows.append(r); cols.append(z); data.append(-sign)
                for i in idxs:
                    for v in cust_vars[i]:
                        rows.append(r); cols.append(v); data.append(sign / (Ng * cust[i].desired_kwh))
                b_ub.append(0.0); r += 1
        c_obj = np.zeros(n_var); c_obj[z] = -1.0
        res = _solve(c_obj, rows, cols, data, b_ub, bounds, n_var, r)
        sat, delivered = _sat_from_x(res.x, cust_vars, cust)
        return _package(sat, delivered, cust, objective, price_by_group)

    # Stage 1: maximize z (worst-off level).
    rows, cols, data, b_ub = list(rows0), list(cols0), list(data0), list(b_ub0)
    er, ec, ed, eb, r1 = epigraph_rows(kind, n_rows0)
    rows += er; cols += ec; data += ed; b_ub += eb
    c_obj = np.zeros(n_var); c_obj[n_xy] = -1.0
    res1 = _solve(c_obj, rows, cols, data, b_ub, bounds, n_var, r1)
    z_star = float(res1.x[n_xy])

    # Stage 2: hold z >= z* - tol, maximize total satisfaction (Pareto completion).
    tol = 1e-6
    bounds2 = [(0.0, u) for u in ub] + [(max(0.0, z_star - tol), 1.0)]
    c_obj2 = np.zeros(n_var)
    for (i, t), v in var_index.items():
        c_obj2[v] = -1.0 / cust[i].desired_kwh / n
    res2 = _solve(c_obj2, rows, cols, data, b_ub, bounds2, n_var, r1)
    sat, delivered = _sat_from_x(res2.x, cust_vars, cust)
    out = _package(sat, delivered, cust, objective, price_by_group)
    out["maximin_value"] = z_star
    return out


def _package(sat, delivered, cust: list, objective: str, price_by_group: tuple = ()) -> dict:
    sat = np.asarray(sat, dtype=float)
    delivered = np.asarray(delivered, dtype=float)
    groups = np.array([c.group for c in cust])
    n_groups = int(groups.max()) + 1 if len(groups) else 1
    group_means = [
        float(sat[groups == g].mean()) if np.any(groups == g) else 0.0
        for g in range(n_groups)
    ]
    group_counts = [int(np.sum(groups == g)) for g in range(n_groups)]
    # Revenue proxy: value-weighted delivered energy (kWh * ability-to-pay
    # multiplier). This is what the operator earns from the allocation, so the
    # profit-based price of fairness can be read off directly.
    if len(price_by_group) and len(delivered):
        mult = np.array([price_by_group[g] for g in groups])
        revenue_value = float(np.sum(delivered * mult))
    else:
        revenue_value = float(np.sum(delivered)) if len(delivered) else 0.0
    return {
        "satisfaction": sat,
        "groups": groups,
        "mean_satisfaction": float(np.mean(sat)) if len(sat) else 0.0,
        "min_satisfaction": float(np.min(sat)) if len(sat) else 0.0,
        "worst_segment_mean": float(min(group_means)) if group_means else 0.0,
        "group_means": group_means,
        "group_counts": group_counts,
        "group_disparity": float(max(group_means) - min(group_means)) if group_means else 0.0,
        "delivered_kwh": float(np.sum(delivered)) if len(delivered) else 0.0,
        "revenue_value": revenue_value,
        "objective": objective,
    }


def oracle_welfare(
    env, key, objective: str = "utilitarian", policy=None, tariff=None
) -> dict:
    """Convenience: extract the realized stream and solve the oracle LP.

    ``tariff`` optionally overrides the per-group price multipliers used by the
    ``"profit"`` objective (see :mod:`chargax.equity.tariffs`); if ``None`` the
    environment's ``price_by_group`` is used.
    """
    customers = extract_arrival_stream(env, key, policy=policy)
    price_by_group = tuple(env.price_by_group) if tariff is None else tuple(tariff)
    out = solve_oracle_lp(
        customers,
        grid_limit_kw=float(env.station.max_kw_throughput),
        minutes_per_timestep=env.minutes_per_timestep,
        horizon_steps=env.max_episode_steps,
        objective=objective,
        price_by_group=price_by_group,
        n_groups=int(env.n_groups),
    )
    out["n_customers"] = len(customers)
    return out
