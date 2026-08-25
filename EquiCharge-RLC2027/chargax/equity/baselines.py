r"""Non-learned baseline policies for welfare-optimal EV charging.

All policies share the signature ``policy(key, env, state, obs) -> action`` and
return an action dict compatible with :class:`~chargax.equity.fair_env.EquiChargax`.
They let us answer the central question of the paper: does a *learned* policy beat
(a) profit-blind max-charging, (b) *myopic* fair heuristics that allocate fairly
among the currently-present cars but cannot anticipate future arrivals, and
(c) a *model-based anticipatory planner* (SAFFE-style) that reserves capacity
against expected future demand?

Policies provided
-----------------
* :func:`random_policy`            -- uniform random actions (floor).
* :func:`max_charge_policy`        -- always request max charge (profit-blind).
* :func:`proportional_fair_policy` -- myopic: request power proportional to each
  present car's *remaining need* (needier cars get a larger share once the grid
  renormalizes). No anticipation.
* :func:`least_laxity_policy`      -- myopic: prioritize the most *urgent* cars
  (smallest laxity = deadline slack). The classic fair online-scheduling rule.
* :func:`margin_greedy_policy`     -- the ONLINE analogue of the revenue-optimal
  offline allocation: serve the connected cars in strict descending tier margin,
  each tier requesting its feasible maximum, lower tiers taking only the residual
  headroom under the grid limit. This is the behaviour Theorem 5.3's chain of
  equalities pins down offline, realized causally, so it is the baseline that says
  what a revenue-maximizing operator would actually deploy.
* :func:`saffe_policy`             -- model-based anticipatory planner: estimates
  expected *future* demand over the remaining horizon and throttles present
  allocation / banks battery energy so future (possibly needier) arrivals are not
  starved. A Chargax instantiation of SAFFE-D (Hassanzadeh et al., ICAIF 2023).

These are deliberately implemented at the action level; the environment's current
renormalization then enforces the shared grid/subtree power limits.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray


def _split_flat_to_evses(env, flat: Array) -> list:
    """Split a ``(num_chargers,)`` array into the per-EVSE action list."""
    sizes = [e.num_chargers for e in env.station.evses]
    out, idx = [], 0
    for s in sizes:
        out.append(flat[idx : idx + s])
        idx += s
    return out


def _evse_action_from_fraction(env, frac: Array, connected: Array) -> Array:
    """Map a per-charger charging fraction in [0,1] to a discrete EVSE action.

    With discharging enabled the idle action is ``num_discretization_levels`` and a
    fraction ``f`` maps to ``round(num_disc * (1 + f))`` (so ``f=1`` is max charge).
    Disconnected chargers are set to idle.
    """
    n = env.num_discretization_levels
    idle = n if env.allow_discharging else 0
    if env.allow_discharging:
        a = jnp.round(n * (1.0 + jnp.clip(frac, 0.0, 1.0))).astype(jnp.int32)
    else:
        a = jnp.round(n * jnp.clip(frac, 0.0, 1.0)).astype(jnp.int32)
    return jnp.where(connected > 0, a, idle).astype(jnp.int32)


def _battery_action(env, state) -> Array:
    """Shared rule: discharge to support demand at peak, charge when slack & cheap.

    Returns a per-battery discrete action (idle = ``num_disc``; ``0`` = full
    discharge; ``2*num_disc`` = full charge)."""
    n = env.num_discretization_levels
    batteries = state.grid.batteries_flat
    evses = state.grid.evses_flat
    connected = evses.charger_is_car_connected.astype(jnp.float32)

    # Present demand (kW) the cars could absorb this step.
    max_cur = jnp.minimum(evses.max_current, evses.car_max_current_intake)
    demand_kw = jnp.sum(evses.voltage * max_cur / 1000.0 * connected)
    grid_limit = env.station.max_kw_throughput

    price = env.get_grid_buy_price(state)
    # Heuristic price reference (median-ish of the NL datasets ~ 0.1-0.2 EUR/kWh).
    cheap = price < 0.12

    soc = batteries.battery_now / (batteries.capacity_kw + 1e-8)
    discharge = demand_kw > 0.85 * grid_limit
    charge = (~discharge) & cheap & (soc < 0.95)

    a = jnp.full(batteries.battery_now.shape, n, dtype=jnp.int32)  # idle
    a = jnp.where(discharge, jnp.int32(0), a)
    a = jnp.where(charge, jnp.int32(2 * n), a)
    return a


def _assemble(env, evse_fracs: Array, connected: Array, state) -> dict:
    evse_actions = _evse_action_from_fraction(env, evse_fracs, connected)
    return {
        "evses": _split_flat_to_evses(env, evse_actions),
        "batteries": list(jnp.atleast_1d(_battery_action(env, state))),
    }


# --------------------------------------------------------------------- policies
def random_policy(key: PRNGKeyArray, env, state, obs) -> dict:
    return env.sample_action(key)


def max_charge_policy(key: PRNGKeyArray, env, state, obs) -> dict:
    evses = state.grid.evses_flat
    connected = evses.charger_is_car_connected.astype(jnp.float32)
    fracs = jnp.ones_like(connected)
    return _assemble(env, fracs, connected, state)


def proportional_fair_policy(key: PRNGKeyArray, env, state, obs) -> dict:
    """Request power proportional to remaining need (myopic proportional fairness)."""
    evses = state.grid.evses_flat
    connected = evses.charger_is_car_connected.astype(jnp.float32)
    remaining = jnp.maximum(evses.car_battery_desired_remaining_kw, 0.0)
    # Normalize so the neediest present car requests full power.
    denom = jnp.max(remaining * connected) + 1e-6
    fracs = jnp.clip(remaining / denom, 0.0, 1.0) * connected
    return _assemble(env, fracs, connected, state)


def least_laxity_policy(key: PRNGKeyArray, env, state, obs) -> dict:
    """Prioritize the most urgent cars (smallest laxity = slack before deadline)."""
    evses = state.grid.evses_flat
    connected = evses.charger_is_car_connected.astype(jnp.float32)

    remaining_kwh = jnp.maximum(evses.car_battery_desired_remaining_kw, 0.0)
    max_rate_kw = jnp.maximum(evses.voltage * evses.car_max_current_intake / 1000.0, 1e-3)
    required_min = remaining_kwh / max_rate_kw * 60.0  # minutes of charging needed
    laxity = evses.car_time_till_leave - required_min  # slack (minutes)

    # Urgent cars (laxity below one timestep of buffer) charge at full; the rest
    # get a small trickle so spare capacity is not wasted.
    urgent = laxity <= env.minutes_per_timestep
    fracs = jnp.where(urgent, 1.0, 0.25) * connected
    return _assemble(env, fracs, connected, state)


def _margin_levels(env) -> list:
    """Tiers grouped by *distinct* margin, ordered strictly descending.

    Returns ``[(margin, [tier indices at that margin]), ...]``. Tiers that pay the
    same margin share a level, so a tariff that compresses two tiers onto the same
    price (a premium cap) makes them peers rather than silently ranking one above
    the other on a float tie. The margins are static Python floats
    (``env.price_by_group``), so the ordering is resolved at trace time and the
    policy stays jittable.
    """
    n_groups = int(env.n_groups)
    margins = ([float(w) for w in env.price_by_group] if env.price_by_group
               else [1.0] * n_groups)
    levels: dict = {}
    for g, w in enumerate(margins):
        levels.setdefault(w, []).append(g)
    return [(w, levels[w]) for w in sorted(levels, key=lambda v: -v)]


def margin_greedy_policy(key: PRNGKeyArray, env, state, obs) -> dict:
    r"""Revenue-greedy rationing: strict priority by tier margin (the status quo).

    At every step the connected cars are served in **strictly descending tier
    margin**. Each car in the current tier requests its *feasible maximum*, the
    lesser of what its charger can push and what its battery will take. Tiers
    further down the price ordering receive only the **residual** headroom under
    the site's grid limit, and a tier that cannot be served in full splits its
    residual pro rata, so the discrimination this policy expresses is strictly
    between tiers and never inside one.

    This is the online counterpart of the revenue-optimal offline allocation. The
    offline optimum saturates the nested chain of top sets (the highest-margin tier
    first, then the two highest, and so on), and this policy realizes the same
    priority causally, one step at a time, without foreknowledge of arrivals. It is
    therefore the baseline that answers "what does a revenue-maximizing operator
    actually deploy", as distinct from the LP ceiling that says what such an
    operator could achieve if it knew the day in advance.

    The allocation is done in the only currency the station can be told to spend,
    whole action levels, and it never exceeds the power actually on offer. That
    matters here in a way it does not for the other baselines. If the request
    overshot, the station's ``distribute()`` would renormalize every charger
    *proportionally* and wash out exactly the strict priority this policy exists to
    express.
    """
    evses = state.grid.evses_flat
    connected = evses.charger_is_car_connected.astype(jnp.float32)
    groups = state.charger_group
    dt_h = env.minutes_per_timestep / 60.0
    n = env.num_discretization_levels

    # A charger's action level buys ``step_kw``, but the car's own intake cap then
    # clips it, so the power a level actually delivers is ``min(L*step_kw, cap_kw)``.
    # Asking for more than the cap is free, which is why the ceiling below is a ceil.
    charger_kw = evses.voltage * evses.max_current / 1000.0
    intake_kw = evses.voltage * evses.car_max_current_intake / 1000.0
    remaining_kwh = jnp.maximum(evses.car_battery_desired_remaining_kw, 0.0)
    active = connected * (remaining_kwh > 1e-6).astype(jnp.float32)
    step_kw = charger_kw / n
    cap_kw = jnp.minimum(charger_kw, intake_kw) * active
    full_level = jnp.minimum(jnp.ceil(cap_kw / (step_kw + 1e-9)), float(n))

    def delivered(levels):
        return jnp.minimum(levels * step_kw, cap_kw)

    # The power on offer this step is the grid connection plus whatever the on-site
    # battery is about to discharge, less whatever it is about to draw to charge.
    # ``_assemble`` applies the shared battery rule below, so we predict it here with
    # the same predicate and the same state-of-charge limits. Ignoring it would make
    # this policy ration against a budget the station does not actually have.
    grid_limit = jnp.asarray(float(env.station.max_kw_throughput), jnp.float32)
    batteries = state.grid.batteries_flat
    hardware_demand_kw = jnp.sum(
        evses.voltage * jnp.minimum(evses.max_current, evses.car_max_current_intake)
        / 1000.0 * connected
    )
    discharging = hardware_demand_kw > 0.85 * grid_limit
    cheap = env.get_grid_buy_price(state) < 0.12
    soc = batteries.battery_now / (batteries.capacity_kw + 1e-8)
    charging = (~discharging) & cheap & (soc < 0.95)
    out_kw = jnp.minimum(batteries.max_kw_throughput, batteries.battery_now / dt_h)
    in_kw = jnp.minimum(
        batteries.max_kw_throughput,
        (batteries.capacity_kw - batteries.battery_now) / dt_h,
    )
    headroom = grid_limit
    headroom = headroom + jnp.where(discharging, jnp.sum(out_kw), 0.0)
    headroom = jnp.maximum(headroom - jnp.sum(jnp.where(charging, in_kw, 0.0)), 0.0)

    alloc = jnp.zeros_like(cap_kw)
    for _margin, tiers in _margin_levels(env):
        in_tier = jnp.zeros_like(active)
        for g in tiers:
            in_tier = in_tier + (groups == g).astype(jnp.float32)
        in_tier = in_tier * active

        request = cap_kw * in_tier
        share = jnp.minimum(1.0, headroom / (jnp.sum(request) + 1e-9))
        target = request * share

        # Largest level whose delivered power stays within this car's target.
        levels = jnp.where(target >= cap_kw - 1e-9, full_level,
                           jnp.floor(target / (step_kw + 1e-9)))
        levels = jnp.clip(levels, 0.0, float(n)) * in_tier
        headroom = headroom - jnp.sum(delivered(levels))

        # The pro-rata split almost never lands on a level boundary, and a car whose
        # intake is finer than one level would otherwise be quantized to nothing.
        # Hand the stranded headroom out by largest remainder (the standard
        # apportionment rule) so the tier spends its residual instead of wasting it.
        step_cost = delivered(jnp.minimum(levels + 1.0, float(n))) - delivered(levels)
        remainder = jnp.where((in_tier > 0) & (levels < full_level), target - delivered(levels), -1.0)
        order = jnp.argsort(-remainder)
        fits = (jnp.cumsum(step_cost[order]) <= headroom + 1e-9) & (remainder[order] >= 0.0)
        extra = jnp.zeros_like(levels).at[order].set(fits.astype(levels.dtype))
        headroom = jnp.maximum(headroom - jnp.sum(extra * step_cost), 0.0)
        alloc = alloc + levels + extra

    return _assemble(env, alloc / n, connected, state)


def saffe_policy(key: PRNGKeyArray, env, state, obs) -> dict:
    r"""SAFFE-D-style anticipatory planner (model-based).

    Estimates the expected total future demand over the remaining horizon from the
    *known* arrival statistics, computes the share of currently-available energy
    that should be reserved for future arrivals, and throttles the present
    allocation accordingly (banking the rest in the battery). This is the
    EV-charging instantiation of "allocate against expected future demand"
    (Hassanzadeh et al. 2023). It assumes a forecastable demand model -- the
    assumption the learned policy does *not* need.
    """
    evses = state.grid.evses_flat
    connected = evses.charger_is_car_connected.astype(jnp.float32)
    remaining = jnp.maximum(evses.car_battery_desired_remaining_kw, 0.0)

    present_demand = jnp.sum(remaining * connected) + 1e-6

    # Expected NEAR-future demand over a short lookahead window (not the whole
    # day -- grid power is use-it-or-lose-it, so only near-term contention matters
    # for the reservation). Read the arrival rate from the env's model.
    lookahead = jnp.minimum(
        60 // env.minutes_per_timestep, jnp.maximum(env.max_episode_steps - state.timestep, 0)
    )  # ~1 hour
    k = jax.random.split(key, 8)
    exp_arrivals_per_step = jnp.mean(
        jnp.stack([env.get_num_cars_arriving(kk, state).astype(jnp.float32) for kk in k])
    )
    mean_desired = present_demand / (jnp.sum(connected) + 1e-6)
    expected_future_demand = exp_arrivals_per_step * lookahead * mean_desired

    # Fair reservation share, floored at 0.5 so present customers are never starved.
    present_share = present_demand / (present_demand + expected_future_demand + 1e-6)
    present_share = jnp.clip(present_share, 0.5, 1.0)

    # Throttle proportional-fair fractions by the reservation share; the battery
    # rule banks the energy held back for the anticipated peak.
    denom = jnp.max(remaining * connected) + 1e-6
    base = jnp.clip(remaining / denom, 0.0, 1.0)
    fracs = base * present_share * connected
    return _assemble(env, fracs, connected, state)


POLICIES = {
    "random": random_policy,
    "max_charge": max_charge_policy,
    "proportional_fair": proportional_fair_policy,
    "least_laxity": least_laxity_policy,
    "margin_greedy": margin_greedy_policy,
    "saffe": saffe_policy,
}
