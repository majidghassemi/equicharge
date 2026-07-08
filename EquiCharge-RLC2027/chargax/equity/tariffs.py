r"""Tariff structures and the rate-design analysis (the actionable policy layer).

The efficiency-equity conflict this paper characterizes is *created by the tariff*:
when the price a customer pays is tied to ability-to-pay (a value-weighted tariff),
a revenue-maximizing operator is incentivized to route scarce power to the
high-paying segments and starve the budget segment. The natural policy question a
utility rate designer or regulator can act on is therefore: **which tariff designs
shrink or remove that equity gap, and at what cost to the operator's revenue?**

This module provides

* named **customer tariff structures** (:data:`TARIFFS`) as callables that return a
  per-segment effective price weight (what the operator earns for serving each
  ability-to-pay segment). These drive the ``"profit"`` oracle objective, so the
  profit-optimal allocation -- and hence the equity gap it opens -- can be
  recomputed under each design; and
* time-of-use / demand-charge **grid-price callables** for the environment, so the
  same tariff structures can be exercised in the online (RL/heuristic) setting.

The headline rate-design result (:func:`subsidy_sweep`) traces the profit-optimal
segment disparity as a **low-income charging subsidy** raises the budget segment's
effective tariff weight toward parity, and reports the subsidy level at which the
profit-optimal allocation stops starving the budget segment (disparity ~ 0). This
is a concrete recommendation -- a lever a utility actually pulls (a low-income EV
tariff / bill credit) -- rather than a restatement of the flat-price identity.

References
----------
Borenstein (2012), "The Redistributional Impact of Nonlinear Electricity Pricing",
AEJ: Economic Policy -- distributional incidence of electricity rate design.
Burger, Knittel, Perez-Arriaga, Schneider, Vom Scheidt (2020), "The Efficiency and
Distributional Effects of Alternative Residential Electricity Rate Designs",
Energy Journal -- ToU vs. demand charges vs. flat, and their equity incidence.
"""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np

# --------------------------------------------------------------------- structures
# Each tariff maps the *base* per-segment ability-to-pay multipliers to the
# per-segment price weight the operator earns. The status-quo value-weighted tariff
# passes the multipliers through; a flat/equity tariff flattens them.

#: Base ability-to-pay multipliers (budget / mid / premium). Grounded in the
#: residential energy-burden gradient (low-income households spend ~3x the income
#: share on energy of higher-income households; see ``segments.py``).
BASE_MULTIPLIERS = (0.6, 1.0, 1.8)


def value_weighted(base: tuple = BASE_MULTIPLIERS) -> tuple:
    """Status-quo tariff: the operator earns each segment's willingness-to-pay."""
    return tuple(base)


def flat(base: tuple = BASE_MULTIPLIERS) -> tuple:
    """A single volumetric price for everyone -- removes ability-to-pay weighting.

    Under this design profit == total delivered energy, so the profit-optimal and
    the (utilitarian) satisfaction-optimal allocations coincide and the equity gap
    vanishes. This is the flat-tariff identity, made explicit as one point on the
    rate-design spectrum rather than a standalone result.
    """
    return tuple(1.0 for _ in base)


def low_income_subsidy(subsidy: float, base: tuple = BASE_MULTIPLIERS) -> tuple:
    r"""Add a per-kWh subsidy to the budget segment's effective tariff weight.

    A regulator/utility credits low-income customers so the operator earns
    ``base[0] + subsidy`` for serving them. As ``subsidy`` raises the budget weight
    to parity with the premium segment, the operator's incentive to starve the
    budget segment disappears and the profit-optimal allocation equalizes.
    """
    b = list(base)
    b[0] = b[0] + subsidy
    return tuple(b)


def cap_premium(cap: float, base: tuple = BASE_MULTIPLIERS) -> tuple:
    """Cap every tier's margin at ``cap`` (a top-tier price cap / compression).

    A regulator can cap the margin an operator may earn on the premium tier rather
    than subsidize the bottom. Note this only removes the *premium* tier's advantage;
    if ``cap`` is above the budget margin the budget tier is still the lowest-paying
    and thus still the profit-maximizer's first to starve, so a top cap alone does not
    reach the flat-tariff floor. Compressing all the way to the budget margin recovers
    the flat tariff.
    """
    return tuple(min(w, cap) for w in base)


TARIFFS: Dict[str, Callable[..., tuple]] = {
    "value_weighted": value_weighted,
    "flat": flat,
    "low_income_subsidy": low_income_subsidy,
    "cap_premium": cap_premium,
}


def premium_cap_sweep(env, key, caps, n_days: int = 8) -> dict:
    """Trace the profit-optimal tier gap as the premium tier's margin is capped."""
    base = tuple(env.price_by_group) if env.price_by_group else BASE_MULTIPLIERS
    rows = []
    for c in caps:
        st = _profit_stats(env, key, cap_premium(float(c), base), n_days)
        st.update({"cap": float(c)})
        rows.append(st)
    return {"rows": rows, "base_multipliers": list(base)}


# ----------------------------------------------------- time-of-use / demand charge
def time_of_use_price(env, peak_price: float = 0.30, offpeak_price: float = 0.10,
                      peak_hours: tuple = (17, 21)):
    """Return a ``get_grid_buy_price(state)`` callable for a time-of-use tariff.

    ToU is *group-agnostic* (the price depends on the hour, not on who is charging),
    so -- unlike a value-weighted tariff -- it does not by itself create
    ability-to-pay discrimination. Its effect is temporal: it shifts the
    profit-optimal *schedule* toward cheap hours. We include it to show that
    temporal rate designs move *when* power is used without closing the
    *distributional* gap, which only ability-to-pay-targeted designs do.
    """
    import jax.numpy as jnp

    lo, hi = peak_hours

    def buy_price(state):
        hour = (state.timestep * env.minutes_per_timestep) / 60.0
        return jnp.where((hour >= lo) & (hour < hi), peak_price, offpeak_price)

    return buy_price


def demand_charge_price(env, base_price: float = 0.15, demand_charge: float = 0.0):
    """Return a ``get_grid_buy_price`` callable with a flat volumetric component.

    A true demand charge prices the *peak* draw over a billing period, which the
    single-day oracle LP does not span; at the daily horizon we approximate its
    incentive with the grid-power cap already in the model (the binding constraint
    that forces rationing). Provided for completeness of the tariff menu.
    """
    import jax.numpy as jnp

    def buy_price(state):
        return jnp.asarray(base_price)

    return buy_price


# ----------------------------------------------------------------- rate-design sweep
SEGMENT_NAMES = ("budget", "mid", "premium")


def _profit_stats(env, key, weights, n_days):
    """Profit-optimal stats under a tariff, computed from day-averaged group means so
    disparity/worst-segment are consistent with the reported group means. Also returns
    *which* segment is worst-off (the identity matters for the justice reading)."""
    import jax
    from chargax.equity import oracle as O
    groups, rev, mean = [], [], []
    for i in range(n_days):
        k = jax.random.fold_in(key, i)
        r = O.oracle_welfare(env, k, objective="profit", tariff=tuple(weights))
        groups.append(r["group_means"]); rev.append(r["revenue_value"])
        mean.append(r["mean_satisfaction"])
    gm = np.mean(np.array(groups), axis=0)
    worst_idx = int(np.argmin(gm))
    return {"disparity": float(np.max(gm) - np.min(gm)),
            "worst_segment": float(np.min(gm)),
            "worst_segment_name": SEGMENT_NAMES[worst_idx] if worst_idx < 3 else str(worst_idx),
            "revenue_value": float(np.mean(rev)),
            "mean_satisfaction": float(np.mean(mean)),
            "group_means": gm.tolist()}


def subsidy_sweep(env, key, subsidies, n_days: int = 8) -> dict:
    """Rate-design analysis: trace the profit-optimal equity gap vs. tariff design.

    We recompute the *profit-optimal* clairvoyant allocation under three designs:

    * the status-quo **value-weighted** tariff (the base multipliers);
    * a **budget-only subsidy** swept over ``subsidies`` (added to the budget weight);
    * the **income-neutral / flat** tariff (all weights equal).

    The key, non-obvious finding this exposes is that a subsidy targeted *only* at
    the lowest-income segment does **not** close the gap -- it merely relocates the
    starvation to whichever segment is now lowest-weighted -- so the effective
    intervention is full income-neutrality, not a partial credit. What income
    neutrality *cannot* remove is the residual, *scarcity-driven* disparity (which
    the grid-capacity lever addresses). Returns the sweep, the value-weighted and
    flat endpoints, and the price-driven vs. scarcity-driven decomposition.
    """
    base = tuple(env.price_by_group) if env.price_by_group else BASE_MULTIPLIERS
    rows = []
    for s in subsidies:
        weights = low_income_subsidy(float(s), base)
        st = _profit_stats(env, key, weights, n_days)
        st.update({"subsidy": float(s), "budget_weight": float(weights[0])})
        rows.append(st)
    vw_disp = rows[0]["disparity"]
    flat_stats = _profit_stats(env, key, flat(base), n_days)
    best = min(rows, key=lambda r: r["disparity"])
    price_driven = max(vw_disp - flat_stats["disparity"], 0.0)
    return {
        "rows": rows,
        "base_multipliers": list(base),
        "value_weighted_disparity": vw_disp,
        "value_weighted_worst_segment_name": rows[0]["worst_segment_name"],
        "flat_tariff_disparity": flat_stats["disparity"],
        "flat_tariff_worst_segment": flat_stats["worst_segment"],
        "flat_tariff_revenue": flat_stats["revenue_value"],
        "price_driven_gap": price_driven,          # removed by income-neutral pricing
        "residual_scarcity_gap": flat_stats["disparity"],  # only capacity can remove this
        "price_driven_reduction_pct": 100.0 * price_driven / (vw_disp + 1e-9),
        "best_budget_only_disparity": best["disparity"],
        "best_budget_only_subsidy": best["subsidy"],
        "best_budget_only_worst_segment": best["worst_segment_name"],
    }
