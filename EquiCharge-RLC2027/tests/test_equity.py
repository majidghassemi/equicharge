"""Unit tests for the welfare-optimal (fair) EV-charging layer.

Run with: ``uv run pytest tests/test_equity.py -q`` (or ``python -m pytest``).
Covers the welfare math (telescoping, EDE bounds, alpha-ordering), the inequality
metrics, and the EquiChargax mechanics (rejection counting, horizon flush,
grouping, bounded sufficient statistic, observation augmentation).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chargax import ChargingStation
from chargax.equity import EquiChargax, metrics as M, welfare as W


# --------------------------------------------------------------------- welfare
def test_utilitarian_equals_mean():
    u = jnp.array([0.2, 0.6, 1.0])
    assert abs(float(W.utilitarian(u)) - float(jnp.mean(u))) < 1e-5


def test_alpha0_equals_utilitarian():
    u = jnp.array([0.1, 0.5, 0.9])
    assert abs(float(W.alpha_fairness(u, alpha=0.0)) - float(W.utilitarian(u))) < 1e-5


def test_softmin_approaches_min():
    u = jnp.array([0.2, 0.7, 0.9])
    assert abs(float(W.rawlsian_softmin(u, beta=300.0)) - float(jnp.min(u))) < 5e-3


def test_ede_bounded_in_unit_interval():
    """EDE welfare stays in [0,1] for every alpha, even with zero utilities."""
    for alpha in [0.0, 0.5, 1.0, 2.0, 5.0]:
        for u in [jnp.array([0.0, 0.0, 1.0]), jnp.array([1.0, 1.0, 1.0]), jnp.array([0.3, 0.6, 0.9])]:
            mk = jnp.mean(W.inner_kernel(u, alpha))
            ede = float(W.equally_distributed_equivalent(mk, alpha))
            assert -1e-6 <= ede <= 1.0 + 1e-6, (alpha, u, ede)


def test_ede_alpha_ordering():
    """For an unequal distribution, EDE decreases as inequality aversion grows."""
    u = jnp.array([0.1, 0.5, 0.9])
    edes = []
    for alpha in [0.0, 1.0, 3.0]:
        mk = jnp.mean(W.inner_kernel(u, alpha))
        edes.append(float(W.equally_distributed_equivalent(mk, alpha)))
    assert edes[0] > edes[1] > edes[2]  # utilitarian > Nash > more-egalitarian
    assert edes[0] == pytest.approx(float(jnp.mean(u)), abs=1e-4)


def test_welfare_from_statistics_pooled_equals_individual():
    """Pooled utilitarian welfare over groups == EDE over all customers."""
    counts = jnp.array([2.0, 3.0])
    # group sums of kernel for alpha=0 (identity)
    g0 = jnp.array([0.2, 0.8])  # sum of u in group 0 (2 customers)
    g1 = jnp.array([0.3, 0.6, 0.9])  # group 1 (3 customers)
    ksum = jnp.array([float(jnp.sum(g0)), float(jnp.sum(g1))])
    w = float(W.welfare_from_statistics(counts, ksum, alpha=0.0, outer="utilitarian"))
    allu = jnp.concatenate([g0, g1])
    assert w == pytest.approx(float(jnp.mean(allu)), abs=1e-5)


def test_welfare_empty_population_is_zero():
    for alpha in [0.0, 2.0]:
        w = float(W.welfare_from_statistics(jnp.zeros(2), jnp.zeros(2), alpha=alpha))
        assert w == pytest.approx(0.0, abs=1e-6)


def test_telescoping_identity():
    """Sum of welfare increments == W(final) - W(initial) for all SWFs."""
    key = jax.random.PRNGKey(0)
    steps = jax.random.uniform(key, (40, 3)) * 0.05
    R = jnp.concatenate([jnp.zeros((1, 3)), jnp.cumsum(steps, axis=0)], axis=0)
    for kind, kw in [("utilitarian", {}), ("nash", {}), ("alpha", {"alpha": 2.0}), ("rawlsian", {"beta": 20.0})]:
        inc = sum(float(W.welfare_increment(R[t + 1], R[t], kind=kind, **kw)) for t in range(len(R) - 1))
        direct = float(W.social_welfare(R[-1], kind=kind, **kw) - W.social_welfare(R[0], kind=kind, **kw))
        assert inc == pytest.approx(direct, abs=1e-3), kind


# --------------------------------------------------------------------- metrics
def test_gini_bounds():
    assert float(M.gini(jnp.ones(10) * 0.5)) == pytest.approx(0.0, abs=1e-6)
    extreme = jnp.array([0.0] * 9 + [1.0])
    assert 0.85 < float(M.gini(extreme)) <= 1.0


def test_atkinson_zero_for_equality():
    assert float(M.atkinson(jnp.ones(8) * 0.7, 1.0)) == pytest.approx(0.0, abs=1e-3)


def test_worst_quantile_and_pof():
    x = jnp.array([0.0, 0.0, 0.5, 0.5, 1.0])
    assert float(M.worst_quantile_mean(x, 0.4)) == pytest.approx(0.0, abs=1e-6)
    assert M.price_of_fairness(profit_fair=80.0, profit_efficient=100.0) == pytest.approx(0.2)


# ------------------------------------------------------------------- env tests
@pytest.fixture
def station():
    return ChargingStation.init_default_station()


def _rollout(env, key):
    obs, state = env.reset_env(key)

    def step(carry, _):
        k, st = carry
        k, ka, ks = jax.random.split(k, 3)
        a = env.sample_action(ka)
        ts, ns = env.step_env(ks, st, a)
        return (k, ns), (ts.reward, ts.info)

    (_, final), (rew, info) = jax.lax.scan(step, (key, state), None, length=env.max_episode_steps)
    return final, rew, info


def test_env_telescopes_to_final_welfare(station):
    """For lambda=1, sum of rewards equals the final welfare (W(R_0)=0)."""
    env = EquiChargax(station=station, welfare_alpha=0.0, lam=1.0, n_groups=1)
    final, rew, info = jax.jit(lambda k: _rollout(env, k))(jax.random.PRNGKey(0))
    assert float(jnp.sum(rew)) == pytest.approx(float(info["welfare"][-1]), abs=1e-2)


def test_rejections_counted_at_zero(station):
    """Counting rejections lowers welfare vs. not counting (same policy/seed)."""
    base = dict(station=station, welfare_alpha=0.0, lam=1.0, n_groups=1)
    env_on = EquiChargax(**base, count_rejections=True)
    env_off = EquiChargax(**base, count_rejections=False)
    k = jax.random.PRNGKey(1)
    _, _, info_on = jax.jit(lambda kk: _rollout(env_on, kk))(k)
    _, _, info_off = jax.jit(lambda kk: _rollout(env_off, kk))(k)
    # With rejections counted at 0, the group count is larger and welfare no higher.
    assert float(info_on["group_count"][-1].sum()) >= float(info_off["group_count"][-1].sum())
    assert float(info_on["welfare"][-1]) <= float(info_off["welfare"][-1]) + 1e-6


def test_horizon_flush_counts_all_connected(station):
    """No car should still be 'connected and uncounted' at the end of the day."""
    env = EquiChargax(station=station, welfare_alpha=0.0, lam=1.0, n_groups=1)
    final, _, _ = jax.jit(lambda k: _rollout(env, k))(jax.random.PRNGKey(2))
    assert int(final.grid.evses_flat.charger_is_car_connected.sum()) == 0


def test_grouping_assigns_by_capacity(station):
    env = EquiChargax(station=station, n_groups=3, group_edges=(45.0, 75.0))
    caps = jnp.array([30.0, 60.0, 90.0])
    groups = env._group_of(caps)
    assert groups.tolist() == [0, 1, 2]


def test_augmentation_present_and_shaped(station):
    env = EquiChargax(station=station, n_groups=3, group_edges=(45.0, 75.0), augment_obs=True)
    obs, _ = env.reset_env(jax.random.PRNGKey(0))
    assert "fair_context" in obs
    assert obs["fair_context"].shape == (3 * 3 + 1,)  # 3*G + 1
    env2 = EquiChargax(station=station, augment_obs=False)
    obs2, _ = env2.reset_env(jax.random.PRNGKey(0))
    assert "fair_context" not in obs2


def test_satisfaction_in_unit_interval(station):
    env = EquiChargax(station=station, welfare_alpha=0.0, lam=1.0, n_groups=1)
    _, _, info = jax.jit(lambda k: _rollout(env, k))(jax.random.PRNGKey(3))
    util = np.array(info["realized_util"])[np.array(info["realized_mask"]) > 0]
    assert (util >= -1e-6).all() and (util <= 1.0 + 1e-6).all()


# ---------------------------------------------------- ability-to-pay segments
def test_price_mult_maps_segments(station):
    env = EquiChargax(station=station, n_groups=3, group_probs=(1 / 3,) * 3,
                      price_by_group=(0.6, 1.0, 1.8))
    mult = env._price_mult(jnp.array([0, 1, 2, 1]))
    assert np.allclose(np.array(mult), [0.6, 1.0, 1.8, 1.0])


def test_charger_price_in_observation(station):
    env = EquiChargax(station=station, n_groups=3, group_probs=(1 / 3,) * 3,
                      price_by_group=(0.6, 1.0, 1.8), augment_obs=True)
    obs, _ = env.reset_env(jax.random.PRNGKey(0))
    assert "charger_price" in obs
    assert obs["charger_price"].shape == (station.num_chargers,)


def test_pricing_changes_profit(station):
    """Value-weighted pricing yields different profit than flat pricing (same seed)."""
    base = dict(station=station, n_groups=3, group_probs=(1 / 3,) * 3, lam=0.0)
    flat = EquiChargax(**base)  # no price_by_group -> flat
    priced = EquiChargax(**base, price_by_group=(0.6, 1.0, 1.8))
    k = jax.random.PRNGKey(5)
    _, _, info_flat = jax.jit(lambda kk: _rollout(flat, kk))(k)
    _, _, info_priced = jax.jit(lambda kk: _rollout(priced, kk))(k)
    assert abs(float(info_flat["profit"][-1]) - float(info_priced["profit"][-1])) > 1e-3


def test_segment_persists_while_connected(station):
    """A connected car keeps its segment; the charger_group field tracks it."""
    env = EquiChargax(station=station, n_groups=3, group_probs=(1 / 3,) * 3)
    final, _, _ = jax.jit(lambda k: _rollout(env, k))(jax.random.PRNGKey(6))
    # charger_group is a valid segment index everywhere.
    cg = np.array(final.charger_group)
    assert cg.min() >= 0 and cg.max() < 3


# ------------------------------------------------------ corrected offline oracle
def _toy_customers():
    """A scarce day: a value-weighted profit optimum starves the low-paying group."""
    from chargax.equity.oracle import Customer
    cust = []
    # group 0 = budget (many, need power), group 1 = premium (fewer, high pay)
    for _ in range(6):
        cust.append(Customer(arrival_step=0, window_steps=6, desired_kwh=20.0,
                             pmax_kw=22.0, group=0))
    for _ in range(3):
        cust.append(Customer(arrival_step=0, window_steps=6, desired_kwh=20.0,
                             pmax_kw=22.0, group=1))
    return cust


def _solve(obj, price=(1.0, 3.0)):
    from chargax.equity.oracle import solve_oracle_lp
    return solve_oracle_lp(_toy_customers(), grid_limit_kw=30.0, minutes_per_timestep=5,
                           horizon_steps=8, objective=obj, price_by_group=price, n_groups=2)


def test_oracle_maximin_does_not_level_below_profit():
    """The corrected two-stage segment maximin's worst segment must be >= the profit
    optimum's worst segment (it is feasible-dominated), i.e. it never levels down."""
    prof = _solve("profit")
    mm = _solve("maximin")
    assert mm["worst_segment_mean"] >= prof["worst_segment_mean"] - 1e-6


def test_oracle_maximin_is_pareto_efficient():
    """True maximin delivers at least as much total energy as strict equalization
    (which wastes energy leveling down)."""
    mm = _solve("maximin")
    eq = _solve("egalitarian_equal")
    assert mm["delivered_kwh"] >= eq["delivered_kwh"] - 1e-6
    assert mm["group_disparity"] <= 0.05  # still (near-)equalized across segments


def test_oracle_profit_starves_low_payer_and_earns_more_revenue():
    prof = _solve("profit")
    util = _solve("utilitarian")
    # Profit favours the high-paying group and earns more value-weighted revenue.
    assert prof["group_means"][1] >= prof["group_means"][0]
    assert prof["revenue_value"] >= util["revenue_value"] - 1e-6


def test_price_of_fairness_is_in_revenue_not_meansat():
    """Equalizing should cost revenue while not cutting mean satisfaction."""
    prof, mm = _solve("profit"), _solve("maximin")
    assert prof["revenue_value"] >= mm["revenue_value"] - 1e-6   # revenue cost
    assert mm["mean_satisfaction"] >= prof["mean_satisfaction"] - 1e-6  # not a service cost


# ------------------------------------------------------ grounded segments / tariffs
def test_segments_derivation_monotone_and_normalized():
    from chargax.equity import segments as S
    m = S.derive_price_multipliers()
    assert m[0] < m[1] < m[2]           # budget < mid < premium
    assert abs(m[1] - 1.0) < 1e-6       # median normalized to 1.0


def test_flat_tariff_does_not_increase_gap_and_vw_starves_low_payer():
    """Value-weighting starves the low-paying group; income-neutral pricing never
    increases the profit-optimal gap (it removes the price-driven component)."""
    from chargax.equity import tariffs as T
    base = (1.0, 3.0)
    vw = _solve("profit", price=base)
    flat = _solve("profit", price=T.flat(base))
    assert vw["group_means"][0] <= vw["group_means"][1] + 1e-9  # low payer starved
    assert flat["group_disparity"] <= vw["group_disparity"] + 1e-6


# ------------------------------------------------- margin-greedy online baseline
def _scarce_env(grid_kw=20.0, n_evses=4, n_groups=3, price=(0.6, 1.0, 1.5)):
    """A power-scarce three-tier site, small enough to roll out inside a test.

    Four EVSEs of two connectors each can request far more than the grid connection,
    so the policy is forced to ration and the tier ordering actually bites.
    """
    from chargax import EVSE, StationBattery, StationSplitter

    station = ChargingStation(
        max_kw_throughput=grid_kw,
        efficiency=1.0,
        connections=[
            StationSplitter(
                max_kw_throughput=grid_kw * 10,
                efficiency=0.99,
                connections=[EVSE(num_chargers=2, voltage=400, max_current=55, efficiency=0.99)
                             for _ in range(n_evses)]
                + [StationBattery(capacity_kw=150.0, max_kw_throughput=40.0, efficiency=0.97)],
            )
        ],
    )
    return EquiChargax(
        station=station, welfare_alpha=0.0, lam=1.0, welfare_outer="rawlsian",
        n_groups=n_groups, group_probs=(1 / n_groups,) * n_groups, price_by_group=price,
        num_discretization_levels=4, allow_discharging=False,
    )


def _greedy_rollout(env, key, steps=None):
    """Roll ``margin_greedy_policy`` out and return the per-tier inclusive means."""
    from chargax.equity import baselines as B

    steps = steps or env.max_episode_steps
    obs, state = env.reset_env(key)

    def step(carry, _):
        k, st, ob = carry
        k, ka, ks = jax.random.split(k, 3)
        a = B.margin_greedy_policy(ka, env, st, ob)
        ts, ns = env.step_env(ks, st, a)
        return (k, ns, ts.observation), (ts.info["realized_util"], ts.info["realized_mask"],
                                         ts.info["realized_group"], ts.info["rejected_per_group"])
    _, (util, mask, grp, rej) = jax.lax.scan(step, (key, state, obs), None, length=steps)
    util, mask, grp = np.array(util), np.array(mask), np.array(grp)
    rejected = np.array(rej).reshape(-1, env.n_groups).sum(axis=0)
    served, served_g = util[mask > 0], grp[mask > 0]
    means = []
    for g in range(env.n_groups):
        s = served[served_g == g]
        denom = len(s) + rejected[g]
        means.append(float(s.sum() / denom) if denom > 0 else np.nan)
    return np.array(means)


def test_margin_levels_ranks_descending_and_groups_ties():
    """Distinct margins rank strictly, equal margins share a level."""
    from chargax.equity.baselines import _margin_levels

    env = _scarce_env(price=(0.6, 1.0, 1.5))
    assert _margin_levels(env) == [(1.5, [2]), (1.0, [1]), (0.6, [0])]
    # A premium cap that compresses the top two tiers must make them peers, not rank
    # one above the other on a floating-point tie.
    capped = _scarce_env(price=(0.6, 1.0, 1.0))
    assert _margin_levels(capped) == [(1.0, [1, 2]), (0.6, [0])]
    flat = _scarce_env(price=(1.0, 1.0, 1.0))
    assert _margin_levels(flat) == [(1.0, [0, 1, 2])]


def test_margin_greedy_action_is_valid_and_jittable():
    from chargax.equity import baselines as B

    env = _scarce_env()
    key = jax.random.PRNGKey(0)
    obs, state = env.reset_env(key)
    action = jax.jit(lambda k, s, o: B.margin_greedy_policy(k, env, s, o))(key, state, obs)
    flat = jnp.concatenate([jnp.atleast_1d(a) for a in action["evses"]])
    assert flat.shape[0] == env.station.num_chargers
    assert int(flat.min()) >= 0 and int(flat.max()) <= env.num_discretization_levels


def test_margin_greedy_never_requests_more_than_the_grid_allows():
    """The whole point of the policy is that it rations, so it must stay in budget.

    If the request overshot, the station would renormalize every charger proportionally
    and the strict tier priority would be washed out. We check the delivered power the
    action asks for against the grid limit at every step of a realized day.
    """
    from chargax.equity import baselines as B

    env = _scarce_env()
    key = jax.random.PRNGKey(3)
    obs, state = env.reset_env(key)
    limit = float(env.station.max_kw_throughput)
    for _ in range(env.max_episode_steps):
        key, ka, ks = jax.random.split(key, 3)
        action = B.margin_greedy_policy(ka, env, state, obs)
        evses = state.grid.evses_flat
        levels = jnp.concatenate([jnp.atleast_1d(a) for a in action["evses"]]).astype(jnp.float32)
        current = jnp.clip(levels / env.num_discretization_levels * evses.max_current,
                           0.0, evses.car_max_current_intake)
        asked_kw = float(jnp.sum(current * evses.voltage) / 1000.0)
        battery_kw = float(jnp.sum(jnp.minimum(state.grid.batteries_flat.max_kw_throughput,
                                               state.grid.batteries_flat.battery_now
                                               / (env.minutes_per_timestep / 60.0))))
        assert asked_kw <= limit + battery_kw + 1e-3, (asked_kw, limit, battery_kw)
        ts, state = env.step_env(ks, state, action)
        obs = ts.observation


def test_margin_greedy_orders_satisfaction_by_tier_margin():
    """The behavioural claim: revenue-greedy rationing starves the low-margin tier.

    Under scarcity the day's per-tier satisfaction must come out ordered the same way
    the tier margins are, budget worst and premium best. This is the online counterpart
    of what the offline revenue optimum does, and it is why the policy is the right
    status-quo baseline to audit.
    """
    env = _scarce_env()
    means = _greedy_rollout(env, jax.random.PRNGKey(11))
    assert np.all(np.isfinite(means)), means
    assert means[0] <= means[1] + 1e-6, means   # budget no better than mid
    assert means[1] <= means[2] + 1e-6, means   # mid no better than premium
    assert means[2] - means[0] > 0.05, means    # and the gap is not a rounding artifact


def test_margin_greedy_gap_shrinks_when_the_grid_stops_binding():
    """The harm is a scarcity phenomenon, so it must fade as capacity grows."""
    scarce = _greedy_rollout(_scarce_env(grid_kw=20.0), jax.random.PRNGKey(11))
    ample = _greedy_rollout(_scarce_env(grid_kw=400.0), jax.random.PRNGKey(11))
    assert (scarce.max() - scarce.min()) > (ample.max() - ample.min())


# ------------------------------------------------ the chain identity (the theorem)
# These exercise the helpers the released scripts actually use, so a regression in
# experiments/verify_theory.py fails here rather than only in a long audit run.
def _chain_helpers():
    from experiments.verify_theory import (_base, build_face, class_energy_range, g_hat,
                                           revenue_optimum, top_sets)
    return _base, build_face, class_energy_range, g_hat, revenue_optimum, top_sets


def _day(margins, grid_kw=30.0, mpt=5, horizon=8):
    """One toy scarce day, plus the LP skeleton the chain results are read off."""
    _base, _bf, _cer, _gh, _ro, _ts = _chain_helpers()
    cust = _toy_customers()
    return cust, _base(cust, grid_kw, mpt, horizon)


def test_top_sets_are_nested_and_strictly_descending():
    _b, _bf, _cer, _gh, _ro, top_sets = _chain_helpers()
    chain = top_sets((0.6, 1.0, 1.5))
    assert [lvl for lvl, _c, _V in chain] == [1.5, 1.0, 0.6]
    assert [V for _l, _c, V in chain] == [(2,), (1, 2), (0, 1, 2)]
    # Tiers priced the same share a margin class rather than being ranked on a tie.
    capped = top_sets((0.6, 1.0, 1.0))
    assert [c for _l, c, _V in capped] == [(1, 2), (0,)]


def test_chain_identity_holds_at_the_revenue_optimum():
    """E(V_l) = g_hat(V_l) for every top set: the structure the audit rests on.

    The revenue optimum saturates the highest-margin class, then the two highest
    together, and so on, so each class receives only what the classes above it left.
    """
    _b, _bf, _cer, g_hat, revenue_optimum, top_sets = _chain_helpers()
    margins = (1.0, 3.0)
    cust, base = _day(margins)
    _x, _rev, per_tier = revenue_optimum(cust, base, margins)
    for _lvl, _cls, V in top_sets(margins):
        assert per_tier[list(V)].sum() == pytest.approx(g_hat(cust, base, V), abs=1e-6)


def test_class_aggregate_identity():
    """E(C_l) = g_hat(V_l) - g_hat(V_{l+1}): every optimum gives each class the same total."""
    _b, _bf, _cer, g_hat, revenue_optimum, top_sets = _chain_helpers()
    margins = (1.0, 3.0)
    cust, base = _day(margins)
    _x, _rev, per_tier = revenue_optimum(cust, base, margins)
    chain = top_sets(margins)
    bounds = [g_hat(cust, base, V) for _l, _c, V in chain]
    for l, (_lvl, cls, _V) in enumerate(chain):
        inner = bounds[l - 1] if l > 0 else 0.0
        assert per_tier[list(cls)].sum() == pytest.approx(bounds[l] - inner, abs=1e-6)


def test_chain_identity_telescopes_to_the_optimal_revenue():
    """revenue = sum_l (v_l - v_{l+1}) g_hat(V_l), with v_{L+1} = 0.

    This is what makes the chain equalities *characterize* the optimal face rather than
    merely hold on it, and it catches a chain assembled in the wrong order.
    """
    _b, _bf, _cer, g_hat, revenue_optimum, top_sets = _chain_helpers()
    margins = (1.0, 3.0)
    cust, base = _day(margins)
    _x, revenue, _per_tier = revenue_optimum(cust, base, margins)
    chain = top_sets(margins)
    levels = [lvl for lvl, _c, _V in chain] + [0.0]
    predicted = sum((levels[l] - levels[l + 1]) * g_hat(cust, base, chain[l][2])
                    for l in range(len(chain)))
    assert revenue == pytest.approx(predicted, rel=1e-9, abs=1e-6)


def test_chain_identity_survives_a_price_cap_and_a_subsidy():
    """The identity is a property of the program, not of one tariff, so it must hold
    under the altered tariffs the Levers corollary re-solves under."""
    _b, _bf, _cer, g_hat, revenue_optimum, top_sets = _chain_helpers()
    for margins in ((1.0, 1.0), (2.0, 1.0), (1.0, 3.0)):
        cust, base = _day(margins)
        per_tier = revenue_optimum(cust, base, margins)[2]
        for _lvl, _cls, V in top_sets(margins):
            assert per_tier[list(V)].sum() == pytest.approx(g_hat(cust, base, V), abs=1e-6), margins


def test_margin_magnitudes_do_not_move_the_optimum():
    """The optimal set depends on the tariff only through the ordering of the classes.

    Two tariffs with the same order must deliver identical per-tier aggregates, which is
    the half of the theorem that makes the elasticity sweep a verification rather than a
    sensitivity analysis.
    """
    _b, _bf, _cer, _gh, revenue_optimum, _ts = _chain_helpers()
    cust, base = _day((1.0, 3.0))
    a = revenue_optimum(cust, base, (1.0, 3.0))[2]
    b = revenue_optimum(cust, base, (1.0, 1.2))[2]
    assert np.allclose(a, b, atol=1e-6), (a, b)


def test_lowest_margin_class_is_the_residual_claimant():
    """Corollary (Levers), base clause: the lowest-paying class gets what is left."""
    _b, _bf, _cer, g_hat, revenue_optimum, _ts = _chain_helpers()
    margins = (1.0, 3.0)
    cust, base = _day(margins)
    per_tier = revenue_optimum(cust, base, margins)[2]
    expected = g_hat(cust, base, (0, 1)) - g_hat(cust, base, (1,))
    assert per_tier[0] == pytest.approx(expected, abs=1e-6)


def test_equal_margins_collapse_the_chain_and_free_every_tier():
    """Corollary (Levers)(iii): with no ordering there is no residual claimant, and the
    optimal face leaves every tier's aggregate free to move."""
    _b, build_face, class_energy_range, g_hat, revenue_optimum, top_sets = _chain_helpers()
    margins = (1.0, 1.0)
    cust, base = _day(margins)
    assert len(top_sets(margins)) == 1
    per_tier = revenue_optimum(cust, base, margins)[2]
    assert per_tier.sum() == pytest.approx(g_hat(cust, base, (0, 1)), abs=1e-6)
    face = build_face(cust, base, margins)
    for g in (0, 1):
        lo, hi = class_energy_range(cust, base, face, (g,))
        assert hi - lo > 1e-6, (g, lo, hi)


def test_ordered_margins_pin_the_lowest_class_on_the_face():
    """The mirror image: with a strict ordering the lowest class is pinned, so its
    shortfall is a property of optimality and not of the solver's tie-break.

    "Pinned" is checked against the face's own numerical slack. ``build_face`` writes
    each chain equality as a pair of inequalities with a relative tolerance, so the
    class aggregate can still wander by that much and no less. What carries the claim is
    the contrast with the unordered tariff, where the same class is free to take
    anything from nothing to the whole total.
    """
    _b, build_face, class_energy_range, g_hat, _ro, _ts = _chain_helpers()
    cust, base = _day((1.0, 3.0))
    total = g_hat(cust, base, (0, 1))

    ordered = build_face(cust, base, (1.0, 3.0))
    lo, hi = class_energy_range(cust, base, ordered, (0,))
    assert hi - lo <= 1e-4 * total, (lo, hi)

    flat = build_face(cust, base, (1.0, 1.0))
    flo, fhi = class_energy_range(cust, base, flat, (0,))
    assert (fhi - flo) > 0.5 * total          # unordered: essentially unconstrained
    assert (hi - lo) < 1e-3 * (fhi - flo)     # ordered: pinned by comparison


def test_lowest_margin_tier_is_the_one_served_out_of_the_residual():
    """The same statement read in satisfaction terms: subsidize past parity and the harm
    moves to whichever tier is now strictly lowest."""
    starved = _solve("profit", price=(1.0, 3.0))["group_means"]
    assert np.argmin(starved) == 0            # budget pays least, budget is starved
    flipped = _solve("profit", price=(3.0, 1.0))["group_means"]
    assert np.argmin(flipped) == 1            # subsidize past parity, the harm moves


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
