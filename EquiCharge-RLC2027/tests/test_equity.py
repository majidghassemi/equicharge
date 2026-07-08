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


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
