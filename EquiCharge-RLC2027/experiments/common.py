r"""Shared infrastructure for the welfare-optimal EV-charging experiments.

Provides station layouts (a power-scarce regime where the *binding* constraint is
grid power shared among many simultaneously-present cars -- the setting in which
distributive choices actually matter -- and an abundant regime for contrast),
environment factories, a unified policy evaluator that collects per-customer
satisfaction for fairness metrics, a thin wrapper to use a trained PPO agent as a
policy, and a ``gamma = 1`` PPO trainer.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from chargax import ChargingStation, EVSE, StationBattery, StationSplitter
from chargax.equity import EquiChargax, metrics as M
from chargax.equity import baselines as B


# --------------------------------------------------------------------- stations
def scarcity_station(grid_kw: float = 60.0, n_evses: int = 16) -> ChargingStation:
    """Many AC chargers behind a *constrained* grid connection.

    ``n_evses`` EVSEs x 2 connectors @ ~22 kW each can request far more than
    ``grid_kw``, so at peak the agent must *ration* scarce power across the cars
    that are present -- this is the fairness lever. An on-site battery allows
    time-shifting power across the day.
    """
    return ChargingStation(
        max_kw_throughput=grid_kw,
        efficiency=1.0,
        connections=[
            StationSplitter(
                max_kw_throughput=grid_kw * 10,  # internal wiring not the bottleneck
                efficiency=0.99,
                connections=[
                    EVSE(num_chargers=2, voltage=400, max_current=55, efficiency=0.99)
                    for _ in range(n_evses)
                ]
                + [
                    StationBattery(
                        capacity_kw=150.0, max_kw_throughput=40.0, efficiency=0.97
                    )
                ],
            )
        ],
    )


def abundant_station(n_evses: int = 16) -> ChargingStation:
    """Same chargers, but the grid is large enough that power is *not* scarce."""
    return scarcity_station(grid_kw=600.0, n_evses=n_evses)


# ----------------------------------------------------------------- env factories
# Real-data calibration (see docs/DATA.md). The environment is driven by Chargax's
# empirically-derived Dutch EV datasets: time-of-day arrival profiles
# (``car_arrival_percentages_*.csv``), per-vehicle energy demand and connection-time
# distributions (``car_energy_demand.csv``, ``car_connection_times.csv``), the EU
# vehicle fleet mix (``car_frequency_and_profiles.csv``), and real 2023 Netherlands
# day-ahead electricity prices (``electricity_prices_kwh_2023_NL.csv``). Only the
# ability-to-pay segmentation is added on top (grounded in income/energy-burden
# data; see :mod:`chargax.equity.segments`). ``ACN_DATA`` is provided as an
# alternative calibration hook for the public Caltech ACN-Data (see
# ``chargax.equity.data_calibration``).
DEFAULT_DATA = {
    "car_profile": "eu",
    "user_profile": "residential",  # longer dwell, more time-sensitive -> cars coexist
    "average_cars_per_day": 30,  # keep chargers from being the binding constraint
    "grid_price_dataset": "2023_NL",  # real day-ahead NL prices
}


def make_env(
    station=None,
    *,
    alpha: float = 0.0,
    lam: float = 1.0,
    outer: str = "utilitarian",
    n_groups: int = 1,
    group_edges: tuple = (),
    group_probs: tuple = (),
    price_by_group: tuple = (),
    augment_obs: bool = True,
    count_rejections: bool = True,
    profit_scale: float = 200.0,  # ~= reference max-charge daily profit; see EquiChargax.profit_scale
    num_discretization_levels: int = 4,
    allow_discharging: bool = True,
    dense_welfare: bool = True,
    shaping_mode: str = "welfare",
    throughput_scale: float = 1.0,
    sat_ref_kwh: float = 20.0,
    data_kwargs: dict | None = None,
    get_num_cars_arriving=None,
    get_new_cars_arriving=None,
) -> EquiChargax:
    """``get_*_cars_arriving`` override the demand stream entirely (used by the
    ACN-Data calibration). They must be passed as top-level env fields: Chargax only
    reads ``car_profile``/``user_profile``/``average_cars_per_day`` out of
    ``default_data_kwargs``, so callables placed in ``data_kwargs`` are ignored."""
    station = station if station is not None else scarcity_station()
    return EquiChargax(
        station=station,
        get_num_cars_arriving=get_num_cars_arriving,
        get_new_cars_arriving=get_new_cars_arriving,
        welfare_alpha=alpha,
        lam=lam,
        welfare_outer=outer,
        n_groups=n_groups,
        group_edges=group_edges,
        group_probs=group_probs,
        price_by_group=price_by_group,
        augment_obs=augment_obs,
        count_rejections=count_rejections,
        profit_scale=profit_scale,
        num_discretization_levels=num_discretization_levels,
        allow_discharging=allow_discharging,
        dense_welfare=dense_welfare,
        shaping_mode=shaping_mode,
        throughput_scale=throughput_scale,
        sat_ref_kwh=sat_ref_kwh,
        default_data_kwargs=data_kwargs or DEFAULT_DATA,
    )


# --------------------------------------------------------- ACN-Data (US) calibration
ACN_ENV_PATH = "EQUICHARGE_ACN_JSON"


def acn_env_or_none(grid_kw=30.0, n_evses=8, num_disc=4, *, path=None, verbose=True,
                    **extra):
    """Reference-geometry env driven by real Caltech ACN-Data sessions, or ``None``.

    Returns ``(env, provenance)`` when ``EQUICHARGE_ACN_JSON`` points at a usable
    dump, else ``None`` so callers run their bundled-data configs unchanged. A
    *present but unusable* dump raises rather than falling back: silently reverting
    to the Dutch data would publish a row labelled as a US session-level calibration
    that contains no ACN data at all.
    """
    import os as _os

    p = path or _os.environ.get(ACN_ENV_PATH)
    if not p:
        if verbose:
            print(f"[acn] {ACN_ENV_PATH} not set -> ACN-Data (US) config skipped.")
        return None
    from chargax.equity import segments as _SEG
    from chargax.equity.data_calibration import acn_scenario

    station = scarcity_station(grid_kw=grid_kw, n_evses=n_evses)
    num_fn, new_fn, prov = acn_scenario(station, p)
    # Same ability-to-pay segmentation as every other site; without it the env has one
    # group and every per-tier quantity collapses to a single column.
    group_kw = dict(n_groups=3, group_probs=_SEG.GROUP_PROBS,
                    price_by_group=_SEG.PRICE_BY_GROUP)
    env = make_env(station=station, num_discretization_levels=num_disc,
                   allow_discharging=False, alpha=0.0, lam=1.0, outer="rawlsian",
                   get_num_cars_arriving=num_fn, get_new_cars_arriving=new_fn,
                   data_kwargs={"car_profile": "us", "grid_price_dataset": "2023_NL"},
                   **{**group_kw, **extra})
    # The failure this guards against is silent: Chargax builds its own Dutch loaders
    # in __post_init__ whenever these fields are left None.
    assert env.get_num_cars_arriving is num_fn and env.get_new_cars_arriving is new_fn, \
        "ACN callables did not reach the env; it would be running on bundled Dutch data."
    if verbose:
        print(f"[acn] {prov['sessions_used']}/{prov['sessions_in_dump']} sessions, "
              f"{prov['date_range_used'][0]}..{prov['date_range_used'][1]}, "
              f"{prov['mean_sessions_per_workday']:.1f} sessions/workday, "
              f"median dwell {prov['median_dwell_minutes']:.0f} min, "
              f"median demand {prov['median_energy_kwh']:.1f} kWh")
    return env, prov


# ------------------------------------------------------------------- evaluation
def make_trained_policy(agent):
    """Wrap a trained PPO agent as a ``policy(key, env, state, obs)`` callable."""
    from jaxnasium.algorithms import PPO

    state = agent.state

    def policy(key, env, st, obs):
        return PPO.get_action(key, state, obs, deterministic=True)

    return policy


def _rollout(env, policy, key):
    obs, state = env.reset_env(key)

    def step(carry, _):
        k, st, ob = carry
        k, ka, ks = jax.random.split(k, 3)
        a = policy(ka, env, st, ob)
        ts, ns = env.step_env(ks, st, a)
        out = (
            ts.info["welfare"],
            ts.info["profit"],
            ts.info["realized_util"],
            ts.info["realized_mask"],
            ts.info["realized_group"],
            ts.info["rejected_per_group"],
        )
        return (k, ns, ts.observation), out

    _, outs = jax.lax.scan(step, (key, state, obs), None, length=env.max_episode_steps)
    return outs


def evaluate_policy(env, policy, key, n_episodes: int = 32) -> dict:
    """Roll out ``policy`` for ``n_episodes`` and compute welfare/fairness metrics.

    Returns per-episode profit/welfare (for CIs) and the pooled per-customer
    satisfaction distribution (served at their realized utility, rejected at 0)
    used for the inequality metrics.
    """
    keys = jax.random.split(key, n_episodes)
    welf, prof, ru, rm, rg, rej = jax.vmap(lambda k: _rollout(env, policy, k))(keys)

    welf = np.array(welf)[:, -1]
    prof = np.array(prof)[:, -1]
    ru, rm, rg, rej = map(np.array, (ru, rm, rg, rej))

    # Pooled per-customer satisfaction (served) + rejected-at-zero.
    served_sat = ru[rm > 0]
    served_grp = rg[rm > 0]
    n_rejected = int(round(rej.sum()))
    inclusive_sat = np.concatenate([served_sat, np.zeros(n_rejected)])

    # Per-group inclusive means (served + rejected-at-zero), if grouped.
    group_means = None
    if env.n_groups > 1:
        rej_per_group = rej.reshape(-1, env.n_groups).sum(axis=0)
        gm = []
        for g in range(env.n_groups):
            s = served_sat[served_grp == g]
            denom = len(s) + rej_per_group[g]
            gm.append(float(s.sum() / denom) if denom > 0 else 0.0)
        group_means = np.array(gm)

    metrics = {
        "welfare_mean": float(welf.mean()),
        "welfare_sem": float(welf.std() / np.sqrt(n_episodes)),
        "profit_mean": float(prof.mean()),
        "profit_sem": float(prof.std() / np.sqrt(n_episodes)),
        "served_mean_sat": float(served_sat.mean()) if len(served_sat) else 0.0,
        "n_served": len(served_sat),
        "n_rejected": n_rejected,
        "rejection_rate": n_rejected / max(len(served_sat) + n_rejected, 1),
        # inequality over the inclusive (served + rejected) population:
        "inclusive_mean_sat": float(inclusive_sat.mean()) if len(inclusive_sat) else 0.0,
        "min_sat": float(M.worst_quantile_mean(inclusive_sat, 0.05)) if len(inclusive_sat) else 0.0,
        "worst10_sat": float(M.worst_quantile_mean(inclusive_sat, 0.1)) if len(inclusive_sat) else 0.0,
        "gini": float(M.gini(inclusive_sat)) if len(inclusive_sat) else 0.0,
        "atkinson": float(M.atkinson(inclusive_sat, 1.0)) if len(inclusive_sat) else 0.0,
    }
    if group_means is not None:
        metrics["group_means"] = [float(x) for x in group_means]
        metrics["group_disparity"] = float(group_means.max() - group_means.min())
    return metrics


# ------------------------------------------------------ multi-seed dispersion
def aggregate_seeds(per_seed: list, keys) -> dict:
    """Aggregate a list of per-seed metric dicts into median/IQR + mean/std.

    Reports the *seed* dispersion (median, inter-quartile range, mean, std) for each
    metric so learned results are interpretable across training seeds -- distinct
    from the within-policy evaluation-day variance (the ``*_sem`` fields), which we
    keep separate to avoid conflating the two sources of noise.
    """
    out = {"n_seeds": len(per_seed)}
    for k in keys:
        vals = np.array([s[k] for s in per_seed if k in s], dtype=float)
        if len(vals) == 0:
            continue
        out[k] = float(np.median(vals))
        out[k + "_median"] = float(np.median(vals))
        out[k + "_iqr"] = float(np.subtract(*np.percentile(vals, [75, 25])))
        out[k + "_mean"] = float(np.mean(vals))
        out[k + "_std"] = float(np.std(vals))
        out[k + "_min"] = float(np.min(vals))
        out[k + "_max"] = float(np.max(vals))
    if "group_means" in per_seed[0]:
        gm = np.array([s["group_means"] for s in per_seed if "group_means" in s])
        mean_gm = gm.mean(axis=0)
        out["group_means"] = mean_gm.tolist()
        out["group_means_std"] = gm.std(axis=0).tolist()
        # Report disparity as max-min of the seed-averaged tier means, the IDENTICAL
        # estimator used for the baselines and the oracle, so the disparity number
        # equals max-min of the displayed group_means row (A3 consistency fix).
        disp = float(mean_gm.max() - mean_gm.min())
        for k in ("group_disparity", "group_disparity_median"):
            out[k] = disp
        out["group_disparity_iqr"] = 0.0
    return out


# --------------------------------------------------------------------- training
def train_ppo(
    env_for_training,
    *,
    total_timesteps: int = 300_000,
    num_envs: int = 16,
    num_steps: int = 288,
    seed: int = 0,
    gamma: float = 1.0,  # undiscounted: the welfare-telescoping objective is exact only at gamma=1
    learning_rate: float = 3e-4,
    normalize_rewards: bool = True,
    ent_coef: float = 0.0,
    anneal_ent_coef: bool = False,
    log: bool = False,
):
    """Train a PPO agent (gamma=1 by default) and return the agent.

    ``ent_coef`` / ``anneal_ent_coef`` control exploration. The aggressive rationing
    the oracle proves is optimal is a low-entropy behavior PPO may never sample under
    the default zero entropy bonus, so raising ``ent_coef`` (optionally annealed) is
    the exploration lever for the equity-control experiment.
    """
    import jaxnasium as jym
    from jaxnasium.algorithms import PPO

    wrapped = jym.LogWrapper(env_for_training)
    agent = PPO(
        num_steps=num_steps,
        num_envs=num_envs,
        total_timesteps=total_timesteps,
        gamma=gamma,
        learning_rate=learning_rate,
        anneal_learning_rate=True,
        ent_coef=ent_coef,
        anneal_ent_coef=anneal_ent_coef,
        normalize_observations=True,
        normalize_rewards=normalize_rewards,
        log_function="simple" if log else None,
        log_interval=0.25,
    )
    agent = agent.train(jax.random.PRNGKey(seed), wrapped)
    return agent
