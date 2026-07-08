r"""EquiChargax: welfare-optimal RL over an *endogenous, streaming* population.

``EquiChargax`` subclasses :class:`chargax.Chargax` to turn the EV-charging
control problem into an **online sequential fair-allocation** problem. The agent
still sets per-charger currents under the same grid/battery/charge-curve
dynamics, but instead of (only) maximizing profit it optimizes a non-linear
social welfare function over the *charging satisfaction* delivered to every
customer in the realized daily arrival stream.

What makes this setting distinct from prior fair-RL (e.g. Siddique et al. 2020,
who optimize a welfare over a *fixed* vector of D users):

1. **Endogenous, dynamic population.** Customers arrive and depart stochastically;
   the agent partially controls *when* a customer leaves (charge-sensitive cars
   leave once satisfied; V2G can extend dwell) and *whether* a customer is served
   at all (rejections).
2. **Departure-realized, non-additive utility.** A customer's utility -- the
   fraction of desired energy delivered -- is only known at departure, and the
   welfare is a non-linear (concave) function of the whole distribution.
3. **Bounded sufficient-statistic state.** Because every welfare here has a
   generalized-mean form, the policy only needs to remember, per customer
   segment, the realized count and accumulated kernel ``(N_g, S_g)`` -- a fixed,
   small statistic regardless of how many cars stream through (see
   :mod:`chargax.equity.welfare`).

Design choices that make the fairness construct well-posed and hack-resistant:

* **Utility over the true arrival stream.** Utility is the fraction of *desired*
  energy delivered, evaluated for **every** arrival: served-and-satisfied,
  served-but-undercharged (left at deadline), still-charging at the end of the
  day (flushed at horizon), and **rejected** customers (utility 0). This prevents
  "looking fair" by completing only easy customers or by turning needy customers
  away. Rejection requires sampling the would-be customer's attributes before
  rejecting (``sample-then-reject``) so its demand is observable.
* **Individual fairness by default** (``n_groups=1``) -- the welfare is over the
  per-customer satisfaction vector, sidestepping contestable demographic proxies.
  An optional **structural-equity** mode groups customers by an *exogenous*,
  policy-independent attribute (battery-capacity tier) and applies an egalitarian
  (Rawlsian) operator across groups.
* **Undiscounted, finite-horizon objective** (one day). The telescoping welfare
  reward is exact only at ``gamma = 1``; we therefore optimize the undiscounted
  return over the bounded daily horizon.
"""

from __future__ import annotations

from typing import Dict, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax_datetime as jdt
from jaxtyping import Array, PRNGKeyArray

from chargax.chargax import Chargax, EnvState
from chargax.equity import welfare as W


class FairEnvState(EnvState):
    """Environment state augmented with the bounded welfare sufficient statistic.

    ``group_count`` and ``group_kernel_sum`` are the per-segment ``(N_g, S_g)``
    statistic accumulated over the day. The ``last_*`` fields expose the customers
    that realized their utility *this step* (for logging/evaluation only).
    """

    # Bounded sufficient statistic for the social welfare function.
    group_count: Array = eqx.field(default=None)  # (G,) realized customers/group
    group_kernel_sum: Array = eqx.field(default=None)  # (G,) accumulated phi(u)
    group_util_sum: Array = eqx.field(default=None)  # (G,) accumulated raw u (eval)
    cf_sum: Array = eqx.field(default=None)  # scalar: accumulated content-free saturating
    #   quantity clip(delivered/sat_ref,0,1) over departed drivers (sat_throughput control).

    # Per-charger ability-to-pay segment of the connected car (assigned at
    # arrival, independent of battery capacity). Persists while connected.
    charger_group: Array = eqx.field(default=None)  # (num_chargers,) int

    # Transient per-step logging of realized utilities (overwritten each step).
    last_realized_util: Array = eqx.field(default=None)  # (num_chargers,)
    last_realized_mask: Array = eqx.field(default=None)  # (num_chargers,) bool
    last_realized_group: Array = eqx.field(default=None)  # (num_chargers,) int
    last_rejected_per_group: Array = eqx.field(default=None)  # (G,)
    last_arrival_mask: Array = eqx.field(default=None)  # (num_chargers,) bool


class EquiChargax(Chargax):
    """Welfare-optimal EV charging environment (see module docstring)."""

    # --- Welfare configuration ---
    welfare_alpha: float = 0.0
    """Inner inequality aversion (individual level). 0=utilitarian, 1=Nash,
    larger=more egalitarian (-> Rawlsian/leximin as alpha grows)."""

    welfare_outer: str = "utilitarian"
    """Cross-group operator: "utilitarian" (pooled individual fairness) or
    "rawlsian" (egalitarian soft-min across exogenous segments)."""

    welfare_beta: float = 20.0
    """Soft-min temperature for the Rawlsian outer operator."""

    lam: float = 1.0
    r"""Equity weight :math:`\lambda \in [0,1]`. Reward is
    ``(1-lam)*profit_term + lam*welfare_term``; sweeping ``lam`` traces the
    profit-equity Pareto front. ``lam=0`` recovers a (scaled) profit maximizer."""

    profit_scale: float = 200.0
    r"""Reward normalization constant for the profit term (EUR/day).

    The blended reward is ``(1-lam)*Dprofit/profit_scale + lam*Dwelfare``. The
    welfare term is a change in a bounded ``[0,1]`` welfare, so the *episode* welfare
    return is O(1). For ``lam`` to be a meaningful, comparable equity weight across
    configs, the *episode* profit return ``sum_t Dprofit_t / profit_scale =
    profit_day / profit_scale`` must also be O(1). We therefore set ``profit_scale``
    to the approximate profit a profit-blind max-charge policy earns per day on the
    reference scarce site (~200 EUR/day; see ``experiments/results/RESULTS.md``), so
    both terms live on a common [0,1]-ish footing and a given ``lam`` denotes the
    same efficiency-equity mix regardless of the welfare operator. This value is
    reported in the experiment configuration; changing the site changes it and it is
    recomputed as the reference max-charge daily profit."""

    welfare_scale: float = 1.0
    """Optional scale on the welfare term."""

    # --- Segmentation (ability-to-pay) ---
    n_groups: int = 1
    """Number of customer segments. 1 == individual fairness (no segments)."""

    group_probs: tuple = ()
    """Arrival probability of each segment (``len == n_groups``); default uniform.
    Each arriving car is independently assigned a segment, so segment membership
    is *exogenous and independent of battery capacity* (avoiding the
    ease-of-service confound of capacity-based grouping)."""

    group_edges: tuple = ()
    """(Optional, unused by the ability-to-pay segmentation.) Capacity tier edges
    kept for the alternative capacity-based :meth:`_group_of` structural analysis."""

    price_by_group: tuple = ()
    """Per-segment price multiplier on ``elec_customer_sell_price`` (ability to
    pay). Empty == flat pricing for all customers. When set (``len == n_groups``),
    revenue is value-weighted, so a profit maximizer is *incentivized* to favour
    high-paying segments -- creating the genuine efficiency-equity tension that
    flat pricing lacks (this is the ability-to-pay axis where real EV-charging
    inequity arises). The welfare objective remains over (price-agnostic)
    charging satisfaction, so equalizing it costs profit."""

    # --- Ablation switches ---
    count_rejections: bool = True
    """If True, rejected customers enter the welfare at utility 0 (over the true
    arrival stream). Ablating this exposes the reject-to-look-fair loophole."""

    augment_obs: bool = True
    """If True, append the bounded welfare statistic + present per-group demand to
    the observation (the state augmentation that makes the objective Markov)."""

    dense_welfare: bool = True
    """If True, the welfare reward is the increment of a *provisional* welfare
    potential that counts currently-connected cars at their current satisfaction
    (in addition to departed/rejected customers). This telescopes to the *same*
    final-welfare objective (connected cars are flushed at the horizon) but yields
    a dense per-step gradient -- charging a behind-schedule car raises the
    potential immediately -- which is far more learnable than the sparse,
    departure-only signal. Set False for the strict departure-only reward."""

    shaping_mode: str = "welfare"
    """Which dense potential the ``lam`` term uses, for the mechanism ablations.

    * ``"welfare"`` (default) -- the Rawlsian welfare potential (concave/saturating AND
      carries per-tier + per-driver satisfaction content).
    * ``"throughput"`` -- a **monotone content-free** control: cumulative delivered energy
      (grid utilization). Removes BOTH the saturating shape and the content, so it only
      shows that *a magnitude-matched monotone signal* does not reproduce the gain.
    * ``"sat_throughput"`` -- a **saturating content-free** control that isolates *shape*
      from *content*: per-connector delivered energy as a fraction of a **constant**
      reference (``sat_ref_kwh``), clipped to ``[0,1]`` so it saturates exactly like
      satisfaction, but referencing no per-driver desired target and no tier. If this
      recovers the gain, the saturating shape (not the welfare content) is the mechanism.

    All controls are magnitude-matched to the welfare potential via ``throughput_scale``."""

    throughput_scale: float = 1.0
    """Magnitude normalizer for the control potentials, set so their per-episode
    contribution matches the welfare potential's."""

    sat_ref_kwh: float = 20.0
    """Constant reference energy (kWh) for the ``"sat_throughput"`` control. A single
    population-level constant (not each driver's own desired target), so the potential
    saturates like satisfaction while carrying no per-driver content."""

    def __post_init__(self):
        super().__post_init__()
        if self.group_probs and len(self.group_probs) != self.n_groups:
            raise ValueError(
                f"len(group_probs) ({len(self.group_probs)}) must equal n_groups "
                f"({self.n_groups})."
            )
        if self.price_by_group and len(self.price_by_group) != self.n_groups:
            raise ValueError(
                f"len(price_by_group) ({len(self.price_by_group)}) must equal "
                f"n_groups ({self.n_groups})."
            )

    # ------------------------------------------------------------------ helpers
    def _group_of(self, capacity_kw: Array) -> Array:
        """Map per-charger battery capacity (kWh) to an exogenous segment index."""
        if self.n_groups == 1:
            return jnp.zeros_like(capacity_kw, dtype=jnp.int32)
        edges = jnp.asarray(self.group_edges)
        # group = number of edges the capacity exceeds.
        return jnp.sum(capacity_kw[..., None] > edges[None, ...], axis=-1).astype(
            jnp.int32
        )

    def _price_mult(self, groups: Array) -> Array:
        """Per-charger price multiplier from segment (1.0 everywhere if flat)."""
        if not self.price_by_group:
            return jnp.ones_like(groups, dtype=jnp.float32)
        mults = jnp.asarray(self.price_by_group, dtype=jnp.float32)
        return mults[groups]

    def _sample_segments(self, key, shape) -> Array:
        """Sample ability-to-pay segment(s) from the arrival mix."""
        if self.n_groups == 1:
            return jnp.zeros(shape, dtype=jnp.int32)
        probs = (
            jnp.asarray(self.group_probs)
            if self.group_probs
            else jnp.ones(self.n_groups) / self.n_groups
        )
        return jax.random.categorical(key, jnp.log(probs + 1e-12), shape=shape).astype(
            jnp.int32
        )

    @staticmethod
    def _per_customer_utility(ports) -> Array:
        """Fraction of desired energy delivered, per charger, in [0, 1].

        ``desired = desired% * capacity - arrival_battery`` (energy the customer
        wanted to add); ``delivered = now - arrival_battery``. Customers that
        arrived already at/above their target have utility 1.
        """
        desired = (
            ports.car_desired_battery_percentage * ports.car_battery_capacity_kw
            - ports.car_arrival_battery_kw
        )
        delivered = jnp.maximum(ports.car_battery_now_kw - ports.car_arrival_battery_kw, 0.0)
        sat = delivered / jnp.maximum(desired, W.EPS)
        sat = jnp.where(desired <= W.EPS, 1.0, sat)
        return jnp.clip(sat, 0.0, 1.0)

    def _scatter_groups(self, values: Array, groups: Array, mask: Array) -> Array:
        """Sum ``values`` (masked) into a length-``n_groups`` vector by ``groups``."""
        contrib = values * mask
        return jnp.stack(
            [jnp.sum(jnp.where(groups == g, contrib, 0.0)) for g in range(self.n_groups)]
        )

    # ------------------------------------------------------------------- reset
    def reset_env(self, key: PRNGKeyArray) -> Tuple[Dict[str, Array], FairEnvState]:
        random_day_of_year = jax.random.randint(key, (), 0, 365)
        year = self.simulation_starting_year
        random_day = jdt.to_datetime(f"{int(year)}-01-01") + jdt.Timedelta(
            days=random_day_of_year
        )
        G = self.n_groups
        nc = self.station.num_chargers
        state = FairEnvState(
            datetime=random_day,
            grid=self.station,
            elec_customer_sell_price=self.elec_customer_sell_price,
            group_count=jnp.zeros(G),
            group_kernel_sum=jnp.zeros(G),
            group_util_sum=jnp.zeros(G),
            cf_sum=jnp.zeros(()),
            last_realized_util=jnp.zeros(nc),
            last_realized_mask=jnp.zeros(nc, dtype=bool),
            last_realized_group=jnp.zeros(nc, dtype=jnp.int32),
            last_rejected_per_group=jnp.zeros(G),
            last_arrival_mask=jnp.zeros(nc, dtype=bool),
            charger_group=jnp.zeros(nc, dtype=jnp.int32),
        )
        return self.get_observation(state), state

    # ------------------------------------------------ departures / utility accrual
    def update_time_and_clear_cars(self, key, state, ports):
        """As base, but on the final timestep *flush* every still-connected car so
        its (possibly partial) utility is counted -- closing the
        complete-only-easy-customers loophole."""
        new_time_till_leave = ports.car_time_till_leave - self.minutes_per_timestep
        new_time_waited = ports.car_time_waited + self.minutes_per_timestep
        ports = ports.replace(
            car_time_till_leave=new_time_till_leave.astype(int),
            car_time_waited=new_time_waited,
        )

        cars_leaving = self.get_cars_departing(key, ports)
        cars_leaving = (cars_leaving * ports.charger_is_car_connected).astype(bool)

        # Horizon flush: count everyone still present at end of day.
        is_final = (state.timestep + 1) >= self.max_episode_steps
        cars_leaving = cars_leaving | (is_final & ports.charger_is_car_connected)

        state = self.set_customer_satisfaction_values(state, ports, cars_leaving)

        ports = ports.replace(
            charger_is_car_connected=ports.charger_is_car_connected * ~cars_leaving,
        )
        return state, ports

    def set_customer_satisfaction_values(self, state, ports, cars_leaving):
        # Keep the base bookkeeping (uncharged_kw, served_customers, ...).
        state = super().set_customer_satisfaction_values(state, ports, cars_leaving)

        mask = cars_leaving.astype(jnp.float32)
        util = self._per_customer_utility(ports)
        groups = state.charger_group  # ability-to-pay segment of each car
        kernel = W.inner_kernel(util, self.welfare_alpha)

        d_count = self._scatter_groups(jnp.ones_like(util), groups, mask)
        d_kernel = self._scatter_groups(kernel, groups, mask)
        d_util = self._scatter_groups(util, groups, mask)
        d_cf = jnp.sum(self._cf_quantity(ports) * mask)  # content-free saturating control

        return state._replace(
            group_count=state.group_count + d_count,
            group_kernel_sum=state.group_kernel_sum + d_kernel,
            group_util_sum=state.group_util_sum + d_util,
            cf_sum=state.cf_sum + d_cf,
            last_realized_util=util * mask,
            last_realized_mask=cars_leaving,
            last_realized_group=groups,
        )

    def _cf_quantity(self, ports) -> Array:
        """Content-free saturating per-connector quantity: delivered energy as a fraction
        of a CONSTANT reference (``sat_ref_kwh``), clipped to [0,1]. Saturates exactly like
        satisfaction but references no per-driver desired target and no tier."""
        delivered = jnp.maximum(ports.car_battery_now_kw - ports.car_arrival_battery_kw, 0.0)
        return jnp.clip(delivered / self.sat_ref_kwh, 0.0, 1.0)

    # ------------------------------------------------------ arrivals / rejections
    def add_new_cars(self, key, state, ports):
        """Sample-then-reject: draw candidate arrivals (each assigned an exogenous
        ability-to-pay segment), place as many as fit, and account the overflow as
        rejected customers with utility 0 attributed to their segment."""
        key1, key2, key3 = jax.random.split(key, 3)
        new_cars_amount = self.get_num_cars_arriving(key1, state)

        not_connected = jnp.logical_not(ports.charger_is_car_connected)
        sort_order = jnp.argsort(not_connected, descending=True)
        required = jnp.arange(self.station.num_chargers) < new_cars_amount
        required_in_order = jnp.zeros_like(required).at[sort_order].set(required)
        arrival_positions = required_in_order * not_connected

        incoming = self.get_new_cars_arriving(key2, state)
        incoming = incoming.replace(charger_is_car_connected=arrival_positions)
        merged = jax.tree.map(
            lambda new, curr: jax.lax.select(arrival_positions, new, curr),
            incoming,
            ports,
        )

        # Assign each arriving car an ability-to-pay segment; persist in state.
        sampled_groups = self._sample_segments(key3, (self.station.num_chargers,))
        new_charger_group = jnp.where(
            arrival_positions, sampled_groups, state.charger_group
        )

        rejected = jnp.maximum(new_cars_amount - not_connected.sum(), 0).astype(
            jnp.int32
        )

        # Attribute rejected customers to segments using the arrival mix
        # (exact for individual fairness; expected attribution for G > 1).
        if self.n_groups == 1:
            group_share = jnp.ones(1)
        elif self.group_probs:
            group_share = jnp.asarray(self.group_probs) / jnp.sum(jnp.asarray(self.group_probs))
        else:
            group_share = jnp.ones(self.n_groups) / self.n_groups
        rejected_per_group = rejected.astype(jnp.float32) * group_share

        if self.count_rejections:
            zero_kernel = W.inner_kernel(jnp.array(0.0), self.welfare_alpha)
            state = state._replace(
                group_count=state.group_count + rejected_per_group,
                group_kernel_sum=state.group_kernel_sum
                + rejected_per_group * zero_kernel,
                last_rejected_per_group=rejected_per_group,
            )
        else:
            state = state._replace(last_rejected_per_group=rejected_per_group)

        state = state._replace(
            rejected_customers=state.rejected_customers + rejected,
            last_arrival_mask=arrival_positions,
            charger_group=new_charger_group,
        )
        return state, merged

    # ------------------------------------------------ value-weighted revenue
    def charge_cars_and_update_batteries(self, state, charging_ports, batteries):
        """As the base method, but revenue is value-weighted by each customer's
        ability-to-pay segment (``price_by_group``). With flat pricing this is
        identical to the base; with heterogeneous prices a profit maximizer is
        incentivized to favour high-paying customers."""
        charging_now = self.kw_to_kw_this_timestep(charging_ports.power_output)
        previous_battery = charging_ports.car_battery_now_kw
        new_battery = (previous_battery + charging_now).clip(
            charging_ports.car_arrival_battery_kw,
            charging_ports.car_battery_capacity_kw,
        )
        real_charged_this_timestep = new_battery - previous_battery

        batteries_throughput_now_kw = self.kw_to_kw_this_timestep(batteries.throughput_now_kw)
        new_station_battery_level = jnp.clip(
            batteries.battery_now + batteries_throughput_now_kw, 0, batteries.capacity_kw
        )
        batteries = batteries.replace(battery_now=new_station_battery_level)

        # --- value-weighted revenue (the only change from the base method) ---
        per_car_sold = jnp.maximum(
            jnp.maximum(real_charged_this_timestep, 0.0)
            - charging_ports.car_discharged_this_session_kw,
            0.0,
        )
        price_mult = self._price_mult(state.charger_group)
        revenue = jnp.sum(per_car_sold * state.elec_customer_sell_price * price_mult)

        discharged_this_session = (
            charging_ports.car_discharged_this_session_kw + -real_charged_this_timestep
        ).clip(0)
        grid_draw_evses = jnp.where(
            real_charged_this_timestep >= 0,
            real_charged_this_timestep / charging_ports.cumulative_efficiency,
            real_charged_this_timestep * charging_ports.cumulative_efficiency,
        )
        grid_draw_batteries = jnp.where(
            batteries_throughput_now_kw >= 0,
            batteries_throughput_now_kw / batteries.cumulative_efficiency,
            batteries_throughput_now_kw * batteries.cumulative_efficiency,
        )
        total_grid_draw = grid_draw_evses.sum() + grid_draw_batteries.sum()
        elec_price = jax.lax.select(
            total_grid_draw >= 0, self.get_grid_buy_price(state), self.get_grid_sell_price(state)
        )
        profit = state.profit + revenue - total_grid_draw * elec_price
        charging_ports = charging_ports.replace(
            car_discharged_this_session_kw=discharged_this_session,
            car_battery_now_kw=new_battery,
        )
        total_charged = jnp.maximum(real_charged_this_timestep, 0.0).sum()
        total_discharged = jnp.maximum(-real_charged_this_timestep, 0.0).sum()
        return (
            state._replace(
                profit=profit,
                total_charged_kw=total_charged + state.total_charged_kw,
                total_discharged_kw=total_discharged + state.total_discharged_kw,
            ),
            charging_ports,
            batteries,
        )

    # ------------------------------------------------------------------- reward
    def _welfare(self, state: FairEnvState) -> Array:
        return W.welfare_from_statistics(
            state.group_count,
            state.group_kernel_sum,
            alpha=self.welfare_alpha,
            outer=self.welfare_outer,
            beta=self.welfare_beta,
        )

    def _provisional_stats(self, state: FairEnvState):
        """Departed/rejected statistic + currently-connected cars at current util."""
        ports = state.grid.evses_flat
        connected = ports.charger_is_car_connected.astype(jnp.float32)
        util = self._per_customer_utility(ports)
        groups = state.charger_group
        kernel = W.inner_kernel(util, self.welfare_alpha)
        d_count = self._scatter_groups(jnp.ones_like(util), groups, connected)
        d_kernel = self._scatter_groups(kernel, groups, connected)
        return state.group_count + d_count, state.group_kernel_sum + d_kernel

    def _welfare_potential(self, state: FairEnvState) -> Array:
        """Provisional welfare Phi(s): the welfare if all connected cars departed now.

        Telescopes to the realized final welfare (connected cars are flushed at the
        horizon, so Phi(terminal) == realized welfare and Phi(reset) == 0)."""
        count, kernel_sum = self._provisional_stats(state)
        return W.welfare_from_statistics(
            count, kernel_sum, alpha=self.welfare_alpha,
            outer=self.welfare_outer, beta=self.welfare_beta,
        )

    def _throughput_potential(self, state: FairEnvState) -> Array:
        """Monotone content-free dense potential (control): normalized cumulative delivered
        energy (grid utilization). Telescopes from 0 (reset) upward, carries NO tier /
        satisfaction information AND no saturation -- monotone in energy."""
        dt_h = self.minutes_per_timestep / 60.0
        max_energy = self.station.max_kw_throughput * self.max_episode_steps * dt_h
        return (state.total_charged_kw / (max_energy + W.EPS)) * self.throughput_scale

    def _saturating_potential(self, state: FairEnvState) -> Array:
        """Saturating content-free dense potential (control): accumulated clip(delivered/
        sat_ref, 0, 1) over departed drivers plus the same for currently-connected ones.
        Saturates like satisfaction (isolating the *shape*) but references no per-driver
        desired target and no tier (removing the *content*)."""
        ports = state.grid.evses_flat
        connected = ports.charger_is_car_connected.astype(jnp.float32)
        prov = jnp.sum(self._cf_quantity(ports) * connected)
        return (state.cf_sum + prov) * self.throughput_scale

    def get_reward(self, old_state: FairEnvState, new_state: FairEnvState) -> Array:
        profit_delta = new_state.profit - old_state.profit
        if self.shaping_mode == "throughput":
            welfare_delta = self._throughput_potential(new_state) - self._throughput_potential(old_state)
        elif self.shaping_mode == "sat_throughput":
            welfare_delta = self._saturating_potential(new_state) - self._saturating_potential(old_state)
        elif self.dense_welfare:
            welfare_delta = self._welfare_potential(new_state) - self._welfare_potential(old_state)
        else:
            welfare_delta = self._welfare(new_state) - self._welfare(old_state)
        profit_term = profit_delta / self.profit_scale
        welfare_term = welfare_delta * self.welfare_scale
        return (1.0 - self.lam) * profit_term + self.lam * welfare_term

    # -------------------------------------------------------------- observation
    def get_observation(self, state: FairEnvState) -> Dict[str, Array]:
        obs = super().get_observation(state)
        if not self.augment_obs:
            return obs

        safe_count = jnp.maximum(state.group_count, 1.0)
        group_score = W.equally_distributed_equivalent(
            state.group_kernel_sum / safe_count, self.welfare_alpha
        )
        total = jnp.maximum(jnp.sum(state.group_count), 1.0)
        group_frac = state.group_count / total

        # Present (unmet) demand per segment, normalized -- lets the policy see who
        # currently needs power and anticipate.
        ports = state.grid.evses_flat
        connected = ports.charger_is_car_connected.astype(jnp.float32)
        remaining = jnp.maximum(ports.car_battery_desired_remaining_kw, 0.0) * connected
        groups = state.charger_group
        active_demand = self._scatter_groups(
            remaining, groups, jnp.ones_like(remaining)
        )
        active_demand = active_demand / (jnp.sum(active_demand) + 1.0)

        progress = jnp.asarray(
            [state.timestep / self.max_episode_steps], dtype=jnp.float32
        )
        # Single concatenated context vector (length 3*G + 1): the bounded
        # sufficient statistic + present per-segment demand + day progress. A
        # single vector (always length > 1) avoids size-1 observation ambiguity.
        obs["fair_context"] = jnp.concatenate(
            [group_score, group_frac, active_demand, progress]
        ).astype(jnp.float32)

        # Per-charger price multiplier of the connected car (0 if empty): the
        # policy must observe ability-to-pay to act on it.
        obs["charger_price"] = (self._price_mult(state.charger_group) * connected).astype(
            jnp.float32
        )
        return obs

    # --------------------------------------------------------------------- info
    def get_info(self, state, actions, old_state=None) -> Dict[str, Array]:
        info = super().get_info(state, actions, old_state=old_state)
        safe_count = jnp.maximum(state.group_count, 1.0)
        ports = state.grid.evses_flat
        am = state.last_arrival_mask.astype(jnp.float32)
        # Attributes of customers that arrived this step (for the oracle stream).
        arr_desired = jnp.maximum(
            ports.car_desired_battery_percentage * ports.car_battery_capacity_kw
            - ports.car_arrival_battery_kw,
            0.0,
        )
        arr_window = ports.car_time_till_leave / self.minutes_per_timestep
        arr_pmax = ports.voltage * ports.max_current / 1000.0
        info.update(
            {
                "welfare": self._welfare(state),
                "group_mean_satisfaction": state.group_util_sum / safe_count,
                "group_count": state.group_count,
                "realized_util": state.last_realized_util,
                "realized_mask": state.last_realized_mask.astype(jnp.float32),
                "realized_group": state.last_realized_group.astype(jnp.float32),
                "rejected_per_group": state.last_rejected_per_group,
                # Arrival stream (masked by arrival_mask) for the offline oracle.
                "arrival_mask": am,
                "arrival_desired_kwh": arr_desired * am,
                "arrival_window_steps": arr_window * am,
                "arrival_pmax_kw": arr_pmax * am,
                "arrival_group": state.charger_group.astype(jnp.float32),
            }
        )
        return info
