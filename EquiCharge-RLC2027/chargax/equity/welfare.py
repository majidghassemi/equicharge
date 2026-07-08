r"""Social Welfare Functions (SWFs) for welfare-optimal reinforcement learning.

This module implements a family of *non-linear* social welfare functions used to
aggregate per-group (or per-individual) utilities into a single scalar that an RL
agent optimizes. These functions encode different ethical stances on distributive
justice:

* **Utilitarian** (:func:`utilitarian`) -- maximize the (population-weighted) sum of
  utilities. Indifferent to inequality. This is the implicit objective of a naive
  profit/throughput maximizer.
* **alpha-fairness** (:func:`alpha_fairness`) -- a one-parameter family that
  interpolates between utilitarianism (``alpha = 0``), proportional / Nash
  fairness (``alpha = 1``), and Rawlsian egalitarianism (``alpha -> inf``). The
  parameter ``alpha`` is the *inequality aversion*.
* **Nash** (:func:`nash_welfare`) -- the (weighted) sum of log-utilities, i.e.
  ``alpha = 1``. Scale-invariant and a classic notion of a "fair" bargaining
  solution.
* **Rawlsian soft-min** (:func:`rawlsian_softmin`) -- a smooth, differentiable
  surrogate for the maximin objective (maximize the welfare of the worst-off
  group), obtained via the log-sum-exp soft-minimum. As the temperature
  ``beta -> inf`` it recovers the hard minimum / leximin stance.

The central object an RL algorithm cares about is the *welfare increment*: if
``W`` is the SWF and ``R_t`` is the vector of accumulated per-group utilities at
time ``t``, the per-step reward ``W(R_{t+1}) - W(R_t)`` telescopes over a
fixed-horizon (undiscounted) episode to ``W(R_T) - W(R_0)``, i.e. exactly the
welfare of the final allocation. This is the mechanism that lets a standard,
additive-reward RL algorithm (e.g. PPO) optimize a *non-additive* welfare
objective on a state-augmented MDP. See :func:`welfare_increment`.

References
----------
Siddique, Weng & Zimmer, "Learning Fair Policies in Multi-Objective (Deep)
Reinforcement Learning with Average and Discounted Rewards", ICML 2020.
Moulin, "Fair Division and Collective Welfare", MIT Press 2003 (alpha-fairness,
Nash, egalitarian welfare).
"""

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

# Satisfaction / utility floor. alpha-fairness and Nash welfare are singular at 0
# (``s ** (1 - alpha)`` diverges for ``alpha > 1`` and ``log(0) = -inf``). We floor
# utilities to keep gradients and rewards finite. A *small* floor preserves the
# strong preference of egalitarian SWFs for lifting the worst-off away from zero.
EPS: float = 1e-3

WelfareKind = Literal["utilitarian", "alpha", "nash", "rawlsian"]


def _normalize_weights(weights: Float[Array, " G"] | None, like: Array) -> Array:
    """Return non-negative weights summing to 1, broadcastable to ``like``."""
    if weights is None:
        weights = jnp.ones_like(like)
    weights = jnp.asarray(weights, dtype=like.dtype)
    total = jnp.sum(weights)
    # If every group is empty (total == 0), fall back to uniform weights so the
    # SWF is well-defined rather than 0/0.
    return jnp.where(total > 0, weights / (total + 1e-12), jnp.ones_like(like) / like.shape[-1])


def utilitarian(
    utilities: Float[Array, " G"], weights: Float[Array, " G"] | None = None
) -> Float[Array, ""]:
    r"""Utilitarian welfare :math:`W = \sum_g w_g\, u_g`.

    Maximizing this is equivalent to maximizing total (weighted) utility and is
    completely indifferent to how utility is distributed across groups.
    """
    u = jnp.clip(utilities, EPS, None)
    w = _normalize_weights(weights, u)
    return jnp.sum(w * u)


def alpha_fairness(
    utilities: Float[Array, " G"],
    weights: Float[Array, " G"] | None = None,
    alpha: float = 1.0,
) -> Float[Array, ""]:
    r"""(Weighted) alpha-fair welfare.

    .. math::
        W_\alpha = \sum_g w_g\, \frac{u_g^{1-\alpha}}{1-\alpha}
        \quad(\alpha \neq 1), \qquad
        W_1 = \sum_g w_g \log u_g.

    ``alpha`` is the inequality-aversion parameter:

    * ``alpha = 0``  -> utilitarian (no inequality aversion);
    * ``alpha = 1``  -> Nash / proportional fairness;
    * ``alpha -> inf`` -> Rawlsian maximin (only the worst-off matters).

    The marginal value of utility to group ``g`` scales as ``u_g**(-alpha)``, so
    larger ``alpha`` shifts marginal reward toward worse-off groups.
    """
    u = jnp.clip(utilities, EPS, None)
    w = _normalize_weights(weights, u)

    def _log_case(_):  # alpha == 1
        return jnp.sum(w * jnp.log(u))

    def _power_case(_):  # alpha != 1
        return jnp.sum(w * (u ** (1.0 - alpha)) / (1.0 - alpha))

    # ``alpha`` is a static Python float in practice; use a host-side branch.
    if abs(alpha - 1.0) < 1e-8:
        return _log_case(None)
    return _power_case(None)


def nash_welfare(
    utilities: Float[Array, " G"], weights: Float[Array, " G"] | None = None
) -> Float[Array, ""]:
    r"""Nash social welfare :math:`W = \sum_g w_g \log u_g` (alpha-fairness, alpha=1)."""
    return alpha_fairness(utilities, weights, alpha=1.0)


def rawlsian_softmin(
    utilities: Float[Array, " G"],
    weights: Float[Array, " G"] | None = None,
    beta: float = 20.0,
) -> Float[Array, ""]:
    r"""Smooth Rawlsian (maximin) welfare via the log-sum-exp soft-minimum.

    .. math::
        W_\beta = -\frac{1}{\beta}\,
        \log\!\Big(\sum_g \tilde w_g\, e^{-\beta u_g}\Big),

    where :math:`\tilde w_g` are normalized weights. As ``beta -> inf`` this
    converges to :math:`\min_g u_g` (the hard Rawlsian objective); finite
    ``beta`` yields a differentiable surrogate with non-vanishing gradients for
    all groups, which is far more stable to optimize than a hard ``min``.
    """
    u = jnp.clip(utilities, EPS, None)
    w = _normalize_weights(weights, u)
    # log-sum-exp with a max-shift for numerical stability.
    z = -beta * u
    log_w = jnp.log(w + 1e-12)
    lse = jax.scipy.special.logsumexp(z, b=jnp.exp(log_w))
    return -lse / beta


def social_welfare(
    utilities: Float[Array, " G"],
    weights: Float[Array, " G"] | None = None,
    kind: WelfareKind = "alpha",
    *,
    alpha: float = 1.0,
    beta: float = 20.0,
) -> Float[Array, ""]:
    """Dispatch to a named social welfare function.

    Parameters
    ----------
    utilities : array of shape (G,)
        Per-group utilities (e.g. mean charging satisfaction), in [0, 1].
    weights : array of shape (G,), optional
        Population weights per group (e.g. customer counts). Defaults to uniform.
    kind : {"utilitarian", "alpha", "nash", "rawlsian"}
        Which welfare function to use.
    alpha : float
        Inequality aversion for ``kind="alpha"``.
    beta : float
        Inverse temperature for ``kind="rawlsian"`` soft-min.
    """
    if kind == "utilitarian":
        return utilitarian(utilities, weights)
    if kind == "alpha":
        return alpha_fairness(utilities, weights, alpha=alpha)
    if kind == "nash":
        return nash_welfare(utilities, weights)
    if kind == "rawlsian":
        return rawlsian_softmin(utilities, weights, beta=beta)
    raise ValueError(f"Unknown welfare kind: {kind!r}")


def welfare_increment(
    utilities_next: Float[Array, " G"],
    utilities_prev: Float[Array, " G"],
    weights: Float[Array, " G"] | None = None,
    kind: WelfareKind = "alpha",
    *,
    alpha: float = 1.0,
    beta: float = 20.0,
) -> Float[Array, ""]:
    r"""Telescoping welfare reward :math:`W(R_{t+1}) - W(R_t)`.

    Summing this over an **undiscounted** (:math:`\gamma=1`), fixed-horizon episode
    yields :math:`W(R_T) - W(R_0)`. Because :math:`R_0` is constant, maximizing the
    undiscounted return is *exactly* maximizing the welfare of the final
    accumulated utilities -- the basis for optimizing a non-additive objective
    with an additive-reward RL algorithm on a state-augmented MDP.

    .. warning::
        The telescoping identity is exact **only for** :math:`\gamma = 1`. With
        :math:`\gamma < 1` the discounted sum does not equal :math:`W(R_T)` and
        this becomes potential-based shaping of an approximate objective. We
        therefore optimize the undiscounted return over the (bounded) one-day
        horizon; see ``experiments/`` for the empirical :math:`\gamma` ablation.
    """
    w_next = social_welfare(utilities_next, weights, kind, alpha=alpha, beta=beta)
    w_prev = social_welfare(utilities_prev, weights, kind, alpha=alpha, beta=beta)
    return w_next - w_prev


# ---------------------------------------------------------------------------
# Bounded sufficient-statistic welfare (the streaming / endogenous-population case)
# ---------------------------------------------------------------------------
#
# In online EV charging the set of beneficiaries is *dynamic*: customers arrive
# and depart, so we cannot keep a fixed-dimension utility vector R (as prior
# fair-RL does). Instead we exploit that every welfare function used here has the
# *generalized-mean* form
#
#     W = Psi_g( {V_g} ),     V_g = (1 / N_g) * sum_{i in group g} phi_a(u_i),
#
# where ``phi_a`` is an inner (individual-level) concave kernel with inequality
# aversion ``a`` and ``Psi_g`` is an outer cross-group operator. The pair
# ``(N_g, S_g = sum phi_a(u_i))`` per group is a *bounded sufficient statistic*
# for W -- it is all the state augmentation the agent needs, regardless of how
# many customers stream through. This is what makes welfare-optimal RL tractable
# for an endogenous population.


def inner_kernel(
    utility: Float[Array, "..."], alpha: float = 0.0
) -> Float[Array, "..."]:
    r"""Individual-level kernel :math:`\phi_\alpha(u)` for the equally-distributed
    equivalent (EDE).

    * ``alpha = 0``  -> ``u``        (identity; risk-neutral / utilitarian);
    * ``alpha = 1``  -> ``log u``    (geometric-mean / Nash kernel);
    * otherwise      -> ``u**(1-alpha)`` (isoelastic / alpha-fair kernel).

    Unlike the raw alpha-fair utility, this kernel omits the ``1/(1-alpha)``
    factor; the inverse transform is applied by :func:`equally_distributed_equivalent`
    so that the resulting welfare is a *bounded* certainty-equivalent in ``[0, 1]``
    for every ``alpha`` (avoiding the numerical blow-up of ``u**(1-alpha)`` for
    ``alpha > 1`` as ``u -> 0``). Utilities are floored at :data:`EPS`.
    """
    u = jnp.clip(utility, EPS, 1.0)
    if abs(alpha) < 1e-8:
        return u
    if abs(alpha - 1.0) < 1e-8:
        return jnp.log(u)
    return u ** (1.0 - alpha)


def equally_distributed_equivalent(
    mean_kernel: Float[Array, "..."], alpha: float = 0.0
) -> Float[Array, "..."]:
    r"""Invert the inner kernel to a certainty-equivalent satisfaction in [0, 1].

    Given ``mean_kernel = (1/N) sum_i phi_alpha(u_i)``, returns the *equally
    distributed equivalent*

    .. math::
        \mathrm{EDE}_\alpha = \Big(\tfrac1N\sum_i u_i^{1-\alpha}\Big)^{1/(1-\alpha)}
        \;(\alpha\neq1), \qquad \exp\!\Big(\tfrac1N\sum_i \log u_i\Big)\;(\alpha=1),

    i.e. the uniform satisfaction level that yields the same welfare as the actual
    (unequal) distribution. ``EDE_0`` is the arithmetic mean (utilitarian),
    ``EDE_1`` the geometric mean (Nash), and ``EDE_alpha -> min_i u_i`` as
    ``alpha -> inf`` (Rawlsian). It is monotone in the underlying alpha-fair
    welfare, so maximizing it is equivalent, but it stays in ``[0, 1]``.
    """
    if abs(alpha) < 1e-8:
        return jnp.clip(mean_kernel, EPS, 1.0)
    if abs(alpha - 1.0) < 1e-8:
        return jnp.clip(jnp.exp(mean_kernel), EPS, 1.0)
    return jnp.clip(mean_kernel ** (1.0 / (1.0 - alpha)), EPS, 1.0)


def welfare_from_statistics(
    group_count: Float[Array, " G"],
    group_kernel_sum: Float[Array, " G"],
    *,
    alpha: float = 0.0,
    outer: WelfareKind = "utilitarian",
    weights: Float[Array, " G"] | None = None,
    beta: float = 20.0,
) -> Float[Array, ""]:
    r"""Evaluate the (bounded, [0,1]) welfare from the sufficient statistic.

    The statistic is, per group ``g``, the realized customer count ``N_g`` and the
    accumulated inner kernel ``S_g = sum_i phi_alpha(u_i)``. The per-group score is
    the EDE ``V_g = EDE_alpha(S_g / N_g)`` (bounded in ``[0,1]``), and the welfare
    is an outer aggregation of the ``V_g``:

    * ``outer="utilitarian"`` -> the *pooled* EDE over all customers, which with a
      single group is exactly individual alpha-fairness (the equally-distributed
      equivalent of the whole population's satisfaction);
    * ``outer="rawlsian"`` -> ``soft-min_g V_g``, egalitarian *across groups*
      (used for the structural-equity analysis over exogenous customer segments).

    Parameters
    ----------
    group_count : (G,)        realized customers per group (served + rejected).
    group_kernel_sum : (G,)   per-group accumulated inner kernel ``S_g``.
    alpha : float             inequality aversion of the inner kernel.
    outer : {"utilitarian","rawlsian"}   cross-group operator.
    weights : (G,), optional  group weights for the Rawlsian operator.
    beta : float              soft-min temperature for ``outer="rawlsian"``.
    """
    total = jnp.sum(group_count)
    # Consistent baseline: welfare of the empty population is 0 (for every alpha),
    # so the telescoped return equals the welfare of the final allocation.
    if outer == "utilitarian":
        # Pooled EDE: EDE_alpha( (1/N) sum_all phi(u_i) ).
        safe_total = jnp.where(total > 0, total, 1.0)
        pooled_mean_kernel = jnp.sum(group_kernel_sum) / safe_total
        w = equally_distributed_equivalent(pooled_mean_kernel, alpha)
        return jnp.where(total > 0, w, 0.0)
    if outer == "rawlsian":
        safe_count = jnp.maximum(group_count, 1.0)
        group_score = equally_distributed_equivalent(
            group_kernel_sum / safe_count, alpha
        )
        present = (group_count > 0).astype(group_score.dtype)
        wts = present if weights is None else weights * present
        w = rawlsian_softmin(group_score, wts, beta=beta)
        return jnp.where(total > 0, w, 0.0)
    raise ValueError(f"Unknown outer operator: {outer!r}")

