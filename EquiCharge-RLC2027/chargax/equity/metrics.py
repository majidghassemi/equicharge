r"""Inequality and fairness metrics for *evaluating* learned charging policies.

These functions are used for reporting/analysis (not optimization). They operate
on a 1-D array of per-customer charging *satisfaction* values in [0, 1] (the
fraction of each customer's desired energy that was delivered before they left,
with rejected customers contributing 0).

Provided metrics:

* :func:`gini` -- the Gini coefficient (0 = perfect equality, 1 = maximal
  inequality).
* :func:`atkinson` -- the Atkinson inequality index with explicit inequality
  aversion ``epsilon`` (the welfare-economics dual of alpha-fairness).
* :func:`worst_quantile_mean` -- mean satisfaction of the worst-off ``q`` fraction
  (a CVaR-style measure of the lower tail; ``q=0.0`` recovers the strict min).
* :func:`coefficient_of_variation` -- std / mean.
* :func:`price_of_fairness` -- the relative loss in an efficiency objective
  (e.g. profit) incurred to obtain a fairer allocation.

All functions are pure NumPy/JAX-compatible (they accept array-likes and use
``jnp``), so they can be called both inside JAX evaluation loops and on host
NumPy arrays.
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float


def _as_1d(x) -> Array:
    x = jnp.asarray(x, dtype=jnp.float32).ravel()
    return x


def gini(satisfaction: Float[Array, " N"]) -> Float[Array, ""]:
    r"""Gini coefficient of a non-negative array.

    .. math:: G = \frac{\sum_i \sum_j |x_i - x_j|}{2 n \sum_i x_i}.

    Returns 0 for a degenerate (all-zero or single-element) input.
    """
    x = _as_1d(satisfaction)
    n = x.shape[0]
    total = jnp.sum(x)
    # Mean absolute difference via the sorted-array formula (O(n log n)-friendly,
    # but here we just use the pairwise form which is clearest for small N).
    abs_diffs = jnp.sum(jnp.abs(x[:, None] - x[None, :]))
    denom = 2.0 * n * total + 1e-12
    g = abs_diffs / denom
    return jnp.where(total > 0, g, 0.0)


def atkinson(
    satisfaction: Float[Array, " N"], epsilon: float = 1.0
) -> Float[Array, ""]:
    r"""Atkinson inequality index with inequality aversion ``epsilon`` >= 0.

    .. math::
        A_\epsilon = 1 - \frac{1}{\mu}
        \Big(\tfrac{1}{n}\sum_i x_i^{1-\epsilon}\Big)^{1/(1-\epsilon)}
        \;(\epsilon \neq 1),\qquad
        A_1 = 1 - \frac{1}{\mu}\Big(\prod_i x_i\Big)^{1/n}.

    0 means perfect equality; values approach 1 as inequality grows. This is the
    welfare-economics counterpart of the alpha-fairness SWF and makes the
    "equity" axis of the paper interpretable independently of the SWF used for
    training.
    """
    x = jnp.clip(_as_1d(satisfaction), 1e-6, None)
    n = x.shape[0]
    mu = jnp.mean(x)
    if abs(epsilon - 1.0) < 1e-8:
        ede = jnp.exp(jnp.mean(jnp.log(x)))  # geometric mean
    else:
        ede = jnp.mean(x ** (1.0 - epsilon)) ** (1.0 / (1.0 - epsilon))
    return 1.0 - ede / (mu + 1e-12)


def worst_quantile_mean(
    satisfaction: Float[Array, " N"], q: float = 0.1
) -> Float[Array, ""]:
    r"""Mean satisfaction of the worst-off ``q`` fraction of customers (CVaR).

    ``q`` is the lower-tail fraction. ``q -> 0`` approaches the strict minimum
    (the quantity Rawlsian fairness most cares about); ``q = 1`` recovers the
    overall mean. Robust to single outliers, unlike the hard ``min``.
    """
    x = jnp.sort(_as_1d(satisfaction))
    n = x.shape[0]
    k = jnp.maximum(1, jnp.floor(q * n).astype(jnp.int32))
    # Mean of the k smallest entries via a prefix mask (jit-friendly).
    idx = jnp.arange(n)
    mask = idx < k
    return jnp.sum(jnp.where(mask, x, 0.0)) / jnp.sum(mask)


def coefficient_of_variation(satisfaction: Float[Array, " N"]) -> Float[Array, ""]:
    """Std / mean. 0 means all customers are served equally well."""
    x = _as_1d(satisfaction)
    mu = jnp.mean(x)
    return jnp.where(mu > 0, jnp.std(x) / (mu + 1e-12), 0.0)


def group_disparity(group_means: Float[Array, " G"]) -> Float[Array, ""]:
    """Max-minus-min satisfaction across groups (largest pairwise gap)."""
    g = _as_1d(group_means)
    return jnp.max(g) - jnp.min(g)


def price_of_fairness(profit_fair: float, profit_efficient: float) -> float:
    r"""Relative efficiency loss from choosing a fairer policy.

    .. math:: \text{PoF} = \frac{\Pi^{\*} - \Pi_{\text{fair}}}{|\Pi^{\*}|},

    where :math:`\Pi^{\*}` is the profit of the efficiency-maximizing policy and
    :math:`\Pi_{\text{fair}}` the profit of the fair policy. PoF = 0 means
    fairness was free; PoF = 0.1 means 10% of profit was sacrificed for equity.
    """
    denom = abs(profit_efficient) + 1e-12
    return float((profit_efficient - profit_fair) / denom)


def summarize(satisfaction, group_means=None) -> dict:
    """Convenience: compute all per-customer metrics at once (host-side floats)."""
    x = _as_1d(satisfaction)
    out = {
        "mean_satisfaction": float(jnp.mean(x)),
        "min_satisfaction": float(jnp.min(x)),
        "worst_10pct_satisfaction": float(worst_quantile_mean(x, 0.1)),
        "gini": float(gini(x)),
        "atkinson_eps1": float(atkinson(x, 1.0)),
        "coeff_variation": float(coefficient_of_variation(x)),
    }
    if group_means is not None:
        out["group_disparity"] = float(group_disparity(group_means))
        out["group_means"] = [float(v) for v in jnp.asarray(group_means).ravel()]
    return out
