"""Welfare-optimal (fair) reinforcement learning for EV charging on Chargax.

This subpackage turns Chargax into an *online sequential fair-allocation* problem
and provides:

* :mod:`~chargax.equity.welfare` -- social welfare functions (utilitarian, Nash,
  alpha-fairness, Rawlsian soft-min) and the bounded sufficient-statistic
  machinery for optimizing them with an additive-reward RL algorithm.
* :class:`~chargax.equity.fair_env.EquiChargax` -- the welfare-optimizing
  environment over an endogenous, streaming customer population.
* :mod:`~chargax.equity.metrics` -- inequality/fairness metrics for evaluation.
* :mod:`~chargax.equity.baselines` -- myopic fair heuristics (proportional /
  least-laxity) and a SAFFE-style anticipatory planner.
* :mod:`~chargax.equity.oracle` -- a clairvoyant offline LP welfare upper bound
  (with a Pareto-efficient two-stage segment maximin and revenue accounting).
* :mod:`~chargax.equity.segments` -- ability-to-pay segments grounded in income /
  energy-burden data.
* :mod:`~chargax.equity.tariffs` -- tariff structures and the rate-design analysis
  (which designs close the equity gap).
* :mod:`~chargax.equity.data_calibration` -- real-data provenance and the ACN-Data
  calibration hook.
"""

from chargax.equity import (
    data_calibration as data_calibration,
    metrics as metrics,
    segments as segments,
    tariffs as tariffs,
    welfare as welfare,
)
from chargax.equity.fair_env import EquiChargax as EquiChargax, FairEnvState as FairEnvState

__all__ = [
    "welfare", "metrics", "segments", "tariffs", "data_calibration",
    "EquiChargax", "FairEnvState",
]
