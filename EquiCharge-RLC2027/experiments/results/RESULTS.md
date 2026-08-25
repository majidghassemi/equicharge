# Results: The Price of a Fair Charge

_Config: 16 chargers behind a 30.0 kW grid; ability-to-pay multipliers [0.6, 1.0, 1.5] (income/energy-burden grounded); profit_scale=200.0 EUR/day; 256 realized days for the oracle; 8 training seed(s); gamma=1, V2G off._

## 1. Offline oracle: the efficiency-equity tradeoff at optimality (clairvoyant LP)

The profit-optimal allocation starves the budget segment to extract price-discrimination revenue; the **true (Pareto-efficient) segment maximin** equalizes the segments *without leveling down* -- it delivers the same total energy as the profit optimum, so equity is nearly free in aggregate satisfaction and is paid for almost entirely in the operator's revenue.

| oracle objective | budget | mid | premium | disparity | mean sat | delivered kWh | revenue |
|---|---|---|---|---|---|---|---|
| profit-optimal | 0.411 | 0.810 | 0.957 | 0.545 | 0.732 | 351 | 406 |
| utilitarian | 0.831 | 0.829 | 0.840 | 0.011 | 0.836 | 351 | 365 |
| segment maximin (true) | 0.820 | 0.822 | 0.820 | 0.002 | 0.821 | 351 | 365 |
| strict equalization | 0.811 | 0.811 | 0.811 | 0.000 | 0.811 | 345 | 359 |

**Price of fairness (profit-optimal -> segment maximin):** **10.3% of operator revenue**, while mean satisfaction *rises* (0.732 -> 0.821, i.e. -12.1%) and the worst segment goes 0.411 -> 0.820. The tension is operator-revenue vs equity, not social-welfare vs equity.

> _Note on the corrected maximin._ A single-epigraph max-min LP finds the right worst-off value but returns an allocation that levels the better-off down to it (the 'strict equalization' row); the two-stage LP holds the maximin value and then Pareto-completes, so the true segment maximin never reports a worst segment below what the profit optimum already achieves.

## 1b. Robustness: the price of fairness is a redistribution premium customers pay for

The mean-satisfaction rise under equalization is a consequence of concave, *saturating* charging utility (a driver does not want more than a full battery), not a free lunch. The premium tier gives up real satisfaction; we report that explicitly and check the effect is not an artifact of one price gradient by sweeping the premium tier's margin.

| premium margin | premium sat (profit→maximin) | premium loss | budget gain | mean sat (profit→maximin) | PoF revenue |
|---|---|---|---|---|---|
| 1.2× | 0.96 → 0.82 | −0.14 | +0.41 | 0.73 → 0.82 | 8% |
| 1.5× | 0.96 → 0.82 | −0.14 | +0.41 | 0.73 → 0.82 | 10% |
| 2.0× | 0.96 → 0.82 | −0.14 | +0.41 | 0.73 → 0.82 | 13% |
| 3.0× | 0.96 → 0.82 | −0.14 | +0.41 | 0.73 → 0.82 | 16% |

_Reading:_ across every price gradient the premium tier loses ≈0.11 of satisfaction and the budget tier gains ≈0.44; equalization is 'nearly free' only in *equal-weighted aggregate*, a value choice we state. The revenue price of fairness grows with the price gradient (8→14%), so the contribution is the magnitude and the price-vs-scarcity decomposition, not the sign (which is the classical implication of concave utility).

## 2. Rate-design finding: the disparate impact is a function of the margin *spread*

Disparity = the gap between best- and worst-served tier in **tier-averaged** satisfaction (an absolute satisfaction-point gap, max−min of the tier means; whether a tier is *systematically* under-served — a disparate impact — not day-to-day noise). Under the status-quo tiered tariff it is **0.545** (worst tier = budget). **Both partial interventions fail, for the same reason.** A subsidy to the lowest tier alone reaches only **0.308** and past parity shifts the burden to the **mid** tier. A cap on the premium tier alone reaches only **0.490**, because the budget tier stays lowest-margin and thus still starved. Only equalizing the margins (income-neutral / full compression) closes it, to **0.012** (−98%). The lever is the **spread**, not either end. _(Revenue is not shown across rows: value-weighted energy at different margins is in different units, so a cross-tariff revenue comparison would be meaningless.)_

| intervention | margins | disparity | worst tier |
|---|---|---|---|
| budget subsidy +0.0 | 0.6, 1.0, 1.5 | 0.545 | budget |
| budget subsidy +0.2 | 0.8, 1.0, 1.5 | 0.545 | budget |
| budget subsidy +0.4 | 1.0, 1.0, 1.5 | 0.308 | mid |
| budget subsidy +0.6 | 1.2, 1.0, 1.5 | 0.527 | mid |
| budget subsidy +0.8 | 1.4, 1.0, 1.5 | 0.528 | mid |
| budget subsidy +1.0 | 1.6, 1.0, 1.5 | 0.521 | mid |
| budget subsidy +1.2 | 1.8, 1.0, 1.5 | 0.521 | mid |
| premium cap 1.5 | 0.6, 1.0, 1.5 | 0.545 | budget |
| premium cap 1.3 | 0.6, 1.0, 1.3 | 0.546 | budget |
| premium cap 1.1 | 0.6, 1.0, 1.1 | 0.546 | budget |
| premium cap 1.0 | 0.6, 1.0, 1.0 | 0.490 | budget |
| premium cap 0.8 | 0.6, 0.8, 0.8 | 0.492 | budget |
| premium cap 0.6 | 0.6, 0.6, 0.6 | 0.013 | budget |

_Actionable reading:_ reduce the **spread** of tier margins — an income-neutral tariff, or a credit that equalizes *all* tiers — removes the 98% price-driven disparate impact. Grid capacity (Section 3) is a second, independent lever: enough capacity removes the scarcity that forces rationing at all. Temporal designs (ToU / demand charges) are tier-agnostic and do not touch the distributional gap.

## 3. Grid capacity: a range, not a sharp threshold

Enough grid capacity removes the scarcity that forces rationing at all (a second, independent lever from pricing). We deliberately state this as a **range, not a sharp threshold**: the oracle tier disparity is noisy and slightly non-monotonic near its floor (~0.02–0.03 above 70 kW), so pinning a single kW value would over-claim. Inequality falls steeply and then flattens — the Gini reaches within 10% of its abundant floor (0.303) by ~**50.0 kW**, and the oracle tier disparity is small (≲0.05) from ~**150.0 kW** upward. For this 16-charger site the disparity is largely removed by roughly **50.0–150.0 kW**; a planner reading the stricter oracle metric should size to the upper end.

_The 30 kW oracle-disparity row below equals the Section-1 oracle exactly (same days, same seed), so the two analyses are consistent._

| grid kW | Gini | mean sat | worst-10% | oracle tier disparity |
|---|---|---|---|---|
| 20 | 0.477 | 0.592 | 0.000 | 0.693 |
| 25 | 0.436 | 0.657 | 0.000 | 0.648 |
| 30 | 0.405 | 0.705 | 0.000 | 0.545 |
| 40 | 0.359 | 0.781 | 0.000 | 0.305 |
| 50 | 0.327 | 0.821 | 0.000 | 0.127 |
| 70 | 0.305 | 0.851 | 0.000 | 0.012 |
| 100 | 0.303 | 0.854 | 0.000 | 0.002 |
| 150 | 0.303 | 0.854 | 0.000 | 0.001 |
| 300 | 0.303 | 0.854 | 0.000 | 0.001 |
| 600 | 0.303 | 0.854 | 0.000 | 0.001 |

## 4. Policies: non-learned baselines and learned welfare policies

One table, matched metrics. Learned rows show the median across 8 seeds with ± half the inter-quartile range; baseline rows are deterministic heuristics evaluated on the same days.

| policy | kind | profit (EUR/day) | mean sat | Gini | worst-10% | segment disp | rejection |
|---|---|---|---|---|---|---|---|
| max_charge | baseline | 178 | 0.705 | 0.405 | 0.000 | 0.040 | 0.21 |
| proportional_fair | baseline | 168 | 0.609 | 0.453 | 0.000 | 0.044 | 0.24 |
| least_laxity | baseline | 169 | 0.695 | 0.417 | 0.000 | 0.037 | 0.21 |
| saffe | baseline | 166 | 0.594 | 0.462 | 0.000 | 0.044 | 0.24 |
| random | baseline | 153 | 0.663 | 0.434 | 0.000 | 0.044 | 0.22 |
| profit_ppo | learned | 143 ± 8 | 0.617 ± 0.024 | 0.457 ± 0.018 | 0.000 ± 0.000 | 0.027 ± 0.006 | 0.233 ± 0.003 |
| util_l050 | learned | 143 ± 4 | 0.624 ± 0.024 | 0.460 ± 0.020 | 0.000 ± 0.000 | 0.040 ± 0.011 | 0.230 ± 0.005 |
| util_l100 | learned | 139 ± 10 | 0.619 ± 0.025 | 0.466 ± 0.029 | 0.000 ± 0.000 | 0.033 ± 0.013 | 0.230 ± 0.005 |
| egal_l025 | learned | 150 ± 7 | 0.626 ± 0.030 | 0.457 ± 0.027 | 0.000 ± 0.000 | 0.026 ± 0.011 | 0.232 ± 0.005 |
| egal_l050 | learned | 141 ± 5 | 0.619 ± 0.023 | 0.465 ± 0.019 | 0.000 ± 0.000 | 0.032 ± 0.012 | 0.233 ± 0.004 |
| egal_l075 | learned | 136 ± 6 | 0.619 ± 0.028 | 0.470 ± 0.024 | 0.000 ± 0.000 | 0.035 ± 0.012 | 0.232 ± 0.002 |
| egal_l100 | learned | 137 ± 8 | 0.615 ± 0.031 | 0.474 ± 0.027 | 0.000 ± 0.000 | 0.036 ± 0.010 | 0.232 ± 0.006 |

_Reading (honest):_ at this **250k-timestep** budget every learned policy is **Pareto-dominated by `random`** (161 profit / 0.439 Gini / 0.681 sat beats all seven on all three axes), and `max_charge` beats `random`. No online policy achieves positive worst-10% satisfaction. A learning-curve diagnostic (`learning_curve.json`) shows this is **undertraining**, not a plateau: the egalitarian policy at 1M steps reaches ~173 profit / 0.428 Gini / 0.698 sat (beating `random`), but is unstable across budget/seed. We therefore **do not present the learned policies as a result** — they are released as a benchmark starting point, and the RL is confined to the paper's appendix. See Section 7.

## 5. Endogeneity audit: the reject-to-look-fair loophole is closed

Defining utility over the *true arrival stream* (rejected customers at 0) is what makes an endogenous, policy-influenced population well-posed. Ablating rejection-counting shows how much apparent welfare a policy could gain by shedding customers it serves poorly.

| policy | welfare (served-only) | welfare (inclusive) | gaming gap | rejection |
|---|---|---|---|---|
| max_charge | 0.641 | 0.505 | 0.136 | 0.21 |
| least_laxity | 0.625 | 0.488 | 0.137 | 0.21 |

## 6. Scarcity induces inequity (profit-blind max-charge)

| grid | mean sat | Gini | worst-10% |
|---|---|---|---|
| 30 kW | 0.705 | 0.405 | 0.000 |
| 600 kW | 0.854 | 0.303 | 0.000 |

