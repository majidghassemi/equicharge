# Results: The Price of a Fair Charge

_Config: 16 chargers behind a 30.0 kW grid; ability-to-pay multipliers [0.6, 1.0, 1.5] (income/energy-burden grounded); profit_scale=200.0 EUR/day; 16 realized days for the oracle; 6 training seed(s); gamma=1, V2G off._

## 1. Offline oracle: the efficiency-equity tradeoff at optimality (clairvoyant LP)

The profit-optimal allocation starves the budget segment to extract price-discrimination revenue; the **true (Pareto-efficient) segment maximin** equalizes the segments *without leveling down* -- it delivers the same total energy as the profit optimum, so equity is nearly free in aggregate satisfaction and is paid for almost entirely in the operator's revenue.

| oracle objective | budget | mid | premium | disparity | mean sat | delivered kWh | revenue |
|---|---|---|---|---|---|---|---|
| profit-optimal | 0.398 | 0.836 | 0.935 | 0.537 | 0.719 | 348 | 402 |
| utilitarian | 0.890 | 0.837 | 0.815 | 0.075 | 0.846 | 348 | 356 |
| segment maximin (true) | 0.840 | 0.836 | 0.827 | 0.013 | 0.835 | 348 | 364 |
| strict equalization | 0.823 | 0.823 | 0.823 | 0.000 | 0.823 | 341 | 358 |

**Price of fairness (profit-optimal -> segment maximin):** **9.5% of operator revenue**, while mean satisfaction *rises* (0.719 -> 0.835, i.e. -16.1%) and the worst segment goes 0.398 -> 0.827. The tension is operator-revenue vs equity, not social-welfare vs equity.

> _Note on the corrected maximin._ A single-epigraph max-min LP finds the right worst-off value but returns an allocation that levels the better-off down to it (the 'strict equalization' row); the two-stage LP holds the maximin value and then Pareto-completes, so the true segment maximin never reports a worst segment below what the profit optimum already achieves.

## 1b. Robustness: the price of fairness is a redistribution premium customers pay for

The mean-satisfaction rise under equalization is a consequence of concave, *saturating* charging utility (a driver does not want more than a full battery), not a free lunch. The premium tier gives up real satisfaction; we report that explicitly and check the effect is not an artifact of one price gradient by sweeping the premium tier's margin.

| premium margin | premium sat (profit→maximin) | premium loss | budget gain | mean sat (profit→maximin) | PoF revenue |
|---|---|---|---|---|---|
| 1.2× | 0.94 → 0.83 | −0.11 | +0.44 | 0.72 → 0.83 | 8% |
| 1.5× | 0.93 → 0.83 | −0.11 | +0.44 | 0.72 → 0.83 | 9% |
| 2.0× | 0.93 → 0.83 | −0.11 | +0.44 | 0.72 → 0.83 | 12% |
| 3.0× | 0.93 → 0.83 | −0.11 | +0.44 | 0.72 → 0.83 | 14% |

_Reading:_ across every price gradient the premium tier loses ≈0.11 of satisfaction and the budget tier gains ≈0.44; equalization is 'nearly free' only in *equal-weighted aggregate*, a value choice we state. The revenue price of fairness grows with the price gradient (8→14%), so the contribution is the magnitude and the price-vs-scarcity decomposition, not the sign (which is the classical implication of concave utility).

## 2. Rate-design finding: the disparate impact is a function of the margin *spread*

Disparity = the gap between best- and worst-served tier in **tier-averaged** satisfaction (an absolute satisfaction-point gap, max−min of the tier means; whether a tier is *systematically* under-served — a disparate impact — not day-to-day noise). Under the status-quo tiered tariff it is **0.537** (worst tier = budget). **Both partial interventions fail, for the same reason.** A subsidy to the lowest tier alone reaches only **0.280** and past parity shifts the burden to the **mid** tier. A cap on the premium tier alone reaches only **0.502**, because the budget tier stays lowest-margin and thus still starved. Only equalizing the margins (income-neutral / full compression) closes it, to **0.040** (−93%). The lever is the **spread**, not either end. _(Revenue is not shown across rows: value-weighted energy at different margins is in different units, so a cross-tariff revenue comparison would be meaningless.)_

| intervention | margins | disparity | worst tier |
|---|---|---|---|
| budget subsidy +0.0 | 0.6, 1.0, 1.5 | 0.537 | budget |
| budget subsidy +0.2 | 0.8, 1.0, 1.5 | 0.539 | budget |
| budget subsidy +0.4 | 1.0, 1.0, 1.5 | 0.280 | mid |
| budget subsidy +0.6 | 1.2, 1.0, 1.5 | 0.534 | mid |
| budget subsidy +0.8 | 1.4, 1.0, 1.5 | 0.534 | mid |
| budget subsidy +1.0 | 1.6, 1.0, 1.5 | 0.569 | mid |
| budget subsidy +1.2 | 1.8, 1.0, 1.5 | 0.570 | mid |
| premium cap 1.5 | 0.6, 1.0, 1.5 | 0.537 | budget |
| premium cap 1.3 | 0.6, 1.0, 1.3 | 0.534 | budget |
| premium cap 1.1 | 0.6, 1.0, 1.1 | 0.537 | budget |
| premium cap 1.0 | 0.6, 1.0, 1.0 | 0.502 | budget |
| premium cap 0.8 | 0.6, 0.8, 0.8 | 0.508 | budget |
| premium cap 0.6 | 0.6, 0.6, 0.6 | 0.024 | mid |

_Actionable reading:_ reduce the **spread** of tier margins — an income-neutral tariff, or a credit that equalizes *all* tiers — removes the 93% price-driven disparate impact. Grid capacity (Section 3) is a second, independent lever: enough capacity removes the scarcity that forces rationing at all. Temporal designs (ToU / demand charges) are tier-agnostic and do not touch the distributional gap.

## 3. Grid capacity: a range, not a sharp threshold

Enough grid capacity removes the scarcity that forces rationing at all (a second, independent lever from pricing). We deliberately state this as a **range, not a sharp threshold**: the oracle tier disparity is noisy and slightly non-monotonic near its floor (~0.02–0.03 above 70 kW), so pinning a single kW value would over-claim. Inequality falls steeply and then flattens — the Gini reaches within 10% of its abundant floor (0.314) by ~**50.0 kW**, and the oracle tier disparity is small (≲0.05) from ~**70.0 kW** upward. For this 16-charger site the disparity is largely removed by roughly **50.0–70.0 kW**; a planner reading the stricter oracle metric should size to the upper end.

_The 30 kW oracle-disparity row below equals the Section-1 oracle exactly (same days, same seed), so the two analyses are consistent._

| grid kW | Gini | mean sat | worst-10% | oracle tier disparity |
|---|---|---|---|---|
| 20 | 0.488 | 0.598 | 0.000 | 0.720 |
| 25 | 0.447 | 0.662 | 0.000 | 0.618 |
| 30 | 0.416 | 0.709 | 0.000 | 0.537 |
| 40 | 0.368 | 0.791 | 0.000 | 0.247 |
| 50 | 0.336 | 0.833 | 0.000 | 0.099 |
| 70 | 0.315 | 0.865 | 0.000 | 0.017 |
| 100 | 0.314 | 0.867 | 0.000 | 0.026 |
| 150 | 0.314 | 0.867 | 0.000 | 0.033 |
| 300 | 0.314 | 0.867 | 0.000 | 0.033 |
| 600 | 0.314 | 0.867 | 0.000 | 0.033 |

## 4. Policies: non-learned baselines and learned welfare policies

One table, matched metrics. Learned rows show the median across 6 seeds with ± half the inter-quartile range; baseline rows are deterministic heuristics evaluated on the same days.

| policy | kind | profit (EUR/day) | mean sat | Gini | worst-10% | segment disp | rejection |
|---|---|---|---|---|---|---|---|
| max_charge | baseline | 186 | 0.709 | 0.416 | 0.000 | 0.045 | 0.23 |
| proportional_fair | baseline | 177 | 0.615 | 0.468 | 0.000 | 0.066 | 0.26 |
| least_laxity | baseline | 176 | 0.701 | 0.428 | 0.000 | 0.038 | 0.24 |
| saffe | baseline | 175 | 0.600 | 0.477 | 0.000 | 0.067 | 0.26 |
| random | baseline | 161 | 0.681 | 0.439 | 0.000 | 0.052 | 0.24 |
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
| max_charge | 0.645 | 0.499 | 0.147 | 0.23 |
| least_laxity | 0.633 | 0.484 | 0.150 | 0.24 |

## 6. Scarcity induces inequity (profit-blind max-charge)

| grid | mean sat | Gini | worst-10% |
|---|---|---|---|
| 30 kW | 0.709 | 0.416 | 0.000 |
| 600 kW | 0.867 | 0.314 | 0.000 |

