# Saved results manifest (for later figure-making)

All raw data is persisted as JSON so figures can be regenerated at any time without
re-running experiments. Timestamped backups live in `archive/`. Figure **style** is
intentionally deferred — these files hold the numbers; plotting is a separate step.

## `results.json` — the offline (RL-free) core
- `meta` — run config (grid_kw, price_by_group `[0.6,1.0,1.5]`, profit_scale, oracle_days, seeds).
- `oracle` — per objective (`profit`, `utilitarian`, `maximin`, `egalitarian_equal`):
  `group_means[3]`, `group_disparity` (max−min of tier means), `worst_segment_mean`,
  `worst_segment_index`, `mean_satisfaction`, `delivered_kwh`, `revenue_value`.
  Plus `price_of_fairness_revenue`, `price_of_fairness_meansat`.
- `robustness.rows[]` — premium-margin sweep (1.2–3.0×): `profit_premium_sat`,
  `maximin_premium_sat`, `premium_sat_loss`, `budget_sat_gain`, `mean_sat_profit`,
  `mean_sat_maximin`, `pof_revenue_pct`.
- `tariffs` — the rate-design spine:
  - `rows[]` (budget-only subsidy sweep): `subsidy`, `budget_weight`, `disparity`,
    `worst_segment_name`, `group_means`, `revenue_value`, `mean_satisfaction`.
  - `premium_cap.rows[]` (premium-cap sweep): `cap`, `disparity`, `worst_segment_name`, `group_means`.
  - scalars: `value_weighted_disparity`, `flat_tariff_disparity`,
    `price_driven_reduction_pct`, `best_budget_only_disparity`,
    `best_partial_cap_disparity`.
- `capacity.rows[]` — grid sweep (20–600 kW): `grid_kw`, `gini`, `served_mean_sat`,
  `worst10_sat`, `profit_mean`, `oracle_profit_disparity`. Plus `threshold_kw_gini`,
  `threshold_kw_oracle`, `abundant_gini`, `abundant_disparity`.
- `baselines` — per policy (`max_charge`, `least_laxity`, `proportional_fair`,
  `saffe`, `random`): full metric dict (profit_mean, served_mean_sat, gini, atkinson,
  worst10_sat, min_sat, rejection_rate, group_means, group_disparity, ...).
- `endogeneity` — per policy: `welfare_served_only`, `welfare_inclusive`, `gaming_gap`, `rejection_rate`.
- `scarcity` — 30 kW vs 600 kW: `served_mean_sat`, `gini`, `worst10_sat`, `profit_mean`.
- `trained` — 250k-step multi-seed benchmark, per config: median/IQR/mean/std/min/max
  per metric + `group_means`, `group_disparity`.

## `learning_curve.json` — RL learning-curve diagnostic
- `reference` — `random`, `max_charge` (profit, gini, sat).
- `curves.{profit_ppo, egal_l100}[]` — per training budget (250k/1M/3M): `budget`,
  `profit`, `gini`, `sat`, `worst10`.

## `gate.json` — RL gate experiment (3M steps, entropy, 5 seeds)
- `budget`, `ent_coef`, `seeds`.
- `baselines.{max_charge, random}` — profit, served_sat, gini, disparity, group_means.
- `configs.{profit_l0, egal_l050, egal_l100}`:
  - `rows[]` — per seed: profit, served_sat, gini, disparity, premium_sat, budget_sat
    (and, in the enriched harness, the FULL metric dict + group_means).
  - aggregates: `profit`, `served_sat`, `gini`, `disparity` each as {median, iqr, min, max}.
- `gate` — verdict: `disparity_beats_maxcharge`, `service_within_15pct`,
  `monotone_frontier`, `PASS` (written at the end of the run).


## `check.json` — survivorship check (egal_l050, 3M, 10 seeds)
- `baselines.{max_charge,...}` — full inclusive metric dicts.
- `seeds[]` — per seed FULL metric dict (profit_mean, inclusive_mean_sat, gini, rejection_rate, welfare_mean, served_mean_sat, worst10_sat, atkinson, group_means, ...).
- `analysis` — per_axis_vs_maxcharge{win_rate,median}, dominate_all4_seeds, dominate_ge3of4_seeds, median_rejection vs maxcharge_rejection, median_inclusive_welfare, SURVIVES_survivorship_check.


## `ablation.json` — welfare-content vs generic-shaping ablation (λ=0.5, 3M, 10 seeds)
- `throughput_scale` — magnitude-matched content-free potential scale.
- `baselines.max_charge`, `welfare_reference` (medians from check.json), `seeds[]` (full metric dicts).
- `analysis` — throughput_median vs welfare_median vs maxcharge, throughput_winrate_vs_maxcharge, `efficiency_gain_survives_content_free_shaping` (False → welfare content matters), interpretation.

## `audit_offline.json` — per-day adversarial audit of the offline pillars (32 days)
- `box1` — profit_optimal_disparity/maximin_disparity (median/iqr/min/max), worst_tier_under_profit_optimal (day counts), delivered_energy_max_dev_across_objectives_kwh (≈0), revenue_price_of_fairness_pct.
- `box2` — flat/budget-subsidy/premium-cap disparity distributions + worst-tier day-counts.


## `shape.json` — shape-vs-content control (λ=0.5, 3M, 10 seeds)
- `baselines.max_charge`, `welfare_reference`/`monotone_throughput_reference` (medians), `seeds[]` (full metric dicts).
- `analysis` — sat_median vs welfare vs monotone vs maxcharge; `saturating_content_free_recovers_gain` (False → demand-aware content matters, not shape); interpretation.


## `audit_robustness.json` — disparate-impact robustness across station configs (oracle, 32 days each)
- `configs.{reference, workplace, shopping, highway}` — per config: layout, mean_customers_per_day, profit_optimal_disparity (median/iqr/min/max), worst_tier_days (day counts), budget_worst_fraction, revenue_pof_pct, delivered_energy_max_dev_kwh.
- `verdict` — budget_worst_fraction_by_config, disparate_impact_structural (False), interpretation.

## `checkpoints/` — serialized trained agents (final/enriched runs only)
- `gate_{config}_seed{n}.eqx` — equinox-serialized PPO agents, re-loadable for
  re-evaluation on any metric without retraining.

## Figures (`results/figures/`, regenerate with `python -m experiments.plot_results`)
`oracle_tradeoff.png`, `rate_design.png`, `capacity_threshold.png`, `robustness_sites.png`, `scarcity_contrast.png`, `learning_curve.png`. Conference style + validated CVD-safe palette (`experiments/plot_style.py`); synced to `paper/figures/` for the paper.
