# Reconciling the manuscript against the 256-day re-run

The released results were regenerated at 256 realized days (they were 32). Every
number below is a number the manuscript prints or derives. The 32-day column is the
run the current manuscript text was written against, snapshotted locally under
`experiments/results/archive/pre-256day-2026-08-24/` (that directory is gitignored, so
the snapshot lives on the machine that ran the regeneration rather than in the repo);
the 256-day column is what the repository now releases.

Nothing structural moved. The revenue optimum still starves the budget tier, the
premium cap still leaves the larger gap despite the smaller spread, the paired
difference is still positive in all ten thousand resamples, and the intervals are
tighter throughout. What moved is the magnitudes, and the flat-tariff residual moved
the most, which strengthens rather than weakens the claim: it now decays monotonically
toward zero as days are added, which is what the manuscript already argues it should.

The manuscript has been updated to the 256-day column, and `make gate` passes against
the released results. This file is the audit trail for that edit, so a reviewer or a
co-author can see exactly which number moved and by how much.

Two caveats that survive the edit.

1. The ACN-Data row of the sites table did NOT move, because `EQUICHARGE_ACN_JSON` was
   not set and that configuration was skipped. Every `*_acn.json` file is still the
   32-day run, and the table caption now says so. Re-run `make acn DAYS=256` with the
   dataset in place before presenting the two runs as one.
2. The learned-controller block of `results.json` was not regenerated either.
   `run_experiments --oracle-only` does not train, so it writes an empty section; the
   previously released multi-seed rows were restored from the archive and the file
   records that provenance. Re-run `make learn` to regenerate them.

| where | quantity | 32 days | 256 days |
|---|---|---|---|
| Table tab:oracle | revenue optimal, tier means | [0.383, 0.807, 0.949] | [0.411, 0.810, 0.957] |
| Table tab:oracle | revenue optimal, gap | 0.567 | 0.545 |
| Table tab:oracle | revenue optimal, gap 95% CI | [0.483, 0.648] | [0.512, 0.577] |
| Table tab:oracle | revenue optimal, revenue | 389.6 | 406.4 |
| Table tab:oracle | revenue optimal, revenue 95% CI | [362.9, 416.3] | [396.2, 416.7] |
| Table tab:oracle | utilitarian, tier means | [0.864, 0.804, 0.834] | [0.831, 0.829, 0.840] |
| Table tab:oracle | utilitarian, gap | 0.059 | 0.011 |
| Table tab:oracle | utilitarian, gap 95% CI | [0.022, 0.122] | [0.003, 0.035] |
| Table tab:oracle | utilitarian, revenue | 347.9 | 365.4 |
| Table tab:oracle | utilitarian, revenue 95% CI | [323.4, 372.4] | [355.6, 375.1] |
| Table tab:oracle | tier maximin, tier means | [0.824, 0.821, 0.816] | [0.820, 0.822, 0.820] |
| Table tab:oracle | tier maximin, gap | 0.008 | 0.002 |
| Table tab:oracle | tier maximin, gap 95% CI | [0.001, 0.019] | [0.001, 0.009] |
| Table tab:oracle | tier maximin, revenue | 351.6 | 364.7 |
| Table tab:oracle | tier maximin, revenue 95% CI | [326.1, 378.0] | [354.9, 374.6] |
| Table tab:oracle | strict equalization, tier means | [0.814, 0.814, 0.814] | [0.811, 0.811, 0.811] |
| Table tab:oracle | strict equalization, gap | 0.000 | 0.000 |
| Table tab:oracle | strict equalization, gap 95% CI | [0.000, 0.000] | [0.000, 0.000] |
| Table tab:oracle | strict equalization, revenue | 348.5 | 358.8 |
| Table tab:oracle | strict equalization, revenue 95% CI | [321.8, 375.7] | [348.8, 369.0] |
| Sec. findings | price of fairness, median % | 8.8 | 10.5 |
| Sec. findings | price of fairness, 95% CI | [7.9, 12.4] | [9.5, 11.3] |
| Table tab:tariff | margins [0.6, 1.0, 1.5], gap | 0.567 | 0.545 |
| Table tab:tariff | margins [0.6, 1.0, 1.5], gap 95% CI | [0.483, 0.648] | [0.512, 0.577] |
| Table tab:tariff | margins [0.6, 1.0, 1.0], gap | 0.527 | 0.490 |
| Table tab:tariff | margins [0.6, 1.0, 1.0], gap 95% CI | [0.455, 0.600] | [0.460, 0.520] |
| Table tab:tariff | margins [1.0, 1.0, 1.5], gap | 0.320 | 0.308 |
| Table tab:tariff | margins [1.0, 1.0, 1.5], gap 95% CI | [0.251, 0.392] | [0.287, 0.338] |
| Table tab:tariff | margins [1.0, 1.0, 1.0], gap | 0.051 | 0.012 |
| Table tab:tariff | margins [1.0, 1.0, 1.0], gap 95% CI | [0.012, 0.118] | [0.003, 0.033] |
| Table tab:tariff | premium_cap_gap | 0.527 | 0.490 |
| Table tab:tariff | budget_parity_gap | 0.320 | 0.308 |
| Table tab:tariff | gap_difference_cap_minus_parity | 0.207 | 0.182 |
| Table tab:tariff | paired difference 95% CI | [0.145, 0.268] | [0.153, 0.203] |
| Table tab:sites | reference_16ch_30kW_residential_eu, budget-worst share | 0.969 | 0.914 |
| Table tab:sites | reference_16ch_30kW_residential_eu, Gamma | 0.567 | 0.545 |
| Table tab:sites | reference_16ch_30kW_residential_eu, rotation entropy H | 0.13 | 0.31 |
| Table tab:sites | workplace_24ch_55kW_us, budget-worst share | 0.385 | 0.386 |
| Table tab:sites | workplace_24ch_55kW_us, Gamma | 0.005 | 0.005 |
| Table tab:sites | workplace_24ch_55kW_us, rotation entropy H | 0.99 | 0.99 |
| Table tab:sites | shopping_8ch_16kW_world, budget-worst share | 0.469 | 0.486 |
| Table tab:sites | shopping_8ch_16kW_world, Gamma | 0.032 | 0.040 |
| Table tab:sites | shopping_8ch_16kW_world, rotation entropy H | 0.94 | 0.95 |
| Table tab:sites | highway_20ch_45kW_eu_highrate, budget-worst share | 0.781 | 0.781 |
| Table tab:sites | highway_20ch_45kW_eu_highrate, Gamma | 0.220 | 0.207 |
| Table tab:sites | highway_20ch_45kW_eu_highrate, rotation entropy H | 0.48 | 0.53 |
| Sec. findings | flat-tariff gap at 32 days | 0.0514 | 0.0514 |
| Sec. findings | flat-tariff gap at 64 days | 0.0435 | 0.0435 |
| Sec. findings | flat-tariff gap at 128 days | 0.0309 | 0.0309 |
| Sec. findings | flat-tariff gap at 256 days | 0.0116 | 0.0116 |
| Sec. two levers | pricing lever d_between_tier_gap | 0.516 | 0.533 |
| Sec. two levers | capacity lever d_between_tier_gap | 0.547 | 0.540 |
| Sec. two levers | pricing lever d_within_tier_gini | 0.065 | 0.057 |
| Sec. two levers | capacity lever d_within_tier_gini | 0.228 | 0.221 |
| App. online (margin-greedy) | profit/day, margin-greedy | 182.447 | 188.620 |
| App. online (margin-greedy) | Gamma | 0.340 | 0.353 |
| App. online (margin-greedy) | budget-worst share | 0.844 | 0.836 |
| App. online (margin-greedy) | rotation entropy H | 0.466 | 0.496 |
| App. online (margin-greedy) | max_charge profit/day | 177.9 | 180.6 |
| App. online (margin-greedy) | max_charge Gamma | 0.039 | 0.005 |
| App. online (margin-greedy) | proportional_fair profit/day | 168.1 | 166.9 |
| App. online (margin-greedy) | proportional_fair Gamma | 0.051 | 0.007 |
| App. online (margin-greedy) | least_laxity profit/day | 169.0 | 170.8 |
| App. online (margin-greedy) | least_laxity Gamma | 0.043 | 0.013 |
| App. online (margin-greedy) | saffe profit/day | 166.1 | 165.0 |
| App. online (margin-greedy) | saffe Gamma | 0.050 | 0.007 |
| App. online (margin-greedy) | Gamma at 30 kW | 0.340 | 0.353 |
| App. online (margin-greedy) | Gamma at 45 kW | 0.070 | 0.106 |
| App. online (margin-greedy) | Gamma at 60 kW | 0.011 | 0.015 |
| App. / Sec. findings | Gamma floor over the optimal face, status quo | (new) | 0.500 |
| App. / Sec. findings | Gamma ceiling over the optimal face, status quo | (new) | 0.612 |
| App. / Sec. findings | Gamma floor over the optimal face, income-neutral | (new) | 0.000 |

Regenerate this table by re-running the audit suite and diffing against the archive.
