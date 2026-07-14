# Data calibration: what is real, what is modelled

The equity study is run on a **real-data-calibrated** environment, not a synthetic
one. This document states the provenance of every distribution the environment
samples from, so a reviewer can see exactly where the numbers come from. Run
`python -m chargax.equity.data_calibration` to print this table from code.

| quantity | source file(s) | status | note |
|---|---|---|---|
| arrival time-of-day | `chargax/data/car_arrival_percentages_workdays.csv`, `_weekends.csv` | **real** | Empirical Dutch time-of-day arrival profiles per user context (residential / workplace / shopping / highway). |
| connection / dwell time | `chargax/data/car_connection_times.csv` | **real** | Empirical dwell-time distribution. |
| per-session energy demand | `chargax/data/car_energy_demand.csv` | **real** | Empirical energy-demand distribution. |
| vehicle fleet & charge curves | `chargax/data/car_frequency_and_profiles.csv` | **real** | EU / US / world vehicle mix, real battery capacities and nonlinear charge curves. |
| electricity price | `chargax/data/electricity_prices_kwh_2021_NL.csv` … `2023_NL.csv` | **real** | Real Netherlands day-ahead wholesale prices, 2021–2023. |
| ability-to-pay segment | `chargax/equity/segments.py` | **grounded** | Modelled *on top of* the real stream. Price weights `(0.6, 1.0, 1.5)` derived from ACS income terciles and the ACEEE/DOE energy-burden gradient with an income elasticity `η≈0.6`; kept exogenous and independent of battery capacity. |
| grid connection limit | `experiments/common.py:scarcity_station` | **design parameter** | The binding scarcity lever; swept explicitly in the capacity-threshold analysis. |

## Alternative calibration: Caltech ACN-Data

For an independent, US, session-level calibration, `chargax.equity.data_calibration.acn_data_kwargs`
adapts the public **Caltech ACN-Data** (Lee, Li & Low, ACM e-Energy 2019;
<https://ev.caltech.edu/dataset>) into Chargax sampling callables (time-of-day
arrival PMF, dwell-time and energy-demand distributions). ACN-Data requires free API
registration and a download, so it is provided as a hook rather than bundled:

```python
from chargax.equity import EquiChargax, data_calibration as dc
kwargs = dc.acn_data_kwargs("acn_sessions.json")   # a downloaded ACN-Data JSON
env = EquiChargax(station=station, get_num_cars_arriving=kwargs["get_num_cars_arriving"],
                  n_groups=3, price_by_group=(0.6, 1.0, 1.5))
```

The adapter never fabricates data: if the file is absent it raises, so a run uses
either the bundled real Dutch data or a real ACN-Data download.

## Grounding the ability-to-pay segments

The one construct added for the equity analysis is the ability-to-pay segmentation.
It is **not** an arbitrary multiplier: see `chargax/equity/segments.py`. Each segment
is a household-income tercile from the **ACS 2024 1-year** data (median household income
≈ \$81.6k; tercile boundaries ≈ \$54k / \$122k; representative within-tercile incomes
≈ \$35k / \$82k / \$161k), and its price weight is its willingness-to-pay per kWh
relative to the median, computed with a sub-proportional income elasticity `η≈0.6`
(residential energy is empirically a necessity, elasticity in (0,1); defensible range
≈0.2–0.9). This yields `(0.6, 1.0, 1.5)`. The energy-burden gradient corroborates the
direction: low-income households (≤200% FPL) spend a median **8.1%** of income on home
energy vs **2.3%** for higher-income households, a ~3.5× gap (Drehobl, Ross & Ayala
2020, AHS 2017; US DOE LEAD). The **A1 grounding sweep**
(`experiments/audit_grounding.py`) shows the disparate impact is invariant across the
plausible elasticity and tier–income-correlation range, so the exact spread is not
load-bearing.

Run `python -m chargax.equity.segments` to reproduce the derivation.
