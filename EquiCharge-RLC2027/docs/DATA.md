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

For an independent, US, session-level calibration, `chargax.equity.data_calibration.acn_scenario`
adapts the public **Caltech ACN-Data** (Lee, Li & Low, ACM e-Energy 2019;
<https://ev.caltech.edu/dataset>) into Chargax sampling callables. It is a hook
rather than a bundled dataset because the data has to be downloaded:

```python
from experiments.common import acn_env_or_none
env, provenance = acn_env_or_none(grid_kw=30.0, n_evses=8)   # reads EQUICHARGE_ACN_JSON
```

What comes from ACN-Data, and what does not:

| quantity | source | status |
|---|---|---|
| arrival rate + time of day | binned session counts / observed days, **workdays and weekends separately** | **real** |
| connection (dwell) time | `disconnectTime − connectionTime` | **real** |
| energy demand | driver-stated `kWhRequested`, falling back to `kWhDelivered` | **real** |
| vehicle fleet | Chargax US fleet mix | **modelled** — ACN-Data records sessions, not vehicle models |
| electricity price | 2023 NL day-ahead | **real, but not US** — see caveat below |

Dwell and energy are drawn **as a pair from the same session**, preserving a real
within-session correlation the bundled loaders (which sample the two independently
from separate CSVs) cannot represent. Energy demand prefers the driver's stated
`kWhRequested` because satisfaction is measured against each driver's own target, and
`kWhDelivered` is an outcome of a possibly power-limited session — using it as the
target would bake the incumbent controller's rationing into the demand distribution.

Two caveats a reviewer should see. First, the default window ends before the March
2020 campus closure (`ACN_PRECOVID_WINDOW`); averaging arrivals across the closure
would quietly turn a power-scarce site into an unconstrained one. Pass
`date_range=None` for the full dump. Second, prices remain the NL day-ahead series,
so this configuration is a US **demand** calibration on an EU price series, not an
end-to-end US site.

The adapter never fabricates data: if the file is absent it raises. The callables must
be passed as **top-level env fields** — Chargax reads only `car_profile`,
`user_profile`, `average_cars_per_day` and `grid_price_dataset` out of
`default_data_kwargs`, so callables placed in that dict are silently dropped and the
env rebuilds the bundled Dutch loaders instead. `acn_env_or_none` wires them up and
asserts they landed, so a misconfiguration fails loudly rather than reporting a "US
session-level" result computed from Dutch data.

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
