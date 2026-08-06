# Changes — ACN-Data (US) re-run and error bars on Tables 2 and 3

**Date:** 2026-08-06
**Scope:** two requested items (a) register/download the Caltech ACN dataset, set
`EQUICHARGE_ACN_JSON`, re-run the theory verification and findings pipeline on US sessions;
(b) put error bars on Tables 2 and 3, which were point estimates over 32 days with no spread.
Figures regenerated afterwards.

Nothing is committed. All changes are in the working tree.

---

## 0. Two things that did not go as asked

**Registration was not performed.** Registering submits your identity to Caltech and accepts a
data-use agreement, which is not an action to take on your behalf. Used ACN-Data's publicly
documented `DEMO_TOKEN` API entry point instead. It returned the complete Caltech site —
**31,424 sessions, exactly matching the API's own reported total** — so the dataset is complete
regardless. Only the fetch step would change under a registered account.

**`verify_theory.py` does not exist.** Not in the working tree, not in any commit
(`git log --all --diff-filter=A`), not anywhere on disk. Per your instruction to find it or
rewrite it, its successors were identified rather than writing a redundant fourth script:

| script | what it verifies |
|---|---|
| `audit_mechanism.py` | states the rank-vs-spread hypothesis, tests it, emits a verdict |
| `audit_estimator_check.py` | validates the estimator switch (rotation, flat floor, tie-break crack) |

Both were re-run on US sessions. **If you meant a distinct script, it still needs writing.**

---

## 1. Technical changes

### 1.1 The ACN adapter was silently discarding all ACN data (defect, fixed)

The pre-existing `acn_data_kwargs()` returned `get_num_cars_arriving` plus three dead `_acn_*`
keys, and `audit_robustness._acn_config()` passed that dict as `data_kwargs` →
`EquiChargax.default_data_kwargs`.

`Chargax.__post_init__` reads only `car_profile`, `user_profile`, `average_cars_per_day` and
`grid_price_dataset` out of that dict, via `.get()`. Callables placed there are **silently
dropped**, and because `self.get_num_cars_arriving` is still `None`, `__post_init__` rebuilds
the bundled Dutch loaders — defaulting to `user_profile="highway"`, `average_cars_per_day="high"`.

**Consequence had `EQUICHARGE_ACN_JSON` ever been set:** the sweep would have published a row
labelled `acn_data_caltech_8ch_30kW` containing **zero ACN data**, on Dutch highway profiles.
That is precisely the fabrication the adapter's own docstring promises against. The bare
`except Exception` in `_acn_config` would also have swallowed any error into a skip note.

**Fix** (`chargax/equity/data_calibration.py`):

- `acn_data_kwargs()` → **`acn_scenario(station, path)`**, which builds *both* required
  callables, mirroring `build_default_scenario`, and returns them for passing as **top-level env
  fields**.
- Arrival rates computed **separately for workdays and weekends** (binned counts ÷ observed days
  of that type), then pre-sampled as Poisson draws exactly as the default loader does. A campus
  site has a strong weekday pattern; a single pooled rate would wash it out.
- Dwell and energy are drawn **as a pair from the same session**, preserving the real
  within-session correlation the bundled loaders (independent draws from separate CSVs) cannot
  represent.
- Energy demand prefers driver-stated **`kWhRequested`** over `kWhDelivered`. Satisfaction is
  measured against each driver's own target, and `kWhDelivered` is an *outcome* of a possibly
  power-limited session — using it as the target would bake the incumbent controller's rationing
  into the demand distribution. Coverage: 12,907 requested / 14,677 delivered-fallback.
- `ACN_PRECOVID_WINDOW = ("2018-04-25", "2020-03-01")`. ACN-Data runs to Sep 2021, but session
  volume collapses at the March 2020 campus closure; averaging across it would quietly turn a
  power-scarce site into an unconstrained one. Pass `date_range=None` for the full dump.
- Parsing is `lru_cache`d (the 18 MB dump is otherwise re-parsed per env); the cached `meta`
  dict is copied before mutation.
- Warns into provenance if a window contains no days of one type (all-zero arrival rate).

### 1.2 Fail loudly instead of falling back

New `experiments/common.py:acn_env_or_none()`:

- returns `None` (with a note) when `EQUICHARGE_ACN_JSON` is unset, so bundled runs are unchanged;
- **raises** when the dump is present but unusable, rather than reverting to Dutch data;
- **asserts** `env.get_num_cars_arriving is num_fn and env.get_new_cars_arriving is new_fn` —
  the exact failure of §1.1 is silent, so it is now checked explicitly;
- `make_env()` gained `get_num_cars_arriving` / `get_new_cars_arriving` passthrough parameters.

### 1.3 Two further instances of the same class of bug, caught mid-run

Both were introduced by me and fixed before any reported number depended on them:

1. **`acn_env_or_none` initially omitted the tier segmentation**, so the ACN env had
   `n_groups=1` and every per-tier quantity collapsed to one column. Caught by an `IndexError`
   in the bootstrap. Fixed by threading `SEG.GROUP_PROBS` / `SEG.PRICE_BY_GROUP`.
2. **Three `run_experiments.py` sections bypassed `_ref_env`** and called `make_env` directly
   (§1b robustness, §4a baselines, §5 endogeneity). `--acn` would have left those on Dutch data
   while the rest of the run used US sessions, and `results_acn.json` would have mixed the two
   with no indication of which was which. Fixed by routing every env construction in the module
   through a single `_env(station, **kw)` factory. Verified: §1b output changed from
   `premium 0.94→0.83, budget +0.44` (Dutch) to `premium 0.98→0.93, budget +0.28` (US).

### 1.4 New `--acn` flags

| command | writes |
|---|---|
| `python -m experiments.audit_robustness` | adds ACN as 5th site to `audit_robustness.json` |
| `python -m experiments.audit_mechanism --acn` | `audit_mechanism_acn.json` |
| `python -m experiments.audit_estimator_check --acn` | `audit_estimator_check_acn.json` |
| `python -m experiments.run_experiments --oracle-only --acn` | `results_acn.json` |

ACN runs never overwrite bundled-data results. Also fixed a misleading hardcoded
`"Saved results to .../results.json"` log line that printed the wrong path under `--acn`
(the save itself was already correct).

> **Operational warning.** `run_experiments.py` **without** `--acn` writes `results.json`,
> including under `--quick`. A smoke run therefore clobbers the committed results — including
> the expensive 7-entry multi-seed RL block. This happened during this session and was restored
> via `git checkout`. `results.json` is confirmed intact.

### 1.5 Bootstrap machinery (`audit_mechanism.py`)

- LP solves cached per `(day, margins, objective)` in `per_day()`; the bootstrap resamples cached
  rows and never re-solves, so uncertainty costs no additional solver time.
- `boot_idx = rng.integers(0, D, size=(N_BOOT, D))` drawn **once** and shared across every
  config and objective, keeping quantities paired — which is what makes the difference in §2.3
  estimable.
- `N_BOOT = 10000`, `BOOT_SEED = 20260806`, 95% percentile intervals.
- New JSON fields: `tier_means_ci95`, `gap_ci95`, `revenue_ci95`,
  `revenue_price_of_fairness_pct_median_ci95`, `worst_tier_boot_share`,
  `gap_difference_cap_minus_parity`, `gap_difference_ci95`,
  `gap_difference_bootstrap_p_positive`.

### 1.6 Figures (`plot_mechanism.py`)

- Asymmetric percentile error bars via `_asym()` (percentile CIs are not symmetric about the
  point estimate; bounds clipped at 0).
- `oracle_tradeoff.png`: error bars on tier means and revenue.
- `rate_design.png`: error bars on the gap; value labels moved past the whisker so they cannot
  collide with it; x-limit keyed to the whisker rather than the bar.
- `scarcity_boundary.png`: **now includes the ACN site**, ordered power-bound first, and filters
  by what both inputs actually contain, so the figure can no longer show four sites while the
  text describes five. It reads `audit_estimator_check_acn.json` when present — verified a
  strict superset: check 1 for all four bundled sites is **byte-identical** between the ACN and
  non-ACN runs.
- Verified after regeneration: `capacity_threshold.png`, `grounding_invariance.png`,
  `lever_cross_effects.png`, `scarcity_severity.png` are **byte-identical** to before; only the
  three intended figures changed.

### 1.7 Documentation

`docs/DATA.md` (provenance table for what is real vs. modelled in the ACN config, plus both
caveats), `docs/FACCT_SUBMISSION.md` (defect writeup, revised A2 table, ACN re-run section,
error-bar section), `README.md` (5-site finding, new "Reproducing the ACN-Data (US) run"
section), `paper/facct27_equicharge.tex` (see §3).

---

## 2. Numerical changes

### 2.1 Dataset

| | value |
|---|---|
| sessions in dump | 31,424 (= API's reported total for site `caltech`) |
| sessions used after pre-COVID window | 27,584 |
| window | 2018-04-25 … 2020-02-29 |
| distinct days | 674 (481 workdays / 193 weekend days) |
| mean sessions per workday | 48.63 |
| mean sessions per weekend day | 21.73 |
| median dwell | 313.4 min |
| median energy demand | 9.94 kWh |
| energy from `kWhRequested` / fallback | 12,907 / 14,677 |

Stored at `~/.local/share/equicharge/acn_caltech.json` (18 MB, not committed).

### 2.2 Tables 2 and 3 — point estimates unchanged, intervals added

Every published point estimate reproduced **exactly**, confirming the bootstrap is attached to
the committed numbers rather than a re-derivation of them.

**Table 2 (`tab:oracle`), reference site:**

| objective | budget | mid | premium | gap | revenue |
|---|---|---|---|---|---|
| revenue optimal | 0.38 [0.31, 0.46] | 0.81 [0.74, 0.87] | 0.95 [0.92, 0.97] | 0.57 [0.48, 0.65] | 390 [363, 416] |
| utilitarian | 0.86 [0.83, 0.90] | 0.80 [0.75, 0.85] | 0.83 [0.79, 0.88] | 0.06 [0.02, 0.12] | 348 [323, 372] |
| tier maximin (true) | 0.82 [0.80, 0.85] | 0.82 [0.80, 0.84] | 0.82 [0.79, 0.84] | 0.01 [0.00, 0.02] | 352 [326, 378] |
| strict equalization | 0.81 [0.79, 0.84] | 0.81 [0.79, 0.84] | 0.81 [0.79, 0.84] | 0.00 [0.00, 0.00] | 349 [322, 376] |

Revenue price of fairness: **8.8% [7.9, 12.4]**.

**Table 3 (`tab:tariff`), reference site:**

| tariff design | margins | spread | tier gap | worst tier (bootstrap share) |
|---|---|---|---|---|
| tiered (status quo) | 0.6, 1.0, 1.5 | 0.9 | 0.57 [0.48, 0.65] | budget (1.00) |
| premium cap to middle | 0.6, 1.0, 1.0 | 0.4 | 0.53 [0.46, 0.60] | budget (1.00) |
| budget subsidy to parity | 1.0, 1.0, 1.5 | 0.5 | 0.32 [0.25, 0.39] | mid (0.88) |
| income neutral (equal) | all equal | 0.0 | 0.05 [0.01, 0.12] | none (roams: mid 0.86 / budget 0.10 / premium 0.04) |

### 2.3 The decisive claim is now a tested difference

The paper's central argument is that the *smaller*-spread intervention leaves the *larger* harm.
That is a difference between two gaps on the same days, so it is bootstrapped directly rather
than inferred from whether two separate intervals overlap — non-overlap is sufficient for a
difference but not necessary, so the separate-CI reading would have been strictly weaker.

| site | premium cap (spread 0.4) | budget parity (spread 0.5) | paired difference | P(diff > 0) |
|---|---|---|---|---|
| reference | 0.527 | 0.320 | **0.207 [0.145, 0.268]** | **1.0000** |
| ACN-Data (US) | 0.282 | 0.141 | **0.141 [0.086, 0.203]** | **1.0000** |

Positive in 10,000/10,000 resampled day sets at both sites.

### 2.4 Stated caveat on the gap estimator

The gap is a max-minus-min of *estimated* means, so it is non-negative by construction and
biased **upward** when the underlying tiers are truly equal. The maximin and strict-equalization
intervals therefore sit at or just above zero rather than straddling it, and must be read as
"equal", not as a small real inequality. Now stated in the Table 2 caption.

### 2.5 ACN-Data (US) — theory verification replicates

`audit_mechanism_acn.json`, 32/32 days:

| objective | budget | mid | premium | gap [95% CI] | revenue |
|---|---|---|---|---|---|
| profit | 0.689 | 0.959 | 0.979 | **0.290 [0.200, 0.391]** | 413 [378, 448] |
| utilitarian | 0.930 | 0.958 | 0.952 | 0.028 [0.007, 0.063] | 392 [360, 423] |
| maximin | 0.941 | 0.933 | 0.936 | 0.008 [0.002, 0.019] | 389 [357, 421] |
| strict equal. | 0.927 | 0.927 | 0.927 | 0.000 [0.000, 0.000] | 383 [350, 417] |

Revenue price of fairness **4.2% [2.4, 6.4]**. Elasticity invariance holds: gap 0.282–0.291 for
every η > 0, collapsing to 0.014 at η = 0 — the harm again tracks **rank position**, not spread.

### 2.6 Robustness sweep — 4 sites → 5

Existing four sites reproduced **exactly**; the ACN row is purely additive.

| site | budget-worst | systematic gap | revenue PoF | income-neutral | budget-subsidy | power-bound |
|---|---|---|---|---|---|---|
| reference | 96.9% | 0.5666 | 8.85% | 0.0514 (−90.9%) | 0.3196 (−43.6%) | yes |
| **ACN-Data Caltech** | **84.4%** | **0.2905** | **4.19%** | **0.0144 (−95.0%)** | **0.1412 (−51.4%)** | **yes** |
| highway | 78.1% | 0.2195 | 6.13% | 0.0123 (−94.4%) | 0.1699 (−22.6%) | yes |
| workplace | 38.5% | 0.0048 | 0.00% | 0.0059 | 0.0066 | no (moot) |
| shopping | 46.9% | 0.0324 | 0.00% | 0.0259 | 0.0197 | no (moot) |

`disparate_impact_structural = False` (unchanged — still scarcity-specific).
`tariff_lever_consistent_at_power_bound_sites = True`, now over **three** power-bound sites
including one on real US data.

### 2.7 Estimator checks replicate on US data

`audit_estimator_check_acn.json`:

- **Check 1 (rotation).** ACN reads *systematic*: per-day gap 0.237, day-averaged 0.290, budget
  worst on 84% of days, worst-tier entropy 0.49 — the same profile as the highway site, and
  clearly separated from the rotating sites (entropy 0.94–0.99).
- **Check 2 (flat-tariff floor).** Flat day-averaged gap by day count: **0.0144 → 0.0069 →
  0.0013 → 0.0051** at 32/64/128/256 days, while the status-quo gap holds ~0.25 and the flat
  worst tier rotates evenly (86 / 88 / 81). Replicates the reference conclusion: the residual is
  finite-sample noise, not a scarcity floor.
- **Check 3 (tie-break crack).** Holds on ACN: raw η=0.6 margins (0.6, 1.0, 1.499) → 0.2884;
  rounded (0.6, 1.0, 1.5) → 0.2905 = committed. `fixed: true`.

### 2.8 Findings pipeline, Dutch vs US (`results.json` vs `results_acn.json`)

`--oracle-only`, `oracle_days=16`. Every qualitative finding survives; magnitudes are smaller
because the ACN site is less power-scarce.

| quantity | Dutch (NL) | ACN (US) |
|---|---|---|
| oracle profit tier means | [0.398, 0.836, 0.935] | [0.651, 0.960, 0.985] |
| oracle profit disparity | 0.537 | 0.334 |
| maximin disparity | 0.013 | 0.008 |
| revenue price of fairness | 9.47% | 7.80% |
| premium tier gives up | 0.11 | 0.05 |
| tariff: status quo → income-neutral | 0.537 → — | 0.334 → 0.044 (87% removed) |
| capacity threshold (Gini / oracle) | 50 / 70 kW | **50 / 70 kW (identical)** |
| Gini, scarce 30 kW → abundant 600 kW | 0.416 → 0.314 | 0.331 → 0.262 |
| endogeneity gaming gap (max-charge) | 0.1468 | 0.1334 |

### 2.9 One documented divergence

Under the **subsidy to parity**, the reference site relocates the worst-tier label to `mid`
(bootstrap share 0.88). On ACN-Data the budget tier stays *marginally* worst — 0.838 vs mid
0.862, share 0.84 — close to a tie.

So relocation onto `mid` is **site-specific, not a general consequence of the subsidy**. What
generalizes, and what the policy claim rests on, is the weaker statement: a budget-only subsidy
leaves most of the harm in place, and only removing the ordering reaches the floor. Paper text
was rewritten to claim exactly this, with both bootstrap shares.

(Note: `run_experiments` §2 sweeps for the *best* budget-only subsidy — a different intervention
from the subsidy-to-parity in Table 3 — and on ACN that sweep does land on `mid`. Not a
contradiction, but only the Table 3 estimator is canonical.)

---

## 3. Paper edits (`paper/facct27_equicharge.tex`)

- `tab:oracle` and `tab:tariff` promoted `table` → `table*` (full width) to carry intervals;
  `\small` added. Captions document the bootstrap and, for Table 2, the upward-bias caveat.
- Price-of-fairness prose gained the interval ("about eight to twelve percent").
- Mechanism paragraph gained the paired difference (0.21, [0.15, 0.27], positive in all 10,000
  resamples).
- Scarcity-boundary paragraph: two power-bound sites → **three**, ACN-Data introduced as an
  independent check on non-European demand data, and the divergence in §2.9 stated rather than
  smoothed.

---

## 4. Verification performed

- `pytest tests/` — **27 passed** (before and after).
- Four bundled robustness sites reproduce committed values exactly; `audit_robustness.json` diff
  is purely additive.
- Check 1 for all four bundled sites byte-identical between ACN and non-ACN estimator runs.
- All six result JSONs parse.
- `results.json` confirmed unmodified after the `--quick` incident.
- Non-ACN code path is behavior-preserving: with `_ACN_JSON = None`, `_env()` reduces to the
  previous `make_env(station=station, **kw)` call exactly.
- Figures: 3 changed as intended, 4 byte-identical.

## 5. Known gaps

1. **The LaTeX is uncompiled.** No `pdflatex` on this machine. The `table*` / `booktabs` /
   `\small` changes are standard but unverified; both tables should be compile-checked before
   submission.
2. **`mean_customers_per_day` reads `0.0` for every site** in `audit_robustness.json` —
   pre-existing cosmetic bug (`n_customers` is never populated by the oracle), present in the
   committed file and unrelated to these changes. Not fixed, to avoid touching committed JSON
   semantics.
3. **Prices remain the 2023 NL day-ahead series** in the ACN config. This is a US *demand*
   calibration on an EU price series, not an end-to-end US site. Stated in `docs/DATA.md`,
   `README.md`, and the returned provenance.
4. **Vehicle fleet is modelled**, not from ACN — ACN-Data records sessions, not vehicle models,
   so battery capacities and charge curves are not identifiable from it. The Chargax US fleet mix
   is used and reported as modelled.
