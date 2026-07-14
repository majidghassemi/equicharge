# FAccT 2027 — submission checklist

Target venue: **ACM FAccT 2027** (Conference on Fairness, Accountability, and
Transparency). This supersedes the AAAI-27/AISI plan in `SUBMISSION.md`; the paper is
`paper/facct27_equicharge.tex`. Confirm every date and limit against the live CFP when the
submission system opens (plan estimate: **system opens October 2026, deadline late October
2026** — confirm the exact day first thing).

## Framing (the venue-carrying items)
- [x] Reframed as a **socio-technical audit**, not an AI-methods paper. Leads with the
  distributional harm and the counterintuitive policy finding (subsidy fails and relocates the
  harm; only removing the price ordering works — rank, not spread). RL demoted to one paragraph
  in the body + an appendix.
- [x] Fairness definitions justified **normatively** (utilitarian / Rawlsian maximin /
  strict equality), framed with distributive- and energy-justice scholarship
  (Rawls, Sen, Sovacool & Dworkin, Jenkins et al.).
- [x] **Disparate impact** used precisely (facially neutral policy + disproportionate
  adverse effect on a protected/marginalized group), with the tier→charging-access→
  tenure/MUD→income/race chain made explicit and a fallback ("regressive allocation")
  where the chain is only partially supported.
- [x] Stakeholders, governance, and levers named (operators, utilities, PUCs; rate
  regulation, low-income tariffs, connection sizing).
- [x] Reflexivity + limitations pass (absent participatory input; dual-use/price-
  discrimination risk; simulation-to-deployment gap).

## Focus areas (select on submission)
- [ ] **Primary: Evaluations and evaluation practices** — audits and evaluations for
  fairness and justice. (Cleanest fit: this paper is an audit instrument + audit.)
- [ ] **Secondary (consider): Law, policy, and governance** — energy-justice/rate-design
  adjacent (the tariff and capacity levers map to real regulatory instruments).

## Required statements (allowed as extra content page)
- [x] **Ethics & adverse-impacts statement** — dual-use of the ability-to-pay model
  (could be repurposed for price discrimination) and how the framing guards against it.
- [x] **Generative AI usage statement** — newly required for the 2027 cycle; disclose
  AI assistance in research and writing. (Review the drafted statement for accuracy
  against actual usage before submitting.)

## Format & hygiene
- [x] ACM format (`\documentclass[sigconf,anonymous,review]{acmart}`). Up to **14 pages
  excluding references + 1 page for statements** — confirm on the CFP.
- [ ] Drop the `acmart` class (TeX Live / ACM) and compile with `pdflatex` twice + bibtex.
  Figures are self-contained PNGs under `paper/figures/`.
- [x] Double-blind: `anonymous` option set; author block anonymized.
- [ ] Scrub the code/data supplement before upload: git history, `pyproject.toml` author
  fields, dataset acknowledgements. The upstream Chargax citation is fine (prior work).
- [ ] Ensure any GitHub/arXiv links in the paper are anonymized or omitted for review.
- [x] **Revision changelog removed** from reviewer-visible docs (moved to the gitignored
  `_private_revision_history/`). Public docs state only the corrected current claims.

## Evidence backing the audit (regenerate before submitting)
- [x] A1 grounding sweep — `experiments/audit_grounding.py` → `results/audit_grounding.json`.
  Shows the harm survives the WTP-elasticity range and the tier→income correlation strength.
- [x] A2 generalization + tariff lever across sites (+ optional ACN-Data hook) —
  `experiments/audit_robustness.py` → `results/audit_robustness.json`.
- [x] Box-1 oracle audit + box-2 tariff — `experiments/audit_offline.py`.
- [x] **Empirical grounding numbers tightened against primary sources** (done 2026-07-13):
  ACS-2024 income terciles (median $81.6k; boundaries $54k/$122k), energy-burden 8.1% vs
  2.3% (Drehobl/Ross/Ayala 2020, AHS 2017; DOE LEAD), elasticity range 0.2–0.9 central
  0.3–0.6 (Schulte & Heindl 2017, Espey & Espey 2004, et al.), home-charging access by
  tenure/dwelling/income/race (Ge 2021 NREL, DOE EERE 2018, Bauer 2021 ICCT, Hsu &
  Fingerman 2021, Yu 2025, Lou 2024), public-vs-home price penalty ~2.5–3× (Borlaug 2026,
  EIA). Folded into `segments.py`, the paper grounding section, and the bibliography.
  Terciles re-anchored to (35k,82k,161k) so the median matches ACS-2024 and the derivation
  still yields the committed (0.6,1.0,1.5) — no downstream re-run needed.
- [x] **Three flagged citations verified** (done 2026-07-13, via Crossref + publisher abstracts):
  Espey & Espey 2004 income elasticities **0.28/0.97 CONFIRMED**; Ermagun & Tian 2024 is
  *Energy Research & Social Science* 115, 103622 (NOT Applied Energy) — **not cited in the paper**;
  Hsu & Fingerman 2021 odds ratios (~0.7× any charger, ~0.5× publicly funded) confirmed, and the
  paper text corrected to frame them as one combined "Black and Hispanic majority" category (CIs
  remain paywalled — check against the full-text table if a CI is ever quoted).

## Open items to confirm on the live CFP
- [ ] Exact submission-open and deadline dates.
- [ ] Page limit (14 + 1) and reference policy.
- [ ] Whether an anonymized code/data supplement is accepted and any separate deadline.
- [ ] Focus-area taxonomy wording (match the CFP's exact labels).

---

# Change log — AAAI-27 RL paper → FAccT 2027 audit (2026-07-13)

Full record of the transition, including technical/code changes. Nothing is committed;
all changes are in the working tree. The AAAI files (`paper/aaai27_equicharge.tex`,
`SUBMISSION.md`) are kept, not deleted.

## New files
| File | Lines | Purpose |
|---|---|---|
| `paper/facct27_equicharge.tex` | 730 | FAccT paper. ACM `acmart` sigconf, double-blind. ~7,300 words, 25 refs, all citations resolve. |
| `experiments/audit_grounding.py` | 215 | A1 grounding sweep (elasticity η + tier↔income coupling ρ). |
| `experiments/results/audit_grounding.json` | — | A1 sweep results. |
| `docs/FACCT_SUBMISSION.md` | this file | FAccT checklist + this change log. |
| `_private_revision_history/RESEARCH_changelog.md` | 26 | Private (gitignored) — revision changelog pulled out of reviewer-visible docs (C3). |

## Modified files (tracked): +345 / −65 lines
`RESEARCH.md`, `chargax/equity/segments.py`, `chargax/equity/tariffs.py`, `docs/DATA.md`,
`experiments/audit_robustness.py`, `experiments/results/audit_robustness.json`.

## Technical changes

### `experiments/audit_grounding.py` (new) — A1 sensitivity run
- **Axis 1, elasticity sweep** η ∈ {0.0..1.0}: derives margins `(y_g/y_median)^η` via
  `SEG.derive_price_multipliers`, re-solves the profit LP per day, records day-averaged
  tier gap, worst-tier `argmin(group_means)`, and revenue PoF `(profit_rev−maximin_rev)/profit_rev`.
- **Axis 2, tier↔income coupling** ρ: mixing model `P(income=m|tier=g)=ρ·1[m=g]+(1−ρ)/3`,
  equal tier priors → confusion matrix `P=(1−ρ)/n·J+ρ·I`, income means `P @ tier_means`,
  which reduces to `income_gap = ρ·tier_gap`, argmin-preserved for ρ>0. Built explicitly
  (`_income_disparity`) and computed numerically, not asserted.
- **Optimizations:** stream extracted once/day (membership is exogenous of η,ρ) then the LP
  re-solved per grounding — guarantees only the grounding varies and skips 10× JAX rollouts;
  maximin PoF read via `_package`'s `revenue_value=Σ delivered·mult` by passing each η's margins.
- **Day filter** matches box-1: skip degenerate days and `min(group_counts)==0` (starvation, not absence).
- **Result (32/32 days):** budget worst 97% across η∈(0,1], gap ~0.556 (invariant), PoF 3.4→13.2%;
  at η=0 gap 0.172 / worst 28% (sanity). Income gap = 0.556·ρ, budget-income worst for all ρ>0.
- **Why invariant:** the profit objective `−Σ mult[g_i]·x_{i,t}` is linear; the optimal vertex
  depends only on tier *ordering*, not margin magnitude — so magnitude moves only the revenue readout.

### `experiments/audit_robustness.py` (+156) — A2 extension
- Signature `audit_config(n_evses,grid_kw,data_kwargs,key)` → `audit_config(env,key)`; new
  `_solve(cust,env,objective,tariff)` calls `solve_oracle_lp` on a pre-extracted stream;
  `main()` builds envs via `_env_for(layout,data)`.
- **Box-2 across configs:** on the same realized days, also solves profit under `T.flat(base)`
  and `T.low_income_subsidy(subsidy_to_parity,base)` with `subsidy_to_parity=max(base[1]−base[0],0)`;
  records disparity + worst-tier per lever in a `tariff_lever` dict.
- **ACN-Data hook** `_acn_config()`: reads `EQUICHARGE_ACN_JSON`, builds a 5th US session-level
  config via `acn_data_kwargs`; skips with a log if unset/unparseable (never fabricates data).
- **Verdict rewrite (fixed a bug):** first-pass binary threshold mis-scored highway/shopping;
  replaced with `_reduction=(a−b)/a` keyed to power-bound sites (`revenue_pof_pct.median>1.0`),
  reporting per-site income-neutral vs budget-subsidy reduction %, `neutral_beats_subsidy` (>5pp),
  and `lever_consistent = all(neutral≥40% AND beats subsidy)` over power-bound sites only.
- **Result:** structural=False (scarcity-specific); lever consistent at power-bound sites=True
  (reference −69% vs −33%; highway −48% vs −16%; workplace/shopping moot).

### `chargax/equity/segments.py` (+92) — grounding constants
- `INCOME_TERCILES` `(30k,67k,130k)` → `(35k,82k,161k)`, solved from `0.6^(1/0.6)=0.427` and
  `1.5^(1/0.6)=1.965` so the median matches ACS-2024 ($81,604) AND `derive_price_multipliers()`
  still returns the committed `(0.6,1.0,1.5)`. **`PRICE_BY_GROUP` verified unchanged** → oracle/
  tariff/RL tables and `results.json` untouched, no re-run.
- `ENERGY_BURDEN` `(0.086,0.045,0.030)` → real `(0.081,0.031,0.023)` (display-only; not in the derivation).
- Docstring rewritten with primary citations (terciles, elasticity range, energy burden, home-charging
  access by tenure/dwelling/income/race, price penalty).

### `chargax/equity/tariffs.py` (+18) — single source of truth
- Added `from chargax.equity import segments as SEG` (no import cycle).
- `BASE_MULTIPLIERS=(0.6,1.0,1.8)` → `= SEG.PRICE_BY_GROUP`. The old `1.8` conflated the ~3×
  energy-*burden* ratio with the WTP *multiplier*; now every tariff magnitude flows from one derivation.

### `RESEARCH.md` (+19) — C3 scrub
- Removed 3 changelog-narrating passages ("earlier version reported…", "only one survived",
  "hypothesis was wrong"); moved to gitignored `_private_revision_history/`.

### `docs/DATA.md` (+19) — grounding paragraph
- Updated stale $30k/$67k/$130k / 8.6% to ACS-2024 + real energy-burden numbers + A1-invariance note.

## Results

**One estimator, one day set.** Every gap below is computed the same way (fixes the earlier
0.537/0.556/0.558 and 0.28/0.374 splits, item 2). The disparate-impact measure is **max−min of
the DAY-AVERAGED per-tier satisfaction** (systematic disadvantage), not the per-day-median gap
(which reads day-to-day noise, e.g. a flat tariff reads ~0.05 day-averaged but ~0.17 per-day).
The reference-site numbers in `audit_mechanism.json` and `audit_robustness.json` now agree
exactly (0.567 / 0.051 / 0.320).

### Canonical mechanism (`results/audit_mechanism.json`, 32/32 days, day-averaged)

**Box-1 oracle** (status-quo margins 0.6/1.0/1.5):

| objective | budget | mid | premium | gap | revenue |
|---|---|---|---|---|---|
| revenue optimal | 0.38 | 0.81 | 0.95 | **0.567** | 390 |
| utilitarian | 0.86 | 0.80 | 0.83 | 0.059 | 348 |
| tier maximin (true) | 0.82 | 0.82 | 0.82 | 0.008 | 352 |
| strict equalization | 0.81 | 0.81 | 0.81 | 0.000 | 349 |

Revenue price of fairness (median): **8.8%**. Same total energy delivered under every objective
(grid binds), so equity costs revenue, not service.

**The mechanism — rank position, not spread magnitude** (`rank_position_beats_spread_magnitude = True`):

| tariff | margins | spread | gap | budget uniquely lowest? | worst tier |
|---|---|---|---|---|---|
| status quo | 0.6/1.0/1.5 | 0.9 | 0.567 | yes | budget |
| premium-capped-to-mid | 0.6/1.0/1.0 | **0.4** | **0.527** | yes | budget |
| budget-tied-to-mid | 1.0/1.0/1.5 | **0.5** | **0.320** | no | **mid** |
| fully flat | 1.0/1.0/1.0 | 0.0 | 0.051 | no | mid |

Takeaway: the premium cap has the **smaller** spread (0.4) yet the **larger** gap (0.527) than
budget-parity (spread 0.5, gap 0.320) — spread magnitude cannot explain that; rank does. The
budget tier is starved exactly while it is the strictly-lowest rank; the worst tier **flips to
mid** once budget escapes uniquely-lowest; only removing the ordering (flat) reaches the floor.

### A1 — elasticity invariance, reframed as structural (not robustness)

Day-averaged gap across the elasticity range, same day set (`audit_mechanism.json`):

| η | margins | gap | budget uniquely lowest? |
|---|---|---|---|
| 0.0 | 1.00/1.00/1.00 | 0.051 | no |
| 0.2 | 0.84/1.00/1.14 | 0.564 | yes |
| 0.6 | 0.60/1.00/1.50 | 0.564 | yes |
| 1.0 | 0.43/1.00/1.96 | 0.565 | yes |

Full 10-point sweep + per-day PoF (3.4%→13.2%) and budget-worst fractions (97%) in
`audit_grounding.json`. Takeaway (item 3): a **flat line is not a sensitivity curve** — it is
the signature of the rank mechanism. Under a linear revenue objective + scarcity, any strict WTP
ordering routes power up the ordering regardless of magnitude, so the harm is a **structural
consequence of ordering + scarcity**, not a calibration artifact. Elasticity moves only the PoF.

Tier↔income coupling (item 4 — **propagation, not evidence**): income-level gap = ρ × tier gap
(0.45 at ρ=0.8, 0.28 at ρ=0.5, 0.19 at ρ=1/3). This is arithmetic translation of an assumed
correlation; the empirical weight sits in the cited access literature (Ge, Bauer, Hsu & Fingerman,
Borlaug), not in the mixing matrix.

### A2 — generalization + scarcity boundary (`results/audit_robustness.json`, day-averaged)

`disparate_impact_structural = False` (scarcity-specific). `tariff_lever_consistent_at_power_bound_sites = True`.
ACN-Data config skipped (`EQUICHARGE_ACN_JSON` unset).

| site | budget-worst (days) | systematic gap | revenue PoF | income-neutral | budget-subsidy | power-bound? |
|---|---|---|---|---|---|---|
| reference (16ch / 30 kW / residential) | 97% | 0.567 | 8.8% | 0.051 (−91%) | 0.320 (−44%) | yes |
| highway (20ch / 45 kW / high-rate) | 78% | 0.220 | 6.1% | 0.012 (−94%) | 0.170 (−23%) | yes |
| workplace (24ch / 55 kW / spread dwell) | 38% | 0.005 | 0.0% | 0.006 | 0.007 | no (moot) |
| shopping (8ch / 16 kW / low-traffic) | 47% | 0.032 | 0.0% | 0.026 | 0.020 | no (moot) |

Takeaway: the boundary separates a **systematic** from a **rotating** disadvantage (see the
stress-test below, which decides whether the day-averaged ~0 is real or an averaging artifact). At
the power-bound sites the harm is systematic (0.567 / 0.220, worst tier = budget on 97% / 78% of
days). Off-boundary the day-averaged gap is ~0 because the worst tier **rotates**, not because the
sites are equal — workplace genuinely has little harm (per-day 0.073) but shopping still carries
substantial **non-systematic** per-day inequality (0.187), a within-population effect, not a
between-tier disparate impact. At both power-bound sites income-neutral pricing removes **>90%** of
the systematic gap and beats the budget-only subsidy (−44%, −23%).

### Estimator stress-test (`results/audit_estimator_check.json`) — required before committing the boundary

Because the estimator and the claim moved together, three checks confirm the day-averaging reveals
structure rather than manufacturing it:

**Check 1 — rotation (is off-boundary ~0 real or averaged away?).** Per-day worst-tier distribution:

| site | per-day gap (median) | day-averaged gap | worst tier | entropy | reading |
|---|---|---|---|---|---|
| reference | 0.558 | 0.567 | budget 97% | 0.13 | **systematic** |
| highway | 0.273 | 0.219 | budget 78% | 0.48 | **systematic** |
| workplace | 0.073 | 0.005 | budget 38% | 0.99 | rotating (harm also *small*) |
| shopping | 0.187 | 0.032 | budget 47% | 0.94 | **rotating (harm substantial, hidden by averaging)** |

Verdict: the day-averaging is correct for a *systematic between-tier* claim, but shopping's
per-day 0.187 is real rotating inequality, not noise. Paper corrected to say "no systematic harm
off-boundary (worst tier rotates); shopping retains substantial non-systematic per-day inequality
= a within-population harm the capacity lever addresses" — not "the harm collapses to ~0."

**Check 2 — flat-tariff floor (real scarcity residual, or sampling noise?).** Flat-tariff
day-averaged gap by day count: 0.051 (32d) → 0.044 (64) → 0.031 (128) → **0.012 (256)** — shrinking
toward **zero**, while the flat *per-day* gap stays ~0.17 and does not shrink. Verdict: the 0.05
"floor" is finite-sample noise; flattening removes the systematic between-tier harm **entirely**.
The ~0.17 that remains is within-population inequality (capacity's job), a distinct harm. Paper's
box-2 and capacity paragraphs corrected: the two levers are complementary (pricing → between-tier,
capacity → within-population), not "pricing leaves a residual only capacity removes."

**Check 3 — the 0.564 vs 0.567 crack.** Confirmed as LP tie-breaking on the eta=0.6 derived margin
(1.499 vs the committed 1.5). Fixed at source: `audit_mechanism.py` now rounds the invariance
margins to the reported 1-decimal precision, so eta=0.6 = the committed reference = 0.567 exactly,
matching the oracle table. A1 table shows 0.563/0.567/0.565 (3 dp) so the flatness is visible
without a rounding-induced false trend.

### Citation corrections (spot-check of 5 more, generalizing the Ermagun mis-attribution risk)
- **Ge 2021 (NREL):** "~25% lack home charging" confirmed; the "~80% charge at home today" is NOT
  confirmed as being in that report → **softened to "the large majority"** in paper + `segments.py`.
- **Borlaug 2026:** $0.42/kWh DCFC confirmed; the $0.53 upper bound and the $0.17 residential are
  NOT in its open text → **paper now cites ~40¢ (Borlaug) and ~17¢ (EIA, new `eia2025` bibitem)**,
  dropping the unverified $0.53.
- Bauer 2021 (all four figures exact), Lou 2024, Yu 2025 (article 5291, already correct) → clean.
- Hsu & Fingerman: reframed to one combined "Black and Hispanic majority" category (round 2).

## Verification
- `pytest tests/test_equity.py` → **27/27 pass** (the `(0.6,1.0,1.8)` fixtures are explicit
  tariff-mechanism inputs, unaffected by the `BASE_MULTIPLIERS` default change).
- `PRICE_BY_GROUP` invariance + `segment_summary` smoke tests.
- A1, A2, the mechanism run, and the estimator stress-test re-run to completion (LP-only, CPU).
- Paper static check: **26** `\bibitem`, `\cite`↔`\bibitem` equal both ways, refs↔labels resolve,
  all environments balanced, no stale numbers.

---

# Change log — round 3: stress-testing the estimator switch (2026-07-13)

The round-2 estimator switch (per-day-median → day-averaged) moved the ruler and the claim at
once, so the boundary and floor numbers were adopted, not tested. New `audit_estimator_check.py`
runs three checks on a fixed day set; results and paper corrections above (Results → stress-test).
Outcome: the mechanism stands, but two headline claims were overstated and are now corrected.

- **Scarcity boundary refined.** Off-boundary day-averaged ~0 is *rotation*, not absence. Workplace
  harm is genuinely small (per-day 0.073); shopping has substantial *non-systematic* per-day
  inequality (0.187) that averaging hid. Paper now claims "no *systematic* between-tier harm
  off-boundary" and names the shopping inequality as a within-population harm (capacity's domain).
- **Flat-tariff floor corrected.** The 0.05 floor is finite-sample noise (→0.012 at 256 days, still
  shrinking), not a scarcity residual. Flattening removes the between-tier harm *entirely*; the
  residual ~0.17 per-day inequality is within-population (capacity). Box-2 + capacity paragraphs
  rewritten: the two levers are complementary across two harms, not redundant on one.
- **0.564/0.567 crack fixed at source.** LP tie-breaking on 1.499 vs 1.5; `audit_mechanism.py` now
  rounds invariance margins to 1 dp so η=0.6 == committed reference == 0.567. A1 table at 3 dp.
- **Five more citations spot-checked.** Ge "~80% at home" softened (not in the cited report);
  Borlaug price split into ~40¢ DCFC (Borlaug) + ~17¢ residential (new `eia2025`), dropping the
  unverified $0.53. Bauer/Lou/Yu clean.

New file: `experiments/audit_estimator_check.py` (+150) → `results/audit_estimator_check.json`.

---

# Change log — round 4: lever cross-effects + full bibliography audit (2026-07-13)

Two verification jobs the round-3 corrections newly required.

## Lever orthogonality — the complementary-levers headline needed a number (`results/audit_levers.json`)
`audit_levers.py` isolates each lever on one fixed 32-day stream set at the reference site.
Metrics: between-tier gap (day-averaged) and within-tier Gini (within-population inequality).

| cell | between-tier gap | within-tier Gini |
|---|---|---|
| status-quo, 30 kW | 0.567 | 0.270 |
| flat, 30 kW | 0.051 | 0.205 |
| status-quo, 90 kW | 0.020 | 0.042 |

- **Pricing** (status-quo→flat @ 30 kW): between −0.516, within **−0.065** → targeted; barely moves within-population.
- **Capacity** (30→90 kW @ status-quo): between **−0.547**, within −0.228 → moves **both**.

**Verdict: the levers are NOT orthogonal** (`orthogonal = false`). The round-3 "two levers for two
distinct harms" framing was itself wrong. Corrected framing (now in the paper, with these numbers):
capacity attacks the root cause and reduces **both** harms; pricing removes the between-tier
disparate impact **specifically** and leaves the within-population inequality that only capacity
reaches. Asymmetric, not orthogonal, not redundant. Fixed in the capacity paragraph, box-2, and the
conclusion; the "second, independent lever" header/line removed.

## Full bibliography audit (all remaining 15 refs, not the flagged subset)
Per the ~50% inspection error rate, every remaining citation was verified against primary
sources/Crossref. Result: **14/15 correct, 1 real error.**
- **Sovacool & Dworkin — year is 2014, not 2016.** Fixed (key renamed `sovacool2016`→`sovacool2014`,
  displayed year 2014). This is the kind of silent error the audit was for.
- Lee et al. 2019 enriched with pages (139–149) + DOI 10.1145/3307772.3328313.
- Schulte & Heindl: ~0.4 confirmed (note: it is an *expenditure* elasticity of residential energy;
  still <1, necessity framing holds). Khan (venue at-risk) confirmed. Atkinson, Bednar, Borenstein,
  Burger, Caulfield, Jenkins, Mo & Walrand, Moulin, Ponse, Rawls, Sen all correct.

New file: `experiments/audit_levers.py` (+140) → `results/audit_levers.json`.

## Standing rule adopted
Soften every number to what its source actually supports, preemptively, not reactively. Verified
the whole bibliography rather than the flagged subset (prior confidence was not calibrated: 3 of the
first 8 spot-checks and 1 of the remaining 15 carried an error).

## Figures regenerated (were stale — a real gap)
The two paper figures (`paper/figures/oracle_tradeoff.png`, `rate_design.png`) had NOT been
re-run since Jul 8 and were built by `plot_results.py` from the RL-era `results.json`, so they
showed pre-mechanism numbers and `rate_design.png` encoded the deleted "spread" framing. New
`experiments/plot_mechanism.py` regenerates both from the canonical `audit_mechanism.json` (same
day set / estimator as every table), reusing the shared `plot_style.py`:
- `oracle_tradeoff.png` → canonical box-1 (budget 0.38 / mid 0.81 / premium 0.95, gap 0.57, −9% PoF).
- `rate_design.png` → **redesigned** as the rank mechanism (gap vs spread per tariff; the premium
  cap's smaller spread 0.4 sits above the subsidy's spread 0.5; color = is budget uniquely lowest).
Verified visually and copied into `paper/figures/`. (The legacy `plot_results.py` was later removed
in the round-5 cleanup; all FAccT figures come from `plot_mechanism.py`.)

**Restyle (palette + less text).** The default blue+orange (Okabe-Ito/Blues, in every ML paper)
was replaced with a distinctive **teal + clay** system, validated with the dataviz palette checker
rather than eyeballed: the tier ordinal ramp (`#66BDB2`→`#2E9284`→`#124F49`) passes the ordinal
checks, and clay `#BF5A38` vs teal `#12A08A` passes CVD ΔE 29.9. Teal = tiers/relief, clay =
cost/harm across both figures. In-figure text was cut to the minimum (removed panel titles, the
gap and −9% arrows, inline spread labels, and the callout); those numbers and the rank-vs-spread
argument now live in the figure captions and body text. Legends and axis labels kept (identity is
never color-alone). Both re-rendered and visually verified.

**Three figures added (2 → 5).** The paper was under-illustrated for an audit; three claims that
lived only in tables/prose now have figures (all from `plot_mechanism.py`, teal/clay, minimal text,
numbers in captions), each visually verified:
- `grounding_invariance.png` (Fig. `fig:grounding`): day-averaged gap vs η — a flat line at ~0.57
  across all η>0 with a cliff to 0.05 at η=0. *Is* the "structural, not calibration artifact"
  argument. (Expanded `audit_mechanism.py`'s invariance sweep to 11 η points for a smooth line.)
- `scarcity_boundary.png` (Fig. `fig:scarcity`): per-site systematic (day-averaged) gap vs typical
  per-day gap. Where equal → systematic disparate impact (reference, highway); where per-day >>
  systematic → rotating within-population inequality (shopping). Visualizes the construct-validity
  distinction.
- `lever_cross_effects.png` (Fig. `fig:levers`): reduction in between-tier vs within-population harm
  under pricing vs capacity — pricing's within-population bar is tiny, capacity's is substantial →
  the levers are not orthogonal.
All five figures are cross-referenced in the text; captions carry the numbers.

**Still worth adding but needs a run:** a capacity-threshold figure (gap/inequality vs grid kW).
The paper's ~50–70 kW claim currently traces to the stale `results.json` and has NOT been
regenerated this session — a fresh kW sweep would make the figure AND verify that number. Flagged,
not yet done.

---

# Change log — round 2: the rank mechanism (2026-07-13)

Reviewer feedback sharpened the box-2 mechanism from "harm scales with the margin spread" to
"harm tracks the tiers' rank ordering by price," and required one consistent day set/estimator.

## New file
| File | Purpose |
|---|---|
| `experiments/audit_mechanism.py` (+215) | Canonical run: box-1, elasticity invariance, and the four-tariff rank test on ONE fixed 32-day set, day-averaged estimator. Emits `results/audit_mechanism.json`. |

## What changed and why
1. **Mechanism established (item 1).** The four-tariff run proves rank beats spread: premium-cap
   (spread 0.4) keeps gap 0.527 while budget-parity (spread 0.5) drops to 0.320 — smaller spread,
   larger gap. Box-2 rewritten around rank position; every "harm scales with the spread" sentence
   deleted from abstract, intro, contributions, findings, path-to-practice, ethics, conclusion.
2. **Numbers reconciled (item 2).** `audit_robustness.py` switched to the day-averaged estimator
   (was per-day median); reference gaps now match the mechanism run exactly (0.567 / 0.051 /
   0.320). One day set, one estimator across box-1, elasticity, and tariff levers.
3. **A1 reframed as structural, not robustness (item 3).** Paper §"disparate impact is structural";
   the flat η-line is presented as the mechanism's signature, not a passed sensitivity check.
4. **ρ demoted to propagation (item 4).** Paper §"a translation and not a finding"; empirical
   weight moved to the cited access literature.
5. **Derivation order fixed (item 5).** Multipliers presented as the *output* of ACS terciles +
   elasticity; noted that only the ordering matters so magnitudes are not load-bearing.
6. **Citations verified (item 6).** See the grounding checklist item above.
7. **Scarcity boundary sharpened (strategic).** Day-averaged A2 shows the *systematic* gap is
   large only at power-bound sites; off-boundary the worst tier rotates. (Refined in round 3 after
   the stress-test — see below — to distinguish rotating from absent harm.)

## Table/number updates in `paper/facct27_equicharge.tex`
- Oracle table → canonical (profit 0.38/0.81/0.95 gap 0.57 rev 390; util 0.06; maximin 0.01).
- Tariff table → rank framing with a `spread` column (status-quo 0.57, premium-cap 0.53,
  budget-subsidy 0.32 worst=mid, flat 0.05).
- Grounding table → η=0 gap 0.17→0.05 (day-averaged, now consistent with box-2 flat).
- Hsu & Fingerman phrasing → single combined "Black and Hispanic majority" category.

---

# Change log — round 5: cleanup, capacity figure, prose pass (2026-07-13)

## Repository cleanup (superseded AAAI artifacts removed, git-tracked so recoverable)
Removed: `paper/aaai27_equicharge.tex`, `docs/SUBMISSION.md`, `RESEARCH.md`, `paper/tables/` (the
three auto-generated RL tables), the AAAI-only figures (`group_means`, `learning_curve`,
`pareto_front`, `robustness_sites`, `scarcity_contrast` under `paper/figures/` and
`experiments/results/figures/`), and the now-dead scripts `experiments/plot_results.py` and
`experiments/make_paper_tables.py` (nothing imported them). `README.md` links repointed to the
FAccT paper + this file, and its reproduce block updated to the `audit_*` + `plot_mechanism`
scripts. **Kept** (load-bearing): all `chargax/` code, the RL `train_*.py` and results JSONs (the
released benchmark the appendix reports), `experiments/` audit/run scripts, `docs/DATA.md`, tests.

## Capacity threshold verified + figure (`results/audit_capacity.json`, `fig:capacity`)
The ~50–70 kW claim traced to the stale `results.json`, so it was recomputed fresh on the oracle
(same day set). New `experiments/audit_capacity.py` sweeps grid kW and reports where each
inequality closes 90% of its distance to the abundant floor. Result: within-population Gini by
**~50 kW** (matches the old claim) but the between-tier gap by **~60 kW, not ~70**. Paper corrected
(the only occurrence) and `fig:capacity` added. This is the sixth figure.

## Six figures total
`oracle_tradeoff`, `rate_design`, `grounding_invariance`, `scarcity_boundary`, `capacity_threshold`,
`lever_cross_effects` — all from `plot_mechanism.py`, teal/clay, minimal text, all cross-referenced.

## Prose pass
Verified the running prose uses no em-dashes, no colons, and no semicolons (fixed one heading colon
and one caption semicolon; cleaned the header comments). The only remaining colons/en-dashes are
inside published titles and page ranges in the bibliography, which must stay. Register kept plain
and academic; no ornate words. Paper is `paper/facct27_equicharge.tex` (~8,500 words, 26 refs,
6 figures, 3 tables), structure resolves clean, tests 27/27.

---

# Change log — round 6: capacity figure on the deployed-policy Gini (2026-07-14)

The capacity figure (`fig:capacity`) was rebuilt to use the **deployed max-charge policy Gini**
(inclusive, rejections at zero) for the within-population line, instead of the oracle Gini, so it
reports the inequality a driver actually experiences. `audit_capacity.py` now also runs a
32-episode max-charge rollout at a station built for each grid kW.

Honest result (and a corrected number): both curves flatten by **~60 kW** (the oracle within-pop
Gini figure had said ~50). The between-tier gap (oracle) → ~0, so capacity removes the disparate
impact. But the **max-charge Gini plateaus at a nonzero floor ~0.30**, not zero, because a larger
grid connection removes power rationing and does not add chargers — the residual is drivers
rejected when all 16 chargers are busy. The oracle reaches far lower with the same power, so that
floor is an online-to-offline / charger-provisioning gap (ties to the RL appendix), not a capacity
limit. Capacity paragraph + caption rewritten to say this plainly.

Consistency: the lever figure keeps the **oracle** within-tier Gini (relabeled "within-tier Gini
(oracle)"); the capacity figure uses the **max-charge** Gini ("within-population Gini
(max-charge)"). Distinct instruments, distinct labels, no same-name-two-values collision. Six
figures, prose still free of em-dashes/colons/semicolons, structure resolves clean.
