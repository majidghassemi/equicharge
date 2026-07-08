# AAAI-27 AI for Social Impact (AISI) — submission checklist

This tracks the format and logistics items (Tier 6 of the revision plan). Verify each
against the live AAAI-27 Author Kit and CFP before submitting — dates and page limits
below are from the plan and **must be confirmed** on the official pages.

## Format
- [x] Paper is in AAAI two-column format (`paper/aaai27_equicharge.tex`, `\documentclass[letterpaper]{article}` + `\usepackage{aaai2027}`).
- [ ] Drop the official **`aaai2027.sty`** and **`aaai.bst`** from the AAAI-27 Author Kit next to the `.tex` and compile with `pdflatex` twice. (Not bundled here; download from the Author Kit.)
- [ ] Confirm the exact **page limit** in the AAAI-27 Author Kit (plan estimate ≈7 pages + references, appendix allowed). Method detail and extended results are written to be movable to an appendix if the body runs long — the body reserves space for impact framing, the grounded setup, and the policy findings.
- [x] Figures are self-contained PNGs under `paper/figures/`; the wide oracle figure uses `figure*` (spans both columns).
- [x] The learned-benchmark table is auto-generated (`experiments/make_paper_tables.py` → `paper/tables/learned_table.tex`) so paper numbers match the released run exactly.

## Anonymization (double-blind)
- [x] Author block is anonymized (`\author{Anonymous Submission}`).
- [x] No author-identifying content in the code release (repo ships the bare simulator + the `chargax.equity` extension; no names/affiliations in the equity code).
- [ ] Before uploading a code/data supplement, scrub git history, `pyproject.toml` author fields, and any dataset acknowledgements for identifying info. The upstream Chargax citation is fine (it is prior published work, not self-identifying).
- [ ] Ensure the arXiv/GitHub links in the paper are anonymized or omitted for review.

## Track and keywords
- [ ] Select the **AI for Social Impact (AISI)** special track (there is **no transfer** between the main track and AISI — committing is final for this cycle).
- [ ] Select AISI keywords: **Energy**, **Mobility and Transportation**, **Social Welfare / Justice / Fairness / Equality**.

## Deadlines (CONFIRM on the CFP)
- [ ] Abstract deadline (plan: late July 2026 for AAAI-27 — confirm exact AISI date, which can differ from the main site).
- [ ] Full-paper deadline (plan: late July 2026 — confirm).
- [ ] Supplementary/code deadline, if separate.

## Reproducibility supplement
- [x] One-command RL-free core: `python -m experiments.run_experiments --oracle-only`.
- [x] Full study: `python -m experiments.run_experiments` (multi-seed) and `--quick` smoke.
- [x] Regeneration: `summarize_results`, `plot_results`, `make_paper_tables`.
- [x] Data provenance documented (`docs/DATA.md`); tests: `pytest tests/test_equity.py -q`.
- [x] Seeds are fixed in the experiment configuration (`experiments/run_experiments.py`).
