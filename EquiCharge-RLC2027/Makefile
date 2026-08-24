# EquiCharge: reproduce the audit, check the paper, build the PDF.
#
# Every target is CPU-only and deterministic except `learn`, which trains PPO.
# Results land in experiments/results/, figures in experiments/results/figures/ and
# are synced into paper/figures/ by the figure target.

PYTHON ?= python
LATEX  ?= pdflatex
#: Days in each realized day set. Every audit script honours EQUICHARGE_N_DAYS, so
#: `make audit DAYS=256` re-runs the whole suite at a larger horizon.
DAYS   ?=
export EQUICHARGE_N_DAYS = $(DAYS)
BIBTEX ?= bibtex
PAPER  ?= facct27_equicharge

.PHONY: all audit theory gate figures test learn acn paper paper-clean clean help

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | sed 's/:.*## /\t/' | column -t -s "$$(printf '\t')"

all: audit theory figures gate test  ## the full RL-free reproduction, then the checks

audit:  ## the offline audit suite (oracle, tariffs, robustness, grounding, levers)
	$(PYTHON) -m experiments.run_experiments --oracle-only
	$(PYTHON) -m experiments.audit_mechanism
	$(PYTHON) -m experiments.audit_robustness
	$(PYTHON) -m experiments.audit_grounding
	$(PYTHON) -m experiments.audit_levers
	$(PYTHON) -m experiments.audit_capacity
	$(PYTHON) -m experiments.audit_offline
	$(PYTHON) -m experiments.audit_estimator_check
	$(PYTHON) -m experiments.audit_scarcity_severity
	$(PYTHON) -m experiments.summarize_results

theory:  ## the structural results: chain identity, optimal-face bounds, online greedy
	$(PYTHON) -m experiments.verify_theory
	$(PYTHON) -m experiments.audit_optimal_face
	$(PYTHON) -m experiments.audit_margin_greedy

figures:  ## regenerate every figure and sync it into paper/figures/
	$(PYTHON) -m experiments.plot_mechanism

gate:  ## pre-submission check that every number in the paper comes from a released run
	$(PYTHON) -m experiments.check_paper_numbers

test:  ## unit tests
	$(PYTHON) -m pytest tests/test_equity.py -q

learn:  ## the multi-seed learning benchmark (slow, GPU-friendly)
	$(PYTHON) -m experiments.run_experiments

acn:  ## the ACN-Data (US) re-run; needs EQUICHARGE_ACN_JSON to point at a sessions dump
	@test -n "$$EQUICHARGE_ACN_JSON" || { echo "set EQUICHARGE_ACN_JSON first"; exit 1; }
	$(PYTHON) -m experiments.audit_robustness
	$(PYTHON) -m experiments.audit_mechanism --acn
	$(PYTHON) -m experiments.audit_estimator_check --acn
	$(PYTHON) -m experiments.run_experiments --oracle-only --acn

paper: gate  ## build paper/$(PAPER).pdf (runs the number gate first)
	@command -v $(LATEX) >/dev/null || { \
	  echo "$(LATEX) not found. Install TeX Live with the acmart class, e.g."; \
	  echo "  sudo apt-get install texlive-full   (or texlive-publishers for acmart alone)"; \
	  exit 1; }
	cd paper && $(LATEX) -interaction=nonstopmode -halt-on-error $(PAPER).tex
	-cd paper && $(BIBTEX) $(PAPER)
	cd paper && $(LATEX) -interaction=nonstopmode -halt-on-error $(PAPER).tex
	cd paper && $(LATEX) -interaction=nonstopmode -halt-on-error $(PAPER).tex
	@echo "built paper/$(PAPER).pdf"
	@grep -c "Overfull" paper/$(PAPER).log | xargs -I{} echo "overfull boxes: {}"

paper-clean:  ## remove LaTeX build artifacts
	cd paper && rm -f *.aux *.bbl *.blg *.log *.out *.toc

clean: paper-clean  ## remove build artifacts
