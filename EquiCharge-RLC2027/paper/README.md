# Manuscript

The live FAccT 2027 manuscript ("Who Charges Last?") is maintained in Overleaf, not here.

`archive_facct27_equicharge_2026-08.tex` is the August 2026 version that used to live in this
folder. It is kept so `make gate` and `make paper` still run, but it is superseded: it predates
the battery-free framing, the optimal-face and storage tables, the priority-neutrality remedy,
and the ACN-Data site.

`figures/` holds the figures the manuscript uses. `make figures` regenerates them from
`experiments/results/` and syncs them here.

To check the live manuscript against the released results, export the `.tex` from Overleaf and run

    make manuscript-check TEX=path/to/main.tex

which runs `experiments/check_manuscript_numbers.py` (the values the current text prints,
recomputed from the result files) and the general numeral gate.
