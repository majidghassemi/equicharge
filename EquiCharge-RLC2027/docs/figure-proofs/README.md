# Figure proofs

Renders of the paper figures at **96 dpi = 100% zoom**, so the type sizes here are the sizes
a reviewer sees on screen. Body text in the proofs is 9pt and captions are 8pt, matching
`acmart[sigconf]`, which is what the figure type should be compared against.

| file | what it shows |
|---|---|
| `before-100pct.png` | the figures as they were: drawn 1.5-2.1x wider than their slot, so LaTeX shrank them and axis labels landed at 4.5-6.4pt |
| `after-100pct.png` | the same page with every figure drawn at its printed width (scale 1.0), nothing below 7pt |
| `grid-2across-100pct.png` | two figures side by side across `\textwidth` (3.40in cells) |
| `grid-3across-100pct.png` | three across `\textwidth` (2.20in cells), with the reduced labelling that width requires |

## Regenerating

```sh
python -m experiments.plot_mechanism                      # single column (what the paper uses)
python -m experiments.plot_mechanism --across 2 --outdir /tmp/fig/g2
python -m experiments.plot_mechanism --across 3 --outdir /tmp/fig/g3

python docs/figure-proofs/render_proof_sheet.py after-100pct.png
python docs/figure-proofs/render_grid_proof.py /tmp/fig
```

## Guard

`experiments/check_figure_legibility.py` enforces the invariants these proofs illustrate:
scale is exactly 1.0, no text below 7pt, no tick-label collisions, nothing spilling the
canvas. Run it at the slot width you actually compose into:

```sh
python -m experiments.check_figure_legibility --across 3
```
