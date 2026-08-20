"""Compose the figures into LaTeX-style grids and render at TRUE 100% zoom (96 dpi).

Generate the per-slot variants first, then point this at their parent directory:

    python -m experiments.plot_mechanism --across 2 --outdir /tmp/fig/g2
    python -m experiments.plot_mechanism --across 3 --outdir /tmp/fig/g3
    python docs/figure-proofs/render_grid_proof.py /tmp/fig
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.image as mpimg, textwrap, sys

DPI, PW = 96, 8.5
TEXTW, GUT = 7.0027, 0.20
LEFT = (PW - TEXTW) / 2
SCR = sys.argv[1] if len(sys.argv) > 1 else "."   # dir holding the g2/ and g3/ variants

def build(rows, title, ph, ncols):
    fig = plt.figure(figsize=(PW, ph), dpi=DPI); fig.patch.set_facecolor("white")
    X = lambda i: i / PW
    Y = lambda i: 1.0 - i / ph
    y = 0.55
    fig.text(X(LEFT), Y(0.30), title, fontsize=11, family="serif", weight="bold")
    tag = iter("abcdefghijkl")
    for src, stems in rows:
        # width comes from the GRID, not from how many cells this row happens to hold,
        # so a short final row keeps the same cell size as a full one
        w = (TEXTW - (ncols - 1) * GUT) / ncols
        hmax = 0
        for k, stem in enumerate(stems):
            img = mpimg.imread(f"{SCR}/{src}/{stem}.png")
            h = w * img.shape[0] / img.shape[1]
            x = LEFT + k * (w + GUT)
            ax = fig.add_axes([X(x), Y(y + h), w / PW, h / ph])
            ax.imshow(img, interpolation="lanczos"); ax.axis("off")
            cap = f"({next(tag)}) " + stem.replace("_", " ")
            fig.text(X(x), Y(y + h + 0.15), cap, fontsize=8, family="serif", va="top")
            hmax = max(hmax, h)
        y += hmax + 0.44
    fig.text(X(LEFT), Y(y + 0.02),
             "Rendered at 96 dpi = 100% zoom. Sub-captions are 8pt, matching acmart.",
             fontsize=8, family="serif", color="#555555", va="top")
    out = f"{SCR}/{title.split()[0].lower()}_grid.png"
    fig.savefig(out, dpi=DPI, facecolor="white"); print("wrote", out, "->", y + 0.4, "in used")
    return out

FIVE = ["rate_design", "scarcity_boundary", "grounding_invariance",
        "capacity_threshold", "lever_cross_effects"]
build([("g2", FIVE[:2]), ("g2", FIVE[2:4])], "Two across \\textwidth (slot 3.40in)", 6.5, 2)
build([("g3", FIVE[:3]), ("g3", FIVE[3:5])], "Three across \\textwidth (slot 2.20in)", 4.6, 3)
