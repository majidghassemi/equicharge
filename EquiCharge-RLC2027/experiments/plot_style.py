r"""Shared publication style for the paper figures, sized for the PRINTED page.

The governing rule is **draw at print size**. A figure is created at exactly the width it
will occupy in the two-column acmart ``sigconf`` layout and included with a matching
``width=`` in the LaTeX, so the scale factor is 1.0 and a point in matplotlib is a point on
the page. Everything else follows from that:

1. **Typography is chosen in final points, not in "big enough to survive shrinking".**
   acmart sigconf sets body at 9pt and captions at 8pt, so figure text sits in a 7.5-8.5pt
   band, never below 7pt. Because nothing is rescaled, those numbers are what the reader
   actually gets at 100% zoom.
2. **No ``bbox="tight"``.** Tight cropping trims the canvas to the drawn content, so the
   saved width no longer equals the requested width and ``width=\columnwidth`` silently
   rescales by an amount that differs per figure. Constrained layout fits the content
   *inside* a fixed canvas instead, which keeps the 1:1 guarantee.
3. **Ink weights are set for a 3.3in canvas**, not a 7in one: ~1.7pt trend lines, ~4.5pt
   markers, hairline spines and grid. Drawn 1:1 these print heavier than the old 2.6pt
   lines did after a 1.5-2x downscale, while the non-data ink stays recessive.
4. **Careful, colorblind-safe color** (validated with the dataviz palette checker):
   a teal ordinal ramp for the ability-to-pay tiers (order encoded by lightness), clay for
   the cost/harm series, teal for relief. Clay vs teal passes CVD dE 29.9.

Colors are assigned by the *entity* (a tier keeps its color across every figure), never by
rank, so the figures read as one system.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# --- printed geometry of acmart[sigconf] -------------------------------------------------
# \the\columnwidth = 241.14749pt, \the\textwidth = 506.295pt, at 72.27pt/in.
COL = 3.34   # single-column figure  -> \includegraphics[width=\columnwidth]
FULL = 7.00  # two-column figure*    -> \includegraphics[width=\textwidth]
GUTTER = 0.20  # horizontal space between panels of a composed grid


def slot(n_across, total=FULL, gutter=GUTTER):
    """Width of one cell when n_across figures are composed side by side.

    A figure dropped into a grid gets a FRACTION of the text width, and if it was drawn for
    the full width LaTeX shrinks it and the type shrinks with it. Drawing at slot() instead
    keeps the placement at scale 1.0 whatever the grid is.

        slot(1)            -> 7.00in   one figure spanning \textwidth
        slot(2)            -> 3.40in   two across \textwidth (~ a single column)
        slot(3)            -> 2.20in   three across \textwidth
        slot(2, total=COL) -> 1.57in   two inside ONE column: too small, see NARROW below
    """
    return (total - (n_across - 1) * gutter) / n_across


# Below this width a chart cannot keep full-length tick labels, a legend and a dense tick
# sequence at >=7pt, so the plots trade them for short labels and fewer ticks.
NARROW = 2.60

# --- validated palette (see module docstring) --------------------------------------------
TIER = {"budget": "#66BDB2", "mid": "#2E9284", "premium": "#124F49"}  # light -> dark
TIER_ORDER = ["budget", "mid", "premium"]
HARM = "#BF5A38"    # clay: the cost / the budget tier still lowest-ranked
RELIEF = "#12A08A"  # teal: relief / the budget tier rescued from lowest rank
INK = "#1a1a1a"     # primary text
MUTED = "#5f5f5f"   # secondary text / reference lines
GRID = "#e3e3e3"
AXIS = "#595959"

# Point sizes, in FINAL printed points (scale is 1.0, so these are literal).
FS_TICK = 7.5
FS_LABEL = 8.5
FS_ANNOT = 7.5
FS_LEGEND = 7.5


def apply_style():
    mpl.rcParams.update({
        "figure.dpi": 400, "savefig.dpi": 400,
        "font.family": "sans-serif",
        "font.sans-serif": ["Nimbus Sans", "Liberation Sans", "DejaVu Sans"],
        "font.size": FS_TICK,
        "axes.titlesize": FS_LABEL, "axes.labelsize": FS_LABEL,
        "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_LEGEND, "legend.title_fontsize": FS_LEGEND,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.7, "text.color": INK,
        "axes.labelcolor": INK, "xtick.color": AXIS, "ytick.color": AXIS,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True,
        "grid.color": GRID, "grid.linewidth": 0.6,
        "lines.linewidth": 1.7, "lines.markersize": 4.5,
        "lines.markeredgewidth": 0.8,
        "patch.linewidth": 0.0,
        "legend.frameon": False, "legend.handlelength": 1.1,
        "legend.handletextpad": 0.5, "legend.columnspacing": 1.0,
        "legend.labelspacing": 0.35, "legend.borderaxespad": 0.2,
        # Fixed canvas + constrained layout: the saved PNG is exactly figsize wide, so
        # width=\columnwidth in the .tex is a 1:1 placement and never rescales the text.
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.015,
        "figure.constrained_layout.w_pad": 0.015,
        "savefig.bbox": "standard", "savefig.pad_inches": 0.0,
        "axes.xmargin": 0.02, "axes.ymargin": 0.04,
    })


def clean(ax, ygrid=True):
    """Recessive axes: no top/right spine, one-direction grid, short outward ticks."""
    ax.grid(axis="y" if ygrid else "x", which="major")
    ax.grid(axis="x" if ygrid else "y", visible=False)
    ax.tick_params(direction="out", length=2.2, width=0.7, pad=2.0)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    return ax


def label_bars(ax, bars, fmt="%.2f", dy=0.0, fs=FS_ANNOT, color=INK):
    """Direct value labels above bars (relief for low-contrast fills + legibility)."""
    for b in bars:
        h = b.get_height()
        ax.annotate(fmt % h, (b.get_x() + b.get_width() / 2, h + dy),
                    ha="center", va="bottom", fontsize=fs, color=color)


def label_line(ax, x, y, text, color, dx=0.0, dy=0.0, ha="left", va="bottom"):
    """Direct label on a curve, which reads faster than a legend and costs no plot width.

    A halo keeps it legible where it crosses the grid or another series."""
    import matplotlib.patheffects as pe
    t = ax.annotate(text, (x, y), (dx, dy), textcoords="offset points",
                    fontsize=FS_ANNOT, color=color, ha=ha, va=va, zorder=6)
    t.set_path_effects([pe.withStroke(linewidth=2.0, foreground="white")])
    return t


def savefig(fig, path):
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)
