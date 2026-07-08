r"""Shared publication style for the paper figures (top-ML-conference look).

Design goals, in order:
1. **Legible when small.** Large fonts, thick 2.6pt trend lines, >=9pt markers, high DPI,
   so trends survive being scaled down in a two-column layout.
2. **Careful, colorblind-safe color** (validated with the dataviz skill's checker):
   - ordered tiers -> a single-hue BLUE ramp (order encoded by lightness, CVD-safe):
     budget #6baed6 -> mid #2171b5 -> premium #08306b (ordinal checks pass; light end
     clears the surface contrast floor).
   - two-line comparisons -> BLUE #0072B2 + VERMILLION #D55E00 (CVD dE ~92, both >=3:1
     contrast) -- the safest high-contrast categorical pair.
   - single-series / reference -> neutral grays.
3. **Recessive non-data ink.** Only left+bottom spines, y-only grid behind the data,
   direct labels instead of dense legends where it declutters.

Colors are assigned by the *entity* (a tier keeps its color across every figure), never
by rank, so the figures read as one system.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# --- validated palette (see module docstring; checked with validate_palette.js) ---
TIER = {"budget": "#6baed6", "mid": "#2171b5", "premium": "#08306b"}
TIER_ORDER = ["budget", "mid", "premium"]
BLUE = "#0072B2"        # primary series / learned / focus
VERMILLION = "#D55E00"  # contrasting series / "cost" / oracle-optimum
GREEN = "#009E73"       # third categorical series when needed
INK = "#1a1a1a"         # primary text
MUTED = "#6b6b6b"       # secondary text / reference lines
GRID = "#e7e7e7"
REF = "#9a9a9a"         # baseline / reference lines (gray)


def apply_style():
    mpl.rcParams.update({
        "figure.dpi": 400, "savefig.dpi": 400,
        "font.family": "sans-serif",
        "font.sans-serif": ["Nimbus Sans", "Liberation Sans", "DejaVu Sans"],
        "font.size": 13,            # large base -> survives downscaling
        "axes.titlesize": 13, "axes.labelsize": 13,
        "xtick.labelsize": 11.5, "ytick.labelsize": 11.5,
        "legend.fontsize": 11, "legend.title_fontsize": 11.5,
        "axes.edgecolor": "#4d4d4d", "axes.linewidth": 1.1, "text.color": INK,
        "axes.labelcolor": INK, "xtick.color": "#4d4d4d", "ytick.color": "#4d4d4d",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True,
        "grid.color": GRID, "grid.linewidth": 0.9,
        "lines.linewidth": 2.6, "lines.markersize": 9,
        "lines.markeredgewidth": 1.4,
        "patch.linewidth": 0.0,
        "legend.frameon": False, "legend.handlelength": 1.5, "legend.columnspacing": 1.2,
        "figure.constrained_layout.use": True,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
        "axes.xmargin": 0.02, "axes.ymargin": 0.04,
    })


def clean(ax, ygrid=True):
    """Recessive axes: no top/right spine, y-only grid, outward light ticks."""
    ax.grid(axis="y" if ygrid else "both", which="major")
    ax.grid(axis="x", visible=False)
    ax.tick_params(direction="out", length=3.5, width=1.0)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#4d4d4d")
    return ax


def label_bars(ax, bars, fmt="%.2f", dy=0.0, fs=10.5, color=INK):
    """Direct value labels above bars (relief for low-contrast fills + legibility)."""
    for b in bars:
        h = b.get_height()
        ax.annotate(fmt % h, (b.get_x() + b.get_width() / 2, h + dy),
                    ha="center", va="bottom", fontsize=fs, color=color)


def savefig(fig, path):
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)
