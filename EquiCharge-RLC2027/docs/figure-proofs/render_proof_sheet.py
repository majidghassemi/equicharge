"""Composite the figures onto a simulated acmart[sigconf] page at TRUE 100% zoom.

Run from the repository root: python docs/figure-proofs/render_proof_sheet.py out.png

Renders at 96 dpi, which is what a PDF viewer at 100% maps a typographic point to on a
standard display, so the pixel sizes here are the sizes a reviewer actually sees. Body text
is set at 9pt and captions at 8pt, matching acmart, so the figure type can be compared
against the prose right next to it.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import textwrap, sys

DPI = 96                       # 100% zoom
PW, PH = 8.5, 15.6             # Letter width; taller canvas so all six fit on one sheet
TEXTW, COLW = 7.0027, 3.3354
COLSEP = TEXTW - 2 * COLW
LEFT = (PW - TEXTW) / 2
TOP = 1.0
FIG = "experiments/results/figures/%s.png"

BODY = ("This is the central and most policy-relevant finding, and it corrects a natural but "
        "wrong intuition, that the harm scales with how far apart the tier margins are. It does "
        "not. Table 3 and Figure 4 recompute the profit-optimal allocation under four tariffs on "
        "one fixed day set, and the pattern is governed by rank, not spread. Under the revenue "
        "objective the operator serves tiers in willingness-to-pay order, so the budget tier is "
        "systematically starved exactly while it is the strictly lowest-paying rank, and by how "
        "much it is lowest does not enter. The two intuitive partial fixes then behave "
        "asymmetrically in a way spread magnitude cannot explain. Capping the premium tier down "
        "to the middle margin compresses the spread to 0.4 but leaves the budget tier still "
        "uniquely lowest, so the gap barely moves, from 0.57 to 0.53, and the budget tier stays "
        "worst-served. Subsidizing the budget tier up to parity with the middle compresses the "
        "spread only to 0.5, a larger spread than the premium cap, yet the gap drops to 0.32, "
        "because the budget tier is no longer uniquely lowest. The smaller-spread intervention "
        "leaves the larger harm. Because both tariffs are evaluated on the same realized days, "
        "we can put an interval on that contrast directly rather than on the two gaps "
        "separately. The paired difference is 0.21 with a 95% bootstrap interval of [0.15, "
        "0.27], and it is positive in every one of ten thousand resampled day sets.")

CAPS = {
 "oracle_tradeoff": "Figure 3: Left, per-tier satisfaction under each audited objective. The revenue "
    "optimum holds the budget tier near 0.38 while giving premium about 0.95, a tier gap of about 0.57, "
    "and the equity objectives equalize the tiers near 0.82. Right, the operator's revenue, where the "
    "price of fairness is paid, a median of about nine percent, and not in aggregate satisfaction.",
 "rate_design": "Figure 4: The harm tracks rank position, not spread magnitude. Bars are coloured by "
    "whether the budget tier remains the strictly lowest-priced rank (clay) or is lifted out of it (teal).",
 "scarcity_boundary": "Figure 5: The scarcity boundary separates a systematic from a rotating "
    "disadvantage. Per site, the day-averaged systematic tier gap (clay) against the typical per-day gap (grey).",
 "capacity_threshold": "Figure 6: Grid capacity and the two inequalities at the reference site. The "
    "between-tier disparate impact under the oracle falls to essentially zero by about sixty kilowatts.",
 "lever_cross_effects": "Figure 7: The two levers are not orthogonal. Pricing removes the between-tier "
    "disparate impact but barely touches the within-tier inequality, while capacity reduces both.",
 "grounding_invariance": "Figure 2: The disparate impact is structural, not a calibration artifact. The "
    "day-averaged tier gap is flat across the willingness-to-pay elasticity and collapses only at zero.",
}

fig = plt.figure(figsize=(PW, PH), dpi=DPI)
fig.patch.set_facecolor("white")


def X(inches):   return inches / PW
def Y(inches):   return 1.0 - inches / PH


def place_image(stem, x_in, y_in, w_in):
    """Place at EXACTLY w_in wide, preserving aspect -- same as \\includegraphics[width=]."""
    img = mpimg.imread(FIG % stem)
    h_in = w_in * img.shape[0] / img.shape[1]
    ax = fig.add_axes([X(x_in), Y(y_in + h_in), w_in / PW, h_in / PH])
    ax.imshow(img, interpolation="lanczos")
    ax.axis("off")
    return y_in + h_in


def place_caption(stem, x_in, y_in, w_in):
    chars = int(w_in / (8.0 * 0.50 / 72))          # ~0.50em average glyph at 8pt
    lines = textwrap.wrap(CAPS[stem], chars)
    fig.text(X(x_in), Y(y_in + 0.055), "\n".join(lines), fontsize=8, va="top", ha="left",
             linespacing=1.35, family="serif", color="#111111")
    return y_in + 0.055 + len(lines) * 8 * 1.35 / 72


def place_body(text, x_in, y_in, w_in, maxlines):
    chars = int(w_in / (9.0 * 0.50 / 72))
    lines = textwrap.wrap(text, chars)[:maxlines]
    fig.text(X(x_in), Y(y_in), "\n".join(lines), fontsize=9, va="top", ha="left",
             linespacing=1.42, family="serif", color="#111111")
    return y_in + len(lines) * 9 * 1.42 / 72


L, R = LEFT, LEFT + COLW + COLSEP

# --- full-width figure* at the top of the page ---
y = place_image("oracle_tradeoff", L, TOP, TEXTW)
y = place_caption("oracle_tradeoff", L, y + 0.04, TEXTW) + 0.22

# --- left column: prose, then a column figure ---
yl = place_body(BODY, L, y, COLW, 26) + 0.16
yl = place_image("rate_design", L, yl, COLW)
yl = place_caption("rate_design", L, yl + 0.04, COLW) + 0.20
yl = place_image("capacity_threshold", L, yl, COLW)
yl = place_caption("capacity_threshold", L, yl + 0.04, COLW)

# --- right column ---
yr = place_image("scarcity_boundary", R, y, COLW)
yr = place_caption("scarcity_boundary", R, yr + 0.04, COLW) + 0.20
yr = place_image("grounding_invariance", R, yr, COLW)
yr = place_caption("grounding_invariance", R, yr + 0.04, COLW) + 0.20
yr = place_image("lever_cross_effects", R, yr, COLW)
yr = place_caption("lever_cross_effects", R, yr + 0.04, COLW)

out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/proof.png"
fig.savefig(out, dpi=DPI, facecolor="white")
print("wrote", out, "at", DPI, "dpi (100%% zoom); bottom of columns: %.2fin / %.2fin" % (yl, yr))
