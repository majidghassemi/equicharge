"""Audit the paper figures for print legibility, at their FINAL printed size.

Run after changing any figure, and at the slot width you actually compose them into:

    python -m experiments.check_figure_legibility              # single column
    python -m experiments.check_figure_legibility --across 3   # three across \textwidth

It fails loudly rather than leaving a figure that only looks right on screen. Checks:
  1. the drawn width equals the width the .tex asks for, so the scale factor is 1.0
  2. every text element is >= MIN_PT once that scale factor is applied
  3. no two tick labels on the same axis overlap
  4. no text element spills outside the figure canvas
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments import plot_mechanism as P
from experiments import plot_style as S

MIN_PT = 7.0        # never smaller than this on the page
CAPTION_PT = 8.0    # acmart sigconf \captionfont

captured = {}


def _capture(fig, path):
    captured[os.path.basename(path).replace(".png", "")] = fig
    fig.savefig(path)


def run(argv=None):
    P.S.savefig = _capture
    S.savefig = _capture
    P.main(argv)

    problems = []
    for stem, fig in captured.items():
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        # the oracle figure is inherently two-panel, so it always spans \textwidth
        want = S.FULL if stem == "oracle_tradeoff" else P.W
        drawn = fig.get_size_inches()[0]
        scale = want / drawn
        if abs(scale - 1.0) > 1e-6:
            problems.append(f"{stem}: SCALE {scale:.3f} (drawn {drawn:.2f}in, tex {want:.2f}in)")

        # tick labels outside the view are retained but clipped; ignore them
        clipped = set()
        for ax in fig.axes:
            for lo, hi, ticks, locs in (
                    (*ax.get_xlim(), ax.get_xticklabels(), ax.get_xticks()),
                    (*ax.get_ylim(), ax.get_yticklabels(), ax.get_yticks())):
                for t, loc in zip(ticks, locs):
                    if loc < min(lo, hi) - 1e-9 or loc > max(lo, hi) + 1e-9:
                        clipped.add(id(t))

        smallest = 999
        fw, fh = fig.canvas.get_width_height()
        for t in fig.findobj(matplotlib.text.Text):
            txt = t.get_text().strip()
            if not txt or not t.get_visible() or id(t) in clipped:
                continue
            pt = t.get_fontsize() * scale
            smallest = min(smallest, pt)
            if pt < MIN_PT - 1e-6:
                problems.append(f"{stem}: text {pt:.1f}pt < {MIN_PT}pt -> {txt!r}")
            bb = t.get_window_extent(r)
            if bb.x0 < -1 or bb.y0 < -1 or bb.x1 > fw + 1 or bb.y1 > fh + 1:
                problems.append(f"{stem}: text spills canvas -> {txt!r}")

        for ai, ax in enumerate(fig.axes):
            for which, ticks in (("x", ax.get_xticklabels()), ("y", ax.get_yticklabels())):
                boxes = [(t.get_text(), t.get_window_extent(r)) for t in ticks
                         if t.get_text().strip() and t.get_visible() and id(t) not in clipped]
                for i in range(len(boxes)):
                    for j in range(i + 1, len(boxes)):
                        (n1, b1), (n2, b2) = boxes[i], boxes[j]
                        ox = min(b1.x1, b2.x1) - max(b1.x0, b2.x0)
                        oy = min(b1.y1, b2.y1) - max(b1.y0, b2.y0)
                        if ox > 0.5 and oy > 0.5:
                            problems.append(f"{stem}: ax{ai} {which}-ticks overlap "
                                            f"{n1!r} / {n2!r} ({ox / fig.dpi * 72:.1f}pt)")
        print(f"  {stem:26s} scale={scale:.3f}  smallest text={smallest:.1f}pt "
              f"(caption is {CAPTION_PT}pt)")

    print()
    if problems:
        print(f"FAIL  {len(problems)} problem(s):")
        for p in problems:
            print("  -", p)
        return 1
    print("PASS  all figures legible at print size")
    return 0


if __name__ == "__main__":
    sys.exit(run())
