r"""Pre-submission gate: every number printed in the paper must come from a released run.

The failure mode this catches is mundane and expensive. A number gets updated in one
place, a table or a figure or a sentence, and not in the others, and the paper ships
with two values for the same quantity. It is not something a reader can be expected to
police and it is not something a proofread reliably catches, because the numbers look
fine individually. So we check it mechanically.

The script pulls every numeral out of ``paper/*.tex``, after stripping the parts of a
LaTeX source where a digit is markup rather than a claim (comments, the preamble,
citation keys, cross-references, figure widths, column specifications, the
bibliography), and then asks whether each surviving numeral appears anywhere under
``experiments/results/``. Matching is numeric rather than textual, a paper value is
matched when a released value rounds to it at the precision the paper prints, so
``0.57`` is satisfied by ``0.5666`` and ``9`` percent by ``8.7``. Percent-versus-
fraction is tried both ways, since the JSON stores some quantities as shares and the
paper prints them as percentages.

A printed number also counts as traceable when it is a standard summary of a released
array, a median, a mean, a min or max, or a quartile under any of the usual quantile
conventions. Results tables are built from per-seed and per-day rows, and the reductions
are the table, so refusing them would flag every honest summary. What the gate will not
accept is a number that no released run produces under any of those readings.

The gate's own report is written into the results directory, and archived runs live
there too. Neither is admitted as evidence: reading the report back would make every
second run pass, and a number that survives only in a superseded run is exactly the
drift this exists to catch.

Run it before submitting::

    python -m experiments.check_paper_numbers            # gate, exits non-zero on a miss
    python -m experiments.check_paper_numbers --verbose  # show where each number matched
    python -m experiments.check_paper_numbers --strict   # also scope prose to cited floats

What the gate is and is not. A numeral inside a labelled table or figure is checked
against that float's own source run, which is a sharp check: a stale table entry has
nowhere to hide. A numeral in running prose is checked against the whole corpus, which
is a weak check, because with twenty released runs almost any two-decimal number in
[0,1] appears somewhere. It still catches the case that matters most, a number that no
run produces at all, which is what a value left behind by a superseded run looks like.
Passing therefore means no printed number is unaccounted for, not that every printed
number is correct. ``--strict`` narrows prose too, at the cost of flagging paragraphs
that legitimately quote two runs at once.

A number that is genuinely not a result, a venue year or a station's step count, goes
in :data:`ALLOWLIST` below with a reason, so the exemption is reviewable rather than
silent.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(__file__)
RESULTS = os.path.join(HERE, "results")
#: Where the report is always written, even when --results points the corpus
#: elsewhere, so aiming the gate at an archived snapshot never writes into it.
_DEFAULT_RESULTS = RESULTS
PAPER = os.path.normpath(os.path.join(HERE, os.pardir, "paper"))

#: Numerals that are legitimately not experimental results. Keyed by the literal as it
#: appears in the source, valued by the reason it is exempt.
ALLOWLIST = {
    "5": "the simulator's five-minute timestep, a fixed environment constant",
    "288": "steps per episode, a fixed environment constant (24h at 5-minute steps)",
    "24": "hours in a day",
    "3": "the number of ability-to-pay tiers, a design constant",
    "2": "connectors per EVSE, a station-layout constant",
    "1": "unit and index literals in prose and math",
    "0": "unit and index literals in prose and math",
}

#: Four-digit calendar years are dates, not measurements. They appear in prose as data
#: vintages and source years, and the citation keys they would otherwise leak from are
#: already stripped, so exempting the shape is safer than listing each year by hand.
_YEAR = re.compile(r"^(?:19|20)\d\d$")

# LaTeX constructs whose digits are markup, not claims.
_STRIP = [
    re.compile(r"(?<!\\)%.*"),                                  # comments
    re.compile(r"\\(?:cite|citep|citet|ref|label|autoref|eqref|subref|input|include)\s*\{[^}]*\}"),
    re.compile(r"\\includegraphics\s*(?:\[[^\]]*\])?\s*\{[^}]*\}"),
    re.compile(r"\\begin\{(?:tabular|tabularx|array)\}\s*(?:\[[^\]]*\])?\s*\{[^}]*\}"),
    re.compile(r"\\ccsdesc\s*(?:\[[^\]]*\])?\s*\{[^}]*\}"),
    re.compile(r"\\(?:setlength|addtolength|vspace|hspace|arraystretch|columnsep"
               r"|tabcolsep|belowrulesep|aboverulesep|cmidrule|scalebox|resizebox)"
               r"\s*(?:\*)?\s*(?:\{[^}]*\})*(?:\[[^\]]*\])?"),
    re.compile(r"\d*\.?\d+\s*\\(?:linewidth|textwidth|columnwidth|paperwidth|height"
               r"|baselineskip|em|ex|pt|cm|in|mm)\b"),           # 0.315\linewidth
    re.compile(r"\\\\\s*\[[^\]]*\]"),                            # row skips \\[2pt]
    re.compile(r"\\begin\{[a-zA-Z*]+\}\s*\[[^\]]*\]"),           # float placement [t]
    re.compile(r"\\(?:documentclass|usepackage)\s*(?:\[[^\]]*\])?\s*\{[^}]*\}"),
]
_NUMBER = re.compile(r"(?<![A-Za-z0-9_.])(\d+(?:\.\d+)?)")
#: LaTeX thousands separators. ``31{,}424`` is one number, not ``31`` and ``424``.
_THOUSANDS = re.compile(r"(?<=\d)\{,\}(?=\d)")


def _blank(match) -> str:
    """Replace a match with as many newlines as it spanned, so line numbers survive."""
    return "\n" * match.group(0).count("\n")


def strip_latex(src: str) -> str:
    """Blank out the preamble, the bibliography, and digit-bearing markup.

    Everything removed is replaced by the same number of newlines, so the line numbers
    the gate reports point at the real lines of the real file. A gate that reports the
    wrong line is worse than no gate, because it sends the reader to innocent text.
    """
    src = _THOUSANDS.sub("", src)
    marker = "\\begin{document}"
    if marker in src:
        cut = src.index(marker) + len(marker)
        src = "\n" * src.count("\n", 0, cut) + src[cut:]
    bib = re.search(r"\\begin\{thebibliography\}", src)
    if bib:
        src = src[:bib.start()] + "\n" * src.count("\n", bib.start())
    src = re.sub(r"\\bibitem\s*(?:\[[^\]]*\])?\s*\{[^}]*\}.*", "", src)
    for pat in _STRIP:
        src = pat.sub(_blank, src)
    return src


def paper_numbers(path: str, strict: bool = False) -> list:
    """Every numeral the paper prints, with its line and its provenance scope.

    ``strict`` additionally scopes numerals in running prose to whatever floats their
    paragraph cites. That is the sharper check and it is also the noisier one, because
    a paragraph routinely quotes its own table and then a cross-calibration from a
    different run in the same breath. It is opt-in for that reason.
    """
    raw = open(path, encoding="utf-8").read()
    scopes = label_scopes(raw)
    para = paragraph_scopes(raw) if strict else {}
    body = strip_latex(raw)
    out = []
    for i, line in enumerate(body.splitlines(), start=1):
        for m in _NUMBER.finditer(line):
            # ``context`` is truncated for display, ``line_text`` is the whole line.
            # Percent detection has to read the whole line, since a table row or a long
            # paragraph can put the "\%" hundreds of characters after the numeral.
            inner = scope_for(i, scopes)
            # Inside a labelled float, that float's source is authoritative. In strict
            # mode, prose falls back to whatever its paragraph cites.
            scope = (inner,) if inner else para.get(i)
            out.append({"literal": m.group(1), "line": i, "scope": scope,
                        "context": line.strip()[:120], "line_text": line})
    return out


#: Which released run backs which table or figure. A numeral printed inside one of
#: these is checked against *that* run only, which is what makes the gate sharp. The
#: global corpus is large enough that almost any two-decimal number in [0,1] appears
#: somewhere in it, so an unscoped existence check catches a number produced by no run
#: at all but will happily wave through a wrong one. Scoping is the difference.
#:
#: A label with no entry here falls back to the global check, as does any numeral in
#: running prose. Add an entry when a table's provenance is known, and the gate gets
#: sharper; a wrong entry shows up immediately as a table full of misses.
SOURCES = {
    "tab:grounding": ("audit_grounding.json", "audit_mechanism.json"),
    "fig:grounding": ("audit_grounding.json", "audit_mechanism.json"),
    "tab:oracle": ("audit_mechanism.json",),
    "fig:oracle": ("audit_mechanism.json",),
    "tab:tariff": ("audit_mechanism.json",),
    "fig:tariff": ("audit_mechanism.json",),
    "tab:sites": ("audit_robustness.json", "audit_estimator_check.json",
                  "audit_mechanism_acn.json"),
    "fig:scarcity": ("audit_robustness.json", "audit_estimator_check.json"),
    "fig:severity": ("scarcity_severity.json",),
    "fig:capacity": ("audit_capacity.json",),
    "fig:levers": ("audit_levers.json",),
    "fig:levers-pair": ("audit_levers.json",),
    "tab:online": ("check.json", "gate.json"),
}

_ENV_BEGIN = re.compile(r"\\begin\{(table\*?|figure\*?)\}")
_ENV_END = re.compile(r"\\end\{(table\*?|figure\*?)\}")
_LABEL = re.compile(r"\\label\{([^}]*)\}")


def label_scopes(raw: str) -> list:
    """``(first_line, last_line, label)`` for every labelled table or figure.

    Read from the RAW source, because the label commands are stripped before numerals
    are extracted. Nested environments (a subfigure inside a figure) all appear, and
    the smallest enclosing one wins.
    """
    lines = raw.splitlines()
    scopes, stack = [], []
    for i, line in enumerate(lines, start=1):
        for _ in _ENV_BEGIN.finditer(line):
            stack.append(i)
        for _ in _ENV_END.finditer(line):
            if not stack:
                continue
            start = stack.pop()
            body = "\n".join(lines[start - 1:i])
            for m in _LABEL.finditer(body):
                scopes.append((start, i, m.group(1)))
    return scopes


_REF = re.compile(r"\\(?:ref|autoref|subref)\{((?:tab|fig):[^}]*)\}")


def paragraph_scopes(raw: str) -> dict:
    """Map each line to the tables and figures its paragraph cites.

    A sentence that says "Table~\\ref{tab:oracle} reports the optimum" and then quotes
    numbers is quoting *that* table, so its numerals are checked against that table's
    source rather than against everything. This is how a reader verifies a claim, and it
    is what lets the gate catch a wrong number in running prose rather than only a
    number that no run produced at all. A paragraph citing several floats is checked
    against the union of their sources, which is the weaker but still honest reading.
    """
    out: dict = {}
    line_no = 1
    for block in raw.split("\n\n"):
        n = block.count("\n") + 1
        labels = tuple(sorted({m.group(1) for m in _REF.finditer(block)} & set(SOURCES)))
        if labels:
            for i in range(line_no, line_no + n):
                out[i] = labels
        line_no += n + 1
    return out


def scope_for(line: int, scopes: list) -> str | None:
    """The innermost labelled environment containing ``line`` that we have a source for."""
    best, best_span = None, None
    for start, end, label in scopes:
        if start <= line <= end and label in SOURCES:
            span = end - start
            if best_span is None or span < best_span:
                best, best_span = label, span
    return best


#: Quantile conventions a paper might legitimately have used. NumPy's default is
#: ``linear``; the classic plotting-position rule is ``weibull``, and a table computed
#: in R or in a spreadsheet can land on either. A number is traceable if ANY of them
#: reproduces it from a released array, since the disagreement is a convention and not
#: a discrepancy in the underlying run.
_QUANTILE_METHODS = ("linear", "weibull", "lower", "higher", "nearest", "midpoint",
                     "median_unbiased", "normal_unbiased")


def _derived(values: list, sink: list) -> None:
    """Standard summary statistics of a released per-seed or per-day array.

    A paper prints medians, quartiles and ranges over the seeds it ran; the JSON stores
    the seeds. Refusing to reduce them would flag every honest summary as untraceable,
    so we admit the reductions a results table is actually built from, and only those.
    """
    if len(values) < 3:
        return
    import numpy as _np

    a = _np.asarray(values, dtype=float)
    sink.extend([float(_np.median(a)), float(a.mean()), float(a.min()), float(a.max())])
    for method in _QUANTILE_METHODS:
        try:
            q = _np.percentile(a, [25.0, 75.0], method=method)
        except (TypeError, ValueError):
            continue
        sink.extend([float(q[0]), float(q[1]), float(q[1] - q[0])])


def _walk_json(node, sink):
    if isinstance(node, dict):
        for k, v in node.items():
            _walk_json(k, sink)
            _walk_json(v, sink)
    elif isinstance(node, list):
        for v in node:
            _walk_json(v, sink)
        nums = [float(v) for v in node
                if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if len(nums) == len(node):
            _derived(nums, sink)
        elif node and all(isinstance(v, dict) for v in node):
            # A list of per-seed or per-day records. Reduce each field across records.
            for key in node[0]:
                col = [r[key] for r in node
                       if isinstance(r.get(key), (int, float)) and not isinstance(r.get(key), bool)]
                if len(col) == len(node):
                    _derived([float(v) for v in col], sink)
    elif isinstance(node, bool):
        pass
    elif isinstance(node, (int, float)):
        sink.append(float(node))
    elif isinstance(node, str):
        for m in _NUMBER.finditer(node):
            sink.append(float(m.group(1)))


#: Written by this script into the results directory, so it must never be read back as
#: evidence. Its ``missing`` list contains exactly the numbers that failed last time,
#: which would make every second run pass.
REPORT_NAME = "check_paper_numbers.json"

#: Superseded runs kept for provenance. A number that only survives in an archived run
#: is precisely the drift this gate exists to catch, so archives are not evidence either.
_EXCLUDED_DIRS = ("archive", "checkpoints")


def result_values() -> dict:
    """Every number under ``experiments/results/``, keyed by the file it came from."""
    found = {}
    for root, dirs, files in os.walk(RESULTS):
        dirs[:] = [d for d in dirs if d not in _EXCLUDED_DIRS]
        for name in sorted(files):
            if name == REPORT_NAME:
                continue
            path = os.path.join(root, name)
            rel = os.path.relpath(path, RESULTS)
            values: list = []
            if name.endswith(".json"):
                try:
                    _walk_json(json.load(open(path, encoding="utf-8")), values)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
            elif name.endswith((".md", ".txt", ".csv")):
                try:
                    text = open(path, encoding="utf-8").read()
                except UnicodeDecodeError:
                    continue
                values = [float(m.group(1)) for m in _NUMBER.finditer(text)]
            if values:
                found[rel] = values
    return found


def _matches(target: str, value: float, percentish: bool) -> bool:
    """Does a released ``value`` round to the ``target`` the paper prints?

    ``percentish`` is set only when the numeral is printed as a percentage, in which
    case a released share is allowed to match it after rescaling. Applying that
    rescaling unconditionally would match almost anything against almost anything,
    which would turn the gate into a rubber stamp.
    """
    decimals = len(target.split(".")[1]) if "." in target else 0
    want = float(target)
    # Compare by distance rather than by re-rounding. Python rounds halves to even,
    # a paper rounds them up, and 348.5 printed as 349 is correct either way.
    tol = 0.5 * 10.0 ** (-decimals) + 1e-9
    candidates = (value, value * 100.0, value / 100.0) if percentish else (value,)
    return any(abs(c - want) <= tol for c in candidates)


def _is_percent(context: str) -> bool:
    return "\\%" in context or "percent" in context.lower()


def find_source(target: str, results: dict, percentish: bool,
                scope: tuple | None = None) -> str | None:
    allowed = {f for label in scope for f in SOURCES.get(label, ())} if scope else None
    for rel, values in results.items():
        if allowed is not None and os.path.basename(rel) not in allowed:
            continue
        for v in values:
            if _matches(target, v, percentish):
                return rel
    return None


#: Keys under which a released run records how many realized days it used. Different
#: scripts spell it differently; all of them are checked.
_DAY_KEYS = ("n_days", "days_used", "used_after_filter", "requested", "oracle_days",
             "days", "n_days_used")


def _day_counts(node, sink):
    """Collect every day count a released run reports, wherever it is nested."""
    if isinstance(node, dict):
        for k, v in node.items():
            if k in _DAY_KEYS and isinstance(v, (int, float)) and not isinstance(v, bool):
                if 1 <= v <= 100000:
                    sink.add(int(v))
            _day_counts(v, sink)
    elif isinstance(node, list):
        for v in node:
            _day_counts(v, sink)


def check_run_lengths(results_dir: str) -> list:
    """Every released run should cover the same realized days. Report the ones that do not.

    This is the check that a numeral-level gate structurally cannot make. Figures are
    images, so no amount of scanning the .tex will notice that a figure was drawn from a
    run of a different length than the table beside it. What it can notice is that two
    files in the released set disagree about how many days they cover, which is the
    upstream cause. The ACN-Data configuration is exempt by name, because it is skipped
    whenever EQUICHARGE_ACN_JSON is unset and is documented as lagging.
    """
    from collections import Counter

    per_file = {}
    for name in sorted(os.listdir(results_dir)):
        if not name.endswith(".json") or name == REPORT_NAME:
            continue
        try:
            data = json.load(open(os.path.join(results_dir, name), encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        found: set = set()
        _day_counts(data, found)
        if found:
            per_file[name] = sorted(found)

    modal = Counter(d for n, ds in per_file.items()
                    if "_acn" not in n for d in ds).most_common(1)
    if not modal:
        return []
    expected = modal[0][0]
    offenders = []
    for name, days in per_file.items():
        if "_acn" in name:
            continue
        odd = [d for d in days if d != expected]
        # A file may legitimately mention other counts (a day-count ladder, a sub-sweep).
        # Only flag one that never mentions the expected horizon at all.
        if expected not in days and odd:
            offenders.append({"file": name, "days_reported": days, "expected": expected})
    return offenders


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--verbose", action="store_true", help="print where each number matched")
    ap.add_argument("--tex", default=None, help="check one .tex file instead of all of paper/")
    ap.add_argument("--strict", action="store_true",
                    help="also check prose numerals against the floats their paragraph "
                         "cites (sharper, and noisier where a paragraph mixes runs)")
    ap.add_argument("--results", default=None,
                    help="check against this results directory instead of experiments/results/")
    args = ap.parse_args()
    if args.results:
        global RESULTS
        RESULTS = os.path.abspath(args.results)

    texs = ([args.tex] if args.tex else
            sorted(os.path.join(PAPER, f) for f in os.listdir(PAPER) if f.endswith(".tex")))
    if not texs:
        print("no .tex found under", PAPER)
        return 1

    results = result_values()
    print("scanning %d result file(s) under %s" % (len(results), os.path.relpath(RESULTS, HERE)))

    total, matched, exempt, missing = 0, 0, 0, []
    for tex in texs:
        nums = paper_numbers(tex, strict=args.strict)
        print("\n%s: %d numerals" % (os.path.basename(tex), len(nums)))
        for item in nums:
            total += 1
            lit = item["literal"]
            reason = ALLOWLIST.get(lit) or ("a calendar year" if _YEAR.match(lit) else None)
            if reason:
                exempt += 1
                if args.verbose:
                    print("  exempt  %-10s line %-4d  (%s)" % (lit, item["line"], reason))
                continue
            src = find_source(lit, results, _is_percent(item["line_text"]), item["scope"])
            if src:
                matched += 1
                if args.verbose:
                    print("  ok      %-10s line %-4d  <- %s%s"
                          % (lit, item["line"], src,
                             "  [scoped to %s]" % ", ".join(item["scope"]) if item["scope"] else ""))
            else:
                checked = sorted({f for label in (item["scope"] or ())
                                  for f in SOURCES.get(label, ())})
                where = ("not in %s (cited as %s)"
                         % ("/".join(checked), ", ".join(item["scope"]))
                         if checked else "in no released run")
                missing.append({"file": os.path.basename(tex), "literal": lit,
                                "line": item["line"],
                                "scope": list(item["scope"]) if item["scope"] else None,
                                "checked_against": checked or "all",
                                "context": item["context"]})
                print("  MISSING %-10s line %-4d  %s\n            %s"
                      % (lit, item["line"], where, item["context"]))

    # A figure cannot be scanned for numerals, so the gate checks the upstream cause
    # instead: whether the released runs agree about how many days they cover.
    stale = check_run_lengths(RESULTS)
    if stale:
        print("\nRUN-LENGTH MISMATCH. These released files do not cover the same realized")
        print("days as the rest of the set, so any figure drawn from them disagrees with the")
        print("tables:")
        for row in stale:
            print("  %-34s reports %s days, the set is at %d"
                  % (row["file"], row["days_reported"], row["expected"]))
    else:
        print("\nrun lengths agree across the released set (ACN-Data files exempt by name)")

    print("\n%d numerals: %d matched, %d exempt, %d unaccounted for"
          % (total, matched, exempt, len(missing)))
    report = os.path.join(_DEFAULT_RESULTS, REPORT_NAME)
    json.dump({"total": total, "matched": matched, "exempt": exempt,
               "missing": missing, "run_length_mismatches": stale,
               "result_files": sorted(results)},
              open(report, "w"), indent=2)
    print("wrote", report)
    if missing or stale:
        print("\nGATE FAILED. Each number above is printed in the paper but is not produced by "
              "any released run.\nEither regenerate the result it should come from, correct the "
              "paper, or add it to ALLOWLIST\nwith a reason.")
        return 1
    print("GATE PASSED. Every printed number traces to a released result.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
