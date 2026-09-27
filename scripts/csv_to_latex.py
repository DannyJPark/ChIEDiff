#!/usr/bin/env python3
r"""Render a comparison CSV as a Springer Nature `tabular*` body, and optionally splice
it straight into paper/main.tex.

Springer Nature requires the manuscript to be a SINGLE .tex file, so `\input{table.tex}`
is not available. The workaround this script implements is marker injection: the table
body lives inside main.tex between

    %% BEGIN AUTO <label>
    ... generated rows ...
    %% END AUTO <label>

and re-running the script with --inject replaces exactly that span. The numbers stay
regenerable from the CSV without ever being retyped, and main.tex stays one file.

Every emitted block carries a provenance line (source path, sha256, row count) so a
number in the manuscript can be traced back to the file it came from.

The output deliberately matches paper/sn-article.tex, the vendor sample, cell for cell:
`tabular*` at `@{\extracolsep\fill}`, `\botrule` (not `\bottomrule`), grouped headers as
`\multicolumn{n}{@{}c@{}}{...}` + bare `\cmidrule{a-b}`, and table notes as
`\footnotemark[n]` / `\footnotetext[n]` rather than a minipage. J. Cheminform. forbids
colour and shading in tables, so row emphasis is a `\midrule` (--rule-after), never a
`\rowcolor`.

Examples
--------
# preview an F1 binding-affinity table
python scripts/csv_to_latex.py \
    --columns label,f1_n_mols,f1_vina_dock_med,f1_qed_med,f1_sa_med \
    --headers 'Model,$n$,Vina Dock,QED,SA' \
    --tiers reference,core --precision 2 \
    --best f1_vina_dock_med:min --label tab:f1

# a landscape table with two header levels, per-column precision and table notes
python scripts/csv_to_latex.py --csv results/comparison/f3_nci/nci_summary.csv \
    --ids reference,targetdiff --env sidewaystable \
    --group-header 'PLIP:2-3|ProLIF:4-5' \
    --precision-map 'heavy_mean:1' --note 'The two panels are not on a common scale.' \
    --columns label,... --label tab:f3 --inject paper/main.tex
"""
import argparse
import csv
import datetime
import decimal
import hashlib
import pathlib
import re
import sys

DEFAULT_CSV = "results/comparison/master_table.csv"
MISSING = {"", "nan", "NaN", "None", "not_measured", "not_applicable"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def escape(text):
    """Escape LaTeX specials, but leave math mode ($...$) alone."""
    parts = re.split(r"(\$[^$]*\$)", str(text))
    out = []
    for i, part in enumerate(parts):
        if i % 2:                       # inside $...$
            out.append(part)
            continue
        for char, rep in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                          ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
                          ("}", r"\}"), ("~", r"\textasciitilde{}"),
                          ("^", r"\textasciicircum{}")):
            part = part.replace(char, rep)
        out.append(part)
    return "".join(out)


def fmt(value, precision):
    """Return (rendered_cell, is_numeric), or (None, False) when the value is missing.

    Integers in the source stay integers -- a count printed as 9036.00 is wrong, not
    merely ugly. Negatives are wrapped in math mode so they typeset as a minus sign
    rather than a hyphen.

    Rounding is ROUND_HALF_UP on the DECIMAL value, not printf's round-half-even on the
    binary one. printf turns 0.145 into 0.14 and 1.795 into 1.79, which reads as an error
    to anyone checking the cell against the CSV. This is the convention the SAC tables
    already documented.
    """
    if value is None or str(value).strip() in MISSING:
        return None, False
    raw = str(value).strip()
    try:
        number = decimal.Decimal(raw)
    except decimal.InvalidOperation:
        return escape(raw), False
    if "." not in raw and "e" not in raw.lower():
        text = str(int(number))
    else:
        quantum = decimal.Decimal(1).scaleb(-precision)
        text = str(number.quantize(quantum, rounding=decimal.ROUND_HALF_UP))
    return (f"${text}$" if text.startswith("-") else text), True


def split_outside_math(spec, sep="|"):
    r"""Split on `sep`, ignoring occurrences inside $...$ or escaped by a backslash.

    Both exemptions are load-bearing on real headers: `|` is the span separator AND the
    absolute-value bar in `$|\Delta_{\mathrm{V}}|$`, while `,` is the header separator AND
    the second character of the thin space `\,` in `2\,\AA{}`. Splitting naively drops
    half a header row without erroring.
    """
    out, buf, in_math, escaped = [], [], False, False
    for char in spec:
        if escaped:
            buf.append(char)
            escaped = False
            continue
        if char == "\\":
            buf.append(char)
            escaped = True
            continue
        if char == "$":
            in_math = not in_math
        if char == sep and not in_math:
            out.append("".join(buf))
            buf = []
        else:
            buf.append(char)
    out.append("".join(buf))
    return out


def parse_group_header(spec, ncols):
    """'Label:2-7|Other:8-13' -> [(label, start, end)], 1-based inclusive, validated."""
    spans = []
    for chunk in split_outside_math(spec):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            sys.exit(f"ERROR: --group-header wants 'Label:first-last', got {chunk!r}")
        label, rng = chunk.rsplit(":", 1)
        m = re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)\s*", rng)
        if not m:
            sys.exit(f"ERROR: --group-header range must be 'first-last', got {rng!r}")
        first, last = int(m.group(1)), int(m.group(2))
        if not 1 <= first <= last <= ncols:
            sys.exit(f"ERROR: --group-header span {first}-{last} is outside 1-{ncols}")
        spans.append((label.strip(), first, last))
    spans.sort(key=lambda s: s[1])
    for (_, _, end), (lab, start, _) in zip(spans, spans[1:]):
        if start <= end:
            sys.exit(f"ERROR: --group-header spans overlap at column {start} ({lab!r})")
    return spans


def vrule_after(align):
    """{column index: '|'} for every vertical rule the --align spec asks for.

    A \\multicolumn cell carries its OWN column spec, so it silently drops any `|` the table
    preamble put at that boundary and the rule breaks in the group-header row. Reading the rules
    back out of the align lets the group headers reinstate them, which is what makes a vertical
    separator run the height of the table instead of appearing only in the body.
    """
    out, col = {}, 0
    for ch in align or "":
        if ch in "lcr":
            col += 1
        elif ch == "|" and col:
            out[col] = "|"
    return out


def group_header_rows(spans, ncols, rules=None):
    r"""One `\multicolumn` header row plus its `\cmidrule` line.

    Columns outside every span emit an empty cell, which is what the sn-article sample
    does for the stub column. Bare `\cmidrule{a-b}` is used rather than booktabs'
    `\cmidrule(lr){a-b}`: sn-jnl.cls overrides `\@@@cmidrule` internally and the vendor
    sample writes it untrimmed.
    """
    by_start = {s[1]: s for s in spans}
    cells, col = [], 1
    while col <= ncols:
        if col in by_start:
            label, first, last = by_start[col]
            width = last - first + 1
            edge = (rules or {}).get(last, "")
            cells.append(f"\\multicolumn{{{width}}}{{@{{}}c@{{}}{edge}}}{{{label}}}"
                         if width > 1 or edge else label)
            col = last + 1
        else:
            cells.append("")
            col += 1
    # No rule under a one-column span: it would underline a single heading and read as a
    # grouping that is not there.
    rule = "".join(f"\\cmidrule{{{first}-{last}}}"
                   for _, first, last in spans if last > first)
    return ["    " + " & ".join(cells) + r" \\" + rule]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=DEFAULT_CSV, help=f"source CSV (default: {DEFAULT_CSV})")
    ap.add_argument("--columns", required=True, help="comma-separated CSV column names, in order")
    ap.add_argument("--headers", help="comma-separated header cells; defaults to the column names")
    ap.add_argument("--group-header", action="append", default=[], metavar="SPEC",
                    help="extra header row above --headers, as 'Label:first-last|Label:first-last' "
                         "with 1-based inclusive column numbers. Repeatable; each occurrence "
                         "adds one more level, outermost first.")
    ap.add_argument("--tiers", help="keep only these tiers, comma-separated (e.g. reference,core)")
    ap.add_argument("--ids", help="keep only these model ids, comma-separated; sets the row order")
    ap.add_argument("--drop-missing", metavar="COL",
                    help="drop rows whose COL is missing (use the family's status column)")
    ap.add_argument("--precision", type=int, default=2, help="decimal places (default: 2)")
    ap.add_argument("--precision-map", default="", metavar="COL:DP,...",
                    help="per-column decimal places, overriding --precision")
    ap.add_argument("--align", help=r"column spec, e.g. 'lrrrr' (default: l then c for the rest)")
    ap.add_argument("--cell-width", metavar="PT", type=float, default=None,
                    help="typeset every non-label cell in a fixed-width right-aligned box of "
                         "this many pt, so all numeric columns come out identical and the "
                         "column GROUPS above them line up. Without it a group of signed "
                         "kcal/mol values is visibly wider than a group of unsigned pK values, "
                         "and no amount of \\extracolsep can even them out because the "
                         "difference is in the cell contents, not the padding. A fixed box is "
                         "also immune to the widening that bold and underline cause, which a "
                         "\\phantom-based fix is not. Set it just above the widest cell and "
                         "re-run scripts/measure_table.py.")
    ap.add_argument("--tabcolsep", metavar="PT", type=float, default=None,
                    help="override \\tabcolsep inside this float, in pt (class default 6). The "
                         "LAST resort for a portrait table, and only when the column set is "
                         "already minimal: it tightens horizontal padding, nothing else. Prefer "
                         "it to shrinking the 8bp table body font, which the class sets and which "
                         "is a legibility change rather than a spacing one. Scoped to the float, "
                         "so it cannot leak into another table. Run scripts/measure_table.py to "
                         "confirm the result actually fits before using this.")
    ap.add_argument("--env", choices=("table", "sidewaystable"), default="table",
                    help="float environment. sidewaystable is the SN idiom for a table too wide "
                         "for the measure, and sets the tabular width to \\textheight.")
    ap.add_argument("--best", action="append", default=[], metavar="COL:min|max",
                    help="bold the best value in COL; repeatable")
    ap.add_argument("--second", action="store_true",
                    help="also underline the runner-up in every --best column")
    ap.add_argument("--best-exclude", default="", metavar="ID,...",
                    help="ids kept out of the --best/--second ranking (e.g. the reference row)")
    ap.add_argument("--rank-within", metavar="COL",
                    help="rank --best/--second separately within each value of COL (e.g. a "
                         "subset or guidance column) instead of across the whole table")
    ap.add_argument("--relabel", action="append", default=[], metavar="ID=TEXT",
                    help="render this id's label cell as TEXT (raw LaTeX, not escaped); "
                         "repeatable. Leaves configs/models.json untouched.")
    ap.add_argument("--rule-after", default="", metavar="ID,...",
                    help=r"emit \midrule after these rows. This is how a row block is set off; "
                         r"J. Cheminform. forbids \rowcolor shading.")
    ap.add_argument("--note", action="append", default=[], metavar="TEXT",
                    help=r"table note; emitted as \footnotetext[n]{TEXT} in order. Put the "
                         r"matching \footnotemark[n] in --headers or --group-header.")
    ap.add_argument("--na", default="--", help="text for a missing cell (default: --)")
    ap.add_argument("--caption", default="CAPTION -- to be written.")
    ap.add_argument("--label", required=True, help="LaTeX label, e.g. tab:f1")
    ap.add_argument("--inject", metavar="TEXFILE",
                    help="replace the %%%% BEGIN/END AUTO <label> span in TEXFILE")
    args = ap.parse_args()

    csv_path = pathlib.Path(args.csv)
    if not csv_path.is_file():
        sys.exit(f"ERROR: no such CSV: {csv_path}")

    columns = [c.strip() for c in args.columns.split(",")]
    headers = ([h.strip() for h in split_outside_math(args.headers, ",")] if args.headers
               else [c.replace("_", r"\_") for c in columns])
    if len(headers) != len(columns):
        sys.exit(f"ERROR: {len(headers)} headers for {len(columns)} columns")

    precision = {}
    for spec in filter(None, (s.strip() for s in args.precision_map.split(","))):
        if ":" not in spec:
            sys.exit(f"ERROR: --precision-map wants COL:DP, got {spec!r}")
        col, dp = spec.rsplit(":", 1)
        if col not in columns:
            sys.exit(f"ERROR: --precision-map column {col!r} is not among --columns")
        precision[col] = int(dp)

    relabel = {}
    for spec in args.relabel:
        if "=" not in spec:
            sys.exit(f"ERROR: --relabel wants ID=TEXT, got {spec!r}")
        rid, text = spec.split("=", 1)
        relabel[rid.strip()] = text.strip()

    with csv_path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        sys.exit(f"ERROR: {csv_path} has no rows")
    unknown = [c for c in columns if c not in rows[0]]
    if unknown:
        sys.exit(f"ERROR: column(s) not in {csv_path}: {', '.join(unknown)}")

    if args.tiers:
        keep = {t.strip() for t in args.tiers.split(",")}
        rows = [r for r in rows if r.get("tier") in keep]
    if args.ids:
        order = [i.strip() for i in args.ids.split(",")]
        by_id = {r.get("id"): r for r in rows}
        missing_ids = [i for i in order if i not in by_id]
        if missing_ids:
            sys.exit(f"ERROR: id(s) not in {csv_path}: {', '.join(missing_ids)}")
        rows = [by_id[i] for i in order]
    if args.drop_missing:
        col = args.drop_missing
        rows = [r for r in rows if str(r.get(col, "")).strip() not in MISSING]
    if not rows:
        sys.exit("ERROR: every row was filtered out")

    excluded = {i.strip() for i in args.best_exclude.split(",") if i.strip()}
    rule_after = {i.strip() for i in args.rule_after.split(",") if i.strip()}

    # A typo in any of these three silently does nothing, which would be invisible in the
    # rendered table -- so refuse instead.
    selected = {r.get("id") for r in rows}
    for name, ids in (("--relabel", relabel), ("--rule-after", rule_after),
                      ("--best-exclude", excluded)):
        stray = sorted(i for i in ids if i not in selected)
        if stray:
            sys.exit(f"ERROR: {name} id(s) not among the selected rows: {', '.join(stray)}")

    if args.rank_within and args.rank_within not in rows[0]:
        sys.exit(f"ERROR: --rank-within column {args.rank_within!r} is not in {csv_path}")

    # Which cells to bold, and optionally underline, in each --best column, as (row, col).
    # Ranking uses the PRINTED value: two cells that both read 0.50 with only one in bold
    # reads as an error, so equal printed values share a rank and the next distinct value
    # is second (dense ranking). A column whose cells all print the same singles nothing
    # out, so it carries no mark.
    bold, second = set(), set()
    for spec in args.best:
        if ":" not in spec:
            sys.exit(f"ERROR: --best wants COL:min|max, got {spec!r}")
        col, direction = spec.rsplit(":", 1)
        if direction not in ("min", "max"):
            sys.exit(f"ERROR: --best direction must be min or max, got {direction!r}")
        if col not in columns:
            sys.exit(f"ERROR: --best column {col!r} is not among --columns")
        groups = {}
        for n, r in enumerate(rows):
            if r.get("id") in excluded:
                continue
            cell, numeric = fmt(r.get(col), precision.get(col, args.precision))
            if not numeric:
                continue
            key = r.get(args.rank_within) if args.rank_within else None
            groups.setdefault(key, []).append((decimal.Decimal(cell.strip("$")), n))
        for vals in groups.values():
            distinct = sorted({v for v, _ in vals}, reverse=(direction == "max"))
            if len(distinct) < 2:
                continue
            bold |= {(n, col) for v, n in vals if v == distinct[0]}
            if args.second:
                second |= {(n, col) for v, n in vals if v == distinct[1]}

    body = []
    for n, row in enumerate(rows):
        cells = []
        for col in columns:
            if col == "label" and row.get("id") in relabel:
                # Verbatim, like --headers: a relabel is authored LaTeX, so \textbf{Ours}
                # must survive rather than become \textbackslash{}textbf\{Ours\}.
                cells.append(relabel[row["id"]])
                continue
            if col == columns[0]:
                # The stub column is a NAME, not a measurement. Routing it through fmt() made a
                # label that happens to equal a missing-value token print as the --na placeholder:
                # the no-energy ablation arm is called "None", and Terms read "--" in every
                # abl table, which says "not measured" about a configuration that was.
                cells.append(escape(str(row.get(col, "")).strip()))
                continue
            cell, numeric = fmt(row.get(col), precision.get(col, args.precision))
            if cell is None:
                cell = args.na
            elif (n, col) in bold:
                # \textbf does not reach inside $...$, so bold maths needs \mathbf.
                cell = (f"$\\mathbf{{{cell.strip('$')}}}$" if numeric
                        else rf"\textbf{{{cell}}}")
            elif (n, col) in second:
                cell = rf"\underline{{{cell}}}"
            if args.cell_width is not None and col != columns[0]:
                cell = rf"\makebox[{args.cell_width:g}pt][r]{{{cell}}}"
            cells.append(cell)
        body.append("    " + " & ".join(cells) + r" \\")
        if row.get("id") in rule_after and n != len(rows) - 1:
            body.append(r"\midrule")

    align = args.align or ("l" + "c" * (len(columns) - 1))
    n_letters = len(re.sub(r"[^lcr]", "", align))
    if n_letters != len(columns):
        sys.exit(f"ERROR: --align has {n_letters} column letters for {len(columns)} columns")
    width = r"\textheight" if args.env == "sidewaystable" else r"\textwidth"

    header_rows = []
    rules = vrule_after(align)
    for spec in args.group_header:
        header_rows += group_header_rows(parse_group_header(spec, len(columns)), len(columns),
                                         rules)
    header_rows.append("    " + " & ".join(headers) + r" \\")

    notes = [f"\\footnotetext[{i}]{{{text}}}" for i, text in enumerate(args.note, 1)]

    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    block = "\n".join([
        f"%% BEGIN AUTO {args.label}",
        f"%% Generated by scripts/csv_to_latex.py on {stamp}. Do not edit by hand --",
        f"%% re-run the script instead, or the number stops being traceable.",
        f"%%   source : {csv_path}",
        f"%%   sha256 : {sha256(csv_path)}",
        f"%%   rows   : {len(rows)}",
        rf"\begin{{{args.env}}}[htbp]",
        f"\\caption{{{args.caption}}}\\label{{{args.label}}}",
        *([rf"\setlength\tabcolsep{{{args.tabcolsep:g}pt}}"] if args.tabcolsep is not None else []),
        rf"\begin{{tabular*}}{{{width}}}{{@{{\extracolsep\fill}}{align}}}",
        r"\toprule",
        *header_rows,
        r"\midrule",
        *body,
        r"\botrule",
        r"\end{tabular*}",
        *notes,
        rf"\end{{{args.env}}}",
        f"%% END AUTO {args.label}",
    ])

    if not args.inject:
        print(block)
        return

    tex_path = pathlib.Path(args.inject)
    if not tex_path.is_file():
        sys.exit(f"ERROR: no such file: {tex_path}")
    text = tex_path.read_text()
    begin = f"%% BEGIN AUTO {args.label}"
    end = f"%% END AUTO {args.label}"
    pattern = re.compile(re.escape(begin) + r".*?" + re.escape(end), re.DOTALL)
    if not pattern.search(text):
        sys.exit(f"ERROR: no '{begin}' ... '{end}' span in {tex_path}.\n"
                 f"       Add those two comment lines where the table belongs, then re-run.")
    tex_path.write_text(pattern.sub(lambda _: block, text, count=1))
    print(f"injected {args.label} into {tex_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
