#!/usr/bin/env python3
"""Verify every rendered cell in paper/main.tex against the CSV it came from.

This is the CSV -> LaTeX consistency gate. paper/build.sh checks that the manuscript
COMPILES and paper/tools/check_manuscript.py checks that the injected blocks match what
the generator would emit today; neither one checks that a printed number equals the
number in the source CSV. That is this script's only job.

The column list, source CSV, row order, relabels, ranked columns and per-column precision
are read from paper/data/table_specs.json, which paper/tables/build_revision_tables.py
writes as it emits each table. Nothing is restated here, so the check cannot drift from
the generator. (It used to parse those facts back out of make_tables.sh with regexes;
that broke the moment make_tables.sh became a wrapper, and silently -- which is exactly
the failure this file exists to prevent.)

Rounding is ROUND_HALF_UP on the decimal value, matching csv_to_latex.fmt.

    python3 scripts/check_paper_tables.py          # exit 0 = every cell traced
"""
import csv
import json
import pathlib
import re
import sys
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP

ROOT = pathlib.Path(__file__).resolve().parent.parent
SPECS = ROOT / "paper/data/table_specs.json"
TEX = ROOT / "paper/main.tex"
MISSING = {"", "nan", "NaN", "None", "not_measured", "not_applicable"}


def rendered_rows(tex, label):
    """The body lines of one %% BEGIN/END AUTO block, as lists of raw cells.

    The first \\midrule separates the header rows (one or two grouped levels plus the
    column names) from the body; --rule-after emits more \\midrule lines inside the body,
    and those are skipped because they do not end in a row terminator.
    """
    block = re.search(r"%% BEGIN AUTO " + re.escape(label) + r"\n.*?%% END AUTO "
                      + re.escape(label), tex, re.S)
    if not block:
        return None
    body = block.group().split(r"\midrule", 1)[1].split(r"\botrule")[0]
    return [[c.strip() for c in line.strip()[:-2].split("&")]
            for line in body.splitlines() if line.strip().endswith(r"\\")]


def shown_value(cell):
    """Strip the emphasis wrappers csv_to_latex adds, leaving the printed number."""
    return re.sub(r"\\mathbf\{|\\underline\{|[$}]", "", cell).strip()


def unwrap_text(cell):
    """Drop a \\textbf wrapper, which is how a bolded NON-numeric cell is emitted."""
    match = re.fullmatch(r"\\textbf\{(.*)\}", cell.strip())
    return match[1] if match else cell.strip()


def latex_escape(text):
    """csv_to_latex.escape's non-math branch, for verbatim text columns.

    Deliberately reimplemented rather than imported: this gate's value is that it
    re-derives every cell independently, so a bug in the generator's own formatting
    still shows up here instead of cancelling out.
    """
    for char, rep in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                      ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
                      ("}", r"\}"), ("~", r"\textasciitilde{}"),
                      ("^", r"\textasciicircum{}")):
        text = text.replace(char, rep)
    return text


def check(spec, tex):
    label = spec["label"]
    path = ROOT / spec["source"]
    csv_rows = list(csv.DictReader(path.open(newline="")))
    # ids fixes the row order; ids=None means the CSV's own order, as --ids does.
    if spec["ids"]:
        by_id = {r["id"]: r for r in csv_rows}
        expected = [(i, by_id[i]) for i in spec["ids"] if i in by_id]
    else:
        expected = [(r.get("id"), r) for r in csv_rows]

    rows = rendered_rows(tex, label)
    if rows is None:
        return 1, 0, [f"{label}: no rendered block in its declared document"]
    if len(rows) != len(expected):
        return 1, 0, [f"{label}: {len(rows)} rendered rows, spec has {len(expected)}"]

    bad, total, notes = 0, 0, []
    columns, precision = spec["columns"], spec["precision"]
    default = spec["default_precision"]
    for (rid, source_row), cells in zip(expected, rows):
        if len(cells) != len(columns):
            bad += 1
            notes.append(f"{label} {rid}: {len(cells)} cells for {len(columns)} columns")
            continue
        for idx, col in enumerate(columns):
            if col == "label":
                # The one place the row WORDING is verified. A generator that reverts
                # "Test set" to "Reference" has to fail here, not in a reader's eye.
                want = spec["relabel"].get(rid)
                if want is not None and cells[idx] != want:
                    bad += 1
                    notes.append(f"{label} {rid:14s} label       "
                                 f"table={cells[idx]!r} spec={want!r}")
                continue
            total += 1
            shown, raw = shown_value(cells[idx]), str(source_row[col]).strip()
            if shown == spec["na"]:
                ok = raw in MISSING
            elif raw in MISSING:
                ok = False
            else:
                try:
                    value = Decimal(raw)
                except InvalidOperation:
                    value = None
                if value is None:
                    # Text column (tab:posebusters `subset`, tab:abl-plip `guidance`):
                    # printed verbatim, LaTeX-escaped, never rounded.
                    ok = unwrap_text(cells[idx]) in (raw, latex_escape(raw))
                elif "." not in raw and "e" not in raw.lower():
                    # Integers stay integers -- a count printed as 9036.00 is wrong,
                    # not merely ugly -- so precision does not apply to them.
                    ok = shown == str(int(value))
                else:
                    dp = len(shown.split(".")[1]) if "." in shown else 0
                    want = value.quantize(Decimal(1).scaleb(-dp), rounding=ROUND_HALF_UP)
                    ok = shown == str(want)
                    # A cell printed at the wrong precision still compares equal above
                    # when the digits happen to agree, so pin the declared precision too.
                    if ok and dp != precision.get(col, default):
                        ok, shown = False, f"{shown} ({dp}dp, declared " \
                                           f"{precision.get(col, default)}dp)"
            if not ok:
                bad += 1
                notes.append(f"{label} {str(rid):14s} {col:24s} "
                             f"table={shown!r} csv={raw!r}")

    # Re-derive from the CSV alone which cells must be bold / underlined: dense ranking on
    # the PRINTED value (ties share a mark), each rank_within group ranked separately, and
    # no mark in a column whose cells all print the same. Every cell's actual mark must
    # equal that expectation, which also catches a stray mark in an unranked column.
    want = {}
    within = spec.get("rank_within")
    for best in spec["best"]:
        col, direction = best.rsplit(":", 1)
        idx, dp = columns.index(col), precision.get(col, default)
        groups = {}
        for r, (rid, source_row) in enumerate(expected):
            raw = str(source_row[col]).strip()
            if rid in spec["best_exclude"] or raw in MISSING:
                continue
            try:
                value = Decimal(raw)
            except InvalidOperation:
                continue
            if "." in raw or "e" in raw.lower():
                value = value.quantize(Decimal(1).scaleb(-dp), rounding=ROUND_HALF_UP)
            groups.setdefault(source_row.get(within) if within else None, []).append((value, r))
        for vals in groups.values():
            distinct = sorted({v for v, _ in vals}, reverse=(direction == "max"))
            if len(distinct) < 2:
                continue
            for value, r in vals:
                if value == distinct[0]:
                    want[(r, idx)] = "bold"
                elif spec["second"] and value == distinct[1]:
                    want[(r, idx)] = "underline"
    for r, cells in enumerate(rows):
        if len(cells) != len(columns):
            continue
        for idx in range(1, len(cells)):
            got = ("bold" if re.search(r"\\mathbf|\\textbf", cells[idx])
                   else "underline" if r"\underline" in cells[idx] else None)
            if got != want.get((r, idx)):
                bad += 1
                notes.append(f"{label} {str(expected[r][0]):14s} {columns[idx]:24s} "
                             f"emphasis={got} expected={want.get((r, idx))}")
    return bad, total, notes


def main():
    if not SPECS.is_file():
        sys.exit(f"ERROR: {SPECS.relative_to(ROOT)} not found -- run "
                 f"'bash paper/make_tables.sh --dry' first to write it.")
    specs = json.loads(SPECS.read_text())
    # Since the supplementary split (2026-09-23) a rendered block lives in the document
    # paper/float_registry.json places it in; checking only main.tex silently skipped the
    # 17 tables of Additional file 1.
    place = {f["label"]: f["place"] for f in
             json.loads((ROOT / "paper/float_registry.json").read_text())["floats"]}
    documents = {"body": TEX.read_text(), "appendix": TEX.read_text(),
                 "supp": (ROOT / "paper/supplementary.tex").read_text()
                 if (ROOT / "paper/supplementary.tex").is_file() else ""}

    total = bad = 0
    # The F4 pocket set is the intersection over the REPORTED models, so the registry's
    # report_roster and the roster the manuscript prints must be the same list. If they
    # drift, tab:f4 is aggregated over a pocket set derived from a different set of models
    # than the rows it shows -- silently, and in the direction that flatters whoever is
    # missing.
    roster = next((s["ids"] for s in specs if s["label"] == "tab:a1"), None)
    declared = ((json.loads((ROOT / "configs/models.json").read_text())["families"]
                 .get("delta") or {}).get("report_roster") or [])
    if declared != roster:
        bad += 1
        print(f"BAD  registry report_roster {declared}\n     != manuscript roster {roster}")
    else:
        print(f"OK   F4 roster in sync with configs/models.json ({len(roster)} models)")

    for spec in specs:
        n_bad, n_cells, notes = check(spec, documents[place.get(spec["label"], "body")])
        bad += n_bad
        total += n_cells
        for note in notes:
            print(f"BAD  {note}")
        if not n_bad:
            print(f"OK   {spec['label']:24s} {n_cells:4d} cells, "
                  f"{len(spec['best'])} ranked column(s)")

    print(f"\n{total} cells compared across {len(specs)} tables, {bad} problem(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
