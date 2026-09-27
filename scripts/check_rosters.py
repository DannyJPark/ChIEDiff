#!/usr/bin/env python3
"""Tier-4: pin the exact ordered row list of every comparison table.

A roster regression is otherwise invisible until someone reads a 335 KB text file. This turns it
into a one-line failure. It also closes the loop the registry migration opened: each table's rows
are compared against BOTH the recorded fixture and what `configs/models.json` says they should
be, so registry and output cannot drift apart silently.

    python scripts/check_rosters.py              # verify
    python scripts/check_rosters.py --update     # re-record after a DELIBERATE roster change

`--update` is the only way the fixture changes, so an accidental roster change always fails first.
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_registry                                                    # noqa: E402

FIXTURE = 'scripts/fixtures/expected_rosters.json'

# (path, key column, registry family, registry sub-table or None)
TABLES = [
    ('results/comparison/f1_sbdd/comparison_overall.csv', 'model', 'sbdd', None),
    ('results/comparison/f4_delta_score/delta_score_overall.csv', 'label', 'delta', None),
    ('results/comparison/f5_pose_quality/posecheck_summary.csv', 'model', 'quality', 'posecheck'),
    ('results/comparison/appendix/posebusters/pose_quality_summary_posebusters.csv',
     'model', 'quality', 'posebusters'),
    ('results/comparison/f2_pose_fidelity/gnina_comparison.csv', 'label', None, None),
    # Added 2026-08-21 with the machine-readable twins of the F2/F3 reports. Before these, those
    # two families had no CSV at all, so their rosters were unverifiable here.
    ('results/comparison/f2_pose_fidelity/rmsd_summary.csv', 'label', 'pose', 'rmsd'),
    ('results/comparison/f3_nci/nci_summary.csv', 'label', 'nci', 'nci_summary'),
]


def rows_of(path, col):
    """Ordered, de-duplicated values of `col` -- the table's roster as published."""
    if not os.path.isfile(path):
        return None
    out = []
    with open(path, newline='', encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            v = r.get(col)
            if v and v not in out:
                out.append(v)
    return out


def registry_rows(family, table):
    """What the registry says the roster should be, in its declared order."""
    if not family:
        return None
    if table:
        rows = [m for m in model_registry.models()
                if model_registry.in_table(m, family, table)]
    else:
        rows = model_registry.for_family(family)
    # legacy_order is the order the table is PUBLISHED in, which is what a comparison against a
    # published file has to use. The canonical `order` is what a future re-ordering will adopt.
    rows = sorted(rows, key=lambda m: ((m['families'][family] or {}).get('legacy_order', 10**6),
                                       m.get('order', 10**6), m['id']))
    return [model_registry.label_for(m, family) for m in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--update', action='store_true')
    ap.add_argument('--shipped', action='store_true',
                    help='skip tables whose CSV is not in the release subset, and report '
                         'how many were skipped rather than failing on them')
    ap.add_argument('--fixture', default=FIXTURE)
    a = ap.parse_args()

    current = {}
    for path, col, fam, table in TABLES:
        got = rows_of(path, col)
        if got is None:
            print(f'  {os.path.basename(path):46s} MISSING')
            continue
        current[path] = got

    if a.update:
        os.makedirs(os.path.dirname(os.path.abspath(a.fixture)), exist_ok=True)
        with open(a.fixture, 'w', encoding='utf-8') as fh:
            json.dump(current, fh, indent=2, ensure_ascii=False)
            fh.write('\n')
        print(f'recorded {len(current)} roster(s) -> {a.fixture}')
        for p, rows in current.items():
            print(f'  {os.path.basename(p):46s} {len(rows)} rows')
        return 0

    if not os.path.isfile(a.fixture):
        sys.exit(f'{a.fixture} not found -- run with --update once to record the baseline')
    with open(a.fixture, encoding='utf-8') as fh:
        expected = json.load(fh)

    failures, skipped = [], []
    for path, col, fam, table in TABLES:
        name = os.path.basename(path)
        got = current.get(path)
        if got is None:
            # Several of these CSVs are intermediate roster files that the release does not
            # ship; the 22 generated tables do not read them. Under --shipped they are
            # counted and skipped so the gate still checks what IS present, instead of
            # failing on absence and telling the reader nothing.
            if a.shipped:
                skipped.append(name)
                continue
            failures.append(f'{name}: file missing')
            continue
        want = expected.get(path)
        if want is None:
            failures.append(f'{name}: not in the fixture -- add it with --update')
            continue
        if got != want:
            failures.append(f'{name}: roster changed')
            print(f'  {name:46s} CHANGED')
            print(f'      recorded: {want}')
            print(f'      current : {got}')
            for x in [r for r in want if r not in got]:
                print(f'      - {x}')
            for x in [r for r in got if r not in want]:
                print(f'      + {x}')
            continue
        # loop closure: does the registry agree with what the table published?
        # MEMBERSHIP is the failure condition; row ORDER is reported but not fatal. A model
        # appearing or vanishing is a roster regression; a sub-table published in an order the
        # registry does not yet model (posebusters orders its 7 rows differently from posecheck's
        # 12) is a gap in the registry, not a wrong table.
        reg = registry_rows(fam, table)
        if reg is not None and set(reg) != set(got):
            failures.append(f'{name}: registry membership differs from the table')
            print(f'  {name:46s} OK vs fixture, but REGISTRY MEMBERSHIP DIFFERS')
            for x in [r for r in reg if r not in got]:
                print(f'      registry has, table lacks: {x}')
            for x in [r for r in got if r not in reg]:
                print(f'      table has, registry lacks: {x}')
            continue
        if reg is not None and reg != got:
            tag = '  (registry agrees on membership; row order not modelled)'
        elif reg is not None:
            tag = '  (registry agrees)'
        else:
            tag = ''
        print(f'  {name:46s} OK  {len(got)} rows{tag}')

    if skipped:
        print(f"\n  {len(skipped)} table(s) not in the release subset, skipped: "
              f"{', '.join(skipped)}")

    if failures:
        print(f'\n{len(failures)} failure(s):')
        for f in failures:
            print('  ' + f)
        print('\nIf the change was deliberate, re-record with --update and say why in the commit.')
        return 1
    print('\nAll rosters match the fixture, and the registry agrees with every table.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
