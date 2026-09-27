#!/usr/bin/env python3
"""Phase A gate: prove `configs/models.json` reproduces both legacy registries EXACTLY.

Nothing may be switched over to the unified registry until this prints IDENTICAL for both. It
compares the parsed structures with key order preserved (Python dicts keep insertion order, and
`json.dumps(..., indent=2)` therefore renders a stable, diffable text), so a reordered key or a
changed count is caught, not just a changed value.

    python scripts/check_registry_sync.py            # exit 0 only if both are IDENTICAL
    python scripts/check_registry_sync.py --verbose  # show the unified diff on mismatch
"""
import argparse
import difflib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_registry                                                    # noqa: E402

TARGETS = [
    ('configs/pose_fidelity_models.json', model_registry.as_pose_fidelity_registry, 'pose'),
    ('configs/delta_score_models.json', model_registry.as_delta_score_registry, 'delta'),
]


def render(obj):
    return json.dumps(obj, indent=2, ensure_ascii=False).splitlines()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--registry', default=model_registry.DEFAULT_PATH)
    ap.add_argument('--verbose', action='store_true')
    a = ap.parse_args()

    ok, present = True, 0
    for legacy_path, shim, family in TARGETS:
        if not os.path.isfile(legacy_path):
            print(f'{family:6s} {legacy_path}: RETIRED (moved to configs/_archive/)')
            continue
        present += 1
        with open(legacy_path, encoding='utf-8') as fh:
            want = json.load(fh)
        got = shim(a.registry)

        w, g = render(want), render(got)
        if w == g:
            print(f'{family:6s} {legacy_path}: IDENTICAL  ({len(got["models"])} models)')
            continue

        ok = False
        print(f'{family:6s} {legacy_path}: DIFFERS')
        # roster-level summary first -- almost always the real cause
        wt = [m.get('tag') for m in want['models']]
        gt = [m.get('tag') for m in got['models']]
        if wt != gt:
            print(f'    roster on disk : {wt}')
            print(f'    roster from reg: {gt}')
            miss, extra = set(wt) - set(gt), set(gt) - set(wt)
            if miss:
                print(f'    MISSING from registry output: {sorted(miss)}')
            if extra:
                print(f'    EXTRA in registry output   : {sorted(extra)}')
            if not miss and not extra:
                print('    same members, different ORDER -- check families.<fam>.legacy_order')
        diff = list(difflib.unified_diff(w, g, 'on_disk', 'from_registry', lineterm='', n=1))
        print(f'    {sum(1 for d in diff if d[:1] in "+-" and d[:3] not in ("+++", "---"))} '
              f'differing line(s)')
        if a.verbose:
            for line in diff:
                print('      ' + line)
        else:
            print('    re-run with --verbose for the full diff')

    print()
    if ok and present == 0:
        # Nothing left to reproduce. Saying "PHASE A OK" here would be a green light earned by
        # having no test to run -- the failure mode this whole exercise exists to prevent.
        print('NOTHING TO CHECK -- both legacy registries are retired and configs/models.json is')
        print('the only roster. This gate has served its purpose; the live invariants are')
        print('scripts/check_registry.py and scripts/check_rosters.py.')
        return 0
    if ok:
        print(f'PHASE A OK -- the unified registry reproduces all {present} legacy registry/ies exactly.')
        print('Consumers can now be switched one at a time with model_registry.load_as().')
        return 0
    print('PHASE A BLOCKED -- fix configs/models.json before switching any consumer.')
    return 1


if __name__ == '__main__':
    sys.exit(main())
