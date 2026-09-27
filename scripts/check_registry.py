#!/usr/bin/env python3
"""Tier-3 invariants for configs/models.json. Exit 1 on any failure; meant for CI / pre-commit.

    python scripts/check_registry.py            # all checks
    python scripts/check_registry.py --skip 3   # skip the reverse-coverage check

Checks
  1  `id` unique, and every ids.<field> unique WITHIN its family.
     Catches the kgdiff/KGDiff collision class that made cross-family joins impossible.
  2  status "ok" implies the paths it promises actually exist. Claiming ok with a missing file
     is a hard error -- a silently empty table is the failure mode this prevents.
  3  REVERSE coverage: every directory under eval_out/, eval_plip/, eval_plip_atom/ and
     results/offtarget/ is claimed by exactly one entry, or matches unregistered_allowlist.
     This is what stops the 13 abl_* dirs re-entering a table by auto-discovery.
  4  tier "core" means every family is ok / not_applicable, or carries an explicit waiver.
  5  Cross-family source consistency: two families reading the SAME .pt must agree on n_mols;
     two families reading DIFFERENT .pt files must say why in `notes`. IPDiff is exactly this
     case -- it reads _chem_ for F1/F2 and the plain merged file for F3.
  6  canonical_pockets.sha256 matches BOTH test_pockets.json and the off-target assignment.
  7  Population rule: no builder writing into results/comparison/ still exposes an opt-out flag
     (--docked_only / --all_molecules / --unfiltered). Without this, the convention rots.
"""
import argparse
import fnmatch
import glob
import hashlib
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_registry                                                    # noqa: E402

SCAN_ROOTS = {'eval_out': 'eval_out', 'plip_tag': 'eval_plip',
              'plip_atom_tag': 'eval_plip_atom', 'delta_tag': 'results/offtarget'}
OPT_OUT_FLAGS = ['--docked_only', '--all_molecules', '--unfiltered']


def check_ids(reg, fail):
    seen_ids = {}
    for m in reg['models']:
        if m['id'] in seen_ids:
            fail(1, f"duplicate id {m['id']!r}")
        seen_ids[m['id']] = m
    for field in ('pt_stem', 'eval_out', 'plip_tag', 'plip_atom_tag', 'delta_tag'):
        seen = {}
        for m in reg['models']:
            v = (m.get('ids') or {}).get(field)
            if v is None:
                continue
            if v in seen:
                fail(1, f'ids.{field} {v!r} claimed by both {seen[v]!r} and {m["id"]!r}')
            seen[v] = m['id']


def check_sources_exist(reg, fail):
    for m in reg['models']:
        for fam in model_registry.FAMILIES:
            if model_registry.status(m, fam) != 'ok':
                continue
            src = model_registry.source_for(m, fam)
            if src and not os.path.exists(src):
                if SHIPPED:
                    SKIPPED.append(f"{m['id']}/{fam}: source {src}")
                else:
                    fail(2, f"{m['id']}/{fam}: status ok but source missing: {src}")
            ids = m.get('ids') or {}
            for field, root in SCAN_ROOTS.items():
                # only assert the directory the family actually consumes
                if (fam, field) not in (('pose', 'eval_out'), ('quality', 'eval_out'),
                                        ('nci', 'plip_tag'), ('delta', 'delta_tag')):
                    continue
                tag = ids.get(field)
                if tag and not os.path.isdir(os.path.join(root, tag)):
                    if SHIPPED:
                        SKIPPED.append(f"{m['id']}/{fam}: {root}/{tag}")
                    else:
                        fail(2, f"{m['id']}/{fam}: status ok but {root}/{tag} does not exist")


def check_reverse(reg, fail):
    allow = reg.get('unregistered_allowlist', [])
    for field, root in SCAN_ROOTS.items():
        if not os.path.isdir(root):
            continue
        claimed = {(m.get('ids') or {}).get(field) for m in reg['models']}
        claimed.discard(None)
        for path in sorted(glob.glob(os.path.join(root, '*'))):
            if not os.path.isdir(path):
                continue
            name = os.path.basename(path)
            if name in claimed:
                continue
            rel = f'{root}/{name}'
            if any(fnmatch.fnmatch(rel, pat) for pat in allow):
                continue
            fail(3, f'{rel} is claimed by no registry entry and matches no allowlist pattern')


def check_core(reg, fail):
    for m in reg['models']:
        if m.get('tier') != 'core':
            continue
        for fam in model_registry.FAMILIES:
            f = (m.get('families') or {}).get(fam) or {}
            st = f.get('status')
            if st in ('ok', 'not_applicable'):
                continue
            if f.get('waiver'):
                continue
            fail(4, f"core model {m['id']} has {fam}={st!r} with no waiver")


def check_source_consistency(reg, fail):
    for m in reg['models']:
        srcs = {}
        for fam in model_registry.FAMILIES:
            if model_registry.status(m, fam) not in ('ok', 'retired'):
                continue
            s = model_registry.source_for(m, fam)
            if s:
                srcs.setdefault(s, []).append(fam)
        if len(srcs) > 1 and not m.get('notes'):
            fail(5, f"{m['id']} reads {len(srcs)} different .pt files across families "
                    f"but has no `notes` explaining why")
        # Compare counts only WITHIN a counting basis. `pose` counts exported SDFs and `delta`
        # counts docking-successful molecules, so ours_vina is legitimately 9857 and 8340 off
        # the same .pt. Comparing across bases would flag that intentional difference as a bug.
        counts = m.get('counts') or {}
        for s, fams in srcs.items():
            by_basis = {}
            for f in fams:
                c = counts.get(f) or {}
                if c.get('n_mols') is None:
                    continue
                basis = c.get('basis')
                if basis is None:
                    fail(5, f"{m['id']}/{f}: counts.n_mols is set but counts.basis is not -- "
                            f"an unlabelled count cannot be compared safely")
                    continue
                by_basis.setdefault(basis, {})[f] = c['n_mols']
            for basis, vals in by_basis.items():
                if len(set(vals.values())) > 1:
                    fail(5, f"{m['id']} families {sorted(vals)} share {s} and the same basis "
                            f"{basis!r} but disagree on n_mols: {vals}")


def check_sha(reg, fail):
    want = (reg.get('canonical_pockets') or {}).get('sha256')
    if not want:
        return fail(6, 'canonical_pockets.sha256 is not set')
    src = reg['canonical_pockets']['source']
    if os.path.isfile(src):
        got = json.load(open(src)).get('sha256_ligand_filenames')
        if got != want:
            fail(6, f'{src}: sha256_ligand_filenames {got} != registry {want}')
    assign = (reg['families'].get('delta') or {}).get('assignment')
    if assign and os.path.isfile(assign):
        got = json.load(open(assign)).get('test_pockets_sha256')
        if got != want:
            fail(6, f'{assign}: test_pockets_sha256 {got} != registry {want}')


def _opt_out_flags(path):
    """Flags that make the docking-successful population optional.

    `--unfiltered` does NOT count when the script also refuses to write into results/comparison/.
    The distinction is which way the default points: `--docked_only` and `--all_molecules` leave
    the filter OFF unless someone remembers the flag, which is how two incomparable versions of a
    table came to ship. A gated `--unfiltered` keeps the filter ON and sends the diagnostic
    variant somewhere it cannot be mistaken for a comparison table.
    """
    try:
        src = open(path, encoding='utf-8').read()
    except (OSError, UnicodeDecodeError):
        return []
    gated = "may not write into results/comparison/" in src
    out = []
    for f in OPT_OUT_FLAGS:
        if not re.search(r"add_argument\(\s*'" + re.escape(f) + r"'", src):
            continue
        if f == '--unfiltered' and gated:
            continue
        out.append(f)
    return out


def check_population_rule(reg, fail, pending):
    """Driven by families.<f>.builder, not by whether a script happens to hardcode an out path.

    The earlier version scanned for scripts containing a family's `out` string, which made it a
    false negative for every builder that takes --out on the command line -- aggregate_posecheck.py
    among them. Declared builders are the reliable handle.
    """
    for fam, f in reg['families'].items():
        builder = f.get('builder')
        if not builder or not os.path.isfile(builder):
            fail(7, f'{fam}: families.{fam}.builder is missing or unset ({builder!r})')
            continue
        flags = _opt_out_flags(builder)
        if f.get('population_enforced'):
            if flags:
                fail(7, f'{fam}: {builder} is declared population_enforced but still exposes '
                        f'{", ".join(flags)} -- the docking-successful population must not be optional')
        elif flags:
            pending.append(f'{fam}: {builder} still exposes {", ".join(flags)} '
                           f'(population_enforced=false; scheduled work)')
        else:
            pending.append(f'{fam}: {builder} has no opt-out flag left -- flip '
                           f'families.{fam}.population_enforced to true')


def check_out_dirs(reg, fail):
    """Each family builder's default output directory must equal families.<f>.out.

    Added after a real miss: `reorg_path_map.py` derived its map from moves.tsv, which lists
    FILE moves, so a default of `results/comparison` (a bare directory, never a move source)
    was left untouched. build_delta_score.py then happily wrote its four outputs back to the
    old top level while f4_delta_score/ kept the stale copies -- silently, because the run
    succeeded. This makes the directory part of the checked contract.
    """
    pat = re.compile(r"add_argument\(\s*'--(?:out|out_dir)'\s*,\s*default\s*=\s*'([^']+)'")
    for fam, f in reg['families'].items():
        builder, want = f.get('builder'), f.get('out')
        if not builder or not want or not os.path.isfile(builder):
            continue
        try:
            src = open(builder, encoding='utf-8').read()
        except (OSError, UnicodeDecodeError):
            continue
        found = [m.group(1).lstrip('./') for m in pat.finditer(src)]
        if not found:
            continue                      # builder takes its destination some other way
        # A default may be the directory itself or a FILE inside it -- build_nci_summary.py
        # writes `<out>/nci_summary.txt` directly, which is just as correct.
        want = want.lstrip('./')
        if not any(d == want or os.path.dirname(d) == want for d in found):
            fail(8, f'{fam}: {builder} defaults its output to {found} but '
                    f'families.{fam}.out is {want!r}')


def check_merged_provenance(reg, fail):
    """A merged .pt is DERIVED, so it can go stale silently -- that is the one real cost of
    materialising the join instead of doing it at read time. This converts "could drift" into
    "drift is detected": scripts/merge_split_pt.py records each parent's sha256 in a sidecar,
    and here we re-hash the parents and compare. A changed parent means the merged file no
    longer represents its inputs and every table built from it is stale.
    """
    for m in reg['models']:
        parents = m.get('merged_from')
        if not parents:
            continue
        out = (m.get('sources') or {}).get('default')
        if not out or not os.path.isfile(out):
            fail(9, f"{m['id']}: merged_from declared but sources.default {out!r} is missing")
            continue
        prov_path = out + '.provenance.json'
        if not os.path.isfile(prov_path):
            fail(9, f"{m['id']}: {out} has no .provenance.json -- cannot tell whether it is stale; "
                    f'regenerate with scripts/merge_split_pt.py --model {m["id"]} --out {out}')
            continue
        with open(prov_path, encoding='utf-8') as fh:
            prov = json.load(fh)
        recorded = [prov['merged_from']['scalars'], prov['merged_from']['molecules']]
        if [r['path'] for r in recorded] != list(parents):
            fail(9, f"{m['id']}: registry merged_from {parents} != provenance "
                    f"{[r['path'] for r in recorded]}")
            continue
        for rec in recorded:
            if not os.path.isfile(rec['path']):
                fail(9, f"{m['id']}: parent {rec['path']} is gone -- merged file is unregeneratable")
            elif _sha256(rec['path']) != rec['sha256']:
                fail(9, f"{m['id']}: parent {rec['path']} CHANGED since the merge -- {out} is stale. "
                        f'Regenerate: python scripts/merge_split_pt.py --model {m["id"]} --out {out}')


def _sha256(path, buf=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(buf), b''):
            h.update(chunk)
    return h.hexdigest()


SHIPPED = False
SKIPPED = []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--registry', default=model_registry.DEFAULT_PATH)
    ap.add_argument('--skip', nargs='*', type=int, default=[])
    ap.add_argument('--shipped', action='store_true',
                    help='the release subset does not contain the raw measurement trees '
                         '(eval_out/, eval_plip/) or the third-party baseline .pt files, so '
                         'skip the checks that assert their presence and report the count')
    a = ap.parse_args()
    global SHIPPED
    SHIPPED = a.shipped

    reg = model_registry.load(a.registry)
    failures, pending = [], []

    def fail(n, msg):
        failures.append((n, msg))

    checks = [(1, 'unique identifiers', check_ids),
              (2, 'declared sources exist', check_sources_exist),
              (3, 'reverse coverage', check_reverse),
              (4, 'core completeness', check_core),
              (5, 'cross-family source consistency', check_source_consistency),
              (6, 'canonical pocket sha256', check_sha),
              (7, 'population rule not optional',
               lambda r, f: check_population_rule(r, f, pending)),
              (8, 'builder output dir matches registry', check_out_dirs),
              (9, 'merged .pt provenance not stale', check_merged_provenance)]

    for num, name, fn in checks:
        if num in a.skip:
            print(f'  [{num}] {name}: SKIPPED')
            continue
        before = len(failures)
        fn(reg, fail)
        n = len(failures) - before
        note = '' if num != 7 or not pending else f' ({len(pending)} pending)'
        print(f'  [{num}] {name}: {"OK" if n == 0 else f"{n} FAILURE(S)"}{note}')

    if SKIPPED:
        print(f'\n{len(SKIPPED)} presence check(s) skipped under --shipped '
              f'(raw measurement trees and third-party baselines are not in the release):')
        for msg in SKIPPED[:8]:
            print(f'    {msg}')
        if len(SKIPPED) > 8:
            print(f'    ... and {len(SKIPPED) - 8} more')

    if pending:
        print(f'\n{len(pending)} pending (declared work, not a failure):')
        for msg in pending:
            print(f'  [7] {msg}')

    if failures:
        print(f'\n{len(failures)} failure(s):')
        for num, msg in failures:
            print(f'  [{num}] {msg}')
        return 1
    print('\nAll registry invariants hold.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
