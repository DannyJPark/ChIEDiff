#!/usr/bin/env python3
"""Generated-pose -> engine-pose RMSD for every engine, one row per molecule.

Primary metrics of the pose-fidelity experiment (analysis/EXPERIMENT_pose_fidelity.md):
    rmsd_gen_min_*   how far local minimisation moves the pose -- the pure "is this already a
                     local minimum" measure, independent of the box and of the global search
    rmsd_gen_dock_*  how far a full re-search moves it (secondary; box-dependent)

RMSD is the non-superposed, symmetry-minimised heavy-atom RMSD (rmsd_core). Superposing would
answer a different question and always flatters the model.

Atom correspondence, and why `method` is recorded per row
---------------------------------------------------------
smina/gnina take SDF in and give SDF out, but they round-trip through PDBQT internally: measured
on pocket000, 0/20 molecules came back with their heavy-atom order intact, and 5/20 came back with
bond orders re-perceived so that an exact graph match is impossible. So `rmsd_core.rmsd_best_effort`
tries exact CalcRMS first and only then falls back to skeleton isomorphism (with the bond-count
gate that makes it a true isomorphism). Which path was used is written to `method_*` so the
fallback's known downward bias (D2) can be measured against the meeko/Vina arm rather than assumed
away.

Coverage is reported per model and must be read alongside
results/pose_fidelity/fragmentation_census.csv: the physics models emit 15-28% disconnected
molecules and the baselines 0%, so equal-looking coverage here is not equal-quality output.

Usage:
    python scripts/pose_rmsd.py --tag vina_fixed
    python scripts/pose_rmsd.py --all --jobs 8
"""
import argparse
import collections
import csv
import glob
import json
import multiprocessing as mp
import os
import sys

import numpy as np
from rdkit import Chem, RDLogger

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rmsd_core import compare                                   # noqa: E402

RDLogger.DisableLog('rdApp.*')

# engine -> (subdir, filename suffix) for each mode we may have run
SOURCES = [
    ('smina',   'min',  'smina',        '_min.sdf'),
    ('smina',   'dock', 'smina',        '_dock.sdf'),
    ('gnina',   'min',  'gnina',        '_min.sdf'),
    ('gnina',   'dock', 'gnina',        '_dock.sdf'),
    ('vinardo', 'min',  'smina_vinardo', '_min.sdf'),
    ('vinardo', 'dock', 'smina_vinardo', '_dock.sdf'),
]


def load_named(*paths):
    """-> {name: mol}, first occurrence wins across all paths. sanitize=False so a bad pose is
    flagged, not lost. Multiple paths so the docking arm's two passes -- the first 30/pocket
    (pocket*_dock.sdf) and the remainder (pocket*_dock_rest.sdf), which are disjoint by molecule
    name -- merge into one full-coverage set."""
    out = {}
    for path in paths:
        if not (os.path.isfile(path) and os.path.getsize(path) > 0):
            continue
        for m in Chem.SDMolSupplier(path, sanitize=False, removeHs=False):
            if m is None or not m.HasProp('_Name'):
                continue
            out.setdefault(m.GetProp('_Name'), m)
    return out


def one_pocket(task):
    model_dir, pi = task
    tag = f'{pi:03d}'
    gen = load_named(os.path.join(model_dir, 'sdf', f'pocket{tag}.sdf'))
    if not gen:
        return []
    engines = {}
    for eng, mode, sub, suf in SOURCES:
        # Glob every pass rather than naming them: dock ran as subset + _rest, and a third
        # `_lost` pass was added 2026-08-05 to recover the molecules that make_subset_sdf.py
        # dropped by sanitising (they were in neither sdf30/ nor sdfrest30/, so no engine ever
        # saw them while they stayed in the reporting denominator). Passes are disjoint by
        # molecule name and load_named keeps the first occurrence, so the union is safe.
        stem = os.path.join(model_dir, sub, f'pocket{tag}{suf[:-4]}')
        paths = [stem + '.sdf'] + sorted(p for p in glob.glob(stem + '_*.sdf'))
        engines[(eng, mode)] = load_named(*paths)

    rows = []
    for name, g in gen.items():
        row = {'name': name, 'pocket_idx': pi, 'heavy': g.GetNumHeavyAtoms()}
        try:
            row['n_frags'] = len(Chem.GetMolFrags(g))
        except Exception:                                       # noqa: BLE001
            row['n_frags'] = -1
        for (eng, mode), pool in engines.items():
            o = pool.get(name)
            key = f'{mode}_{eng}'
            if o is None:
                row[f'rmsd_gen_{key}'] = ''
                row[f'method_{key}'] = 'absent'
                row[f'cdisp_{key}'] = ''
                row[f'jensen_{key}'] = ''
                continue
            c = compare(g, o)
            row[f'rmsd_gen_{key}'] = '' if c['rmsd'] is None else round(c['rmsd'], 4)
            row[f'method_{key}'] = c['method'] + ('+maxed' if c['match_maxed_out'] else '')
            row[f'cdisp_{key}'] = ('' if c['centroid_disp'] is None
                                   else round(c['centroid_disp'], 4))
            row[f'jensen_{key}'] = '' if c['jensen_ok'] is None else int(c['jensen_ok'])
        rows.append(row)
    return rows


def fields():
    f = ['name', 'pocket_idx', 'heavy', 'n_frags']
    for eng, mode, _, _ in SOURCES:
        k = f'{mode}_{eng}'
        f += [f'rmsd_gen_{k}', f'method_{k}', f'cdisp_{k}', f'jensen_{k}']
    return f


def run_model(model, jobs):
    sdf_dir = os.path.join(model['dir'], 'sdf')
    tasks = [(model['dir'], int(f[6:9])) for f in sorted(os.listdir(sdf_dir))
             if f.startswith('pocket') and f.endswith('.sdf')]
    if jobs > 1:
        with mp.Pool(jobs) as pool:
            rows = [r for chunk in pool.imap(one_pocket, tasks) for r in chunk]
    else:
        rows = [r for t in tasks for r in one_pocket(t)]
    rows.sort(key=lambda r: (r['pocket_idx'], r['name']))
    if not rows:
        print(f'{model["tag"]:14s} no molecules found')
        return

    F = fields()
    out = os.path.join(model['dir'], 'pose_rmsd.csv')
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=F, lineterminator='\n')
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in F})

    print(f'\n{model["tag"]}  (n={len(rows)}) -> {out}')
    for eng, mode, _, _ in SOURCES:
        k = f'{mode}_{eng}'
        vals = [r[f'rmsd_gen_{k}'] for r in rows if isinstance(r.get(f'rmsd_gen_{k}'), float)]
        meth = collections.Counter(r.get(f'method_{k}', 'absent') for r in rows)
        if meth.get('absent', 0) == len(rows):
            continue
        avail = len(rows) - meth.get('absent', 0)
        jv = sum(1 for r in rows if r.get(f'jensen_{k}') == 0)
        v = np.array(vals) if vals else np.array([np.nan])
        print(f'  {k:14s} n={len(vals):6d}/{avail:6d} avail  med={np.nanmedian(v):6.3f}A  '
              f'<1A={100*np.mean(v < 1):5.1f}%  exact={meth.get("calcrms",0):5d} '
              f'fallback={meth.get("skeleton_iso",0):4d} failed={meth.get("failed",0):3d}  '
              f'jensen_viol={jv}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--registry', default='configs/models.json')
    ap.add_argument('--tag', default='')
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--jobs', type=int, default=4)
    args = ap.parse_args()

    import sys as _sys, os as _os

    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

    import model_registry                                    # noqa: E402

    # Unified registry; load_as() renders the legacy 'pose' view, proven

    # byte-identical by scripts/check_registry_sync.py.

    reg = model_registry.load_as(args.registry, 'pose')
    models = [m for m in reg['models'] if args.all or m['tag'] == args.tag]
    if not models:
        raise SystemExit('nothing selected: pass --tag <t> or --all')
    for m in models:
        run_model(m, args.jobs)
    print('\nmethod=skeleton_iso rows used the bond-order-agnostic fallback: correct isomorphism,')
    print('but it can only bias RMSD DOWNWARD. Compare against the meeko/Vina arm before quoting.')
    return 0


if __name__ == '__main__':
    mp.set_start_method('fork', force=True)
    sys.exit(main())
