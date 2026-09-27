#!/usr/bin/env python3
"""
Aggregate the per-pocket PoseCheck CSVs into the three metrics the CBYG paper reports
(its Table 5): Avg. Clash, Avg. Strain Energy, Avg. Interaction.

    python scripts/aggregate_posecheck.py \
        --dirs eval_out/vina_fixed eval_out/pignet_fixed eval_out/pgdiff_fixed eval_out/novdw \
        --labels vina_fixed pignet_fixed pgdiff_fixed novdw \
        --out results/comparison

Reads  <dir>/posecheck/pocket*.csv  (+ .timeouts sidecars)
Writes <out>/posecheck_summary.csv and <out>/posecheck_summary.txt

Strain energy is reported as mean AND median. The paper gives means only, and its values
(1e7 - 1e16) show those means are set by a handful of pathological molecules rather than by the
bulk of the sample; the median says what a typical molecule looks like. Both are printed so neither
picture is hidden. Molecules that timed out or that PoseCheck could not process are counted and
printed too, so a model cannot look good merely by having dropped its worst poses.
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np


def _f(x):
    """float or None -- never let one missing field discard a molecule's other metrics."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def molecule_index(src_pt):
    """p<NNN>_m<MMMM> -> (docked?, heavy_atom_count) for a sampling/consolidated .pt.

    The join key is the SDF title eval_export_sdf.py writes: pocket index + the molecule's ORIGINAL
    position in that pocket's list, so it survives molecules being skipped.
    """
    if not src_pt or not os.path.isfile(src_pt):
        return {}
    sys.path.insert(0, os.path.abspath('.'))
    from scripts.eval_export_sdf import load_grouped

    def docked(e):
        v = e.get('vina')
        if isinstance(v, dict) and v.get('dock'):
            return v['dock'][0].get('affinity') is not None
        if isinstance(v, list) and v:
            return v[0].get('affinity') is not None
        return False

    out = {}
    for pi, pocket in enumerate(load_grouped(src_pt)):
        for mi, e in enumerate(pocket):
            mol = e.get('mol')
            if mol is None or mol.GetNumConformers() == 0:
                continue
            out[f'p{pi:03d}_m{mi:04d}'] = (docked(e), mol.GetNumHeavyAtoms())
    return out


def load_model(d, src_pt=None, docked_only=False):
    """Return (rows, n_timeout, n_pockets, n_dropped_nondocked).

    PoseCheck runs on everything eval_export_sdf.py exported, which is NOT the same set for every
    model: the reference .pt files (targetdiff / kgdiff / pidiff / novdw) were saved
    docking-successful-only, while consolidate_docking.py keeps docking-FAILED molecules too. Left
    unfiltered, our physics runs are therefore scored over a superset that the baselines never
    include -- so `docked_only=True` restricts every model to its docking-successful molecules,
    which is a no-op for the baselines and the only way to compare like with like.
    """
    rows, n_timeout, n_nondocked = [], 0, 0
    meta = molecule_index(src_pt) if src_pt else {}
    files = sorted(glob.glob(os.path.join(d, 'posecheck', 'pocket*.csv')))
    for f in files:
        with open(f) as fh:
            for r in csv.DictReader(fh):
                name = r.get('molecule')
                is_docked, heavy = meta.get(name, (None, None))
                if docked_only and meta and not is_docked:
                    n_nondocked += 1
                    continue
                rec = {'clashes': _f(r.get('clashes')),
                       'strain': _f(r.get('strain_energy')),
                       'inter': _f(r.get('n_interactions')),
                       'heavy': heavy}
                rec.update({k: _f(v) for k, v in r.items() if k.startswith('int_')})
                if rec['clashes'] is None and rec['strain'] is None and rec['inter'] is None:
                    continue
                rows.append(rec)
        t = f + '.timeouts'
        if os.path.isfile(t):
            with open(t) as fh:
                n_timeout += sum(1 for line in fh if line.strip())
    return rows, n_timeout, len(files), n_nondocked


def stats(rows):
    """Per-metric statistics, each over its OWN non-missing subset.

    Each metric carries its own n: a molecule whose strain-energy relaxation failed still has a
    valid clash and interaction count, and dropping the whole molecule (as an earlier version did)
    silently discarded those. Reporting n per metric makes any such gap visible instead.
    """
    if not rows:
        return {}
    col = lambda k: np.array([r[k] for r in rows if r.get(k) is not None], dtype=float)
    clash, inter, heavy = col('clashes'), col('inter'), col('heavy')
    strain = col('strain')
    strain = strain[np.isfinite(strain)]
    # interaction per heavy atom, molecule-wise -- interaction counts scale with ligand size
    per_atom = np.array([r['inter'] / r['heavy'] for r in rows
                         if r.get('inter') is not None and r.get('heavy')], dtype=float)
    out = {
        'n': len(rows),
        'n_clash': clash.size, 'n_strain': strain.size, 'n_inter': inter.size,
        'clash_mean': clash.mean() if clash.size else float('nan'),
        'clash_median': float(np.median(clash)) if clash.size else float('nan'),
        'strain_mean': strain.mean() if strain.size else float('nan'),
        'strain_median': float(np.median(strain)) if strain.size else float('nan'),
        'inter_mean': inter.mean() if inter.size else float('nan'),
        'inter_median': float(np.median(inter)) if inter.size else float('nan'),
        'heavy_mean': heavy.mean() if heavy.size else float('nan'),
        'inter_per_atom': per_atom.mean() if per_atom.size else float('nan'),
    }
    # Per-pocket CSVs only carry an int_<type> column if SOME molecule in that pocket fired it, so a
    # molecule that made ZERO of a type has no cell rather than a 0 -- for NATIVE (1 mol/pocket) that
    # means 23/100 mols silently vanish from the HBAcceptor mean, biasing it UP (2.64 over 77 vs the
    # true 2.03 over 100). Any molecule that WAS measured (has an interaction count) but lacks the
    # column made 0 of that type: count it as 0.0, do not drop it. Matches build_nci_summary.load_prolif.
    int_cols = sorted({k for r in rows for k in r if k.startswith('int_')})
    for c in int_cols:
        v = np.array([(r.get(c) if r.get(c) is not None else 0.0)
                      for r in rows if r.get('inter') is not None], dtype=float)
        out[c + '_mean'] = float(v.mean()) if v.size else float('nan')
    return out


def main():
    ap = argparse.ArgumentParser()
    # Roster comes from configs/models.json (Phase B, 2026-08-21). This was the ONE family with
    # no roster at all: --dirs/--labels/--src were typed by hand at every invocation, which is why
    # posecheck_summary.csv had 12 rows, pose_quality_summary_posebusters.csv had 7 (two of them
    # models retired months earlier), and posecheck_summary_PREV/_unfiltered had a third list.
    ap.add_argument('--registry', default='configs/models.json')
    ap.add_argument('--table', default='posecheck',
                    help="which families.quality.tables list to render")
    ap.add_argument('--dirs', nargs='+', default=None,
                    help='ad-hoc override; bypasses the registry (diagnostics only)')
    ap.add_argument('--labels', nargs='*', default=None)
    ap.add_argument('--src', nargs='*', default=None,
                    help='sampling/consolidated .pt per dir; needed for the docking-successful '
                         'filter and for the heavy-atom (size) column')
    # D5: the docking-successful population is NOT optional. It used to be an opt-in flag, and
    # forgetting it compared our full generated set against baselines that only ever stored their
    # docking-successful molecules -- the exact reversal documented in posecheck-denominator-mismatch.
    ap.add_argument('--unfiltered', action='store_true',
                    help='DIAGNOSTIC ONLY: skip the docking-successful filter. Refuses to write '
                         'into results/comparison/ -- use results/diagnostics/.')
    ap.add_argument('--out', default='results/comparison/f5_pose_quality')
    ap.add_argument('--basename', default='posecheck_summary')
    args = ap.parse_args()

    args.docked_only = not args.unfiltered
    if args.unfiltered and 'results/comparison' in os.path.abspath(args.out):
        raise SystemExit('--unfiltered may not write into results/comparison/ -- '
                         'point --out at results/diagnostics/')

    ROW_ID = {}          # published label -> registry id, the join key for the master table
    if args.dirs:
        labels = args.labels or [os.path.basename(d.rstrip('/')) for d in args.dirs]
        srcs = args.src or [None] * len(args.dirs)
        per_model_docked = [args.docked_only] * len(args.dirs)
    else:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import model_registry                                            # noqa: E402
        rows = [m for m in model_registry.models(args.registry)
                if model_registry.in_table(m, 'quality', args.table)]
        rows.sort(key=lambda m: (m['families']['quality'].get('legacy_order', 10**6), m['id']))
        args.dirs = [f"eval_out/{m['ids']['eval_out']}" for m in rows]
        labels = [model_registry.label_for(m, 'quality') for m in rows]
        srcs = [model_registry.source_for(m, 'quality') for m in rows]
        # docked_policy 'connected_is_docked': the source was rebuilt by load_samples, which drops
        # fragmented molecules first, so every measured molecule is ALREADY docking-successful by
        # the docked<=>connected identity -- and the file carries no vina field to filter on.
        # Applying the filter would drop all of them. Same rule build_nci_summary applies.
        per_model_docked = [args.docked_only and m.get('docked_policy') != 'connected_is_docked'
                            for m in rows]
        ROW_ID = {model_registry.label_for(m, 'quality'): m['id'] for m in rows}
        print(f'[roster] {len(rows)} model(s) from {args.registry} (table={args.table!r})')
    os.makedirs(args.out, exist_ok=True)

    res = {}
    for d, lab, src, dok in zip(args.dirs, labels, srcs, per_model_docked):
        rows, n_to, n_pk, n_nd = load_model(d, src, dok)
        s = stats(rows)
        s['n_timeout'] = n_to
        s['n_pockets'] = n_pk
        s['n_nondocked_dropped'] = n_nd
        res[lab] = s
        print(f'{lab:16s} pockets={n_pk:3d} mols={s.get("n", 0):5d} timeouts={n_to}'
              + (f' dropped_nondocked={n_nd}' if n_nd else ''))

    # timeout rate over ATTEMPTED molecules, so the strain numbers are always read next to how
    # many of the hardest molecules never finished
    for s in res.values():
        attempted = s.get('n', 0) + s.get('n_timeout', 0)
        s['n_attempted'] = attempted
        s['timeout_pct'] = round(100.0 * s.get('n_timeout', 0) / attempted, 2) if attempted else ''

    int_cols = sorted({k for s in res.values() for k in s if k.startswith('int_')})
    cols = (['n_pockets', 'n_attempted', 'n', 'n_timeout', 'timeout_pct', 'n_nondocked_dropped',
             'n_clash', 'n_strain', 'n_inter',
             'clash_mean', 'clash_median',
             'strain_mean', 'strain_median',
             'inter_mean', 'inter_median', 'heavy_mean', 'inter_per_atom'] + int_cols)

    csv_path = os.path.join(args.out, args.basename + '.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['id', 'model'] + cols)
        for lab in labels:
            s = res[lab]
            w.writerow([ROW_ID.get(lab, ''), lab] + [s.get(c, '') for c in cols])

    lines = [
        'POSECHECK  (clashes / strain energy / interactions)',
        '=' * 108, '',
        'These are the metrics the CBYG paper reports in its Table 5, despite that table being',
        'captioned "Posebusters". Definitions below are read off the installed posecheck 1.3.1',
        'sources, so they describe what these numbers actually are here.', '',
        'Clash (lower better)   Count of protein-ligand ATOM PAIRS with',
        '                       d(i,j) < Rvdw(lig i) + Rvdw(prot j) - 0.5 A   (RDKit GetRvdw radii;',
        '                       0.5 A is posecheck\'s default tolerance). Hydrogens count -- the',
        '                       receptor is protonated with `reduce`. It is a PAIR count, not a count',
        '                       of offending atoms, and is not normalised by ligand or pocket size.',
        'Strain (lower better)  E(pose, locally relaxed) - min E(51 freely relaxed conformers), UFF.',
        '                       Local = UFF minimisation with each atom constrained to move <= 0.1 A;',
        '                       global = 50 re-embedded conformers + one unconstrained relaxation from',
        '                       the pose. So: how much worse this pose is than the best conformer found',
        '                       for the same molecule. Heavy-tailed -- the MEAN is set by a handful of',
        '                       pathological molecules, the MEDIAN is the typical one. Both are given.',
        'Inter (higher better)  ProLIF `Fingerprint()` on the pose: number of (residue, interaction-',
        '                       type) cells that fire. Only HBAcceptor, HBDonor, Hydrophobic and',
        '                       VdWContact ever fire in our data.', '',
        'READ THE TIMEOUT COLUMN. n_mol counts molecules with a PoseCheck result; t/o counts those',
        'dropped when the strain-energy force-field relaxation exceeded the per-molecule limit.',
        'Those are disproportionately the molecules with the WORST strain, so every strain number',
        'here is optimistic by an amount that grows with t/o. The limit is identical across models,',
        'so the comparison stays fair -- and a model that times out more is itself evidence of worse',
        'geometry. t/o% is over attempted molecules (n_mol + t/o).', '',
        'Interaction counts scale with ligand size, so heavy-atom mean (HA) and interactions per',
        'heavy atom are printed next to the raw count -- compare those, not the raw count alone.',
        'HA differs a lot (Pocket2Mol 17.7, DeepICL 20.6, PharDiff 26.3), so I/HA -- not raw Inter --',
        'is the only size-fair interaction column.', '',
        'VERSION / SETTINGS: every row was measured with the SAME PoseCheck 1.3.1 / ProLIF 2.2.0',
        'env and the same eval_posecheck.py (clash tolerance 0.5 A, UFF strain over 51 conformers),',
        'verified by identical CSV schema across all models. Interaction = ProLIF default count=False',
        '(per-(residue,type) bool); per-contact counts and PLIP live in results/comparison/nci_summary.txt.',
        '',
        'DeepICL H FIX: DeepICL ships OpenBabel SDFs with no explicit H, which made RDKit set',
        'noImplicit=True on donor N so AddHs added no donor H and ProLIF HBDonor read 0 for ALL mols.',
        'Repaired (clear noImplicit + sanitize) before this run -- HBDonor 0.00 -> 0.33, VdWContact',
        '5.79 -> 7.48. No other model is affected (only DeepICL is OpenBabel-sourced). Full write-up:',
        'analysis/DEEPICL_HYDROGEN_BUG.md.', '',
        f"{'model':16s} {'n_pk':>5s} {'n_mol':>6s} {'t/o':>5s} {'t/o%':>6s} "
        f"{'Clash avg':>10s} {'Clash med':>10s} {'Strain avg':>12s} {'Strain med':>11s} "
        f"{'Inter avg':>10s} {'HA':>6s} {'I/HA':>6s}",
        '-' * 122,
    ]
    for lab in labels:
        s = res[lab]
        if not s.get('n'):
            lines.append(f'{lab:16s}  (no results)')
            continue
        attempted = s['n'] + s['n_timeout']
        to_pct = 100.0 * s['n_timeout'] / attempted if attempted else 0.0
        lines.append(
            f"{lab:16s} {s['n_pockets']:5d} {s['n']:6d} {s['n_timeout']:5d} {to_pct:5.1f}% "
            f"{s['clash_mean']:10.2f} {s['clash_median']:10.2f} "
            f"{s['strain_mean']:12.3g} {s['strain_median']:11.2f} "
            f"{s['inter_mean']:10.2f} {s['heavy_mean']:6.1f} {s['inter_per_atom']:6.2f}")
        if s['n_clash'] != s['n'] or s['n_strain'] != s['n'] or s['n_inter'] != s['n']:
            lines.append(f"{'':16s}   (per-metric n: clash {s['n_clash']}, strain {s['n_strain']},"
                         f" inter {s['n_inter']})")
        if s.get('n_nondocked_dropped'):
            lines.append(f"{'':16s}   (dropped {s['n_nondocked_dropped']} non-docked molecules)")

    if int_cols:
        lines += ['', 'Interaction breakdown (mean count per molecule):', '',
                  f"{'model':16s} " + ' '.join(f'{c[4:]:>14s}' for c in int_cols), '-' * 100]
        for lab in labels:
            s = res[lab]
            if not s.get('n'):
                continue
            lines.append(f'{lab:16s} ' + ' '.join(f'{s.get(c, 0.0):14.2f}' for c in int_cols))

    txt_path = os.path.join(args.out, args.basename + '.txt')
    with open(txt_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f'\nWrote:\n  {csv_path}\n  {txt_path}')


if __name__ == '__main__':
    main()
