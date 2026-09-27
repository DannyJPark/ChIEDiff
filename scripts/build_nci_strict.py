#!/usr/bin/env python3
"""NCI counts on the INTERSECTION of the PLIP-measured and ProLIF-measured molecules.

WHY. results/comparison/f3_nci/nci_summary.csv gives each instrument its own denominator, and the
two are not the same set. Measured on the manuscript roster, ProLIF loses between 0.10% (MolCRAFT)
and 6.05% (KGDiff) of the PLIP molecules, because PoseCheck caps its strain relaxation and the
molecules that stall are the geometrically pathological ones -- a loss that is not missing at
random. PIDiff runs the other way: PLIP has 804 molecules against ProLIF's 831, so neither arm is
a subset of the other and the intersection is smaller than both.

That is exactly why F3 ships as TWO tables. Putting the two instruments in ONE table with a single
n is only honest if the n is real, so this builder makes it real: a molecule counts only if BOTH
instruments measured it, and then one n and one heavy-atom mean serve the whole row. It is the
same device scripts/build_pose_table.py already uses for its strict denominator.

WHAT THIS DOES NOT FIX. The intersection makes the POPULATION common; it does not put the two
instruments on a common SCALE. PLIP counts interaction objects at the atom-pair level; ProLIF
counts boolean (residue, interaction-type) cells, so several contacts with one residue collapse
into one. The two halves of a row still must not be differenced or summed -- see the notes in
results/comparison/README.md and the footnotes on the generated table.

Outputs (to --out_dir):
  nci_summary_strict.csv   one row per model, one n, one heavy_mean
  nci_summary_strict.txt   the table plus a coverage section showing what the intersection cost

Usage:
    conda run -n kgdiff python scripts/build_nci_strict.py
"""

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import build_nci_summary as bns                                          # noqa: E402
import model_registry                                                    # noqa: E402


# The PLIP arm and the ProLIF arm do not always number pockets the same way. A FLAT .pt is
# enumerated by order of appearance, so PIDiff's eval_plip/ export calls canonical pocket 47
# "46" and is shifted by one from index 46 on -- 48 of its 99 pockets. Nothing is wrong with
# either measurement: every PLIP row names its own receptor correctly and the file index matches
# the row index. The two arms simply live in different key spaces, and joining them on the raw
# p<NNN>_m<MMMM> string paired 356 of PIDiff's 785 molecules with a DIFFERENT molecule, up to 54
# heavy atoms apart. Re-keying the PLIP side through the canonical pocket order -- the same
# targetdiff-derived order build_comparison_tables uses -- drops that to zero, with no
# re-measurement. It is a no-op for the other thirteen models, whose two arms already agree.
def canonical_maps(path='results/sampling_results/targetdiff_vina_docked.pt'):
    import build_comparison_tables as _bct
    obj = _bct.load_pt(path)
    idx2name, full2idx, dir2idx = _bct.build_pocket_maps(_bct.canonical_index(obj))
    del obj
    return full2idx, dir2idx


def plip_key(r, full2idx=None, dir2idx=None):
    """Canonical p<NNN>_m<MMMM> for a PLIP row, or None when the pocket cannot be placed."""
    if full2idx is None:
        return f"p{int(r['pocket_idx']):03d}_m{int(r['mi']):04d}"
    i = full2idx.get(r.get('pk'))
    if i is None and dir2idx:
        i = dir2idx.get(str(r.get('pk', '')).split('/')[0])
    return None if i is None else f"p{i:03d}_m{int(r['mi']):04d}"


def mean(rows, fn):
    vals = [v for v in (fn(r) for r in rows) if v is not None]
    return round(float(np.mean(vals)), 4) if vals else ''


def measure(label, ptag, pdir, pt, grp, full2idx=None, dir2idx=None):
    """-> (row dict, coverage dict). Empty row when either instrument has no run."""
    keep, heavy = bns.docked_keys(pt) if os.path.isfile(pt) else (set(), {})
    if ptag in bns.CONNECTED_IS_DOCKED and heavy:
        keep = set(heavy)
    # Load PLIP UNFILTERED, then re-key, THEN apply the docking filter. The order matters:
    # bns.load_plip(tag, keep) tests each row's RAW p<NNN>_m<MMMM> against `keep`, and `keep`
    # comes from the .pt, which is canonical space. For PIDiff -- the one flat .pt here whose
    # eval_plip export numbers pockets by appearance -- those are two different key spaces, and
    # the test silently kept an arbitrary 804 of its 850 measured molecules. PLIP had in fact
    # measured all 850. Filtering after the re-key restores them.
    plip_all = bns.load_plip(ptag, None) if ptag and os.path.isdir(f'eval_plip/{ptag}') else []
    prolif = bns.load_prolif(pdir, keep, heavy) if pdir else []

    pmap = {}
    unplaced = 0
    for r in plip_all:
        k = plip_key(r, full2idx, dir2idx)
        if k is None:
            unplaced += 1
        elif keep is None or k in keep:
            pmap[k] = r
    rmap = {r['mol']: r for r in prolif}
    both = sorted(set(pmap) & set(rmap))
    cov = {'n_plip': len(pmap), 'n_prolif': len(rmap), 'n_both': len(both),
           'n_unplaced': unplaced,
           'n_plip_only': len(set(pmap) - set(rmap)), 'n_prolif_only': len(set(rmap) - set(pmap)),
           'n_docked': (len(keep) if keep is not None else None)}
    if not both:
        return None, cov

    P = [pmap[k] for k in both]
    R = [rmap[k] for k in both]

    # Heavy-atom count reaches us twice: PLIP reads it off the molecule it parsed, ProLIF takes it
    # from the source .pt. On one population they must agree, so the disagreement is reported
    # rather than assumed away -- it would mean the two arms are keyed to different molecules.
    hp = mean(P, lambda r: r.get('heavy'))
    hr = mean(R, lambda r: r.get('heavy'))
    cov['heavy_plip'], cov['heavy_prolif'] = hp, hr
    cov['heavy_gap'] = (abs(hp - hr) if isinstance(hp, float) and isinstance(hr, float) else None)
    # Per-MOLECULE agreement, not just the means: offsetting errors can leave two means equal
    # while every pairing is wrong. This is the gate that caught the PIDiff key-space mismatch.
    cov['n_heavy_mismatch'] = sum(
        1 for k in both
        if pmap[k].get('heavy') is not None and rmap[k].get('heavy') is not None
        and pmap[k]['heavy'] != rmap[k]['heavy'])

    row = {'label': label, 'group': grp, 'plip_tag': ptag, 'eval_out': pdir,
           'n': len(both), 'heavy_mean': hp}
    # `.get(c, 0)`: a molecule with none of that interaction type carries no key at all, and
    # reading that as missing instead of zero once inflated the reference hydrophobic mean.
    for c, _ in bns.PLIP_TYPES:
        row[f'plip_{c}'] = mean(P, lambda r, _c=c: r.get(_c, 0))
    row['plip_total'] = mean(P, lambda r: sum(r.get(c, 0) for c, _ in bns.PLIP_TYPES))
    row['plip_per_heavy'] = mean(
        P, lambda r: (sum(r.get(c, 0) for c, _ in bns.PLIP_TYPES) / r['heavy'])
        if r.get('heavy') else None)
    for c, _ in bns.PROLIF_TYPES:
        row[f'prolif_{c.replace("int_", "")}'] = mean(R, lambda r, _c=c: r.get(_c, 0))
    row['prolif_total'] = mean(R, lambda r: r.get('n_interactions'))
    row['prolif_per_heavy'] = mean(
        R, lambda r: (r['n_interactions'] / r['heavy'])
        if r.get('heavy') and r.get('n_interactions') is not None else None)
    return row, cov


def columns():
    return (['id', 'label', 'group', 'plip_tag', 'eval_out', 'n', 'heavy_mean']
            + [f'plip_{c}' for c, _ in bns.PLIP_TYPES] + ['plip_total', 'plip_per_heavy']
            + [f'prolif_{c.replace("int_", "")}' for c, _ in bns.PROLIF_TYPES]
            + ['prolif_total', 'prolif_per_heavy'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', default='results/comparison/f3_nci')
    args = ap.parse_args()

    ptag2id = {(m.get('ids') or {}).get('plip_tag'): m['id']
               for m in model_registry.models() if (m.get('ids') or {}).get('plip_tag')}

    full2idx, dir2idx = canonical_maps()
    rows, covs = [], []
    for label, ptag, pdir, pt, grp in bns.MODELS:
        row, cov = measure(label, ptag, pdir, pt, grp, full2idx, dir2idx)
        cov['label'] = label
        covs.append(cov)
        if row is None:
            print(f'  [skip] {label:22s} no overlap '
                  f'(plip {cov["n_plip"]}, prolif {cov["n_prolif"]})')
            continue
        row['id'] = ptag2id.get(ptag, ptag)
        rows.append(row)
        print(f'  {label:22s} plip {cov["n_plip"]:5d}  prolif {cov["n_prolif"]:5d}  '
              f'-> both {cov["n_both"]:5d}  '
              f'(-{100 * (1 - cov["n_both"] / max(cov["n_plip"], 1)):.2f}% vs plip)')

    os.makedirs(args.out_dir, exist_ok=True)
    cpath = os.path.join(args.out_dir, 'nci_summary_strict.csv')
    cols = columns()
    with open(cpath, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in rows:
            w.writerow([r.get(c, '') for c in cols])

    L = ['=' * 118,
         'NON-COVALENT INTERACTIONS -- STRICT (INTERSECTED) DENOMINATOR',
         '=' * 118, '',
         'A molecule counts only if BOTH PLIP and ProLIF measured it, so one n and one heavy-atom',
         'mean serve the whole row. nci_summary.csv next door gives each instrument its own, larger',
         'denominator and stays the file to cite for a single-instrument number.',
         '',
         'The intersection makes the POPULATION common. It does NOT put the two instruments on a',
         'common SCALE: PLIP counts atom-pair interaction objects, ProLIF counts boolean',
         '(residue, type) cells. Never difference or sum the two halves of a row.',
         '']
    hdr = ['Model', 'n plip', 'n prolif', 'n both', 'plip only', 'prolif only', 'lost vs plip',
           'heavy (plip)', 'heavy (prolif)', 'mis-paired']
    crows = []
    for c in covs:
        if not c['n_both']:
            continue
        lost = 100 * (1 - c['n_both'] / max(c['n_plip'], 1))
        crows.append([c['label'], c['n_plip'], c['n_prolif'], c['n_both'], c['n_plip_only'],
                      c['n_prolif_only'], f'{lost:.2f}%',
                      c.get('heavy_plip', ''), c.get('heavy_prolif', ''),
                      c.get('n_heavy_mismatch', 0)])
    w0 = [max([len(str(h))] + [len(str(r[i])) for r in crows]) for i, h in enumerate(hdr)]

    def fmt(cs):
        return '  '.join(str(c).ljust(w0[i]) if i == 0 else str(c).rjust(w0[i])
                         for i, c in enumerate(cs)).rstrip()

    L += ['COVERAGE -- what the intersection cost', '',
          fmt(hdr), fmt(['-' * x for x in w0])] + [fmt(r) for r in crows]
    mism = [c for c in covs if c.get('n_heavy_mismatch')]
    L += ['', ('PAIRING GATE: every molecule agrees on its heavy-atom count across the two arms, '
               'on every row.' if not mism else
               'PAIRING GATE FAILED -- the two arms are keyed to different molecules on: '
               + ', '.join(f'{c["label"]} ({c["n_heavy_mismatch"]}/{c["n_both"]})' for c in mism))]
    gaps = [c for c in covs if c.get('heavy_gap') not in (None, '') and c['heavy_gap'] > 0.01]
    L += ['', ('HEAVY-ATOM CROSS-CHECK: the two arms agree to <= 0.01 on every row.' if not gaps
               else 'HEAVY-ATOM MISMATCH on: ' + ', '.join(f'{c["label"]} ({c["heavy_gap"]:.3f})'
                                                           for c in gaps))]
    tpath = os.path.join(args.out_dir, 'nci_summary_strict.txt')
    with open(tpath, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(L) + '\n')

    print(f'\nWrote:\n  {cpath}\n  {tpath}')


if __name__ == '__main__':
    main()
