#!/usr/bin/env python3
"""Join every pose-fidelity metric onto one row per generated molecule, for all models.

Inputs per model (each optional -- whatever has finished is used, and what is missing is counted):
    <dir>/strain.csv                  T4-a  MMFF94s internal strain + size/flexibility confounders
    <dir>/pose_rmsd.csv               T1/T4-c  RMSD to smina / gnina / vinardo poses
    <dir>/vina_meeko/pocket*_ref.csv  Phase 1B  AutoDock Vina + EXACT meeko atom mapping
    <dir>/posebusters/pocket*.csv     T4-b  per-check physical validity
    <dir>/posecheck/pocket*.csv       T4-d  protein-ligand clashes + interactions

Output:
    results/pose_fidelity/master.csv  long format, one row per (model, molecule)

Join key is the SDF title p<pocket>_m<index>, which every tool preserves, so a molecule dropped by
one tool never shifts another tool's rows onto the wrong molecule.

Nothing is imputed. A metric a tool did not produce stays empty and is reported as missing,
because the missingness is itself the finding here: PoseBusters/PoseCheck time out on
pathological molecules (see the .timeouts sidecars) and meeko cannot prepare disconnected ones,
and both failure modes hit the physics models far harder than the baselines. Treating them as
"not measured" rather than "absent" is what keeps the model comparison honest (defect D3).

Usage:
    python scripts/aggregate_pose_fidelity.py
"""
import argparse
import csv
import glob
import json
import os
from collections import defaultdict

PB_CHECKS = [
    'sanitization', 'inchi_convertible', 'all_atoms_connected', 'no_radicals',
    'bond_lengths', 'bond_angles', 'internal_steric_clash', 'aromatic_ring_flatness',
    'non-aromatic_ring_non-flatness', 'double_bond_flatness', 'internal_energy',
    'protein-ligand_maximum_distance', 'minimum_distance_to_protein',
    'minimum_distance_to_organic_cofactors', 'minimum_distance_to_inorganic_cofactors',
    'minimum_distance_to_waters', 'volume_overlap_with_protein',
    'volume_overlap_with_organic_cofactors', 'volume_overlap_with_inorganic_cofactors',
    'volume_overlap_with_waters',
]

# molecule-intrinsic vs pose(protein-context) split: these move in OPPOSITE directions for the
# physics models, so a single pass-rate hides the trade-off entirely.
PB_MOLECULE = ['sanitization', 'inchi_convertible', 'all_atoms_connected', 'no_radicals',
               'bond_lengths', 'bond_angles', 'internal_steric_clash',
               'aromatic_ring_flatness', 'non-aromatic_ring_non-flatness',
               'double_bond_flatness', 'internal_energy']
PB_POSE = [c for c in PB_CHECKS if c not in PB_MOLECULE]


def read_csv_keyed(path, key='name'):
    if not (os.path.isfile(path) and os.path.getsize(path) > 0):
        return {}
    with open(path) as f:
        return {r[key]: r for r in csv.DictReader(f) if r.get(key)}


def read_glob_keyed(pattern, key):
    out = {}
    for p in sorted(glob.glob(pattern)):
        if os.path.getsize(p) == 0:
            continue
        with open(p) as f:
            for r in csv.DictReader(f):
                if r.get(key):
                    out.setdefault(r[key], r)
    return out


def count_timeouts(pattern):
    n = 0
    for p in glob.glob(pattern):
        with open(p) as f:
            n += sum(1 for line in f if line.strip())
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--registry', default='configs/models.json')
    ap.add_argument('--out', default='results/pose_fidelity/master.csv')
    args = ap.parse_args()

    import sys as _sys, os as _os

    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

    import model_registry                                    # noqa: E402

    # Unified registry; load_as() renders the legacy 'pose' view, proven

    # byte-identical by scripts/check_registry_sync.py.

    reg = model_registry.load_as(args.registry, 'pose')
    rows, cov = [], {}

    for m in reg['models']:
        d, tag = m['dir'], m['tag']
        # The docking arms were capped at 30 molecules/pocket partway through, but 102 pockets
        # had already been docked at full depth. Without this flag those pockets would contribute
        # ~100 molecules while the rest contribute 30, silently overweighting them inside each
        # model. subset_30.txt is the authoritative list of what every engine was asked to dock.
        sub_path = os.path.join(d, 'subset_30.txt')
        subset = set()
        if os.path.isfile(sub_path):
            subset = {ln.strip() for ln in open(sub_path) if ln.strip()}
        # docking-successful set (the reporting population; see scripts/mark_docked.py)
        dk_path = os.path.join(d, 'docked_names.txt')
        docked = ({ln.strip() for ln in open(dk_path) if ln.strip()}
                  if os.path.isfile(dk_path) else None)
        strain = read_csv_keyed(os.path.join(d, 'strain.csv'))
        prmsd = read_csv_keyed(os.path.join(d, 'pose_rmsd.csv'))
        meeko = read_glob_keyed(os.path.join(d, 'vina_meeko', 'pocket*_ref.csv'), 'name')
        pb = read_glob_keyed(os.path.join(d, 'posebusters', 'pocket*.csv'), 'molecule')
        pc = read_glob_keyed(os.path.join(d, 'posecheck', 'pocket*.csv'), 'molecule')

        names = set(strain) | set(prmsd) | set(meeko)
        for name in sorted(names):
            s, pr, mk = strain.get(name, {}), prmsd.get(name, {}), meeko.get(name, {})
            b, c = pb.get(name, {}), pc.get(name, {})
            r = {
                'model': tag, 'group': m['group'], 'name': name,
                'in_subset30': int(name in subset) if subset else 1,
                'orig_docked': (1 if docked is None else int(name in docked)),
                'pocket_idx': s.get('pocket_idx') or pr.get('pocket_idx') or mk.get('pocket_idx', ''),
                'heavy': s.get('heavy', ''), 'n_rot': s.get('n_rot', ''),
                'tpsa': s.get('tpsa', ''), 'n_frags': s.get('n_frags', ''),
                # T4-a
                'strain_mmff': s.get('strain_mmff', ''),
                'strain_per_heavy': s.get('strain_per_heavy', ''),
                'strain_status': s.get('strain_status', ''),
                # Phase 1B: Vina with exact atom mapping
                'vm_prep_ok': mk.get('prep_ok', ''), 'vm_prep_reason': mk.get('prep_reason', ''),
                'vm_score_only': mk.get('vina_score_only', ''),
                'vm_minimize': mk.get('vina_minimize', ''), 'vm_dock': mk.get('vina_dock', ''),
                'vm_dLocal': mk.get('dLocal', ''), 'vm_dGlobal': mk.get('dGlobal', ''),
                'vm_rmsd_gen_min': mk.get('rmsd_gen_min', ''),
                'vm_rmsd_gen_dock': mk.get('rmsd_gen_dock', ''),
                # T1 / T4-c
                'rmsd_min_smina': pr.get('rmsd_gen_min_smina', ''),
                'rmsd_dock_smina': pr.get('rmsd_gen_dock_smina', ''),
                'rmsd_min_gnina': pr.get('rmsd_gen_min_gnina', ''),
                'rmsd_dock_gnina': pr.get('rmsd_gen_dock_gnina', ''),
                # Vinardo: the scoring-function-swap control. `min` is the box- and
                # search-independent form of the test, so it must travel with `dock`.
                'rmsd_min_vinardo': pr.get('rmsd_gen_min_vinardo', ''),
                'rmsd_dock_vinardo': pr.get('rmsd_gen_dock_vinardo', ''),
                'method_dock_smina': pr.get('method_dock_smina', ''),
                'method_dock_gnina': pr.get('method_dock_gnina', ''),
                'method_min_vinardo': pr.get('method_min_vinardo', ''),
                'method_dock_vinardo': pr.get('method_dock_vinardo', ''),
                # T4-d
                'pc_clashes': c.get('clashes', ''),
                'pc_strain': c.get('strain_energy', ''),
                'pc_n_interactions': c.get('n_interactions', ''),
            }
            # T4-b: per-check booleans + the two group pass rates
            if b:
                got = [k for k in PB_CHECKS if b.get(k) not in ('', None)]
                r['pb_measured'] = 1
                r['pb_pass_all'] = int(all(b[k] == 'True' for k in got)) if got else ''
                r['pb_mol_pass'] = int(all(b[k] == 'True' for k in PB_MOLECULE
                                           if b.get(k) not in ('', None))) if got else ''
                r['pb_pose_pass'] = int(all(b[k] == 'True' for k in PB_POSE
                                            if b.get(k) not in ('', None))) if got else ''
                for k in PB_CHECKS:
                    v = b.get(k)
                    r[f'pb_{k}'] = '' if v in ('', None) else int(v == 'True')
            else:
                r['pb_measured'] = 0
                r['pb_pass_all'] = r['pb_mol_pass'] = r['pb_pose_pass'] = ''
                for k in PB_CHECKS:
                    r[f'pb_{k}'] = ''
            rows.append(r)

        cov[tag] = {
            'n': len(names), 'strain': len(strain), 'pose_rmsd': len(prmsd),
            'vina_meeko': len(meeko), 'posebusters': len(pb), 'posecheck': len(pc),
            'pb_timeouts': count_timeouts(os.path.join(d, 'posebusters', '*.timeouts')),
            'pc_timeouts': count_timeouts(os.path.join(d, 'posecheck', '*.timeouts')),
        }

    if not rows:
        raise SystemExit('no rows -- has anything finished yet?')
    F = list(rows[0].keys())
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=F, lineterminator='\n')
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in F})

    print(f'{"model":14s} {"n":>7s} {"strain":>7s} {"rmsd":>7s} {"vina_mk":>8s} '
          f'{"PB":>6s} {"PC":>6s} {"PBt/o":>6s} {"PCt/o":>6s}')
    print('-' * 78)
    for t, c in cov.items():
        print(f'{t:14s} {c["n"]:7d} {c["strain"]:7d} {c["pose_rmsd"]:7d} {c["vina_meeko"]:8d} '
              f'{c["posebusters"]:6d} {c["posecheck"]:6d} {c["pb_timeouts"]:6d} {c["pc_timeouts"]:6d}')
    print(f'\n-> {args.out}  ({len(rows)} rows)')
    print('PBt/o & PCt/o are molecules the tool TIMED OUT on and never scored -- they are excluded')
    print('from that tool\'s rates, so quote them next to any pass-rate.')


if __name__ == '__main__':
    main()
