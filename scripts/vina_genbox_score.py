#!/usr/bin/env python3
"""AutoDock Vina score_only of each generated pose, in a box fitted to the molecule itself.

The box is centred on the bounding box of the molecule with its hydrogens added, and each edge
is that molecule's extent plus --buffer. The ligand is prepared through Open Babel and meeko
0.1.dev3, and the receptor is the <pdb>_rec.pdbqt beside the manifest's receptor. This is the
score the generated-molecule .pt stores as vina['score_only'].

The molecules are read from the .pt rather than from the exported SDFs, because an SDF record
does not carry the explicit hydrogen count of a hypervalent phosphorus. Rows are named
p<NNN>_m<MMMM> as eval_export_sdf.py names them. A fragmented molecule is written with status
'fragmented' and no score, never dropped.

Run with the ged-train environment, which holds Open Babel 3.1.1, meeko 0.1.dev3 and vina 1.2.2:
    "$CONDA_BASE/envs/ged-train/bin/python" scripts/vina_genbox_score.py \
        --pt mymodel.pt --dir eval_out/mymodel --pockets 0 1 2
"""
import argparse
import contextlib
import csv
import io
import os
import sys
import tempfile
from pathlib import Path

from meeko import MoleculePreparation
from openbabel import pybel
from rdkit import Chem, RDLogger
from vina import Vina

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_export_sdf import load_grouped                         # noqa: E402

RDLogger.DisableLog('rdApp.*')
ROOT = Path(__file__).resolve().parents[1]
FIELDS = ['name', 'pocket_idx', 'status', 'vina_score_only']


def resolve(path):
    """A manifest path relative to the current directory or to the repository root."""
    p = Path((path or '').strip())
    for cand in (p, ROOT / p):
        if str(p) not in ('', '.') and cand.exists():
            return cand
    return None


def ligand_pdbqt(mol_h, tmpdir):
    """The H-added molecule through an SDF file, Open Babel and meeko 0.1.dev3."""
    path = os.path.join(tmpdir, 'ligand.sdf')
    w = Chem.SDWriter(path)
    w.write(mol_h)
    w.close()
    ob = next(pybel.readfile('sdf', path))
    prep = MoleculePreparation()
    with contextlib.redirect_stdout(io.StringIO()):            # meeko 0.1.dev3 prints its setup
        prep.prepare(ob.OBMol)
        return prep.write_pdbqt_string()


def score_only(mol, receptor_pdbqt, buffer_, tmpdir):
    mol_h = Chem.AddHs(mol, addCoords=True)
    pos = mol_h.GetConformer().GetPositions()
    center = ((pos.max(0) + pos.min(0)) / 2).tolist()
    size = ((pos.max(0) - pos.min(0)) + buffer_).tolist()
    v = Vina(sf_name='vina', seed=0, verbosity=0)
    v.set_receptor(str(receptor_pdbqt))
    v.set_ligand_from_string(ligand_pdbqt(mol_h, tmpdir))
    v.compute_vina_maps(center=center, box_size=size)
    return float(v.score()[0])


def run_pocket(base, row, entries, buffer_):
    pi = int(row['pocket_idx'])
    tag = f'{pi:03d}'
    receptor = resolve(row['receptor'].strip()[:-4] + '.pdbqt')
    if receptor is None:
        print(f'pocket{tag} vina genbox: receptor pdbqt for {row["receptor"]} not found, skipping',
              flush=True)
        return None
    rows = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for mi, entry in enumerate(entries):
            mol = entry.get('mol')
            if mol is None or mol.GetNumConformers() == 0:
                continue
            out_row = {'name': f'p{tag}_m{mi:04d}', 'pocket_idx': pi,
                       'status': 'ok', 'vina_score_only': ''}
            if len(Chem.GetMolFrags(mol)) > 1:
                out_row['status'] = 'fragmented'
            else:
                try:
                    out_row['vina_score_only'] = score_only(mol, receptor.resolve(), buffer_,
                                                            tmpdir)
                except Exception as e:                       # noqa: BLE001
                    out_row['status'] = f'error:{type(e).__name__}:{str(e)[:60]}'
            rows.append(out_row)
    outdir = base / 'vina_genbox'
    outdir.mkdir(exist_ok=True)
    out = outdir / f'pocket{tag}.csv'
    tmp = outdir / f'.pocket{tag}.csv.{os.getpid()}'
    with open(tmp, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator='\n')
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, out)
    n_ok = sum(r['status'] == 'ok' for r in rows)
    print(f'pocket{tag} vina genbox: {n_ok}/{len(rows)} scored -> {out}', flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pt', required=True, help='the grouped .pt the SDFs were exported from')
    ap.add_argument('--dir', required=True, help='eval_out/<model> (holds manifest.csv)')
    ap.add_argument('--pockets', type=int, nargs='+', help='restrict to these pocket_idx')
    ap.add_argument('--buffer', type=float, default=5.0, help='added to each box edge (A)')
    a = ap.parse_args()

    base = Path(a.dir)
    with open(base / 'manifest.csv') as fh:
        manifest = list(csv.DictReader(fh))
    todo = [r for r in manifest
            if a.pockets is None or int(r['pocket_idx']) in set(a.pockets)]
    if not todo:
        sys.exit('no pockets selected')
    data = load_grouped(a.pt)
    for row in todo:
        run_pocket(base, row, data[int(row['pocket_idx'])], a.buffer)
    return 0


if __name__ == '__main__':
    sys.exit(main())
