#!/usr/bin/env python3
"""Heavy-atom-only protein-ligand clash count per generated molecule.

PoseCheck 1.3.1 (posecheck/utils/clashes.py::count_clashes) counts every protein-ligand atom pair
with

    d_ij + 0.5 A < r_vdw(i) + r_vdw(j)        radii from RDKit's periodic table

over all atoms, including the hydrogens RDKit adds to the ligand and hydride adds to the receptor.
This applies the same inequality, with the same radii, to heavy atoms on both sides, so the count
does not depend on where a hydrogen was placed after generation. The ligand is the exported
generated pose (<dir>/sdf/, read as eval_posecheck.py reads it); the receptor is the manifest's
receptor file, whole. Run it with the ged-posecheck interpreter so the radius table is the one
PoseCheck reads.

Output: <dir>/heavy_clash/pocket<NNN>.csv
    molecule, pocket_idx, n_lig_heavy, heavy_clashes, heavy_clashes_ion
heavy_clashes_ion is the subset of pairs whose protein atom is a monatomic ion: a ligand atom
coordinating a metal sits inside the vdW sum by design.
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[1]

OUT_SUB = "heavy_clash"
FIELDS = ["molecule", "pocket_idx", "n_lig_heavy", "heavy_clashes", "heavy_clashes_ion"]
TOLERANCE = 0.5
# monatomic ions a test-set receptor may carry as HETATM
ION_ELEMENTS = frozenset({"LI", "NA", "K", "RB", "CS", "MG", "CA", "SR", "BA", "MN", "FE", "CO",
                          "NI", "CU", "ZN", "CD", "HG", "CL", "BR", "I"})

_PT = Chem.GetPeriodicTable()


def vdw_radii(atomic_numbers):
    return np.array([_PT.GetRvdw(int(z)) for z in atomic_numbers], dtype=float)


def read_receptor(path):
    """Heavy atoms of a receptor PDB, typed from the element column, which every test-set record
    carries. A record without one is an error rather than a guess."""
    xyz, z, ion = [], [], []
    with open(path) as fh:
        for line in fh:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            el = line[76:78].strip().upper()
            if not el:
                raise ValueError(f"{path}: record without an element column: {line.rstrip()}")
            if el in ("H", "D"):
                continue
            xyz.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
            z.append(_PT.GetAtomicNumber(el[:1] + el[1:].lower()))
            ion.append(line.startswith("HETATM") and el in ION_ELEMENTS
                       and line[17:20].strip().upper() == el)
    z = np.asarray(z, dtype=int)
    return {"xyz": np.asarray(xyz, dtype=float), "r": vdw_radii(z),
            "ion": np.asarray(ion, dtype=bool)}


def heavy_clash_counts(receptor, mol):
    """(ligand heavy atoms, heavy-heavy clash pairs, of which with a monatomic ion)."""
    z = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=int)
    keep = z > 1
    lig_xyz = mol.GetConformer().GetPositions()[keep]
    lig_r = vdw_radii(z[keep])
    prot_xyz, prot_r = receptor["xyz"], receptor["r"]
    if len(lig_xyz) == 0 or len(prot_xyz) == 0:
        return int(keep.sum()), 0, 0
    # Drop only protein atoms that cannot satisfy the inequality: a hit has |dx|, |dy|, |dz| <= d
    # < r_lig + r_prot - tolerance, which is within the margin.
    reach = float(lig_r.max() + prot_r.max() - TOLERANCE)
    lo, hi = lig_xyz.min(0) - reach, lig_xyz.max(0) + reach
    near = np.nonzero(np.all((prot_xyz >= lo) & (prot_xyz <= hi), axis=1))[0]
    if len(near) == 0:
        return int(keep.sum()), 0, 0
    # norm(prot - lig), the expression count_clashes uses
    d = np.linalg.norm(prot_xyz[near][:, np.newaxis, :] - lig_xyz[np.newaxis, :, :], axis=-1)
    hit = d + TOLERANCE < lig_r[np.newaxis, :] + prot_r[near][:, np.newaxis]
    pj, _ = np.nonzero(hit)
    return int(keep.sum()), int(len(pj)), int(receptor["ion"][near[pj]].sum())


def evaluate(base, pockets=None):
    base = Path(base)
    outdir = base / OUT_SUB
    outdir.mkdir(exist_ok=True)
    t0, n_mol, n_noconf, n_pk = time.time(), 0, 0, 0
    receptor_path, receptor = None, None
    with open(base / "manifest.csv") as fh:
        manifest = list(csv.DictReader(fh))
    for row in manifest:
        pk = int(row["pocket_idx"])
        if pockets is not None and pk not in pockets:
            continue
        tag = f"{pk:03d}"
        sdf = base / "sdf" / f"pocket{tag}.sdf"
        if not sdf.is_file():
            continue
        if row["receptor"] != receptor_path:
            receptor_path, receptor = row["receptor"], read_receptor(ROOT / row["receptor"])
        rows, names = [], []
        # read exactly as eval_posecheck.py does, including its title fallback
        for m in Chem.SDMolSupplier(str(sdf), sanitize=False, removeHs=False):
            if m is None:
                continue
            name = m.GetProp("_Name") if m.HasProp("_Name") else f"p{tag}_m{len(names):04d}"
            names.append(name)
            if m.GetNumConformers() == 0:
                n_noconf += 1
                continue
            nh, hc, hi = heavy_clash_counts(receptor, m)
            rows.append({"molecule": name, "pocket_idx": pk, "n_lig_heavy": nh,
                         "heavy_clashes": hc, "heavy_clashes_ion": hi})
        with open(outdir / f"pocket{tag}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        n_mol += len(rows)
        n_pk += 1
    print(f"{base}: {n_pk} pockets, {n_mol} molecules, {n_noconf} without a conformer, "
          f"{time.time() - t0:.1f} s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", required=True, help="eval_out/<tag> (holds manifest.csv and sdf/)")
    ap.add_argument("--pockets", type=int, nargs="+", help="restrict to these pocket_idx")
    a = ap.parse_args()
    evaluate(a.dir, set(a.pockets) if a.pockets else None)


if __name__ == "__main__":
    main()
