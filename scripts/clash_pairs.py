"""Protein-ligand clash pairs under PoseCheck's criterion, a heavy-atom-only count, and a hydrogen
breakdown.

PoseCheck 1.3.1 (posecheck/utils/clashes.py::count_clashes) counts every protein-ligand atom pair
with

    d_ij + tolerance < r_vdw(i) + r_vdw(j)        tolerance 0.5 A, radii = RDKit periodic table

over ALL atoms, including the hydrogens RDKit adds to the ligand and Hydride adds to the receptor.
`clash_pairs` reproduces that inequality exactly (same distance expression, same radii, same
comparison), only vectorised.

  * heavy_clash_counts  applies it to heavy atoms on both sides -- the pairs that do not depend on
    where a hydrogen was placed after generation. Evaluated on the receptor file's heavy atoms it
    equals the heavy-heavy subset of PoseCheck's pairs, provided PoseCheck's loaded protein holds
    exactly those heavy atoms (scripts/check_posecheck_receptor_heavy.py checks that per receptor).
  * classify_pairs  labels the full PoseCheck pair list by hydrogen involvement. CODE ONLY: the
    categories rest on our own H-bond geometry rule, which is not a published metric, so nothing in
    the repo stores or reports its output (see scripts/classify_clash_pairs.py).
"""
from collections import Counter

import numpy as np
from rdkit import Chem

POSECHECK_TOLERANCE = 0.5
# monatomic ions a receptor file may carry as HETATM (the test-set receptors hold Zn/Co/Mg/Ca/Cu/K/
# Na/Cl); a ligand atom coordinating one sits inside the vdW sum, so those pairs are counted apart
ION_ELEMENTS = frozenset({"LI", "NA", "K", "RB", "CS", "MG", "CA", "SR", "BA", "MN", "FE", "CO",
                          "NI", "CU", "ZN", "CD", "HG", "CL", "BR", "I"})
HB_DONOR_HEAVY = (7, 8)           # a polar hydrogen sits on N or O
HB_ACCEPTOR = (7, 8, 9)           # N, O, F

_PT = Chem.GetPeriodicTable()


def atomic_number(symbol):
    s = symbol.strip()
    return _PT.GetAtomicNumber(s[:1].upper() + s[1:].lower())


def vdw_radii(atomic_numbers):
    """RDKit periodic-table radii: the table count_clashes reads through pt.GetRvdw."""
    return np.array([_PT.GetRvdw(int(z)) for z in atomic_numbers], dtype=float)


def clash_pairs(prot_xyz, prot_r, lig_xyz, lig_r, tolerance=POSECHECK_TOLERANCE):
    """(ligand index, protein index, distance) of every pair with d + tolerance < r_lig + r_prot.

    Distances are taken as norm(prot - lig), the expression count_clashes uses. The bounding-box
    prefilter drops only protein atoms that cannot satisfy the inequality: a hit has
    |dx|, |dy|, |dz| <= d < r_lig + r_prot - tolerance <= the box margin.
    """
    prot_xyz = np.asarray(prot_xyz, dtype=float)
    lig_xyz = np.asarray(lig_xyz, dtype=float)
    empty = (np.zeros(0, dtype=int), np.zeros(0, dtype=int), np.zeros(0))
    if len(prot_xyz) == 0 or len(lig_xyz) == 0:
        return empty
    reach = float(lig_r.max() + prot_r.max() - tolerance)
    lo, hi = lig_xyz.min(0) - reach, lig_xyz.max(0) + reach
    near = np.nonzero(np.all((prot_xyz >= lo) & (prot_xyz <= hi), axis=1))[0]
    if len(near) == 0:
        return empty
    d = np.linalg.norm(prot_xyz[near][:, np.newaxis, :] - lig_xyz[np.newaxis, :, :], axis=-1)
    hit = d + tolerance < lig_r[np.newaxis, :] + prot_r[near][:, np.newaxis]
    pj, li = np.nonzero(hit)
    return li, near[pj], d[pj, li]


def read_receptor(path):
    """Heavy atoms of a receptor PDB, typed from the element column (present on every test-set
    record; a record without one is an error rather than a guess)."""
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
            z.append(atomic_number(el))
            ion.append(line.startswith("HETATM") and el in ION_ELEMENTS
                       and line[17:20].strip().upper() == el)
    z = np.asarray(z, dtype=int)
    return {"xyz": np.asarray(xyz, dtype=float), "z": z, "r": vdw_radii(z),
            "ion": np.asarray(ion, dtype=bool)}


def heavy_clash_counts(receptor, mol, tolerance=POSECHECK_TOLERANCE):
    """(ligand heavy atoms, heavy-heavy clash pairs, of which with a monatomic ion)."""
    z = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=int)
    keep = z > 1
    lig_xyz = mol.GetConformer().GetPositions()[keep]
    li, pj, _ = clash_pairs(receptor["xyz"], receptor["r"], lig_xyz, vdw_radii(z[keep]),
                            tolerance)
    return int(keep.sum()), int(len(li)), int(receptor["ion"][pj].sum())


# --------------------------------------------------------------------------- hydrogen breakdown
def _heavy_neighbor(atom):
    return next((n for n in atom.GetNeighbors() if n.GetAtomicNum() > 1), None)


def _angle(a, b, c):
    v1, v2 = a - b, c - b
    cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def _category(la, pa, lpos, ppos, hb_da_max, hb_dha_min):
    lh, ph = la.GetAtomicNum() == 1, pa.GetAtomicNum() == 1
    if not lh and not ph:
        return "heavy_heavy"
    if lh != ph:                                        # exactly one hydrogen in the pair
        h, hpos, acc, apos = (la, lpos, pa, ppos) if lh else (pa, ppos, la, lpos)
        donor = _heavy_neighbor(h)
        if (donor is not None and donor.GetAtomicNum() in HB_DONOR_HEAVY
                and acc.GetAtomicNum() in HB_ACCEPTOR):
            D, Hx, A = hpos[donor.GetIdx()], hpos[h.GetIdx()], apos[acc.GetIdx()]
            if np.linalg.norm(D - A) <= hb_da_max and _angle(D, Hx, A) >= hb_dha_min:
                return "hbond_geometry"
    if lh:
        return "ligand_h"
    donor = _heavy_neighbor(pa)
    if donor is not None and donor.GetAtomicNum() in (7, 8, 16):
        return "protein_polar_h"
    return "protein_nonpolar_h"


def classify_pairs(prot, lig, tolerance=POSECHECK_TOLERANCE, hb_da_max=3.5, hb_dha_min=130.0):
    """Every PoseCheck clash pair of (prot, lig), both carrying hydrogens exactly as PoseCheck holds
    them, labelled with the first matching category:

      heavy_heavy         no hydrogen in the pair
      hbond_geometry      one partner is a polar H (bonded to N/O), the other an N/O/F on the other
                          molecule, with D...A <= hb_da_max and angle D-H...A >= hb_dha_min
                          (defaults are ProLIF 2.2.0's HBond thresholds, 3.5 A and 130 deg)
      ligand_h            the ligand atom is a hydrogen (placed by RDKit AddHs), H-H pairs included
      protein_polar_h     protein H on N/O/S outside H-bond geometry
      protein_nonpolar_h  protein H on carbon

    Our own rule for inspection only -- not a published metric, not stored anywhere.
    Returns (rows, Counter of categories).
    """
    ppos = prot.GetConformer().GetPositions()
    lpos = lig.GetConformer().GetPositions()
    pz = np.array([a.GetAtomicNum() for a in prot.GetAtoms()], dtype=int)
    lz = np.array([a.GetAtomicNum() for a in lig.GetAtoms()], dtype=int)
    pr, lr = vdw_radii(pz), vdw_radii(lz)
    li, pj, dist = clash_pairs(ppos, pr, lpos, lr, tolerance)
    rows = []
    for i, j, d in zip(li, pj, dist):
        la, pa = lig.GetAtomWithIdx(int(i)), prot.GetAtomWithIdx(int(j))
        info = pa.GetPDBResidueInfo()
        rows.append({
            "category": _category(la, pa, lpos, ppos, hb_da_max, hb_dha_min),
            "distance": round(float(d), 4), "rvdw_sum": round(float(lr[i] + pr[j]), 3),
            "ligand_index": int(i), "ligand_element": la.GetSymbol(),
            "protein_index": int(j), "protein_element": pa.GetSymbol(),
            "protein_atom": info.GetName().strip() if info else "",
            "protein_residue": (f"{info.GetResidueName().strip()}{info.GetResidueNumber()}"
                                f"{info.GetChainId().strip()}") if info else ""})
    return rows, Counter(r["category"] for r in rows)
