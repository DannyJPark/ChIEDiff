#!/usr/bin/env python3
"""Shared, correct RMSD primitives for the pose-fidelity experiment.

One definition, used by every arm (smina/SDF, gnina/SDF, meeko/Vina) and by the refactored
compare_gen_docked.py, so no two tables can silently mean different things by "RMSD".

    RMSD_sym(a, b) = min over graph automorphisms of the NON-superposed heavy-atom RMSD

`rdMolAlign.CalcRMS` is exactly that: it does NOT align the molecules and it does minimise over
symmetry-equivalent atom mappings. `GetBestRMS` is the superposing variant and must never be used
here -- superposing would turn "how far did the pose move in the pocket" into "how different is
the shape", which is a different question and always flatters the model.

What this module deliberately does NOT do
-----------------------------------------
It does not re-derive an atom correspondence from element+connectivity (the old `_skeleton()` +
`GetSubstructMatches` approach in compare_gen_docked.py). That was substructure matching, not
graph isomorphism, so it accepted wrong mappings, and taking `min` over them biased RMSD DOWNWARD
(defects D1/D2). Every caller here feeds molecules whose correspondence is already exact --
either the same SDF record round-tripped through smina/gnina, or a meeko PDBQT carrying its
`REMARK SMILES IDX` map -- and CalcRMS only has to resolve genuine symmetry.
"""
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolAlign

RDLogger.DisableLog('rdApp.*')


def heavy(mol):
    """Heavy atoms only. Never raises: a pose we want to FLAG as bad must not vanish because
    RDKit refuses to sanitize it."""
    if mol is None:
        return None
    try:
        mol = Chem.RemoveHs(mol, sanitize=False)
    except Exception:                                         # noqa: BLE001
        pass
    if any(a.GetAtomicNum() == 1 for a in mol.GetAtoms()):
        rw = Chem.RWMol(mol)
        for i in sorted((a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 1),
                        reverse=True):
            rw.RemoveAtom(i)
        mol = rw.GetMol()
    return mol


def coords(mol):
    """(N,3) float array of the single conformer (RDKit build-agnostic)."""
    c = mol.GetConformer()
    return np.array([list(c.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())],
                    dtype=np.float64)


def centroid_disp(a, b):
    """||centroid(a) - centroid(b)||. Defined whenever both have coordinates."""
    if a is None or b is None or a.GetNumAtoms() == 0 or b.GetNumAtoms() == 0:
        return None
    return float(np.linalg.norm(coords(a).mean(0) - coords(b).mean(0)))


def positional_rmsd(a, b):
    """Index-order RMSD, no symmetry. Only meaningful when the two mols are KNOWN to share atom
    order (smina/gnina SDF round-trip). Meeko does NOT preserve order -- it returns canonical
    SMILES order -- so this is None-ing out rather than lying is the caller's job."""
    if a is None or b is None or a.GetNumAtoms() != b.GetNumAtoms() or a.GetNumAtoms() == 0:
        return None
    d = coords(a) - coords(b)
    return float(np.sqrt((d * d).sum(1).mean()))


def rmsd_sym(probe, ref):
    """min-over-automorphisms, non-superposed heavy-atom RMSD, or None if incomparable."""
    if probe is None or ref is None:
        return None
    if probe.GetNumAtoms() != ref.GetNumAtoms() or probe.GetNumAtoms() == 0:
        return None
    try:
        return float(rdMolAlign.CalcRMS(probe, ref))
    except Exception:                                         # noqa: BLE001
        return None


def _skeleton(mol):
    """Element + connectivity only: all bonds single, aromaticity/charge cleared. Order preserved."""
    rw = Chem.RWMol(mol)
    for b in rw.GetBonds():
        b.SetBondType(Chem.BondType.SINGLE)
        b.SetIsAromatic(False)
    for a in rw.GetAtoms():
        a.SetIsAromatic(False)
        a.SetFormalCharge(0)
        a.SetNoImplicit(True)
    m = rw.GetMol()
    m.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(m)
    return m


def rmsd_skeleton_iso(probe, ref, max_matches=10000):
    """Fallback for engines that re-perceive bond orders (smina/gnina round-trip through PDBQT).

    Returns (rmsd, maxed_out) or (None, False).

    This is the legacy approach with its two defects closed:
      D1  the old code used GetSubstructMatches, which is a SUBGRAPH match and so accepted
          skeletons with different bond counts. Requiring |V| and |E| to be EQUAL turns a
          subgraph monomorphism into a genuine isomorphism, which is what RMSD_sym is defined
          over.
      D3  hitting the match cap silently truncated the minimisation; `maxed_out` is returned so
          the caller can record it per row instead of quietly reporting an optimistic number.

    The remaining known bias is D2: erasing bond orders and aromaticity can merge atoms that are
    chemically distinct, creating symmetries that do not really exist and therefore pulling the
    minimum DOWN. That is why this is a fallback and never the headline: the meeko/Vina arm has a
    real index map, and comparing the two on the same molecules is what quantifies this bias.
    """
    if probe is None or ref is None:
        return None, False
    if probe.GetNumAtoms() != ref.GetNumAtoms() or probe.GetNumAtoms() == 0:
        return None, False
    if probe.GetNumBonds() != ref.GetNumBonds():          # D1: enforce isomorphism, not subgraph
        return None, False
    ps, rs = _skeleton(probe), _skeleton(ref)
    matches = ps.GetSubstructMatches(rs, uniquify=False, maxMatches=max_matches)
    if not matches:
        return None, False
    pc, rc = coords(probe), coords(ref)
    best = None
    for mt in matches:
        d = rc - pc[list(mt)]
        r = float(np.sqrt((d * d).sum(1).mean()))
        best = r if best is None else min(best, r)
    return best, len(matches) >= max_matches


def rmsd_best_effort(probe, ref):
    """-> (rmsd, method, maxed_out). Prefers the exact graph match; falls back to skeleton
    isomorphism only when bond re-perception makes the exact match impossible."""
    r = rmsd_sym(probe, ref)
    if r is not None:
        return r, 'calcrms', False
    r, maxed = rmsd_skeleton_iso(probe, ref)
    if r is not None:
        return r, 'skeleton_iso', maxed
    return None, 'failed', False


def compare(gen_mol, other_mol, same_atom_order=False):
    """Full comparison of one generated pose against one engine pose.

    Returns a dict with rmsd / centroid_disp / jensen_ok (and sym_gain when the atom order is
    known to match, which is a free consistency probe: the index-order RMSD can never be BELOW
    the symmetry-minimised one, and their gap is how much symmetry actually bought).

    jensen_ok encodes the invariant  RMSD >= ||centroid_a - centroid_b||  (Jensen). A valid atom
    correspondence can never violate it, so a False here means the mapping is wrong -- it is a
    self-check on the pipeline, not a property of the pose.
    """
    g, o = heavy(gen_mol), heavy(other_mol)
    r, method, maxed = rmsd_best_effort(o, g)
    cd = centroid_disp(g, o)
    out = {'rmsd': r, 'method': method, 'match_maxed_out': maxed,
           'centroid_disp': cd, 'sym_gain': None, 'jensen_ok': None}
    if r is not None and cd is not None:
        out['jensen_ok'] = bool(r >= cd - 1e-3)
    if same_atom_order:
        # Only meaningful if the caller KNOWS the orders match. smina/gnina do NOT preserve
        # order (they round-trip through PDBQT's torsion-tree ordering), so a negative sym_gain
        # here is the tell that the assumption is false.
        p = positional_rmsd(o, g)
        if p is not None and r is not None:
            out['sym_gain'] = round(p - r, 6)
    return out
