#!/usr/bin/env python3
"""Shared RMSD primitives for the pose-fidelity measurements.

Every engine arm (smina/SDF, gnina/SDF, meeko/Vina) uses this one definition, so no two tables
can mean different things by "RMSD".

    RMSD_sym(a, b) = min over graph automorphisms of the non-superposed heavy-atom RMSD

`rdMolAlign.CalcRMS` is exactly that. It does not align the molecules and it does minimise over
symmetry-equivalent atom mappings. `GetBestRMS` is the superposing variant and is not used here.
Superposing replaces the question "how far did the pose move in the pocket" with "how different
is the shape", which is a different question and one that flatters the model.

This module deliberately does not re-derive an atom correspondence from element and connectivity
by substructure matching. A substructure match is not a graph isomorphism, so it accepts wrong
mappings, and taking the minimum over them biases RMSD downward. Every caller here feeds
molecules whose correspondence is already exact, either the same SDF record round-tripped
through smina or gnina, or a meeko PDBQT carrying its `REMARK SMILES IDX` map, so CalcRMS only
has to resolve genuine symmetry.
"""
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolAlign

RDLogger.DisableLog('rdApp.*')


def heavy(mol):
    """Heavy atoms only. Never raises, because a pose that should be flagged as bad must not
    vanish from the table just because RDKit refuses to sanitize it."""
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
    """Index-order RMSD, with no symmetry minimisation. It is meaningful only when the two
    molecules are known to share atom order. Meeko does not preserve order, it returns canonical
    SMILES order, so a caller working with meeko output must leave this unused rather than
    report it."""
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
    """Fallback for engines that re-perceive bond orders, which is what happens when a pose
    round-trips through PDBQT inside smina or gnina.

    Returns (rmsd, maxed_out) or (None, False).

    Two properties keep the fallback honest. Requiring the atom count and the bond count to be
    equal turns the substructure search into a genuine graph isomorphism, which is what RMSD_sym
    is defined over. A bare substructure match would accept skeletons with different bond counts.
    Hitting the match cap truncates the minimisation, so `maxed_out` is returned and the caller
    can record it per row instead of reporting an optimistic number as a complete one.

    The known bias is that erasing bond orders and aromaticity can merge atoms that are
    chemically distinct. That creates symmetries which do not exist, and extra symmetries can
    only pull the minimum down. For that reason this is a fallback and never the headline. The
    meeko/Vina arm has a real index map, and comparing the two on the same molecules is what
    quantifies the bias.
    """
    if probe is None or ref is None:
        return None, False
    if probe.GetNumAtoms() != ref.GetNumAtoms() or probe.GetNumAtoms() == 0:
        return None, False
    if probe.GetNumBonds() != ref.GetNumBonds():          # isomorphism, not a subgraph match
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
    """-> (rmsd, method, maxed_out). Prefers the exact graph match and falls back to skeleton
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

    Returns a dict with rmsd, centroid_disp and jensen_ok, plus sym_gain when the atom order is
    known to match. sym_gain is a free consistency probe, because the index-order RMSD can never
    fall below the symmetry-minimised one and their gap is how much symmetry bought.

    jensen_ok encodes the invariant  RMSD >= ||centroid_a - centroid_b||  (Jensen). A valid atom
    correspondence cannot violate it, so a False here means the mapping is wrong. It is a
    self-check on the pipeline rather than a property of the pose.
    """
    g, o = heavy(gen_mol), heavy(other_mol)
    r, method, maxed = rmsd_best_effort(o, g)
    cd = centroid_disp(g, o)
    out = {'rmsd': r, 'method': method, 'match_maxed_out': maxed,
           'centroid_disp': cd, 'sym_gain': None, 'jensen_ok': None}
    if r is not None and cd is not None:
        out['jensen_ok'] = bool(r >= cd - 1e-3)
    if same_atom_order:
        # Only meaningful if the caller knows the orders match. smina and gnina do not preserve
        # order, because they round-trip through PDBQT's torsion-tree ordering, so a negative
        # sym_gain here is the sign that the assumption is false.
        p = positional_rmsd(o, g)
        if p is not None and r is not None:
            out['sym_gain'] = round(p - r, 6)
    return out
