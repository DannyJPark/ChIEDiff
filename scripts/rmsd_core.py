#!/usr/bin/env python3
"""Shared RMSD primitives for the pose-fidelity measurements. Every engine arm uses this one
definition, so no two tables can mean different things by RMSD.

    RMSD_sym(a, b) = min over graph automorphisms of the non-superposed heavy-atom RMSD

rdMolAlign.CalcRMS is exactly that. GetBestRMS is the superposing variant and is not used here,
because superposing replaces the question of how far the pose moved in the pocket with how
different the shape is, which flatters the model. Callers feed molecules whose atom
correspondence is already exact, either the same SDF record round-tripped through smina or
gnina, or a meeko PDBQT carrying its REMARK SMILES IDX map, so CalcRMS only resolves symmetry.
"""
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolAlign

RDLogger.DisableLog('rdApp.*')


def heavy(mol):
    """Heavy atoms only, never raising, so a bad pose does not vanish from the table."""
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
    """(N, 3) float array of the single conformer."""
    c = mol.GetConformer()
    return np.array([list(c.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())],
                    dtype=np.float64)


def centroid_disp(a, b):
    """||centroid(a) - centroid(b)||."""
    if a is None or b is None or a.GetNumAtoms() == 0 or b.GetNumAtoms() == 0:
        return None
    return float(np.linalg.norm(coords(a).mean(0) - coords(b).mean(0)))


def positional_rmsd(a, b):
    """Index-order RMSD with no symmetry minimisation, meaningful only when the two molecules
    share atom order. Meeko returns canonical SMILES order, so meeko callers leave this unused."""
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
    """Element and connectivity only, with atom order preserved."""
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
    """Fallback, returning (rmsd, maxed_out), for engines that re-perceive bond orders when a
    pose round-trips through PDBQT inside smina or gnina.

    Requiring equal atom and bond counts turns the substructure search into a genuine graph
    isomorphism, which is what RMSD_sym is defined over, and maxed_out reports that the match
    cap truncated the minimisation. Erasing bond orders and aromaticity can merge chemically
    distinct atoms, inventing symmetries that only pull the minimum down, so this is a fallback
    and never the headline. The meeko and Vina arm has a real index map, and comparing the two
    on the same molecules quantifies the bias.
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
    """Returns (rmsd, method, maxed_out), preferring the exact graph match and falling back to
    skeleton isomorphism only when bond re-perception blocks it."""
    r = rmsd_sym(probe, ref)
    if r is not None:
        return r, 'calcrms', False
    r, maxed = rmsd_skeleton_iso(probe, ref)
    if r is not None:
        return r, 'skeleton_iso', maxed
    return None, 'failed', False


def compare(gen_mol, other_mol, same_atom_order=False):
    """Compare one generated pose against one engine pose.

    sym_gain is the gap between the index-order RMSD and the symmetry-minimised one and can
    never be negative, so it doubles as a check that the atom orders really do match. jensen_ok
    encodes the invariant RMSD >= ||centroid_a - centroid_b||, which a valid atom correspondence
    cannot violate, so False means the mapping is wrong rather than the pose being bad.
    """
    g, o = heavy(gen_mol), heavy(other_mol)
    r, method, maxed = rmsd_best_effort(o, g)
    cd = centroid_disp(g, o)
    out = {'rmsd': r, 'method': method, 'match_maxed_out': maxed,
           'centroid_disp': cd, 'sym_gain': None, 'jensen_ok': None}
    if r is not None and cd is not None:
        out['jensen_ok'] = bool(r >= cd - 1e-3)
    if same_atom_order:
        # smina and gnina round-trip through PDBQT's torsion-tree ordering and do not preserve
        # atom order, so a negative sym_gain means the caller's assumption is false.
        p = positional_rmsd(o, g)
        if p is not None and r is not None:
            out['sym_gain'] = round(p - r, 6)
    return out
