"""
AutoDock Vina atom typing (XS types), ported from the Vina source so the `vina` physics loss
gates its hydrophobic / H-bond terms by the SAME rules the real scoring function uses.

Reference (AutoDock-Vina, src/lib/atom_constants.h):

    inline bool xs_is_hydrophobic(sz xs) {
        return xs == XS_TYPE_C_H || xs == XS_TYPE_F_H || xs == XS_TYPE_Cl_H ||
               xs == XS_TYPE_Br_H || xs == XS_TYPE_I_H;
    }
    inline bool xs_is_acceptor(sz xs) {
        return xs == XS_TYPE_N_A || xs == XS_TYPE_N_DA || xs == XS_TYPE_O_A || xs == XS_TYPE_O_DA;
    }
    inline bool xs_is_donor(sz xs) {
        return xs == XS_TYPE_N_D || xs == XS_TYPE_N_DA || xs == XS_TYPE_O_D ||
               xs == XS_TYPE_O_DA || xs == XS_TYPE_Met_D;
    }
    inline bool xs_h_bond_possible(sz t1, sz t2) {
        return (xs_is_donor(t1) && xs_is_acceptor(t2)) || (xs_is_donor(t2) && xs_is_acceptor(t1));
    }

Reference (src/lib/model.cpp, model::assign_types):

    case EL_TYPE_C: x = bonded_to_heteroatom(a) ? XS_TYPE_C_P : XS_TYPE_C_H; break;
    bool acceptor   = (a.ad == AD_TYPE_OA || a.ad == AD_TYPE_NA);
    bool donor_NorO = (a.el == EL_TYPE_Met || bonded_to_HD(a));
    case EL_TYPE_N: x = (acceptor && donor_NorO) ? XS_TYPE_N_DA : (acceptor ? XS_TYPE_N_A :
                        (donor_NorO ? XS_TYPE_N_D : XS_TYPE_N_P)); break;   // O_* identical

So, in words:
  * hydrophobic  = a CARBON with NO heteroatom neighbour, or F / Cl / Br / I.
                   (a carbon bonded to N,O,S,P,halogen is XS_TYPE_C_P -> NOT hydrophobic)
  * acceptor     = N or O typed OA/NA by AutoDock (S is deliberately excluded:
                   "X-Score formulation apparently ignores SA" -- Vina source comment)
  * donor        = N or O carrying a polar hydrogen (bonded to an AD_TYPE_HD), or a metal
  * H-bond fires ONLY on a donor-acceptor pair, never acceptor-acceptor / donor-donor.

Also exported: XS_RADIUS_BY_Z, Vina's `xs_radius` per element (atom_kind_data).  These are NOT
the Bondi radii in utils/vina_rules.ATOM_VANDER_R -- Vina uses C 1.9 / N 1.8 / O 1.7, which sets
the surface distance d = r - (R_i + R_j) that every Vina term is a function of.  With Vina's
radii an N...O pair has R_i+R_j = 3.5 A, so the H-bond ramp (active for d in [-0.7, 0]) spans
r in [2.8, 3.5] A -- exactly the real H-bond range.
"""

import torch

# --- Vina xs_radius (src/lib/atom_constants.h, atom_kind_data) -------------------------------
# C/A 1.9 | N/NA 1.8 | O/OA 1.7 | P 2.1 | S/SA 2.0 | H/HD 1.5 | F 1.5 | Cl 1.8 | Br 2.0 | I 2.2
# Se is absent from Vina's table; we map it to sulfur (2.0), its closest congener.
XS_RADIUS_BY_Z = {1: 1.5, 6: 1.9, 7: 1.8, 8: 1.7, 9: 1.5, 15: 2.1,
                  16: 2.0, 17: 1.8, 34: 2.0, 35: 2.0, 53: 2.2}
XS_RADIUS_DEFAULT = 1.9

HALOGEN_Z = (9, 17, 35, 53)      # F, Cl, Br, I  -> always XS_TYPE_*_H (hydrophobic)

# --- Protein XS typing table ------------------------------------------------------------------
# flags = (hydrophobic, donor, acceptor).  Derived by applying Vina's rules to the standard
# residue topologies: a carbon is hydrophobic iff every heavy neighbour is a carbon.
_H = (1, 0, 0)   # hydrophobic carbon (XS_TYPE_C_H)
_X = (0, 0, 0)   # polar carbon (C_P) / sulfur (S_P) / proline N (N_P) -- inert for both terms
_D = (0, 1, 0)   # donor only        (N_D / O_D)
_A = (0, 0, 1)   # acceptor only     (N_A / O_A)
_DA = (0, 1, 1)  # donor + acceptor  (N_DA / O_DA)

# Backbone: N carries an amide H (donor); CA is bonded to N and C is bonded to O/N, so both are
# C_P; O is a carbonyl acceptor.  Proline's ring N has no H -> N_P (overridden below).
_BACKBONE = {'N': _D, 'CA': _X, 'C': _X, 'O': _A, 'OXT': _A, 'OT1': _A, 'OT2': _A}

_SIDECHAIN = {
    'ALA': {'CB': _H},
    'ARG': {'CB': _H, 'CG': _H, 'CD': _X, 'NE': _D, 'CZ': _X, 'NH1': _D, 'NH2': _D},
    'ASN': {'CB': _H, 'CG': _X, 'OD1': _A, 'ND2': _D},
    'ASP': {'CB': _H, 'CG': _X, 'OD1': _A, 'OD2': _A},
    'CYS': {'CB': _X, 'SG': _X},
    'GLN': {'CB': _H, 'CG': _H, 'CD': _X, 'OE1': _A, 'NE2': _D},
    'GLU': {'CB': _H, 'CG': _H, 'CD': _X, 'OE1': _A, 'OE2': _A},
    'GLY': {},
    # Both imidazole N are H-bond capable; which one is the donor depends on the HIS tautomer,
    # which the heavy-atom-only pocket files do not record.  Treat both as N_DA.
    'HIS': {'CB': _H, 'CG': _X, 'ND1': _DA, 'CD2': _X, 'CE1': _X, 'NE2': _DA},
    'ILE': {'CB': _H, 'CG1': _H, 'CG2': _H, 'CD1': _H, 'CD': _H},
    'LEU': {'CB': _H, 'CG': _H, 'CD1': _H, 'CD2': _H},
    'LYS': {'CB': _H, 'CG': _H, 'CD': _H, 'CE': _X, 'NZ': _D},
    'MET': {'CB': _H, 'CG': _X, 'SD': _X, 'CE': _X},
    'PHE': {'CB': _H, 'CG': _H, 'CD1': _H, 'CD2': _H, 'CE1': _H, 'CE2': _H, 'CZ': _H},
    'PRO': {'N': _X, 'CB': _H, 'CG': _H, 'CD': _X},   # ring N has no polar H -> N_P
    'SER': {'CB': _X, 'OG': _DA},
    'THR': {'CB': _X, 'OG1': _DA, 'CG2': _H},
    'TRP': {'CB': _H, 'CG': _H, 'CD1': _X, 'NE1': _D, 'CD2': _H, 'CE2': _X,
            'CE3': _H, 'CZ2': _H, 'CZ3': _H, 'CH2': _H},
    'TYR': {'CB': _H, 'CG': _H, 'CD1': _H, 'CD2': _H, 'CE1': _H, 'CE2': _H, 'CZ': _X, 'OH': _DA},
    'VAL': {'CB': _H, 'CG1': _H, 'CG2': _H},
}

# AA index order used by utils.data.PDBProtein.AA_NAME_NUMBER
_AA_ORDER = ['ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 'LEU',
             'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER', 'THR', 'VAL', 'TRP', 'TYR']


def _protein_atom_flags(res_name, atom_name):
    """(hydrophobic, donor, acceptor) for one protein heavy atom. Unknown names -> inert."""
    side = _SIDECHAIN.get(res_name, {})
    if atom_name in side:            # sidechain (and PRO's N override) wins over the backbone table
        return side[atom_name]
    return _BACKBONE.get(atom_name, _X)


def protein_vina_flags(atom_name, atom_to_aa_type):
    """Vina XS flags for every protein atom.

    Args:
        atom_name:       list[str] of length N (PDB atom names, e.g. 'CA', 'OD1', 'NE2')
        atom_to_aa_type: (N,) long tensor of residue indices into PDBProtein.AA_NAME_NUMBER
    Returns:
        (N, 3) bool tensor -- columns [is_hydrophobic, is_donor, is_acceptor]
    """
    aa = atom_to_aa_type.tolist()
    flags = [_protein_atom_flags(_AA_ORDER[a], n) if 0 <= a < len(_AA_ORDER) else _X
             for a, n in zip(aa, atom_name)]
    return torch.tensor(flags, dtype=torch.bool)


def ligand_vina_flags(element, bond_index, atom_feature):
    """Vina XS flags for every ligand heavy atom.

    hydrophobic reproduces Vina exactly (bonded_to_heteroatom over the real bond graph).
    donor/acceptor use RDKit's BaseFeatures Donor/Acceptor families restricted to N,O -- RDKit's
    `Donor` SMARTS ([N;!H0], [O;H1]) is the same predicate as Vina's `bonded_to_HD` for N/O.

    Args:
        element:      (N,) long tensor of atomic numbers (H already removed)
        bond_index:   (2, E) long tensor, heavy-atom bonds, stored in BOTH directions
        atom_feature: (N, 8) tensor over utils.data.ATOM_FAMILIES; col 0 = Acceptor, col 1 = Donor
    Returns:
        (N, 3) bool tensor -- columns [is_hydrophobic, is_donor, is_acceptor]
    """
    n = element.numel()
    device = element.device

    # Vina: is_heteroatom() == not carbon and not hydrogen
    is_hetero = (element != 6) & (element != 1)
    bonded_to_hetero = torch.zeros(n, dtype=torch.long, device=device)
    if bond_index.numel() > 0:
        src, dst = bond_index[0], bond_index[1]
        bonded_to_hetero.index_add_(0, src, is_hetero[dst].long())
    bonded_to_hetero = bonded_to_hetero > 0

    is_halogen = torch.zeros(n, dtype=torch.bool, device=device)
    for z in HALOGEN_Z:
        is_halogen |= element == z

    hydrophobic = ((element == 6) & ~bonded_to_hetero) | is_halogen

    # Vina restricts donor/acceptor to N and O (sulfur is explicitly ignored)
    is_n_or_o = (element == 7) | (element == 8)
    acceptor = atom_feature[:, 0].bool() & is_n_or_o
    donor = atom_feature[:, 1].bool() & is_n_or_o

    return torch.stack([hydrophobic, donor, acceptor], dim=-1)
