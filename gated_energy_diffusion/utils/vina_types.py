"""AutoDock Vina atom typing (XS types), ported from src/lib/atom_constants.h and
src/lib/model.cpp so the interaction energy gates its hydrophobic and H-bond terms by the rules
the scoring function uses. Hydrophobic means a carbon whose heavy neighbours are all carbon, or
F, Cl, Br or I. Acceptor means an N or O typed OA or NA by AutoDock, sulfur being excluded.
Donor means an N or O carrying a polar hydrogen, or a metal. An H-bond fires only on a
donor-acceptor pair.

XS_RADIUS_BY_Z holds Vina's xs_radius per element, which are not Bondi radii. Vina uses C 1.9,
N 1.8 and O 1.7, so an N and O pair sums to 3.5 A and the H-bond ramp, active over
d in [-0.7, 0], spans r in [2.8, 3.5] A.
"""

import torch

# Se is absent from Vina's table and is mapped to sulfur, its closest congener.
XS_RADIUS_BY_Z = {1: 1.5, 6: 1.9, 7: 1.8, 8: 1.7, 9: 1.5, 15: 2.1,
                  16: 2.0, 17: 1.8, 34: 2.0, 35: 2.0, 53: 2.2}
XS_RADIUS_DEFAULT = 1.9

HALOGEN_Z = (9, 17, 35, 53)

# flags = (hydrophobic, donor, acceptor), derived by applying Vina's rules to the standard
# residue topologies.
_H = (1, 0, 0)
_X = (0, 0, 0)   # inert for both terms: polar carbon, sulfur or proline N
_D = (0, 1, 0)
_A = (0, 0, 1)
_DA = (0, 1, 1)

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
    # Which imidazole N donates depends on the tautomer, which the heavy-atom-only pocket
    # files do not record, so both are marked donor and acceptor.
    'HIS': {'CB': _H, 'CG': _X, 'ND1': _DA, 'CD2': _X, 'CE1': _X, 'NE2': _DA},
    'ILE': {'CB': _H, 'CG1': _H, 'CG2': _H, 'CD1': _H, 'CD': _H},
    'LEU': {'CB': _H, 'CG': _H, 'CD1': _H, 'CD2': _H},
    'LYS': {'CB': _H, 'CG': _H, 'CD': _H, 'CE': _X, 'NZ': _D},
    'MET': {'CB': _H, 'CG': _X, 'SD': _X, 'CE': _X},
    'PHE': {'CB': _H, 'CG': _H, 'CD1': _H, 'CD2': _H, 'CE1': _H, 'CE2': _H, 'CZ': _H},
    'PRO': {'N': _X, 'CB': _H, 'CG': _H, 'CD': _X},
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
    """Unknown atom names are inert."""
    side = _SIDECHAIN.get(res_name, {})
    if atom_name in side:            # the sidechain table, including PRO's N, wins over the backbone
        return side[atom_name]
    return _BACKBONE.get(atom_name, _X)


def protein_vina_flags(atom_name, atom_to_aa_type):
    """(N, 3) bool tensor of [is_hydrophobic, is_donor, is_acceptor] per protein atom.
    atom_to_aa_type indexes PDBProtein.AA_NAME_NUMBER."""
    aa = atom_to_aa_type.tolist()
    flags = [_protein_atom_flags(_AA_ORDER[a], n) if 0 <= a < len(_AA_ORDER) else _X
             for a, n in zip(aa, atom_name)]
    return torch.tensor(flags, dtype=torch.bool)


def ligand_vina_flags(element, bond_index, atom_feature):
    """(N, 3) bool tensor of [is_hydrophobic, is_donor, is_acceptor] per ligand heavy atom.

    bond_index must carry every heavy-atom bond in both directions, because hydrophobic applies
    Vina's bonded_to_heteroatom over the real bond graph. donor and acceptor read column 1 and
    column 0 of atom_feature over utils.data.ATOM_FAMILIES, RDKit's Donor SMARTS being the same
    predicate as Vina's bonded_to_HD for N and O.
    """
    n = element.numel()
    device = element.device

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

    is_n_or_o = (element == 7) | (element == 8)
    acceptor = atom_feature[:, 0].bool() & is_n_or_o
    donor = atom_feature[:, 1].bool() & is_n_or_o

    return torch.stack([hydrophobic, donor, acceptor], dim=-1)
