#!/usr/bin/env python3
"""
Shared loading / alignment helpers for the Delta Score (selectivity) experiment.

Imported by build_offtarget_assignment.py, dock_offtarget.py and build_delta_score.py
so all three see exactly the same molecules in exactly the same order -- the molecule
subsample is stored as an INDEX into the per-pocket list this module returns, so any
disagreement between the three would silently dock the wrong molecules.

Why not reuse build_comparison_tables.normalize_model: it funnels every record through
extract_mol(), which keeps only the metric scalars and throws the RDKit mol away. We need
the mol itself, so the alignment RULE is reused (full2idx, exact ligand_filename match)
while the record stays intact.
"""

import json
import os
import sys

sys.path.append(os.path.abspath('./'))

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_POCKETS = 'analysis/nci_benchmark/test_pockets.json'
# Single switch for all four Delta Score scripts (Phase B, 2026-08-21). Their --registry defaults
# reference this constant instead of hardcoding a path, so the family moves in one edit.
# load_registry() renders configs/models.json into the legacy delta view, and
# scripts/check_registry_sync.py proves that view is byte-identical to the retired
# configs/delta_score_models.json -- which is what makes this switch a no-op.
REGISTRY = 'configs/models.json'
ASSIGNMENT_TMPL = 'configs/offtarget_assignment_seed{seed}.json'
OFFTARGET_ROOT = 'results/offtarget'


# --------------------------------------------------------------------------- #
# config / SSOT
# --------------------------------------------------------------------------- #

def load_registry(path=REGISTRY):
    """The delta-family view of the model registry.

    Accepts either configs/models.json or the legacy configs/delta_score_models.json; a legacy
    path is honoured verbatim so a half-migrated checkout still runs.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import model_registry                                                     # noqa: E402
    return model_registry.load_as(path, 'delta')


def model_by_tag(cfg, tag):
    for m in cfg['models']:
        if m['tag'] == tag:
            return m
    raise KeyError(f"model tag {tag!r} not in registry (have: "
                   f"{[m['tag'] for m in cfg['models']]})")


def load_test_pockets(path=TEST_POCKETS, registry=None):
    """Return (tp, entries, idx2full, full2idx) from the sha256-gated SSOT.

    "sha256-gated" was aspirational until 2026-08-21: the docstring said it, but nothing here
    recomputed the hash. scripts/pockets.verify_sha256() now does, and the registry's declared
    canonical_pockets.sha256 is checked against it too, so an edited pocket list cannot slip
    past any of the three places that record it.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import pockets as _pockets                                                # noqa: E402
    recorded = _pockets.verify_sha256(path)

    with open(path) as fh:
        tp = json.load(fh)
    entries = tp['pockets']
    idx2full = {e['pocket_idx']: e['ligand_filename'] for e in entries}
    full2idx = {v: k for k, v in idx2full.items()}
    if len(full2idx) != len(idx2full):
        raise RuntimeError('duplicate ligand_filename in test_pockets.json')

    reg_path = registry or REGISTRY
    if os.path.isfile(reg_path):
        import model_registry                                                 # noqa: E402
        declared = (model_registry.load(reg_path).get('canonical_pockets') or {}).get('sha256')
        if declared and declared != recorded:
            raise RuntimeError(f'{reg_path}: canonical_pockets.sha256 {declared} != '
                               f'{path} {recorded}')
    return tp, entries, idx2full, full2idx


def load_assignment(seed=None, registry=REGISTRY):
    if seed is None:
        seed = load_registry(registry)['seed']
    path = ASSIGNMENT_TMPL.format(seed=seed)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'{path} not found -- run scripts/build_offtarget_assignment.py first')
    with open(path) as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# molecule loading
# --------------------------------------------------------------------------- #

def get_vina_triplet(vina):
    """(score_only, minimize, dock) affinities, or Nones. Mirrors
    build_comparison_tables.get_vina but returns a dict."""
    out = {'score_only': None, 'minimize': None, 'dock': None}
    if isinstance(vina, dict):
        for mode in out:
            stage = vina.get(mode)
            if stage:
                if isinstance(stage, dict):              # MolCraft-style single dict
                    out[mode] = _f(stage.get('affinity'))
                else:
                    out[mode] = _f(stage[0].get('affinity'))
    elif isinstance(vina, list) and vina:                # qvina-style bare list
        out['dock'] = _f(vina[0].get('affinity'))
    return out


def _f(x):
    try:
        return None if x is None else float(x)
    except (TypeError, ValueError):
        return None


def _has_3d(mol):
    return mol is not None and mol.GetNumConformers() > 0


def load_model_pockets(model, full2idx, idx2full, verbose=True):
    """Return {pocket_idx: [record, ...]}.

    A record is {'mol', 'ligand_filename', 'on', 'src'} where 'on' is the reused
    on-target Vina triplet and 'src' locates the molecule in the source .pt.
    Only molecules that have BOTH a 3-D mol and an on-target Vina Dock score are
    kept -- those are the only ones the delta score can ever pair.

    The per-pocket list order is deterministic (source file order), which is what
    the stored molecule subsample indexes into.
    """
    obj = torch.load(model['pt'])
    layout = model['layout']

    if layout == 'flat':
        records = obj['all_results'] if isinstance(obj, dict) else obj
        groups = None
    else:
        groups = obj

    per_pocket = {}
    n_seen = n_kept = 0
    mismatched_lf = 0

    if layout == 'flat':
        for j, rec in enumerate(records):
            n_seen += 1
            lf = rec.get('ligand_filename')
            i = full2idx.get(lf)
            if i is None:
                continue
            kept = _make_record(rec, rec.get('mol'), lf, ('flat', j))
            if kept is not None:
                per_pocket.setdefault(i, []).append(kept)
                n_kept += 1
    else:
        for i, elem in enumerate(groups):
            group = elem if isinstance(elem, list) else [elem]     # 'single' layout
            # Every model now stores its molecules in the same file as its scalars. `ours_noguide`
            # was the last exception (scalars and mols in two files, paired by docked_order);
            # scripts/merge_split_pt.py materialised that pairing once -- verifying it against
            # recomputed QED over all 2145 pairs rather than a 3-molecule sample -- so the join
            # branch that used to live here has no remaining user. Retired 2026-08-25; F4 output
            # verified byte-identical across the change.
            pairs = [(r, r.get('mol'), ('grouped', i, k)) for k, r in enumerate(group)]

            for rec, mol, src in pairs:
                n_seen += 1
                lf = rec.get('ligand_filename') or idx2full.get(i)
                if lf and full2idx.get(lf) not in (None, i):
                    mismatched_lf += 1
                kept = _make_record(rec, mol, idx2full.get(i, lf), src)
                if kept is not None:
                    per_pocket.setdefault(i, []).append(kept)
                    n_kept += 1

    if mismatched_lf:
        raise RuntimeError(
            f"{model['tag']}: {mismatched_lf} molecules whose stored ligand_filename "
            'disagrees with their list index -- the pocket order is not canonical. '
            'See the flat-.pt pocket renumbering bug before trusting this file.')

    expected = model.get('n_mols')
    if expected is not None and n_kept != expected:
        raise RuntimeError(
            f"{model['tag']}: kept {n_kept} docked molecules, registry says {expected}. "
            'The source .pt changed -- update configs/delta_score_models.json deliberately.')

    if verbose:
        print(f"[load] {model['tag']:20s} pockets={len(per_pocket):3d} "
              f'molecules={n_kept} (of {n_seen} records)')
    return per_pocket


def _make_record(rec, mol, ligand_filename, src):
    on = get_vina_triplet(rec.get('vina'))
    if on['dock'] is None or not _has_3d(mol):
        return None
    return {'mol': mol, 'ligand_filename': ligand_filename, 'on': on, 'src': list(src)}


# --------------------------------------------------------------------------- #
# output paths
# --------------------------------------------------------------------------- #

def offtarget_out_path(tag, pocket_idx, root=OFFTARGET_ROOT):
    return os.path.join(root, tag, f'id{pocket_idx}', 'offtarget_docked.pt')
