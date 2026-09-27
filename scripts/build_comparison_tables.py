#!/usr/bin/env python3
"""
Build cross-model comparison tables from docked-result .pt files.

Metrics per molecule:
    vina_score = vina['score_only'][0]['affinity']
    vina_min   = vina['minimize'][0]['affinity']
    vina_dock  = vina['dock'][0]['affinity']
    qed        = chem_results['qed']
    sa         = chem_results['sa']

Handles the two on-disk layouts found in results/sampling_results/:
  1. grouped-by-pocket:  list[N_pockets] of list[mols]   (targetdiff, ar, cvae, ...)
                         or list[N_pockets] of a single dict (crossdocked native)
  2. flat all_results:   dict{'all_results': [mols, ...]}  (PIDiff, DiffSBDD, ...)

Pockets are aligned across models by the ligand-file directory (first path
component of ligand_filename), using targetdiff's ordering as the canonical
0..99 index (identical to the CrossDock test-set / sampling data_id order).

Outputs (to --out_dir):
  comparison_overall.csv      one row per model, pooled over all molecules
  comparison_per_target.csv   one row per (pocket, model)
  comparison_tables.txt       ablation sections + readable overall / per-pocket tables

Note: comparison_tables.md in that directory is NOT written by this script. It is a
leftover from an older markdown-emitting version, predates our models entirely, and is
not regenerated -- read comparison_tables.txt instead.

Usage:
    python scripts/build_comparison_tables.py
    python scripts/build_comparison_tables.py --out_dir results/comparison
"""

import argparse
import os
import csv
from collections import defaultdict, Counter

import numpy as np
import torch
from rdkit import Chem, DataStructs, RDLogger

RDLogger.DisableLog('rdApp.*')


# SUPERSEDED 2026-08-21 by configs/models.json (ids.pt_stem -> label). Still consulted for the
# --ablation_file path only, whose rows keep their own guidance-ablation names. Every main and
# physics-ablation label now comes from model_registry.label_for(); adding an entry here will
# NOT put a model in a table.
# Pretty display names; anything not listed falls back to the file stem.
DISPLAY_NAMES = {
    'crossdocked_test_vina_docked': 'Reference (native)',
    'targetdiff_vina_docked': 'TargetDiff',
    'ar_vina_docked': 'AR',
    'cvae_vina_docked': 'liGAN (cvae)',
    'CVAE_test_docked_sf1.5': 'CVAE (sf1.5)',
    'pocket2mol_vina_docked': 'Pocket2Mol',
    'PIDiff_vina_docked': 'PIDiff',
    'PIDiff_vina_docked_complete': 'PIDiff',
    'DiffSBDD_vina_dock': 'DiffSBDD',
    'DrugGPS_vina_dock': 'DrugGPS',
    'ResGen_vina_dock': 'ResGen',
    'our_vina_score_docked': 'kgdiff',
    # DecompDiff ships a bare flat list[dict]; scripts/decompdiff_wrap.py wraps it into
    # {'all_results': ...} (-> results/decompdiff_gen/) so it aligns by ligand_filename
    # like PIDiff. Already docked + chem-scored; docking-successful only.
    'decompdiff_vina_docked': 'DecompDiff',
    # IPDiff's official release is raw sampler output with no docking scores and no
    # chem_results; this file is the in-house Vina pass (ipdiff_merge_docking.py) plus
    # scripts/ipdiff_add_chem.py. Grouped-by-pocket and index-aligned to targetdiff.
    'ipdiff_official_chem_vina_docked': 'IPDiff',
    # MolCraft also ships a bare flat list[dict], and additionally stores each Vina stage as
    # a single dict instead of a ranked list; scripts/molcraft_wrap.py wraps + normalises it
    # (-> results/molcraft_gen/). Already docked + chem-scored; docking-successful only.
    'molcraft_vina_docked': 'MolCraft',
    # AliDiff ships six disjoint per-pocket-range shards; scripts/alidiff_merge_shards.py
    # concatenates them into this one flat all_results file. Docking-successful only,
    # like every other external baseline.
    'alidiff_vina_docked': 'AliDiff',
    # DeepICL ships raw SDFs (no docking); scripts/dock_deepicl.py docked them in-house
    # (own-bbox, exh 16) into this grouped list[100]x[100] file with vina + chem inline.
    'deepicl_official_vina_docked': 'DeepICL',
    'head1_dock_342k_vina_docked': 'Ours (head1_dock_342k)',
    'pgdiff_3term_342k_vina_docked': 'Ours (pgdiff_3term_342k)',
    'vina_675k_vina_docked': 'Ours (vina_675k)',
    'pignet_675k_vina_docked': 'Ours (pignet_675k)',
    # fixed Vina/PIGNet type gating, best (val-loss) checkpoint. vina/pignet: 100 pockets x 100 mols;
    # pgdiff (unit_d0 False + squared force reduction, best iter 675k): 100 pockets x 24 mols.
    'vina_fixed_best_vina_docked': 'Ours (vina_fixed_best)',
    'pignet_fixed_best_vina_docked': 'Ours (pignet_fixed_best)',
    'pgdiff_fixed_best_vina_docked': 'Ours (pgdiff_fixed_best)',
    'novdw_gen_vina_docked': 'Ours (novdw / no-physics)',
    # --- guidance ablation (guide_mode: no_guide) ---
    # vina_675k was trained with the buggy element-level type gating and is superseded by
    # vina_fixed_best (correct Vina donor-acceptor / hydrophobic gating, best checkpoint).
    'vina_fixed_best_noguide_vina_docked': 'vina_fixed_best',
    'pignet_fixed_best_noguide_vina_docked': 'pignet_fixed_best',
    # atom-type guidance ONLY (head1_type_grad_weight 100, head1_pos_grad_weight 0)
    'vina_fixed_best_typeonly_vina_docked': 'vina_fixed_best [type-only]',
    # position guidance ONLY (head1_type_grad_weight 0, head1_pos_grad_weight 25)
    'vina_fixed_best_posonly_vina_docked': 'vina_fixed_best [pos-only]',
    'pgdiff_noguide_vina_docked': 'pgdiff_3term_342k',
    'pidiff_noguide_vina_docked': 'head1_dock_342k (pidiff)',
}

# Guidance ablation: (guided display name, ablation display name, printed label).
# The guided rows are re-aggregated over the SAME pockets the no_guide run covers,
# so the two conditions are strictly comparable.
# Optional third guidance condition per model: atom-type gradient only (position gradient 0).
# Maps guided display name -> ablation display name of its type-only run.
def _L(model_id, registry='configs/models.json'):
    """F1 display label for a registry id.

    These ablation maps used to hardcode the guided model's display name as a KEY. Renaming the
    label would then empty the ablation sections silently -- the same label-as-join-key mistake
    the registry exists to prevent, sitting inside the builder itself.
    """
    import sys as _s, os as _o
    _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
    import model_registry                                                # noqa: E402
    return model_registry.label_for(model_registry.by_id(model_id, registry), 'sbdd')


TYPE_ONLY = {
    _L('ours_vina'): 'vina_fixed_best [type-only]',
}

# Optional fourth condition: position gradient only (atom-type gradient 0).
POS_ONLY = {
    _L('ours_vina'): 'vina_fixed_best [pos-only]',
}

ABLATION_PAIRS = [
    (_L('ours_vina'),   'vina_fixed_best',          'vina_fixed_best  (energy)'),
]

# ---- PHYSICS ABLATION: which physics term is attached to the SAME kgdiff guidance ----
# Both rows run guide_mode head1_only with identical guidance weights, so the only thing
# that differs between them is the physics loss. Restricted to the pockets both cover.
#   (display name, printed label)
PHYSICS_ABLATION = [
    (_L('ours_vina'),        'vina_fixed_best  (Vina energy loss + kgdiff guidance)'),
    (_L('head1_dock_342k'),  'head1_dock_342k  (vdW force loss   + kgdiff guidance)'),
]

# DEAD as of 2026-08-21 -- kept only as documentation of what each stem is. Skipping is now
# implicit: the roster is iterated from configs/models.json, so a .pt nobody registered is
# simply never opened, and `supersedes` records which stem a registered model replaces.
# Files in --sampling_dir that must NOT be auto-loaded as-is.
# - PIDiff_vina_docked (partial, no vina_dock) is replaced by PIDiff_vina_docked_complete.
# - decompdiff_vina_docked_pose_checked is a bare flat list[dict] that the auto-scan would
#   mis-group into thousands of pockets; it is loaded via the wrapped copy in our_file instead.
# - decompdiff_ref_vina_docked_pose_checked docks against the REFERENCE-ligand box (off-target
#   per our own-bbox convention), so it is excluded entirely, not shown as another DecompDiff row.
SKIP_STEMS = {'PIDiff_vina_docked',
              'decompdiff_vina_docked_pose_checked',
              'decompdiff_ref_vina_docked_pose_checked'}

# Baselines staged into sampling_dir but NOT yet integrated. They are bare flat list[dict]
# files (same shape as DecompDiff's raw file), so the auto-scan would mis-group each into
# thousands of single-molecule pockets. Skipped for now so they don't pollute the tables;
# to add any of them, wrap it like scripts/decompdiff_wrap.py and register it the same way.
# molpilot_ref additionally has 0 dock scores (ref-box only).
SKIP_STEMS |= {
    'binddm_vina_docked_pose_checked',
    'flag_vina_docked_pose_checked',
    'flag_official_vina_docked_pose_checked',
    # integrated via the wrapped copy in our_file (results/molcraft_gen/), like DecompDiff
    'molcraft_vina_docked_pose_checked',
    'moljo_vina_sa_vina_docked_pose_checked',
    'molpilot_ref_vina_docked',
    'tagmol_vina_docked_pose_checked',
}

# Loaded (so the ablation sections can use them) but kept OUT of the OVERALL and
# PER-TARGET tables. Those two tables are the cross-method comparison: external
# baselines plus only our current models. Superseded checkpoints and models that
# exist purely as an ablation arm live in their own section instead.
#   *_675k          - trained with the buggy element-level type gating, superseded
#                     by vina_fixed_best / pignet_fixed_best
#   *_342k          - older checkpoints; head1_dock_342k is now the vdW-loss arm of
#                     the PHYSICS ABLATION section
#   novdw           - the no-physics control, an ablation arm, not a method
#   pgdiff_fixed_best - superseded, not part of the current model set
#   pignet_fixed_best - RETIRED 2026-08-05 by user decision, together with pgdiff_fixed_best.
#                     Both are dropped from every generated table (main, per-target and the
#                     guidance ablation) and from the pose-fidelity registry. The raw sampling
#                     .pt files and eval_out/ docking outputs are KEPT -- only the reporting is
#                     removed, so the decision is reversible by restoring these entries.
MAIN_TABLE_EXCLUDE = {
    'Ours (pignet_fixed_best)',      # retired 2026-08-05 (user decision), with pgdiff_fixed_best
    'Ours (vina_675k)',
    'Ours (pignet_675k)',
    'Ours (pgdiff_3term_342k)',
    'Ours (head1_dock_342k)',
    'Ours (novdw / no-physics)',
    'Ours (pgdiff_fixed_best)',
}

METRICS = ['vina_score', 'vina_min', 'vina_dock', 'qed', 'sa']
METRIC_LABELS = {
    'vina_score': 'Vina S',
    'vina_min': 'Vina M',
    'vina_dock': 'Vina D',
    'qed': 'QED',
    'sa': 'SA',
}

# ---- HA% / Div, defined as in ConDitar (results/comparison/
# ConDitar_SBDD_with_property_optimization.pdf, "Evaluation Metrics" p.5 + Table 1 caption p.6) ----
#
# HA%  per target, the % of generated ligands whose Vina D beats that target's REFERENCE
#      ligand; the table reports Avg./Med. of that percentage across targets.
# Div  per target, the average pairwise Tanimoto DISTANCE (1 - similarity) over 2048-bit
#      RDKit fingerprints; Avg./Med. across targets. Needs >= 2 molecules in the pocket.
#
# SR% (the pooled Vina D < -8.18 / QED > 0.25 / SA > 0.59 success rate) was DROPPED from this
# family: it is a molecule-pooled rate whose denominator is the docking-successful set, which is
# the full generated set for every external baseline but a survivor population for ours, so the
# column read optimistically for us by construction and could not be repaired inside F1.
#
# SA convention matches the paper: utils/evaluation/sascorer.compute_sa_score returns the
# NORMALISED (10 - sa) / 9 in [0,1], higher = better.
# (Cross-check: our native reference SA 0.728 vs the paper's 0.73.)


def load_pt(path):
    """torch.load that tolerates RDKit objects under torch>=2.6 (weights_only)."""
    try:
        return torch.load(path, map_location='cpu')
    except Exception:
        return torch.load(path, map_location='cpu', weights_only=False)


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def get_vina(vina):
    """Return (score, min, dock) affinities; any may be None."""
    s = m = d = None
    if isinstance(vina, dict):
        if vina.get('score_only'):
            s = _f(vina['score_only'][0].get('affinity'))
        if vina.get('minimize'):
            m = _f(vina['minimize'][0].get('affinity'))
        if vina.get('dock'):
            d = _f(vina['dock'][0].get('affinity'))
    elif isinstance(vina, list) and vina:
        # qvina-style: single ranked list -> treat best as dock
        d = _f(vina[0].get('affinity'))
    return s, m, d


def extract_mol(mol, need_fp=False):
    s, m, d = get_vina(mol.get('vina'))
    chem = mol.get('chem_results') or {}
    md = {
        'vina_score': s,
        'vina_min': m,
        'vina_dock': d,
        'qed': _f(chem.get('qed')),
        'sa': _f(chem.get('sa')),
        'fp': None,
    }
    # Fingerprints feed Div only, and Div is reported over the docking-successful set,
    # so undocked molecules never need one.
    if need_fp and d is not None:
        rdmol = mol.get('mol')
        if rdmol is not None:
            try:
                md['fp'] = Chem.RDKFingerprint(rdmol)     # RDKit default fpSize = 2048
            except Exception:
                pass
    return md


def pocket_dir(mol):
    lf = mol.get('ligand_filename') or ''
    return lf.split('/')[0] if lf else None


def canonical_index(canonical_obj):
    """pocket index -> full ligand_filename, from targetdiff's per-pocket order."""
    idx2full = {}
    for i, elem in enumerate(canonical_obj):
        mols = elem if isinstance(elem, list) else [elem]
        for mol in mols:
            lf = mol.get('ligand_filename')
            if lf:
                idx2full[i] = lf
                break
    return idx2full


def build_pocket_maps(idx2full):
    """Derive the alignment/display maps from idx2full.

    The CrossDock test set lists some binding sites twice under the SAME pocket
    directory but with a DIFFERENT reference ligand (e.g. NOS1 at index 53 uses
    3tym..., index 54 uses 4d7o...). These are distinct targets, so aligning
    flat all_results models by directory alone wrongly merges the two. We key on
    the full ligand_filename instead, and only fall back to the directory for
    directories that map to a single index.

    Returns:
      idx2name - display name per index (dir, tagged with the receptor code when
                 the dir is shared by >1 index so the two rows stay distinct)
      full2idx - full ligand_filename -> index (exact, 1:1)
      dir2idx  - pocket dir -> index, only for dirs unique to one index
    """
    dir_of = {i: lf.split('/')[0] for i, lf in idx2full.items()}
    dir_counts = Counter(dir_of.values())
    idx2name, full2idx, dir2idx = {}, {}, {}
    for i, lf in idx2full.items():
        d = dir_of[i]
        full2idx[lf] = i
        if dir_counts[d] == 1:
            dir2idx[d] = i
            idx2name[i] = d
        else:                                    # shared dir -> tag by receptor
            ref = os.path.basename(lf).split('_')[0]
            idx2name[i] = f'{d} [{ref}]'
    return idx2name, full2idx, dir2idx


def normalize_model(obj, full2idx=None, dir2idx=None, need_fp=False, extract=None):
    """Return (per_pocket, flat_list).

    Grouped-by-pocket models (list indexed by the shared CrossDock test-set
    order) are aligned by list index -> per_pocket populated. Flat all_results
    models are aligned by the full ligand_filename (exact target), falling back
    to the pocket directory only for directories unique to one index. Molecules
    with no match contribute to the overall (pooled) table only.

    `extract` overrides the per-molecule reader (default extract_mol). It exists so a
    sibling builder can pull MORE per-molecule fields out of the same .pt files while
    reusing this function's pocket alignment, which is the part that is hard to get
    right. Passing nothing leaves this file's behaviour bit-identical.
    """
    extract = extract or extract_mol
    per_pocket = defaultdict(list)
    flat = []

    if isinstance(obj, dict) and 'all_results' in obj:
        for mol in obj['all_results']:
            md = extract(mol, need_fp)
            flat.append(md)
            lf = mol.get('ligand_filename')
            i = None
            if lf and full2idx:
                i = full2idx.get(lf)
                if i is None and dir2idx:              # unique-dir fallback
                    i = dir2idx.get(lf.split('/')[0])
            if i is not None:
                per_pocket[i].append(md)
    elif isinstance(obj, list):
        for i, elem in enumerate(obj):
            mols = elem if isinstance(elem, list) else [elem]
            for mol in mols:
                md = extract(mol, need_fp)
                per_pocket[i].append(md)
                flat.append(md)
    return per_pocket, flat


def agg(values):
    vals = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not vals:
        return None, None
    return float(np.mean(vals)), float(np.median(vals))


def docked_only(mols):
    """Keep only molecules that succeeded through docking (have a Vina Dock score).

    Every model's numbers are reported over this set: the reference files (targetdiff / kgdiff /
    pidiff, and novdw) hold docking-successful molecules ONLY -- confirmed by the author of those
    runs, and visible as n_mols here matching their full entry count exactly (targetdiff 9036,
    kgdiff 8813, pidiff 850) -- so this is a no-op for them and n_mol is unchanged, while our runs
    also contain fragmented / docking-failed molecules, which this drops so n_mol and all metrics
    are the docking-successful count for every model alike. A molecule that docked is necessarily
    an intact, reconstructed molecule, so this is also the QED/SA denominator.
    """
    return [md for md in mols if md.get('vina_dock') is not None]


def pocket_ha(mols, ref_dock):
    """% of this pocket's docking-successful molecules beating the reference ligand's Vina D."""
    if ref_dock is None:
        return None
    vals = [md['vina_dock'] for md in docked_only(mols)]
    if not vals:
        return None
    return 100.0 * sum(1 for v in vals if v < ref_dock) / len(vals)


def pocket_div(mols):
    """Average pairwise Tanimoto DISTANCE among this pocket's docking-successful molecules."""
    fps = [md['fp'] for md in docked_only(mols) if md.get('fp') is not None]
    if len(fps) < 2:
        return None
    total, n = 0.0, 0
    for i in range(len(fps) - 1):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:])
        total += sum(1.0 - s for s in sims)
        n += len(sims)
    return total / n if n else None


def extra_metrics(per_pocket, ref_dock):
    """HA% / Div: per-pocket first, then avg+med across pockets.

    Both are per-TARGET measurements, so a model whose molecules never map onto a canonical
    pocket (per_pocket empty -- DrugGPS / ResGen / DiffSBDD) gets blanks for both.
    """
    ha = [pocket_ha(mols, ref_dock.get(i)) for i, mols in per_pocket.items()]
    dv = [pocket_div(mols) for mols in per_pocket.values()]
    return {
        'ha': agg([x for x in ha if x is not None]),
        'div': agg([x for x in dv if x is not None]),
    }


def overall_row(flat, per_pocket):
    """Pool docking-successful molecules, return {metric: (avg, med)} + pocket/mol counts."""
    valid = docked_only(flat)
    pooled = {k: [md[k] for md in valid] for k in METRICS}
    row = {k: agg(pooled[k]) for k in METRICS}
    return row, len(per_pocket), len(valid)


def restricted_row(per_pocket, idxs):
    """Pool docking-successful molecules that live in `idxs`. Returns (row, n_pk, n_mol)."""
    valid = docked_only([md for i in idxs for md in per_pocket.get(i, [])])
    pooled = {k: [md[k] for md in valid] for k in METRICS}
    n_pk = sum(1 for i in idxs if any(m.get('vina_dock') is not None for m in per_pocket.get(i, [])))
    return {k: agg(pooled[k]) for k in METRICS}, n_pk, len(valid)


def pocket_row(mols):
    valid = docked_only(mols)
    per = {k: [md[k] for md in valid] for k in METRICS}
    return {k: agg(per[k]) for k in METRICS}, len(valid)


def fmt(x, nd=3):
    return '' if x is None else f'{x:.{nd}f}'


def metric_columns():
    cols = []
    for m in METRICS:
        cols += [f'{m}_avg', f'{m}_med']
    return cols


# Table column order, matching ConDitar Table 1 minus SR%:
#   Vina S | Vina M | Vina D | HA% | QED | SA | Div
_VINA_KEYS = ('vina_score', 'vina_min', 'vina_dock')
_CHEM_KEYS = ('qed', 'sa')


def metric_cells(row, extras=None, per_pocket_scalar=False):
    """Cells in table order. extras=None -> the plain 10-column layout (ablation tables).

    per_pocket_scalar: a PER-TARGET row is a single pocket, so its HA% / Div are single
    numbers rather than an avg/med pair -- emit one column each instead of two.
    """
    out = []
    for m in _VINA_KEYS:
        a, md = row[m]
        out += [fmt(a), fmt(md)]
    if extras is not None:
        a, md = extras['ha']
        out += [fmt(a, 1)] if per_pocket_scalar else [fmt(a, 1), fmt(md, 1)]
    for m in _CHEM_KEYS:
        a, md = row[m]
        out += [fmt(a), fmt(md)]
    if extras is not None:
        a, md = extras['div']
        out += [fmt(a)] if per_pocket_scalar else [fmt(a), fmt(md)]
    return out


def write_overall_csv(path, models, extras=None):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        # `id` first: a LABEL IS NOT A JOIN KEY. Downstream joins (the integrated
        # table) key on this, never on the display name.
        w.writerow(['id', 'model', 'n_pockets', 'n_mols'] + metric_columns()
                   + ['ha_pct_avg', 'ha_pct_med', 'div_avg', 'div_med'])
        for name, (row, n_pk, n_mols) in models:
            vals = []
            for m in METRICS:
                a, md = row[m]
                vals += [fmt(a), fmt(md)]
            ex = (extras or {}).get(name) or {'ha': (None, None), 'div': (None, None)}
            ha_a, ha_m = ex['ha']
            dv_a, dv_m = ex['div']
            vals += [fmt(ha_a, 1), fmt(ha_m, 1), fmt(dv_a), fmt(dv_m)]
            w.writerow([_row_id(name), name, n_pk, n_mols] + vals)


def write_per_target_csv(path, model_pockets, idx2name, ref_dock=None):
    ref_dock = ref_dock or {}
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['pocket', 'pocket_name', 'id', 'model', 'n_mols'] + metric_columns()
                   + ['ref_vina_dock', 'ha_pct', 'div'])
        all_idx = sorted({i for pp in model_pockets.values() for i in pp})
        for i in all_idx:
            for model_name, pp in model_pockets.items():
                if i not in pp:
                    continue
                mols = pp[i]
                row, n_mols = pocket_row(mols)
                vals = []
                for m in METRICS:
                    a, md = row[m]
                    vals += [fmt(a), fmt(md)]
                vals += [fmt(ref_dock.get(i)), fmt(pocket_ha(mols, ref_dock.get(i)), 1),
                         fmt(pocket_div(mols))]
                w.writerow([i, idx2name.get(i, ''), _row_id(model_name), model_name, n_mols] + vals)


def render_table(headers, rows, aligns):
    """Monospace fixed-width table (2-space gutter). aligns: 'l' or 'r' per col."""
    widths = [len(str(h)) for h in headers]
    for r in rows:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(str(c)))

    def fmt_row(cells):
        out = []
        for i, c in enumerate(cells):
            s = str(c)
            out.append(s.ljust(widths[i]) if aligns[i] == 'l' else s.rjust(widths[i]))
        return '  '.join(out).rstrip()

    lines = [fmt_row(headers), fmt_row(['-' * w for w in widths])]
    for r in rows:
        lines.append(fmt_row(r))
    return lines


# Compact column headers for the txt tables.
# TXT_METRIC_COLS   - the plain 10-column layout, used by the ablation sections.
# FULL_METRIC_COLS  - OVERALL, with HA% after Vina D and Div after SA.
# PK_METRIC_COLS    - PER-TARGET; one pocket per row, so HA% and Div are single columns.
TXT_METRIC_COLS = []
for _m in METRICS:
    TXT_METRIC_COLS += [f'{METRIC_LABELS[_m]}(avg)', f'{METRIC_LABELS[_m]}(med)']


def _cols(ha_div_scalar):
    cols = []
    for m in _VINA_KEYS:
        cols += [f'{METRIC_LABELS[m]}(avg)', f'{METRIC_LABELS[m]}(med)']
    cols += ['HA%'] if ha_div_scalar else ['HA%(avg)', 'HA%(med)']
    for m in _CHEM_KEYS:
        cols += [f'{METRIC_LABELS[m]}(avg)', f'{METRIC_LABELS[m]}(med)']
    cols += ['Div'] if ha_div_scalar else ['Div(avg)', 'Div(med)']
    return cols


FULL_METRIC_COLS = _cols(False)
PK_METRIC_COLS = _cols(True)


def ablation_lines(all_pockets, abl_pockets, sub):
    """head1 guidance ON vs OFF on an identical pocket set."""
    pairs = [(g, a, lab) for g, a, lab in ABLATION_PAIRS
             if g in all_pockets and a in abl_pockets]
    if not pairs:
        return []
    # common pockets: present in every guided AND every ablation model
    common = None
    for g, a, _ in pairs:
        have = {i for i, v in all_pockets[g].items() if v} & {i for i, v in abl_pockets[a].items() if v}
        common = have if common is None else (common & have)
    common = sorted(common or [])
    if not common:
        return []

    has_type_only = any(TYPE_ONLY.get(g) in abl_pockets for g, _, _ in pairs)
    lines = [sub, 'GUIDANCE ABLATION - head1 classifier guidance ON vs OFF', sub, '',
             f'Identical pocket set for every row: {len(common)} pockets (data_id '
             f'{common[0]}-{common[-1]}). The guided rows are re-aggregated over exactly',
             'these pockets, so the conditions are strictly comparable.',
             '',
             '  ON   = guide_mode head1_only, head1_type_grad_weight 100, head1_pos_grad_weight 25',
             '  TYPE = atom-type gradient only (head1_type_grad_weight 100, head1_pos_grad_weight 0)',
             '  POS  = position gradient only  (head1_type_grad_weight 0,   head1_pos_grad_weight 25)',
             '  OFF  = guide_mode no_guide     (both guidance weights 0)',
             '']
    if has_type_only:
        lines += ['TYPE and POS split the guidance in half: TYPE keeps only the atom-type gradient and',
                  'POS only the position gradient, so reading them against ON and OFF shows which half',
                  'of the guidance is doing the work.', '']
    headers = ['Model', 'guide', 'n_pk', 'n_mol'] + TXT_METRIC_COLS
    aligns = ['l', 'l', 'r', 'r'] + ['r'] * len(TXT_METRIC_COLS)
    rows = []
    for g, a, lab in pairs:
        gr, gpk, gm = restricted_row(all_pockets[g], common)
        ar, apk, am = restricted_row(abl_pockets[a], common)

        cells = metric_cells

        rows.append([lab, 'ON', gpk, gm] + cells(gr))
        t_name = TYPE_ONLY.get(g)
        if t_name in abl_pockets:
            tr, tpk, tm = restricted_row(abl_pockets[t_name], common)
            rows.append(['', 'TYPE', tpk, tm] + cells(tr))
        p_name = POS_ONLY.get(g)
        if p_name in abl_pockets:
            pr, ppk, pm = restricted_row(abl_pockets[p_name], common)
            rows.append(['', 'POS', ppk, pm] + cells(pr))
        rows.append(['', 'OFF', apk, am] + cells(ar))
    lines += render_table(headers, rows, aligns)
    lines += ['']
    return lines


def physics_ablation_lines(all_pockets, sub):
    """Same kgdiff guidance, different physics loss, on an identical pocket set."""
    rows_in = [(n, lab) for n, lab in PHYSICS_ABLATION if n in all_pockets]
    if len(rows_in) < 2:
        return []

    common = None
    for n, _ in rows_in:
        have = {i for i, v in all_pockets[n].items()
                if any(m.get('vina_dock') is not None for m in v)}
        common = have if common is None else (common & have)
    common = sorted(common or [])
    if not common:
        return []

    lines = [sub, 'PHYSICS ABLATION - which physics loss, same kgdiff guidance', sub, '',
             f'Identical pocket set for both rows: {len(common)} pockets (data_id '
             f'{common[0]}-{common[-1]}), re-aggregated',
             'over exactly those pockets so the two conditions are strictly comparable.',
             '',
             'Both rows use the SAME sampler settings (guide_mode head1_only,',
             'head1_type_grad_weight 100, head1_pos_grad_weight 25). The only difference is',
             'the physics term in the training loss:',
             '',
             '  Vina energy loss - a differentiable Vina score on the predicted x0 (an ENERGY:',
             '                     it pulls the ligand toward the protein)',
             '  vdW force loss   - a Lennard-Jones force residual on the predicted x0 (a FORCE:',
             '                     it pushes overlapping atoms apart)',
             '',
             'n_mol is docking-successful molecules only, so the fragmented share each physics',
             'loss produces is already excluded here -- quote it from',
             'analysis/RESULTS_pose_fidelity_2026-07.md alongside these numbers, never on its own.',
             '']
    headers = ['Model', 'n_pk', 'n_mol'] + TXT_METRIC_COLS
    aligns = ['l', 'r', 'r'] + ['r'] * len(TXT_METRIC_COLS)
    rows = []
    for n, lab in rows_in:
        row, n_pk, n_mol = restricted_row(all_pockets[n], common)
        rows.append([lab, n_pk, n_mol] + metric_cells(row))
    lines += render_table(headers, rows, aligns)
    lines += ['']
    return lines


def write_txt(path, models, model_pockets, idx2name, abl_pockets=None, all_pockets=None,
              extras=None, ref_dock=None):
    W = 190
    bar = '=' * W
    sub = '-' * W
    lines = [bar,
             'MODEL COMPARISON - VINA DOCKING & MOLECULAR PROPERTIES',
             bar,
             '',
             'Vina affinities in kcal/mol (lower = better).  QED / SA / HA% / Div higher = better.',
             'Blank cell = that metric could not be computed for the model (e.g. liGAN stores only',
             'vina_dock; DrugGPS / ResGen / DiffSBDD are not pocket-aligned, so HA% and Div, which are',
             'per-TARGET measurements, have no target to attach to).',
             '',
             'HA%  % of a target\'s ligands whose Vina D beats that target\'s REFERENCE ligand,',
             '     averaged (avg) / median (med) across targets.',
             'Div  average pairwise Tanimoto distance (1 - similarity, 2048-bit RDKit fingerprints)',
             '     within a target, then avg / med across targets.',
             '',
             'Definitions follow results/comparison/ConDitar_SBDD_with_property_optimization.pdf,',
             'minus its SR%: that pooled success rate is deliberately NOT reported here -- see the',
             'CAVEAT under OVERALL AVERAGE.',
             '']

    if all_pockets is None:
        all_pockets = model_pockets

    if abl_pockets:
        lines += ablation_lines(all_pockets, abl_pockets, sub)
    lines += physics_ablation_lines(all_pockets, sub)

    # ---- Overall table ----
    lines += [sub, 'OVERALL AVERAGE', sub, '',
              'n_mol = docking-successful molecules (those with a Vina Dock score); all metrics are',
              'averaged over exactly this set. The reference files were already docking-successful-only,',
              'so their n_mol is unchanged; our rows drop the fragmented/undocked share of the',
              '24/pocket generated.',
              '',
              'External baselines plus our current models only. Superseded checkpoints and',
              'ablation-only arms (the *_675k / *_342k runs, novdw, pgdiff_fixed_best) are',
              'excluded here and appear in the ablation sections above instead.',
              '',
              'CAVEAT on HA%. It is a RATE, so its denominator is the docking-successful set here',
              'too. That set is the full generated set for every external baseline -- their .pt files',
              'never stored a docking failure -- but NOT for ours: 15-28% of what the physics losses',
              'generate is disconnected, never docks, and is therefore absent from the denominator.',
              'Our HA% is consequently measured on a survivor population and reads optimistically',
              'against the baselines. Always quote the fragmentation rate from',
              'analysis/RESULTS_pose_fidelity_2026-07.md next to this column.',
              '',
              'SR% was REMOVED from this family for the same reason, without the mitigation: it is a',
              'pooled rate over molecules with no per-pocket structure to fall back on, so unlike HA%',
              'it could not be qualified, only misread.', '']
    headers = ['Model', 'n_pk', 'n_mol'] + FULL_METRIC_COLS
    aligns = ['l', 'r', 'r'] + ['r'] * len(FULL_METRIC_COLS)
    rows = []
    for name, (row, n_pk, n_mols) in models:
        rows.append([name, n_pk, n_mols]
                    + metric_cells(row, (extras or {}).get(name)))
    lines += render_table(headers, rows, aligns)
    lines += ['']

    # ---- Per-target tables ----
    lines += ['', bar, 'PER-TARGET RESULTS', bar, '',
              'Models stored grouped-by-pocket are index-aligned; flat all_results',
              'models are pocket-aligned by their ligand_filename directory. Molecules',
              'whose pocket dir is not in the canonical set appear in OVERALL only.', '']

    all_idx = sorted({i for pp in model_pockets.values() for i in pp})
    pk_headers = ['Model', 'n_mol'] + PK_METRIC_COLS
    pk_aligns = ['l', 'r'] + ['r'] * len(PK_METRIC_COLS)
    ref_dock = ref_dock or {}
    for i in all_idx:
        lines += [sub, f'Pocket {i:>3d}  -  {idx2name.get(i, "?")}'
                       f'   [reference Vina D {fmt(ref_dock.get(i))}]', sub]
        prows = []
        for model_name, pp in model_pockets.items():
            if i not in pp:
                continue
            row, n_mols = pocket_row(pp[i])
            # One pocket per row, so HA% / Div are scalars, not an avg/med pair.
            pk_extra = {'ha': (pocket_ha(pp[i], ref_dock.get(i)), None),
                        'div': (pocket_div(pp[i]), None)}
            prows.append([model_name, n_mols]
                         + metric_cells(row, pk_extra, per_pocket_scalar=True))
        lines += render_table(pk_headers, prows, pk_aligns)
        lines += ['']

    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


# Row order for all tables: Reference on top, then the remaining models sorted
# by performance (best Vina Dock first), then these key methods at the bottom.
# ROW_HEAD is still the fallback for an unregistered --our_file override; ROW_TAIL is DEAD --
# row order comes from the registry's `order` field (see order_models below).
ROW_HEAD = 'Reference (native)'
ROW_TAIL = ['TargetDiff', 'PIDiff', 'DeepICL', 'DecompDiff', 'IPDiff', 'MolCraft', 'AliDiff', 'kgdiff',
            'Ours (vina_fixed_best)']


def _row_id(label, registry='configs/models.json'):
    """Registry id for a published label, or '' when nothing claims it.

    Empty rather than the label itself: a blank id says "this row is not in the registry", which
    a join can act on. Echoing the label back would silently reintroduce label-joining.
    """
    import sys as _s, os as _o
    _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
    import model_registry                                                # noqa: E402
    for m in model_registry.models(registry):
        for fam in model_registry.FAMILIES:
            if model_registry.label_for(m, fam) == label:
                return m['id']
    return ''


def order_models(models, registry='configs/models.json'):
    """Return display names in the desired top-to-bottom table order.

    Order is the registry's declared `order`, not a sort on the data. The old version ranked the
    middle block by mean Vina Dock, which made ROW ORDER A FUNCTION OF THE NUMBERS -- re-running
    after any re-dock could silently reshuffle rows, and a reader diffing two versions of the
    table could not tell a reordering from a result change.

    Anything the registry does not know about keeps its relative position at the end, so an
    --our_file override still renders.
    """
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    import model_registry                                                # noqa: E402

    rank = {}
    for m in model_registry.models(registry):
        for fam in ('sbdd',):
            rank[model_registry.label_for(m, fam)] = m.get('order', 10**6)
    names = [name for name, _ in models]
    return sorted(names, key=lambda n: (rank.get(n, 10**6), names.index(n)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sampling_dir', type=str, default='./results/sampling_results')
    p.add_argument('--our_file', type=str, nargs='+',
                   default=['./results/vina_fixed_best_gen/vina_fixed_best_vina_docked.pt',
                            './results/ipdiff_official_gen/ipdiff_official_chem_vina_docked.pt',
                            './results/AliDiff_evaluation_results/alidiff_vina_docked.pt',
                            './results/decompdiff_gen/decompdiff_vina_docked.pt',
                            './results/molcraft_gen/molcraft_vina_docked.pt',
                            './results/deepicl_gen/deepicl_official_vina_docked.pt',
                            # PIDiff retrained from scratch; replaces the pidiff_pub row in the
                            # F1 roster from 2026-09-10 (pidiff_pub sbdd -> `retired`). It lives
                            # outside sampling_dir because it is our own run, not a shipped
                            # baseline, and it is grouped-by-pocket like every other file here.
                            './results/pidiff_retrain_gen/pidiff_retrain_vina_docked.pt',
                            # loaded for the PHYSICS ABLATION section only; MAIN_TABLE_EXCLUDE
                            # keeps it out of OVERALL / PER-TARGET.
                            './results/head1_dock_342k/head1_dock_342k_vina_docked.pt'],
                   help='extra grouped-by-pocket result file(s) living outside sampling_dir')
    p.add_argument('--ablation_file', type=str, nargs='*',
                   default=['./results/noguide/vina_fixed_best_noguide/vina_fixed_best_noguide_vina_docked.pt',
                            './results/typeonly/vina_fixed_best_typeonly/vina_fixed_best_typeonly_vina_docked.pt',
                            './results/posonly/vina_fixed_best_posonly/vina_fixed_best_posonly_vina_docked.pt'],
                   help='guide_mode=no_guide runs; shown only in the GUIDANCE ABLATION section, '
                        'never mixed into the 100-pocket OVERALL / PER-TARGET tables')
    p.add_argument('--canonical', type=str,
                   default='targetdiff_vina_docked.pt',
                   help='grouped-by-pocket file used to fix the 0..99 pocket order')
    p.add_argument('--out_dir', type=str, default='./results/comparison/f1_sbdd')
    p.add_argument('--registry', type=str, default='configs/models.json',
                   help='single source of truth for the roster, labels, tiers and row order')
    p.add_argument('--audit_unregistered', action='store_true',
                   help='list .pt files in sampling_dir that no registry entry claims. Reports '
                        'only -- an unregistered file can never become a table row.')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ---- Roster comes from configs/models.json, not from whatever .pt happens to be on disk.
    # The old code globbed sampling_dir and then subtracted SKIP_STEMS, so a newly staged
    # baseline silently became a table row -- and one did: PharDiff had no DISPLAY_NAMES entry,
    # so its raw file stem 'PharDiff_vina_dock' shipped as the row label. Iterating the registry
    # inverts that: an unregistered .pt is inert, and a REGISTERED model whose file is missing is
    # now a loud error instead of a quietly absent row.
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    import model_registry                                                # noqa: E402

    # Guidance-ablation rows are owned by --ablation_file, which re-aggregates them over the
    # 25-pocket common set under their own names. Loading them here as well would put the same
    # data in all_pockets twice under two different labels.
    roster = [m for m in model_registry.for_family(
                  'sbdd', tiers=('reference', 'core', 'extended', 'ablation'),
                  statuses=('ok',), sections=None, path=args.registry)
              if model_registry.section(m, 'sbdd') != 'guidance_ablation']

    def _norm(p):
        return os.path.normpath(p)

    files, missing, seen = [], [], set()
    for m in roster:
        src = model_registry.source_for(m, 'sbdd')
        if not src:
            continue
        if os.path.isfile(src):
            files.append((src, m))
            seen.add(_norm(src))
        else:
            missing.append((m['id'], src))
    for extra in (args.our_file or []):          # explicit override still wins
        if not os.path.isfile(extra):
            print(f'[note] not found yet: {extra} (run dock_and_eval.py first)')
        elif _norm(extra) not in seen:           # normpath: './results/x' == 'results/x'
            files.append((extra, None))
            seen.add(_norm(extra))
    if missing:
        print(f'[warn] {len(missing)} registered model(s) have no source file on disk:')
        for mid, src in missing:
            print(f'         {mid}: {src}')

    # The old auto-scan, kept as an AUDIT that only prints. This is how a newly staged baseline
    # gets noticed without it being able to enter a table by itself.
    if args.audit_unregistered:
        known = {(m.get('ids') or {}).get('pt_stem') for m in model_registry.models(args.registry)}
        superseded = {s for m in model_registry.models(args.registry)
                      for s in (m.get('supersedes') or [])}
        strays = [fn[:-3] for fn in sorted(os.listdir(args.sampling_dir))
                  if fn.endswith('.pt') and fn[:-3] not in known]
        if strays:
            print(f'[audit] {len(strays)} .pt in {args.sampling_dir} not in the registry '
                  f'(none of them entered a table):')
            for s in strays:
                why = ' (superseded)' if s in superseded else ''
                print(f'          {s}{why}')

    # Canonical pocket names from targetdiff's per-pocket ordering.
    canon_path = os.path.join(args.sampling_dir, args.canonical)
    canon_obj = load_pt(canon_path)
    idx2full = canonical_index(canon_obj)
    idx2name, full2idx, dir2idx = build_pocket_maps(idx2full)
    print(f'Canonical pockets: {len(idx2name)}')

    # Guidance-ablation runs live in their own section (25 pockets only), so they
    # are loaded separately and never enter the 100-pocket OVERALL table.
    abl_pockets = {}
    for path in (args.ablation_file or []):
        if not os.path.isfile(path):
            print(f'[note] ablation file missing: {path}')
            continue
        stem = os.path.basename(path)[:-3]
        name = DISPLAY_NAMES.get(stem, stem)
        try:
            per_pocket, flat = normalize_model(load_pt(path), full2idx, dir2idx)
        except Exception as e:
            print(f'[skip] ablation {stem}: load failed ({e})')
            continue
        if flat:
            abl_pockets[name] = per_pocket
            print(f'  [ablation] {name:28s} pockets={len(per_pocket):3d} mols={len(flat)}')

    # Pass 1: load every model. HA% needs the native reference's per-pocket Vina D, which is
    # itself one of these files, so the extras can only be computed once all of them are in.
    loaded = []            # (display_name, per_pocket, flat, registry_entry)
    for path, entry in files:
        stem = os.path.basename(path)[:-3]
        # Labels come from the registry. An --our_file override with no entry keeps the old
        # stem fallback, but says so rather than shipping a filename as a row heading.
        if entry is None:
            try:
                entry = model_registry.resolve('sbdd', stem, path=args.registry)
            except KeyError:
                entry = None
        if entry is None:
            name = stem
            print(f'[warn] {stem}: no registry entry -- falling back to the file stem as the '
                  f'row label. Register it in {args.registry}.')
        else:
            name = model_registry.label_for(entry, 'sbdd')
        try:
            obj = load_pt(path)
        except Exception as e:
            print(f'[skip] {stem}: load failed ({e})')
            continue
        # Div is a per-target metric, so only pocket-aligned models can ever report it;
        # skipping the fingerprints for the rest saves the RDKit pass entirely.
        per_pocket, flat = normalize_model(obj, full2idx, dir2idx, need_fp=True)
        if not flat:
            print(f'[skip] {stem}: no usable molecules')
            continue
        loaded.append((name, per_pocket, flat, entry))

    # Reference ligand Vina D per pocket -- the HA% threshold.
    ref_dock = {}
    for name, per_pocket, _, entry in loaded:
        # tier 'reference' identifies the crystal-ligand row; matching on the label would break
        # the moment the label is edited.
        is_ref = entry.get('tier') == 'reference' if entry else name == ROW_HEAD
        if not is_ref:
            continue
        for i, mols in per_pocket.items():
            vals = [md['vina_dock'] for md in mols if md['vina_dock'] is not None]
            if vals:
                ref_dock[i] = float(np.mean(vals))
    if not ref_dock:
        print('[warn] no reference (native) row loaded -- HA% will be blank for every model')
    else:
        print(f'Reference Vina D available for {len(ref_dock)} pockets (HA% threshold)')

    # Pass 2: build the table rows.
    models = []            # (display_name, (overall_row, n_pk, n_mols)) -- main tables only
    model_pockets = {}     # display_name -> dict[pocket_idx] -> [metric dicts], main tables only
    all_pockets = {}       # same, but INCLUDING MAIN_TABLE_EXCLUDE (for the ablation sections)
    extras = {}            # display_name -> {'ha': (avg, med), 'div': (avg, med)}
    for name, per_pocket, flat, entry in loaded:
        row, n_pk, n_mols = overall_row(flat, per_pocket)
        if per_pocket:
            all_pockets[name] = per_pocket
        # Main-table membership is now a registry property, not a hand-maintained name list:
        # tier 'ablation'/'quarantine' and any model assigned to a sub-table section stay out of
        # OVERALL / PER-TARGET and appear only in their own block.
        if entry is not None:
            # tier 'ablation' is excluded BY DEFAULT, but that is a default and not a law: novdw
            # is an ablation arm and also a legitimate comparison point, so families.sbdd.main_row
            # promotes one deliberately without pretending it is not an ablation.
            fam = (entry.get('families') or {}).get('sbdd') or {}
            excluded = (entry.get('tier') == 'quarantine'
                        or model_registry.section(entry, 'sbdd') is not None
                        or (entry.get('tier') == 'ablation' and not fam.get('main_row')))
        else:
            excluded = name in MAIN_TABLE_EXCLUDE
        if excluded:
            print(f'  {name:28s} pockets={n_pk:3d} mols(docked)={n_mols}'
                  f'  (ablation sections only, excluded from OVERALL/PER-TARGET)')
            continue
        models.append((name, (row, n_pk, n_mols)))
        extras[name] = extra_metrics(per_pocket, ref_dock)
        if per_pocket:                       # only index-aligned models
            model_pockets[name] = per_pocket
        tag = '' if per_pocket else '  (overall-only, not pocket-aligned: HA%/Div blank)'
        print(f'  {name:28s} pockets={n_pk:3d} mols(docked)={n_mols}{tag}')

    # The reference ligand is the HA% yardstick and there is 1 molecule per pocket, so its own
    # HA% (it never beats itself) and Div (needs >= 2 molecules) are undefined, not zero.
    if ROW_HEAD in extras:
        extras[ROW_HEAD]['ha'] = (None, None)
        extras[ROW_HEAD]['div'] = (None, None)

    # Apply the requested row order to every table.
    order = order_models(models)
    rank = {n: i for i, n in enumerate(order)}
    models.sort(key=lambda x: rank[x[0]])
    model_pockets = {n: model_pockets[n] for n in order if n in model_pockets}

    write_overall_csv(os.path.join(args.out_dir, 'comparison_overall.csv'), models, extras)
    write_per_target_csv(os.path.join(args.out_dir, 'comparison_per_target.csv'),
                         model_pockets, idx2name, ref_dock)
    write_txt(os.path.join(args.out_dir, 'comparison_tables.txt'),
              models, model_pockets, idx2name, abl_pockets=abl_pockets,
              all_pockets=all_pockets, extras=extras, ref_dock=ref_dock)

    print(f'\nWrote:')
    print(f'  {os.path.join(args.out_dir, "comparison_overall.csv")}')
    print(f'  {os.path.join(args.out_dir, "comparison_per_target.csv")}')
    print(f'  {os.path.join(args.out_dir, "comparison_tables.txt")}')


if __name__ == '__main__':
    main()
