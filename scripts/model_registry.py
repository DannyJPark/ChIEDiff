#!/usr/bin/env python3
"""Single source of truth for the model roster: `configs/models.json`.

Phase A of the registry migration. This module can already REPRODUCE the two legacy per-family
registries exactly, so consumers can be switched over one at a time with a one-line change:

    reg = json.load(open('configs/pose_fidelity_models.json'))     # before
    reg = model_registry.load_as('configs/models.json', 'pose')    # after

`scripts/check_registry_sync.py` proves the reproduction is exact before anything is switched.

Why per-family shims instead of one shape: the families disagree on purpose. `n_mols` for our
main model is 9857 under `pose` (molecules exported) and 8340 under `delta` (docking-successful),
and `delta_score_common.load_model_pockets` asserts on its value -- the strongest data-integrity
check in the repo. Namespacing the counts keeps that assertion meaningful.
"""
import json
import os

DEFAULT_PATH = 'configs/models.json'
FAMILIES = ('sbdd', 'pose', 'nci', 'delta', 'quality')

# Which family view each legacy registry corresponds to.
LEGACY = {
    'configs/pose_fidelity_models.json': 'pose',
    'configs/delta_score_models.json': 'delta',
}

_CACHE = {}


def load(path=DEFAULT_PATH):
    """Parsed registry, cached by absolute path."""
    key = os.path.abspath(path)
    if key not in _CACHE:
        with open(path, encoding='utf-8') as fh:
            _CACHE[key] = json.load(fh)
    return _CACHE[key]


def models(path=DEFAULT_PATH):
    return load(path)['models']


def by_id(model_id, path=DEFAULT_PATH):
    for m in models(path):
        if m['id'] == model_id:
            return m
    raise KeyError(f'no model with id {model_id!r} in {path}')


def status(m, family):
    return (m.get('families', {}).get(family) or {}).get('status')


def source_for(m, family):
    """Per-family source .pt, falling back to 'default'.

    IPDiff is why this exists: F1/F2 read ipdiff_official_chem_vina_docked.pt (it carries
    chem_results) while F3 reads ipdiff_official_vina_docked.pt (its p<NNN>_m<MMMM> positions
    match eval_plip/ipdiff). Same molecules, different extra fields.
    """
    src = m.get('sources') or {}
    return src.get(family, src.get('default'))


def section(m, family):
    """Sub-table this model belongs to within a family, or None for the main table.

    `ours_noguide` is the reason this exists: its F1 status is `ok`, but it is loaded via
    --ablation_file and only ever appears in the guidance-ablation block, never as an OVERALL
    row. Counting it as a main row would report 16 where the table has 15.
    """
    return (m.get('families', {}).get(family) or {}).get('section')


def for_family(family, tiers=('reference', 'core', 'extended'), statuses=('ok',),
               sections=(None,), path=DEFAULT_PATH):
    """Registry entries for one family, in `order`, filtered by tier, status and sub-table.

    `sections=(None,)` -> main-table rows only (the default). Pass `sections=None` for every
    row regardless of sub-table, or e.g. `sections=('guidance_ablation',)` for one block.
    """
    if family not in FAMILIES:
        raise ValueError(f'unknown family {family!r}; expected one of {FAMILIES}')
    # families.<f>.main_row is an explicit opt-in that overrides the TIER filter: an ablation arm
    # can still be a legitimate main-table row (novdw is both). Honouring it here rather than in
    # each builder is what keeps the builders and check_rosters.py from disagreeing -- they did,
    # and the roster gate caught it.
    def _keep(m):
        fam = (m.get('families', {}).get(family) or {})
        tier_ok = m.get('tier') in tiers or bool(fam.get('main_row'))
        return (tier_ok
                and (statuses is None or status(m, family) in statuses)
                and (sections is None or section(m, family) in sections))

    out = [m for m in models(path) if _keep(m)]
    return sorted(out, key=lambda m: (m.get('order', 10**6), m['id']))


def resolve(family, identifier, path=DEFAULT_PATH):
    """Reverse lookup: any ids.* string (or an id) -> the registry entry.

    This is what makes cross-family joins possible at all. `kgdiff` (F1) and `KGDiff` (F4) are
    the same model; so are `vina_fixed`, `vina_fixed_best` and `vina_fixed_on`.
    """
    for m in models(path):
        if m['id'] == identifier or identifier in (m.get('ids') or {}).values():
            return m
    raise KeyError(f'{identifier!r} matches no id or ids.* value in {path}')


# --------------------------------------------------------------------- legacy shims
def _legacy_rows(family, path):
    """Models a legacy registry contained: reportable (`ok`) only, in the legacy row order.

    Membership comes from `status == 'ok'`, NOT from the presence of legacy_order -- otherwise
    the check would be circular. That is exactly what makes `retired` load-bearing: novdw and
    pignet_fixed are both tier 'ablation', but novdw is reported and pignet_fixed was withdrawn
    on 2026-08-05, so only `status` can tell them apart.
    """
    rows = [m for m in models(path) if status(m, family) == 'ok']
    return sorted(rows, key=lambda m: ((m['families'][family] or {}).get('legacy_order', 10**6),
                                       m.get('order', 10**6), m['id']))


def label_for(m, family):
    """Display label for one family.

    `label` is the canonical name; `families.<f>.legacy_label` pins the string a table is
    currently published with, so a roster migration can be verified as row-changes-only before
    any rename lands. Dropping the legacy_label is what adopts the canonical name.
    """
    return (m.get('families', {}).get(family) or {}).get('legacy_label', m['label'])


_label = label_for            # internal alias used by the legacy shims


def group_for(m, family):
    """Group code for one family. `families.<f>.legacy_group` pins a family's own vocabulary.

    The families genuinely disagree: F3 uses ref/ours/abl/ext/ng while the pose registry uses
    reference/ours/ablation/baseline. Neither is wrong, so the registry keeps the canonical
    `group` and lets a family declare its own spelling rather than forcing a translation table
    into every builder.
    """
    return (m.get('families', {}).get(family) or {}).get('legacy_group', m.get('group'))


def labels_by(family, key='eval_out', path=DEFAULT_PATH):
    """{ids.<key>: label} for one family -- replaces the private LABEL dicts the builders kept.

    Three builders each carried their own copy keyed by eval_out tag, and they disagreed:
    build_rmsd_summary said "Ours (Vina loss)", build_redock_comparison said "Ours(vina_fixed)",
    for the same model in the same family. One lookup, one answer.
    """
    out = {}
    for m in models(path):
        k = (m.get('ids') or {}).get(key)
        if k:
            out[k] = label_for(m, family)
    return out


def in_table(m, family, table):
    """Is this model rendered in a named sub-table of `family`?

    A family can own more than one table over overlapping rosters -- F3 has nci_summary (absolute
    contact counts, 13 rows) and the IFP appendix (19 rows), sharing only 7. A single `section`
    string cannot express membership in both, so `families.<f>.tables` is a list.
    """
    tables = (m.get('families', {}).get(family) or {}).get('tables')
    return bool(tables) and table in tables


def as_pose_fidelity_registry(path=DEFAULT_PATH):
    """Reproduce configs/pose_fidelity_models.json, key-for-key."""
    reg = load(path)
    out = []
    for m in _legacy_rows('pose', path):
        if m['id'] == 'reference':
            continue                    # reference enters F2 as the ceiling row, not as a model
        entry = {
            'tag': m['ids']['eval_out'],
            'label': _label(m, 'pose'),
            'dir': f"eval_out/{m['ids']['eval_out']}",
            'pt': source_for(m, 'pose'),
            'group': m.get('group'),
            'n_mols': (m.get('counts', {}).get('pose') or {}).get('n_mols'),
            'n_pockets': (m.get('counts', {}).get('pose') or {}).get('n_pockets'),
        }
        note = (m['families']['pose'] or {}).get('legacy_note')
        if note:
            entry['note'] = note
        out.append(entry)
    return {'_comment': reg['_legacy_comments']['pose'],
            'models': out,
            'protocol': reg['families']['pose']['protocol']}


def as_delta_score_registry(path=DEFAULT_PATH):
    """Reproduce configs/delta_score_models.json, key-for-key."""
    reg = load(path)
    out = []
    for m in _legacy_rows('delta', path):
        entry = {
            'tag': m['ids']['delta_tag'],
            'label': _label(m, 'delta'),
            'pt': source_for(m, 'delta'),
        }
        # mol_pt/join used to be forwarded here for ours_noguide. Retired 2026-08-25: its two
        # files were merged into one canonical .pt, so no model declares them any more.
        entry['layout'] = m['layout']
        entry['group'] = m.get('group')
        counts = m.get('counts', {}).get('delta') or {}
        entry['n_mols'] = counts.get('n_mols')
        entry['n_pockets'] = counts.get('n_pockets')
        entry['rationale'] = m.get('rationale')
        out.append(entry)
    fam = reg['families']['delta']
    return {'_comment': reg['_legacy_comments']['delta'],
            'n_per_pocket': fam['n_per_pocket'],
            'seed': fam['seed'],
            'models': out}


SHIMS = {'pose': as_pose_fidelity_registry, 'delta': as_delta_score_registry}


def load_as(path, family):
    """Drop-in for `json.load(open(<legacy registry>))`.

    Accepts either the unified registry or a legacy path; a legacy path is honoured as-is so a
    half-migrated tree still works.
    """
    if os.path.basename(path) in {os.path.basename(p) for p in LEGACY}:
        with open(path, encoding='utf-8') as fh:
            return json.load(fh)
    if family not in SHIMS:
        raise ValueError(f'no legacy shim for family {family!r}')
    return SHIMS[family](path)


if __name__ == '__main__':
    import sys
    reg = load()
    print(f'{len(reg["models"])} models in {DEFAULT_PATH}')
    for fam in FAMILIES:
        rows = for_family(fam)
        core = [m['id'] for m in rows if m['tier'] == 'core']
        print(f'  {fam:8s} {len(rows):2d} reportable rows  (core: {len(core)})')
    if len(sys.argv) > 1 and sys.argv[1] == '--core':
        print('\ncore roster:')
        for m in models():
            if m['tier'] == 'core':
                miss = [f for f in FAMILIES if status(m, f) != 'ok']
                print(f"  {m['id']:16s} {m['label']:26s} gaps: {miss or 'none'}")
