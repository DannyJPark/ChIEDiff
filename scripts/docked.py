#!/usr/bin/env python3
"""One home for "which molecules succeeded through docking".

Four implementations of this question had drifted apart across the builders. They are collected
here as ONE primitive plus the projections each family actually needs. **Every body is a verbatim
transplant** -- the point is to stop the drift, not to change a number.

    has_affinity(vina, key)      the primitive        (was mark_docked.has_aff)
    resolve_key(data)            dock-or-minimize sniffing, hoisted out of two copies
    docked_records(mols)         F1  normalized records   (was build_comparison_tables.docked_only)
    docked_index(src_pt, ...)    F5  {name: (docked, heavy)} (was aggregate_posecheck.molecule_index)
    docked_keys(pt_path)         F3  (keep, heavy)        (was build_nci_summary.docked_keys)
    docked_names(model_dir)      F2  from docked_names.txt (was build_gnina_comparison.docked_set)

THE POLICIES ARE NOT INTERCHANGEABLE, which is why `policy` is an explicit argument and not a
default that quietly wins:

  'dock'              a molecule counts only if it has a Vina DOCK affinity.
                      aggregate_posecheck.molecule_index has always used this.
  'dock_or_minimize'  fall back to the MINIMIZE affinity when the .pt never ran the dock mode.
                      build_nci_summary and build_interaction_tables have always used this.

For the current roster the two agree, because every registered source .pt carries dock scores --
but that is a fact about today's data, not an invariant. `scripts/check_docked_equivalence.py`
measures it rather than assuming it. The plain `PIDiff_vina_docked.pt` is the counter-example
that shows the difference is real: score_only + minimize, zero dock scores. It is superseded by
`PIDiff_vina_docked_complete.pt` for exactly this reason.
"""
import os
import sys

POLICIES = ('dock', 'dock_or_minimize')


def _load_grouped(pt_path):
    """The shared loader. Its pocket-alignment fallback is load-bearing -- do not reimplement.

    PIDiff / MolCRAFT / PharDiff are flat .pt sets that only land on the right receptor because
    load_grouped keys on the FULL ligand_filename and falls back to the directory only when that
    directory is unique. Deriving the indexing any other way risks a silent off-by-one, which is
    precisely the bug that once paired 77 PIDiff pockets with the wrong protein.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from eval_export_sdf import load_grouped                                 # noqa: E402
    return load_grouped(pt_path)


def has_affinity(vina, key):
    """A valid AutoDock Vina result for that mode. Verbatim from mark_docked.has_aff."""
    e = (vina or {}).get(key)
    if isinstance(e, list) and e:
        e = e[0]
    return isinstance(e, dict) and e.get('affinity') is not None


def resolve_key(data, policy='dock_or_minimize'):
    """Which vina mode decides docking success for this .pt."""
    if policy not in POLICIES:
        raise ValueError(f'unknown policy {policy!r}; expected one of {POLICIES}')
    if policy == 'dock':
        return 'dock'
    return 'dock' if any(has_affinity(e.get('vina'), 'dock')
                         for mols in data for e in mols) else 'minimize'


# ------------------------------------------------------------------ F1: normalized records
def docked_records(mols):
    """Keep only molecules that succeeded through docking (have a Vina Dock score).

    Verbatim from build_comparison_tables.docked_only. Note the different INPUT SHAPE: these are
    already-normalized records carrying a flat `vina_dock`, not raw .pt entries carrying `vina`.

    Every model's numbers are reported over this set: the reference files (targetdiff / kgdiff /
    pidiff, and novdw) hold docking-successful molecules ONLY, so this is a no-op for them and
    n_mol is unchanged, while our runs also contain fragmented / docking-failed molecules, which
    this drops so n_mol and all metrics are the docking-successful count for every model alike. A
    molecule that docked is necessarily an intact, reconstructed molecule, so this is also the
    QED/SA denominator.
    """
    return [md for md in mols if md.get('vina_dock') is not None]


# ------------------------------------------------------------------ F5: name -> (docked, heavy)
def docked_index(src_pt, policy='dock'):
    """p<NNN>_m<MMMM> -> (docked?, heavy_atom_count) for a sampling/consolidated .pt.

    Verbatim from aggregate_posecheck.molecule_index, with the policy made explicit (it was
    hard-wired to 'dock'). The join key is the SDF title eval_export_sdf.py writes: pocket index
    plus the molecule's ORIGINAL position in that pocket's list, so it survives molecules being
    skipped.
    """
    if not src_pt or not os.path.isfile(src_pt):
        return {}
    data = _load_grouped(src_pt)
    key = resolve_key(data, policy)

    def docked(e):
        v = e.get('vina')
        if isinstance(v, dict) and v.get(key):
            entry = v[key]
            if isinstance(entry, list) and entry:
                entry = entry[0]
            return isinstance(entry, dict) and entry.get('affinity') is not None
        if isinstance(v, list) and v:
            return v[0].get('affinity') is not None
        return False

    out = {}
    for pi, pocket in enumerate(data):
        for mi, e in enumerate(pocket):
            mol = e.get('mol')
            if mol is None or mol.GetNumConformers() == 0:
                continue
            out[f'p{pi:03d}_m{mi:04d}'] = (docked(e), mol.GetNumHeavyAtoms())
    return out


# ------------------------------------------------------------------ F3: (keep, heavy)
def docked_keys(pt_path, policy='dock_or_minimize'):
    """Return (keep, heavy) for a sampling .pt. Verbatim from build_nci_summary.docked_keys.

    keep = set of docking-successful 'pNNN_mMMMM' names. heavy = per-molecule heavy-atom count
    for every molecule with a 3D conformer.

    A model whose .pt carries NO vina field at all is raw sampler output not yet docked in-house.
    For it `keep` is None, meaning "no docking filter available yet" -- downstream that model is
    reported on ALL measured molecules and FLAGGED, not silently dropped to zero and not compared
    on a different-looking but unlabelled population.
    """
    data = _load_grouped(pt_path)
    has_vina = any(e.get('vina') for mols in data for e in mols)
    key = resolve_key(data, policy)
    keep, heavy = (None if not has_vina else set()), {}
    for pi, mols in enumerate(data):
        for mi, e in enumerate(mols):
            mol = e.get('mol')
            if mol is None or mol.GetNumConformers() == 0:
                continue
            n = f'p{pi:03d}_m{mi:04d}'
            heavy[n] = mol.GetNumHeavyAtoms()
            if keep is not None and has_affinity(e.get('vina'), key):
                keep.add(n)
    return keep, heavy


# ------------------------------------------------------------------ F2: marker file
def docked_names(model_dir):
    """-> set of docking-successful molecule names, or None if the model has no marker file.

    Verbatim from build_gnina_comparison.docked_set. Written by scripts/mark_docked.py; the names
    are the SDF titles eval_export_sdf.py used, which are also the `name` column of
    per_molecule.csv and the `molecule` column of the PoseBusters CSVs, so one set filters every
    section.
    """
    p = os.path.join(model_dir, 'docked_names.txt')
    if not os.path.isfile(p):
        return None
    return {ln.strip() for ln in open(p) if ln.strip()}
