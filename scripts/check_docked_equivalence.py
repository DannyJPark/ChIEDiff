#!/usr/bin/env python3
"""Prove scripts/docked.py reproduces the four original implementations EXACTLY, on real data.

Nothing may be switched over to the shared module until this passes. A byte diff of the reports
would not be enough here: the four originals disagree with each other on purpose (see below), so
"the report did not change" could hide a projection that silently adopted the wrong policy.

Also audits the one real disagreement:
  aggregate_posecheck.molecule_index  -> 'dock' only
  build_nci_summary.docked_keys       -> 'dock', falling back to 'minimize'
For today's roster the two should agree, because every registered source .pt carries dock scores.
This measures that rather than assuming it -- and prints any model where the policies diverge.

    python scripts/check_docked_equivalence.py
    python scripts/check_docked_equivalence.py --limit 3     # quick pass over 3 models
"""
import argparse
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath('.'))
import docked                                                            # noqa: E402
import model_registry                                                    # noqa: E402


def originals():
    """Import the four original implementations, still in place during Phase A."""
    out = {}
    ap = importlib.import_module('aggregate_posecheck')
    out['molecule_index'] = ap.molecule_index
    nci = importlib.import_module('build_nci_summary')
    out['nci_docked_keys'] = nci.docked_keys
    it = importlib.import_module('build_interaction_tables')
    out['it_docked_keys'] = it.docked_keys
    gn = importlib.import_module('build_gnina_comparison')
    out['docked_set'] = gn.docked_set
    bct = importlib.import_module('build_comparison_tables')
    out['docked_only'] = bct.docked_only
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0, help='check only the first N models')
    a = ap.parse_args()

    orig = originals()
    failures, notes = [], []

    # ---- projection 1: docked_records vs build_comparison_tables.docked_only (pure function)
    sample = [{'vina_dock': -9.1}, {'vina_dock': None}, {}, {'vina_dock': 0.0}]
    if docked.docked_records(sample) != orig['docked_only'](sample):
        failures.append('docked_records != build_comparison_tables.docked_only')
    print(f'  docked_records        : {"OK" if not failures else "FAIL"}  (pure function)')

    # ---- projections 2-3: each is exercised only where its family actually uses it.
    # docked_keys is F3's; running it on an F1-only model would test a path production never
    # takes -- and would fail, because the two projections disagree on .pt shape (see the
    # shape-tolerance check below).
    models = [m for m in model_registry.models()
              if (m.get('sources') or {}).get('default')
              and m.get('docked_policy') == 'vina_dock'
              and model_registry.status(m, 'nci') in ('ok', 'retired')]
    if a.limit:
        models = models[:a.limit]

    print(f'\n  checking {len(models)} model .pt file(s)\n')
    hdr = f'  {"model":18s} {"n_mol":>6s} {"idx==orig":>10s} {"keys==orig":>11s} {"it==orig":>9s} {"dock vs +min":>13s}'
    print(hdr)
    print('  ' + '-' * (len(hdr) - 2))

    for m in models:
        src = model_registry.source_for(m, 'pose') or m['sources']['default']
        if not os.path.isfile(src):
            notes.append(f"{m['id']}: source not on disk, skipped ({src})")
            continue
        try:
            new_idx = docked.docked_index(src, policy='dock')
            old_idx = orig['molecule_index'](src)
            idx_ok = new_idx == old_idx

            new_keep, new_heavy = docked.docked_keys(src)
            old_keep, old_heavy = orig['nci_docked_keys'](src)
            keys_ok = (new_keep == old_keep) and (new_heavy == old_heavy)

            it_keep = orig['it_docked_keys'](src)
            it_ok = (new_keep or set()) == it_keep

            # the policy audit
            dock_only = {k for k, (d, _) in new_idx.items() if d}
            both = new_keep or set()
            same = dock_only == both
            delta = '' if same else f'  DIFFER +{len(both - dock_only)}/-{len(dock_only - both)}'

            print(f"  {m['id']:18s} {len(new_heavy):6d} {str(idx_ok):>10s} {str(keys_ok):>11s} "
                  f"{str(it_ok):>9s} {('same' if same else 'DIFFER'):>13s}{delta}")

            if not idx_ok:
                failures.append(f"{m['id']}: docked_index != aggregate_posecheck.molecule_index")
            if not keys_ok:
                failures.append(f"{m['id']}: docked_keys != build_nci_summary.docked_keys")
            if not it_ok:
                failures.append(f"{m['id']}: docked_keys != build_interaction_tables.docked_keys")
            if not same:
                notes.append(f"{m['id']}: 'dock' and 'dock_or_minimize' select DIFFERENT sets "
                             f"-- the two policies are not interchangeable for this model")
        except Exception as exc:                                  # noqa: BLE001
            failures.append(f"{m['id']}: raised {type(exc).__name__}: {exc}")

    # ---- projection 4: docked_names vs build_gnina_comparison.docked_set
    dirs = [f"eval_out/{(m.get('ids') or {}).get('eval_out')}" for m in model_registry.models()
            if (m.get('ids') or {}).get('eval_out')]
    n_checked = 0
    for d in dirs:
        if docked.docked_names(d) != orig['docked_set'](d):
            failures.append(f'docked_names != docked_set for {d}')
        n_checked += 1
    print(f'\n  docked_names          : {"OK" if not failures else "see below"}  '
          f'({n_checked} eval_out dirs)')

    # ---- shape tolerance: pin the pre-existing disagreement rather than paper over it.
    # Some .pt files store entry['vina'] as a LIST, not a dict (CVAE_test_docked_sf1.5.pt does).
    # molecule_index has an explicit isinstance(v, list) branch and copes; docked_keys does
    # `(vina or {}).get(key)` and raises AttributeError. That asymmetry predates this module.
    # It is latent only because the list-shaped models are F1-only and F3 never loads them.
    # Asserting BOTH still behave identically is what makes the transplant verbatim.
    shape_src = 'results/sampling_results/CVAE_test_docked_sf1.5.pt'
    if os.path.isfile(shape_src):
        def outcome(fn, *args):
            try:
                return ('ok', len(fn(*args)))
            except Exception as exc:                              # noqa: BLE001
                return ('raise', type(exc).__name__)
        pairs = [('docked_index', outcome(docked.docked_index, shape_src),
                  outcome(orig['molecule_index'], shape_src)),
                 ('docked_keys', outcome(docked.docked_keys, shape_src),
                  outcome(orig['nci_docked_keys'], shape_src))]
        print('\n  shape tolerance on a list-shaped `vina` field '
              '(CVAE_test_docked_sf1.5.pt):')
        for name, new, old in pairs:
            same = new == old
            print(f'    {name:14s} new={new}  original={old}  '
                  f'{"MATCH" if same else "DIVERGED"}')
            if not same:
                failures.append(f'{name}: behaviour on list-shaped vina differs from the original '
                                f'(new={new}, original={old})')
        notes.append('docked_keys raises on a list-shaped `vina` where docked_index copes. '
                     'Pre-existing, and harmless today because the only list-shaped sources '
                     '(cvae, ligan) are F1-only. Fix it before any such model enters F3.')

    if notes:
        print(f'\n{len(notes)} note(s):')
        for n in notes:
            print('  ' + n)
    if failures:
        print(f'\n{len(failures)} FAILURE(S):')
        for f in failures:
            print('  ' + f)
        return 1
    print('\nEQUIVALENT -- scripts/docked.py reproduces all four originals exactly.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
