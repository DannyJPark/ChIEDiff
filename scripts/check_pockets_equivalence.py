#!/usr/bin/env python3
"""Prove scripts/pockets.py agrees with all four legacy canonical-order code paths.

The plan called the three-notions-of-canonical-order problem a reconciliation. It is not: the
paths already agree, so this measures the agreement and turns it into a standing assertion.
Consolidating without this check would be exactly the assumption that produced the
pocket-renumbering bug in the first place.

    python scripts/check_pockets_equivalence.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath('.'))
import pockets                                                           # noqa: E402

CANON_PT = 'results/sampling_results/targetdiff_vina_docked.pt'


def main():
    failures = []
    print(f'  SSOT sha256           : {pockets.verify_sha256()[:16]}...  OK')
    ours = pockets.ligand_files()
    idx2full, full2idx, dir2idx, idx2name = pockets.canonical()

    # ---- 1/2: the two canonical_ligand_files copies, both keyed off targetdiff's .pt
    for modname in ('eval_export_sdf', 'plip_interactions'):
        try:
            mod = __import__(modname)
            got = mod.canonical_ligand_files()
        except Exception as exc:                                  # noqa: BLE001
            failures.append(f'{modname}.canonical_ligand_files raised {type(exc).__name__}: {exc}')
            print(f'  {modname:22s}: RAISED {type(exc).__name__}')
            continue
        same = got == ours
        n_diff = sum(1 for a, b in zip(got, ours) if a != b) + abs(len(got) - len(ours))
        print(f'  {modname:22s}: {"IDENTICAL" if same else f"{n_diff} DIFFER"}  ({len(got)} pockets)')
        if not same:
            failures.append(f'{modname}.canonical_ligand_files != pockets.ligand_files()')
            for i, (a, b) in enumerate(zip(got, ours)):
                if a != b:
                    print(f'      idx {i}: legacy={a!r}  ssot={b!r}')
                    break

    # ---- 3: build_comparison_tables' maps, derived from the .pt
    try:
        import build_comparison_tables as B
        obj = B._torch_load(CANON_PT) if hasattr(B, '_torch_load') else None
        if obj is None:
            from eval_export_sdf import _torch_load
            obj = _torch_load(CANON_PT)
        legacy_idx2full = B.canonical_index(obj)
        l_name, l_full, l_dir = B.build_pocket_maps(legacy_idx2full)
        checks = [('idx2full', legacy_idx2full, idx2full), ('full2idx', l_full, full2idx),
                  ('dir2idx', l_dir, dir2idx), ('idx2name', l_name, idx2name)]
        for label, legacy, new in checks:
            same = legacy == new
            print(f'  build_pocket_maps.{label:9s}: {"IDENTICAL" if same else "DIFFERS"} '
                  f'({len(legacy)} vs {len(new)} entries)')
            if not same:
                failures.append(f'build_pocket_maps.{label} != pockets.canonical()')
                extra = set(legacy) ^ set(new)
                if extra:
                    print(f'      key mismatch: {sorted(extra)[:5]}')
                else:
                    bad = [k for k in legacy if legacy[k] != new[k]][:3]
                    print(f'      value mismatch at {bad}')
    except Exception as exc:                                      # noqa: BLE001
        failures.append(f'build_comparison_tables path raised {type(exc).__name__}: {exc}')
        print(f'  build_pocket_maps      : RAISED {type(exc).__name__}: {exc}')

    # ---- 4: delta_score_common, which already reads the SSOT
    try:
        import delta_score_common as D
        _tp, _entries, d_idx2full, d_full2idx = D.load_test_pockets()
        for label, legacy, new in (('idx2full', d_idx2full, idx2full),
                                   ('full2idx', d_full2idx, full2idx)):
            same = legacy == new
            print(f'  delta_score_common.{label:8s}: {"IDENTICAL" if same else "DIFFERS"}')
            if not same:
                failures.append(f'delta_score_common.{label} != pockets.canonical()')
    except Exception as exc:                                      # noqa: BLE001
        failures.append(f'delta_score_common path raised {type(exc).__name__}: {exc}')
        print(f'  delta_score_common     : RAISED {type(exc).__name__}: {exc}')

    print(f'\n  {len(dir2idx)}/{len(idx2full)} pockets are addressable by directory; '
          f'{len(idx2full) - len(dir2idx)} need the full ligand_filename '
          f'({len(pockets.shared_dirs())} directories host two binding sites).')

    if failures:
        print(f'\n{len(failures)} FAILURE(S):')
        for f in failures:
            print('  ' + f)
        return 1
    print('\nEQUIVALENT -- every canonical-order code path agrees with the SSOT.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
