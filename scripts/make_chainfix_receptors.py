#!/usr/bin/env python3
"""Write chain-remapped copies of the test-set receptors, for PoseCheck and ProLIF.

PoseCheck 1.3.1 loads a receptor through Biopython's PDBParser, which keys atoms on
(chainID, resSeq+iCode, atom name, altLoc) and never reads segID. Twelve CrossDocked test
receptors carry their chain copies only in segID, so the copies collide and Biopython keeps
one atom per key, leaving a partial receptor. PLIP has the same defect, and
scripts/plip_interactions.remap_chains fixes it by rewriting column 22 from the distinct
(chainID, segID) pairs. This applies that function to every receptor.

A receptor with a single (chainID, segID) pair comes back untouched, but every file is still
rewritten, so all receptors come from one directory and one code path.
"""
import argparse
import hashlib
import shutil
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import plip_interactions as P                                       # noqa: E402  (remap_chains)


def biopython_key_stats(lines):
    """Returns (heavy atoms, distinct Biopython atom keys). Biopython keeps one atom per key, so
    fewer keys than heavy atoms means atoms are dropped on load."""
    heavy = [l for l in lines if l[76:78].strip().upper() not in ("H", "D")]
    keys = {(l[21], l[22:27], l[12:16], l[16]) for l in heavy}
    return len(heavy), len(keys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test_set', default='data/test_set')
    ap.add_argument('--out', default=None, help='default: <test_set>_chainfix')
    args = ap.parse_args()

    test_set = Path(args.test_set)
    OUT = Path(args.out) if args.out else test_set.parent / (test_set.name + '_chainfix')
    receptors = sorted(test_set.glob('*/*_rec.pdb'))
    if not receptors:
        sys.exit(f'no */*_rec.pdb under {test_set}')
    report, bad = {}, []
    for src in receptors:
        rec = str(src)
        with open(src) as fh:
            orig = [l.rstrip("\n").ljust(80) for l in fh if l.startswith(("ATOM", "HETATM"))]
        remapped, n_pairs = P.load_receptor(str(src))
        assert len(remapped) == len(orig), rec
        # remap_chains may only touch column 22.
        assert all(a[:21] == b[:21] and a[22:] == b[22:] for a, b in zip(orig, remapped)), rec
        h0, k0 = biopython_key_stats(orig)
        h1, k1 = biopython_key_stats(remapped)
        dest = OUT / src.relative_to(test_set)
        dest.parent.mkdir(parents=True, exist_ok=True)
        text = "\n".join(remapped) + "\nEND\n"
        dest.write_text(text)
        report[rec] = {
            "chainfix_path": str(dest),
            "n_chain_segid_pairs": n_pairs,
            "chain_column_changed": any(a[21] != b[21] for a, b in zip(orig, remapped)),
            "heavy_atoms": h1, "biopython_keys_before": k0, "biopython_keys_after": k1,
            "collisions_before": h0 - k0, "collisions_after": h1 - k1,
            "sha256": hashlib.sha256(text.encode()).hexdigest()}
        if h1 != k1:
            bad.append(rec)
    # Copy the other files each pocket needs, so this directory can be handed to the export step
    # as a whole. The PDBQT is taken unmodified, carrying atom types and coordinates with no
    # chain semantics, so the chain remapping does not apply to it.
    n_aux = 0
    for aux in sorted(list(test_set.glob('*/*.sdf')) + list(test_set.glob('*/*.pdbqt'))):
        dest = OUT / aux.relative_to(test_set)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(aux, dest)
        n_aux += 1

    (OUT / "manifest.json").write_text(json.dumps(
        {"source": "scripts/make_chainfix_receptors.py (plip_interactions.remap_chains)",
         "n_receptors": len(receptors), "receptors": report}, indent=1) + "\n")
    fixed = [r for r, v in report.items() if v["collisions_before"] > 0]
    relabeled = [r for r, v in report.items() if v["chain_column_changed"]]
    print(f"{len(receptors)} receptors and {n_aux} auxiliary files written to {OUT}")
    print(f"  had Biopython key collisions before: {len(fixed)}")
    for r in fixed:
        v = report[r]
        print(f"    {Path(r).parent.name:22s} collisions {v['collisions_before']:6d} -> "
              f"{v['collisions_after']}  ({v['n_chain_segid_pairs']} chain/segID pairs)")
    print(f"  chain column rewritten (any change): {len(relabeled)}")
    if bad:
        sys.exit(f"collisions remain after remapping: {bad}")
    print("  no receptor has a Biopython key collision after remapping")


if __name__ == "__main__":
    main()
