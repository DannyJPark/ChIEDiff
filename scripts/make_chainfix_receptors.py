#!/usr/bin/env python3
"""Chain-remapped copies of the test-set receptors, for re-measuring PoseCheck clashes and ProLIF.

    ~/anaconda3/envs/kgdiff/bin/python scripts/make_chainfix_receptors.py

WHY. PoseCheck 1.3.1 loads a receptor through Biopython's PDBParser, which keys atoms on
(chainID, resSeq+iCode, atom name, altLoc) and never reads segID. Twelve CrossDocked test receptors
carry their chain copies only in segID, so the copies collide and PoseCheck keeps one atom per key:
it measured those pockets against 1/2, 1/3 or 1/4 of the receptor
(results/pose_fidelity/heavy_clash/receptor_check/). PLIP had the same defect and was fixed by
scripts/plip_interactions.remap_chains, which rewrites column 22 from the distinct
(chainID, segID) pairs. This applies that very function, unchanged, to every receptor.

A receptor with a single (chainID, segID) pair is returned untouched by remap_chains; every file is
still rewritten here (ATOM/HETATM records + END) so all 100 receptors come from one directory and
one code path. The re-measurement then proves on the unaffected pockets that this rewrite changes
nothing PoseCheck sees.

Output:
  data/test_set_chainfix/<pocket>/<receptor>.pdb
  data/test_set_chainfix/manifest.json   per receptor: chain pairs, collisions before/after, sha256
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import eval_heavy_clash as EH                                       # noqa: E402  (model_dirs)
import plip_interactions as P                                       # noqa: E402  (remap_chains)

OUT = ROOT / "data" / "test_set_chainfix"


def biopython_key_stats(lines):
    """(heavy atoms, distinct (chainID, resSeq+iCode, name, altLoc) keys) -- Biopython keeps one
    atom per key, so keys < heavy atoms means atoms are dropped on load."""
    heavy = [l for l in lines if l[76:78].strip().upper() not in ("H", "D")]
    keys = {(l[21], l[22:27], l[12:16], l[16]) for l in heavy}
    return len(heavy), len(keys)


def main():
    receptors = sorted({r["receptor"] for d in EH.model_dirs()
                        for r in csv.DictReader(open(ROOT / "eval_out" / d / "manifest.csv"))})
    report, bad = {}, []
    for rec in receptors:
        src = ROOT / rec
        with open(src) as fh:
            orig = [l.rstrip("\n").ljust(80) for l in fh if l.startswith(("ATOM", "HETATM"))]
        remapped, n_pairs = P.load_receptor(str(src))
        assert len(remapped) == len(orig), rec
        # remap_chains may only touch column 22
        assert all(a[:21] == b[:21] and a[22:] == b[22:] for a, b in zip(orig, remapped)), rec
        h0, k0 = biopython_key_stats(orig)
        h1, k1 = biopython_key_stats(remapped)
        dest = OUT / Path(rec).relative_to("data/test_set")
        dest.parent.mkdir(parents=True, exist_ok=True)
        text = "\n".join(remapped) + "\nEND\n"
        dest.write_text(text)
        report[rec] = {
            "chainfix_path": str(dest.relative_to(ROOT)),
            "n_chain_segid_pairs": n_pairs,
            "chain_column_changed": any(a[21] != b[21] for a, b in zip(orig, remapped)),
            "heavy_atoms": h1, "biopython_keys_before": k0, "biopython_keys_after": k1,
            "collisions_before": h0 - k0, "collisions_after": h1 - k1,
            "sha256": hashlib.sha256(text.encode()).hexdigest()}
        if h1 != k1:
            bad.append(rec)
    (OUT / "manifest.json").write_text(json.dumps(
        {"source": "scripts/make_chainfix_receptors.py (plip_interactions.remap_chains)",
         "n_receptors": len(receptors), "receptors": report}, indent=1) + "\n")
    fixed = [r for r, v in report.items() if v["collisions_before"] > 0]
    relabeled = [r for r, v in report.items() if v["chain_column_changed"]]
    print(f"{len(receptors)} receptors written to {OUT.relative_to(ROOT)}")
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
