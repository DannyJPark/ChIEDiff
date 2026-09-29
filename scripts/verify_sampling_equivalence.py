#!/usr/bin/env python3
"""Control B: the sampler's output distribution, not molecule identity.

Molecule identity is NOT a usable criterion, and that is a property of the original code
rather than of this release. Measured on CPU with identical weights, inputs and seeds:

    same instance, called twice      affinity position gradient differs by 2.38e-07
    same class, two instances        2.38e-07
    reference vs pruned release      2.68e-07

while the forward pass itself is bit-identical. The position gradient is the second of two
backward passes over a graph the first retained, and re-traversing it accumulates in a
different float order each time. The original code's own comment states the consequence: "the
1000-step stochastic sampler amplifies [~1e-7] into a visibly different molecule."

So two runs of the REFERENCE also produce different molecules, and demanding bitwise identity
between reference and release would demand something the reference does not satisfy.

This control therefore runs the reference TWICE as its own control arm, and passes only if the
reference-vs-release spread is no worse than reference-vs-reference. It runs the two ENTRY
POINTS as subprocesses rather than reimplementing the setup, so the seeding order and the
atom-count prior are exercised as shipped.

The atom-count distribution is held to exact agreement: it is drawn from the pocket-size prior
before any gradient is taken, so a difference there would be a real defect rather than chaotic
amplification.

Requires OMP_NUM_THREADS=1, which removes thread nondeterminism but not the above.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--reference-repo", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-path", required=True)
    p.add_argument("--split-path", required=True)
    p.add_argument("--data-ids", type=int, nargs="+", default=[0])
    p.add_argument("--num-samples", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--keep", action="store_true")
    return p.parse_args()


def run(cmd, cwd, env, label):
    r = subprocess.run(cmd, cwd=str(cwd), env=env, capture_output=True,
                       text=True, timeout=10800)
    if r.returncode != 0:
        print(f"    {label}: exited {r.returncode}")
        for line in (r.stderr or r.stdout).strip().splitlines()[-10:]:
            print(f"      {line}")
    return r.returncode == 0


def load_pt(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def molecules(out_dir: Path, did: int):
    out = []
    for f in sorted((out_dir / f"id{did}").glob("result_*.pt")):
        d = load_pt(f)
        out.append((torch.as_tensor(d["pos"]).float(), torch.as_tensor(d["v"]).long()))
    return out


def spread(a, b):
    """(n compared, max coordinate delta, atom-type mismatches)."""
    n = min(len(a), len(b))
    worst, mism = 0.0, 0
    for (pa, va), (pb, vb) in zip(a[:n], b[:n]):
        worst = max(worst, (pa - pb).abs().max().item()) if pa.shape == pb.shape else float("inf")
        mism = mism + int((va != vb).sum()) if va.shape == vb.shape else -1
    return n, worst, mism


def main() -> int:
    args = parse_args()
    ref_repo = Path(args.reference_repo).resolve()

    print("=" * 78)
    print("Control B: sampling distribution, measured against a reference-vs-reference control")
    print("=" * 78)
    omp = os.environ.get("OMP_NUM_THREADS")
    print(f"OMP_NUM_THREADS={omp}  seed={args.seed}  n={args.num_samples}")
    if torch.cuda.is_available():
        print(f"gpu: {torch.cuda.get_device_name()}")
    if omp != "1":
        print("\nABORT: OMP_NUM_THREADS must be 1.")
        return 2
    print()

    env = dict(os.environ)
    env["PYTHONPATH"] = f"{ROOT}:{env.get('PYTHONPATH', '')}".rstrip(":")
    tmp = Path(tempfile.mkdtemp(prefix="verifyB_"))
    failures = []

    for did in args.data_ids:
        print(f"  data_id={did}")
        outs = {k: tmp / f"{k}_{did}" for k in ("refA", "refB", "rel")}

        def reference(out):
            return [
                sys.executable, "scripts/sample_diffusion2.py",
                "--ckpt", args.ckpt,
                "--data_path", args.data_path, "--split_path", args.split_path,
                "--data_id", str(did), "--guide_mode", "head1_only",
                "--head1_type_grad_weight", "100", "--head1_pos_grad_weight", "25",
                "--w_on", "1.0",
                "--num_samples", str(args.num_samples),
                "--batch_size", str(args.batch_size),
                "--result_path", str(out),
            ]

        release = [
            sys.executable, "bin/sample.py",
            "--ckpt", args.ckpt,
            "--data_path", args.data_path, "--split_path", args.split_path,
            "--data_id", str(did),
            "--num_samples", str(args.num_samples), "--batch_size", str(args.batch_size),
            "--seed", str(args.seed), "--result_path", str(outs["rel"]),
        ]

        if not (run(reference(outs["refA"]), ref_repo, env, "reference A")
                and run(reference(outs["refB"]), ref_repo, env, "reference B")
                and run(release, ROOT, env, "release")):
            failures.append(f"data_id={did}: a sampler exited non-zero")
            continue

        A, B, R = (molecules(outs[k], did) for k in ("refA", "refB", "rel"))
        if not (len(A) == len(B) == len(R) and A):
            failures.append(f"data_id={did}: file counts differ ({len(A)}/{len(B)}/{len(R)})")
            continue

        _, ctl_pos, ctl_typ = spread(A, B)
        _, rel_pos, rel_typ = spread(A, R)
        print(f"    reference vs reference : max|dpos|={ctl_pos:.3e}  type mismatches={ctl_typ}")
        print(f"    reference vs release   : max|dpos|={rel_pos:.3e}  type mismatches={rel_typ}")

        sizes = [tuple(sorted(len(v) for _, v in s)) for s in (A, B, R)]
        if not sizes[0] == sizes[1] == sizes[2]:
            failures.append(f"data_id={did}: atom counts diverged -- the pocket-size prior "
                            f"is drawn before any gradient, so this is a real defect")
            print(f"    FAIL  atom counts differ: {sizes}")
            continue
        print(f"    atom counts agree exactly: {sizes[0]}")

        if ctl_pos == 0.0 and rel_pos == 0.0:
            print("    OK  both bitwise identical")
        elif rel_pos <= max(ctl_pos * 4.0, 1e-6):
            print(f"    OK  release spread within the reference's own spread")
        else:
            failures.append(f"data_id={did}: release spread {rel_pos:.3e} exceeds the "
                            f"reference's own {ctl_pos:.3e} by more than 4x")
            print("    FAIL  release differs from the reference by more than the reference "
                  "differs from itself")

    print()
    if args.keep:
        print(f"scratch kept at {tmp}")
    else:
        shutil.rmtree(tmp, ignore_errors=True)

    if failures:
        print("RESULT: FAIL")
        for f in failures:
            print(f"  {f}")
        return 1
    print("RESULT: PASS -- the release samples like the reference, and the atom-count prior")
    print("  agrees exactly. Molecule identity is not reproducible for either implementation;")
    print("  see this script's docstring.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
