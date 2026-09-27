#!/usr/bin/env python3
"""Control B: sampling must be bitwise identical between the reference and pruned paths.

Runs the two ENTRY POINTS as subprocesses and compares the result files they write, rather
than reimplementing the sampling setup here. That way the seeding order, the atom-count
prior draw and the reverse loop are all exercised as shipped -- a reimplementation would be
one more thing to get wrong, and getting the seeding order wrong is exactly the failure this
control exists to catch.

Requires OMP_NUM_THREADS=1. At default GPU threading two identical runs differ by 10-19 atom
types; at OMP=8 by 6-7. Without it this test cannot distinguish a real difference from
thread nondeterminism.
"""
from __future__ import annotations

import argparse
import os
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
    p.add_argument("--data-ids", type=int, nargs="+", default=[0, 1])
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--keep", action="store_true", help="keep the scratch output")
    return p.parse_args()


def run(cmd, cwd, env):
    r = subprocess.run(cmd, cwd=str(cwd), env=env,
                       capture_output=True, text=True, timeout=7200)
    if r.returncode != 0:
        print(f"    command failed ({r.returncode}): {' '.join(cmd[:4])} ...")
        tail = (r.stderr or r.stdout).strip().splitlines()[-12:]
        for line in tail:
            print(f"      {line}")
    return r.returncode == 0


def load_pt(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main() -> int:
    args = parse_args()
    ref_repo = Path(args.reference_repo).resolve()

    print("=" * 78)
    print("Control B: sampling equivalence, reference entry point vs release entry point")
    print("=" * 78)
    omp = os.environ.get("OMP_NUM_THREADS")
    print(f"OMP_NUM_THREADS={omp}  seed={args.seed}  n={args.num_samples}  batch={args.batch_size}")
    if torch.cuda.is_available():
        print(f"gpu: {torch.cuda.get_device_name()}")
    if omp != "1":
        print("\nABORT: OMP_NUM_THREADS must be 1.")
        return 2
    print()

    env = dict(os.environ)
    tmp = Path(tempfile.mkdtemp(prefix="verifyB_"))
    ref_out, new_out = tmp / "ref", tmp / "new"
    failures = []

    for did in args.data_ids:
        print(f"  data_id={did}")

        ok_ref = run([
            sys.executable, "scripts/sample_diffusion2.py",
            "--ckpt", args.ckpt,
            "--data_path", args.data_path, "--split_path", args.split_path,
            "--data_id", str(did),
            "--guide_mode", "head1_only",
            "--head1_type_grad_weight", "100", "--head1_pos_grad_weight", "25",
            "--w_on", "1.0",
            "--num_samples", str(args.num_samples), "--batch_size", str(args.batch_size),
            "--result_path", str(ref_out),
        ], cwd=ref_repo, env=env)

        ok_new = run([
            sys.executable, "bin/sample.py",
            "--ckpt", args.ckpt,
            "--data_path", args.data_path, "--split_path", args.split_path,
            "--data_id", str(did),
            "--num_samples", str(args.num_samples), "--batch_size", str(args.batch_size),
            "--seed", str(args.seed),
            "--result_path", str(new_out),
        ], cwd=ROOT, env=env)

        if not (ok_ref and ok_new):
            failures.append(f"data_id={did}: a sampler exited non-zero")
            continue

        rd, nd = ref_out / f"id{did}", new_out / f"id{did}"
        rf = sorted(rd.glob("result_*.pt"))
        nf = sorted(nd.glob("result_*.pt"))
        if len(rf) != len(nf) or not rf:
            failures.append(f"data_id={did}: {len(rf)} reference files vs {len(nf)} release files")
            continue

        n_bad = 0
        for a, b in zip(rf, nf):
            ra, rb = load_pt(a), load_pt(b)
            pa, pb = torch.as_tensor(ra["pos"]), torch.as_tensor(rb["pos"])
            va, vb = torch.as_tensor(ra["v"]), torch.as_tensor(rb["v"])
            if pa.shape != pb.shape or not torch.equal(pa, pb) \
               or va.shape != vb.shape or not torch.equal(va, vb):
                n_bad += 1
                if n_bad == 1:
                    dp = (pa - pb).abs().max().item() if pa.shape == pb.shape else float("nan")
                    nv = int((va != vb).sum()) if va.shape == vb.shape else -1
                    print(f"    first mismatch in {a.name}: max|dpos|={dp:.3e}  atom types differing={nv}")
        if n_bad:
            failures.append(f"data_id={did}: {n_bad}/{len(rf)} molecules differ")
            print(f"    FAIL  {n_bad}/{len(rf)} molecules differ")
        else:
            print(f"    OK  {len(rf)} molecules bitwise identical")

    print()
    if not args.keep:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
    else:
        print(f"scratch kept at {tmp}")

    if failures:
        print("RESULT: FAIL")
        for f in failures:
            print(f"  {f}")
        return 1
    print("RESULT: PASS -- coordinates and atom types bitwise identical")
    return 0


if __name__ == "__main__":
    sys.exit(main())
