#!/usr/bin/env python3
"""Control B: sampling must be bitwise identical between the reference and pruned models.

Requires OMP_NUM_THREADS=1. At default GPU threading two identical runs differ by 10-19
atom types; at OMP=8 by 6-7. Only at one thread is this test meaningful.

Compares the head1_only path only, which is the one the release keeps.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--reference-repo", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-path", required=True)
    p.add_argument("--split-path", required=True)
    p.add_argument("--data-ids", type=int, nargs="+", default=[0, 1])
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--s-pos", type=float, default=25.0)
    p.add_argument("--s-type", type=float, default=100.0)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def setup(ref_repo: Path, args):
    sys.path.insert(0, str(ref_repo))
    from datasets import get_dataset
    from datasets.pl_data import FOLLOW_BATCH
    import utils.transforms as trans
    from torch_geometric.transforms import Compose
    from easydict import EasyDict

    pf, lf = trans.FeaturizeProteinAtom(), trans.FeaturizeLigandAtom("add_aromatic")
    transform = Compose([pf, lf, trans.FeaturizeLigandBond()])
    cfg = EasyDict({"name": "pl", "path": args.data_path, "split": args.split_path})
    _, subsets = get_dataset(config=cfg, transform=transform)
    return subsets["test"], pf.feature_dim, lf.feature_dim, FOLLOW_BATCH


def load(ref_repo: Path, ckpt_path, p_dim, l_dim, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"].model

    sys.path.insert(0, str(ref_repo))
    from models.molopt_score_model2 import ScorePosNet3D as Ref

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from gated_energy_diffusion.models.score_model import ScorePosNet3D as New

    out = []
    for M in (Ref, New):
        m = M(cfg, protein_atom_feature_dim=p_dim, ligand_atom_feature_dim=l_dim).to(device)
        m.load_state_dict(ckpt["model"], strict=False)
        m.eval()
        out.append(m)
    return out


def sample_one(model, data, args, follow_batch, device, is_ref: bool):
    """Reproduce the sampler's own sequence: seed, then atom count, then the reverse loop."""
    from torch_geometric.data import Batch
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from gated_energy_diffusion.utils import misc
    from gated_energy_diffusion.utils.evaluation import atom_num

    # Seeded before anything consumes randomness, matching the entry point's ordering.
    misc.seed_all(args.seed)

    n = args.batch_size
    batch = Batch.from_data_list([data.clone() for _ in range(n)],
                                 follow_batch=follow_batch).to(device)
    centre = batch.protein_pos.mean(dim=0)
    pocket_size = atom_num.get_space_size(
        data.protein_pos.detach().cpu().numpy() - data.protein_pos.mean(dim=0).numpy())
    counts = [atom_num.sample_atom_num(pocket_size).astype(int) for _ in range(n)]

    batch_ligand = torch.repeat_interleave(
        torch.arange(n, device=device),
        torch.tensor(counts, device=device)).long()
    total = int(sum(counts))
    init_pos = torch.randn(total, 3, device=device) + centre

    num_classes = model.num_classes
    uniform = torch.zeros(total, num_classes, device=device)
    from gated_energy_diffusion.models.score_model import log_sample_categorical as ls_new
    if is_ref:
        from models.molopt_score_model2 import log_sample_categorical as ls_ref
        init_v = ls_ref(uniform)
    else:
        init_v = ls_new(uniform)

    kwargs = dict(
        protein_pos=batch.protein_pos,
        protein_v=batch.protein_atom_feature.float(),
        batch_protein=batch.protein_element_batch,
        init_ligand_pos=init_pos,
        init_ligand_v=init_v,
        batch_ligand=batch_ligand,
        num_steps=args.num_steps,
        center_pos_mode="protein",
        guide_mode="head1_only",
        head1_type_grad_weight=args.s_type,
        head1_pos_grad_weight=args.s_pos,
        w_on=1.0,
    )
    if is_ref:
        kwargs.update(head2_type_grad_weight=0.0, head2_pos_grad_weight=0.0, w_off=1.0)

    with torch.no_grad():
        r = model.sample_diffusion_with_guidance(**kwargs)
    pos = r["pos"] if isinstance(r, dict) else r[0]
    v = r["v"] if isinstance(r, dict) else r[1]
    return pos, v


def main() -> int:
    import os
    args = parse_args()
    ref_repo = Path(args.reference_repo).resolve()

    omp = os.environ.get("OMP_NUM_THREADS")
    print("=" * 78)
    print("Control B: sampling equivalence, reference vs pruned")
    print("=" * 78)
    print(f"OMP_NUM_THREADS={omp}  seed={args.seed}  s_pos={args.s_pos}  s_type={args.s_type}")
    if torch.cuda.is_available():
        print(f"gpu: {torch.cuda.get_device_name()}")
    if omp != "1":
        print("\nABORT: OMP_NUM_THREADS must be 1 or this test cannot distinguish a real")
        print("difference from thread nondeterminism.")
        return 2
    print()

    test_set, p_dim, l_dim, follow_batch = setup(ref_repo, args)
    ref, new = load(ref_repo, args.ckpt, p_dim, l_dim, args.device)

    failures = []
    for did in args.data_ids:
        data = test_set[did]
        rp, rv = sample_one(ref, data, args, follow_batch, args.device, is_ref=True)
        np_, nv = sample_one(new, data, args, follow_batch, args.device, is_ref=False)

        pos_ok = rp.shape == np_.shape and torch.equal(rp, np_)
        v_ok = rv.shape == nv.shape and torch.equal(rv, nv)
        if pos_ok and v_ok:
            print(f"  data_id={did}: OK  ({rp.shape[0]} atoms, bitwise identical)")
        else:
            dp = (rp - np_).abs().max().item() if rp.shape == np_.shape else float("nan")
            nd = int((rv != nv).sum()) if rv.shape == nv.shape else -1
            print(f"  data_id={did}: FAIL  max|dpos|={dp:.3e}  atom types differing={nd}")
            failures.append(did)

    print()
    if failures:
        print(f"RESULT: FAIL on data_ids {failures}")
        return 1
    print("RESULT: PASS -- coordinates and atom types bitwise identical")
    return 0


if __name__ == "__main__":
    sys.exit(main())
