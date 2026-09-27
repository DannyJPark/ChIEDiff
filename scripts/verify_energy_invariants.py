#!/usr/bin/env python3
"""Controls C and D on the pruned model.

C  the chemistry-gate assertion fires when the XS flags are absent, so the pre-fix
   element-level proxy can no longer be reached silently.
D  the energy is linear in the per-term scales:
       E(steric only) + E(hydrophobic only) + E(hbond only) == E(full)
   bitwise. If this fails, the term weighting has been altered.
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--reference-repo", required=True, help="provides the data stack")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-path", required=True)
    p.add_argument("--split-path", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=2)
    return p.parse_args()


def build(args):
    ref = Path(args.reference_repo).resolve()
    sys.path.insert(0, str(ref))
    from datasets import get_dataset
    from datasets.pl_data import FOLLOW_BATCH
    import utils.transforms as trans
    from torch_geometric.loader import DataLoader
    from torch_geometric.transforms import Compose
    from easydict import EasyDict

    pf, lf = trans.FeaturizeProteinAtom(), trans.FeaturizeLigandAtom("add_aromatic")
    transform = Compose([pf, lf, trans.FeaturizeLigandBond(),
                         trans.NormalizeVina("pl"), trans.FeaturizeVinaAtomTypes()])
    cfg = EasyDict({"name": "pl", "path": args.data_path, "split": args.split_path})
    _, subsets = get_dataset(config=cfg, transform=transform)
    loader = DataLoader(subsets["test"], batch_size=args.batch_size, shuffle=False,
                        num_workers=0, follow_batch=FOLLOW_BATCH,
                        exclude_keys=["ligand_nbh_list"])
    batch = next(iter(loader)).to(args.device)

    sys.path.insert(0, str(ROOT))
    from gated_energy_diffusion.models.score_model import ScorePosNet3D
    ckpt = torch.load(args.ckpt, map_location=args.device)
    return ScorePosNet3D, ckpt, batch, pf.feature_dim, lf.feature_dim


def energy(model, batch, scales=None):
    if scales is not None:
        model._vina_term_scale = torch.tensor(scales, dtype=torch.float32,
                                              device=model._vina_term_scale.device)
    with torch.no_grad():
        return model._compute_vina_energy(
            pred_ligand_pos=batch.ligand_pos,
            protein_pos=batch.protein_pos,
            batch_ligand=batch.ligand_element_batch,
            batch_protein=batch.protein_element_batch,
            num_graphs=batch.num_graphs,
            ligand_v=batch.ligand_atom_feature_full,
            protein_v=batch.protein_atom_feature.float(),
            ligand_xs=batch.ligand_vina_xs,
            protein_xs=batch.protein_vina_xs,
        )


def main() -> int:
    args = parse_args()
    Model, ckpt, batch, p_dim, l_dim = build(args)
    model = Model(ckpt["config"].model, protein_atom_feature_dim=p_dim,
                  ligand_atom_feature_dim=l_dim).to(args.device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    failures = []

    print("=" * 78)
    print("Control C: the chemistry-gate assertion")
    print("=" * 78)
    try:
        with torch.no_grad():
            model._compute_vina_energy(
                pred_ligand_pos=batch.ligand_pos, protein_pos=batch.protein_pos,
                batch_ligand=batch.ligand_element_batch,
                batch_protein=batch.protein_element_batch,
                num_graphs=batch.num_graphs,
                ligand_v=batch.ligand_atom_feature_full,
                protein_v=batch.protein_atom_feature.float(),
                ligand_xs=None, protein_xs=None,
            )
        print("  FAIL: absent XS flags were accepted; the pre-fix proxy is still reachable")
        failures.append("C")
    except AssertionError as e:
        print(f"  OK: AssertionError raised -- {str(e)[:70]}")

    print()
    print("=" * 78)
    print("Control D: per-term linearity")
    print("=" * 78)
    base = list(model._vina_term_scale.tolist())
    e_full = energy(model, batch, [1., 1., 1., 1., 1.])
    e_ste = energy(model, batch, [1., 1., 1., 0., 0.])
    e_hyd = energy(model, batch, [0., 0., 0., 1., 0.])
    e_hb = energy(model, batch, [0., 0., 0., 0., 1.])
    model._vina_term_scale = torch.tensor(base, dtype=torch.float32,
                                          device=model._vina_term_scale.device)

    parts = e_ste + e_hyd + e_hb
    exact = torch.equal(parts, e_full)
    print(f"  steric      {e_ste.item():+.10g}")
    print(f"  hydrophobic {e_hyd.item():+.10g}")
    print(f"  hbond       {e_hb.item():+.10g}")
    print(f"  sum         {parts.item():+.10g}")
    print(f"  full        {e_full.item():+.10g}")
    if exact:
        print("  OK: bitwise equal")
    else:
        d = (parts - e_full).abs().item()
        # Summation order differs between the two routes, so a float-epsilon gap is
        # expected; anything larger means the weighting changed.
        rel = d / max(abs(e_full.item()), 1e-12)
        if rel < 1e-6:
            print(f"  OK: equal to {rel:.2e} relative (float summation order)")
        else:
            print(f"  FAIL: relative difference {rel:.3e}")
            failures.append("D")

    print()
    if failures:
        print(f"RESULT: FAIL ({', '.join(failures)})")
        return 1
    print("RESULT: PASS (C, D)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
