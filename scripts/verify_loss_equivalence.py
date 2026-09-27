#!/usr/bin/env python3
"""Control A: the pruned model's loss and gradients must match the original bitwise.

No randomness inside the comparison: one fixed batch, an explicit time_step (so
sample_time() is never called), and a pre-drawn pocket-noise tensor, all replayed into both
models. Otherwise the two would consume the RNG stream differently and a difference would
be uninterpretable.

Tolerance is exact equality. The deleted branches are unreachable under this configuration
and the removed initialisers are conditionally called, so neither model consumes
initialisation randomness the other does not. A nonzero difference is a real change --
do not widen the tolerance.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--reference-repo", required=True,
                   help="the original working repository, providing models/molopt_score_model2.py")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-path", required=True)
    p.add_argument("--split-path", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--timesteps", type=int, nargs="+",
                   default=[1, 100, 300, 500, 700, 900, 999],
                   help="sweep so a timestep-dependent branch cannot hide")
    p.add_argument("--pos-noise-std", type=float, default=0.1)
    return p.parse_args()


def build_batch(ref_repo: Path, data_path: str, split_path: str, batch_size: int, device: str):
    """Build one minibatch using the ORIGINAL repository's data stack.

    Using the original loader on purpose: if the release's dataset code had drifted, feeding
    both models from the original removes that as a confound and isolates the model itself.
    The release's own loader is exercised separately by Control A-prime.
    """
    sys.path.insert(0, str(ref_repo))
    from datasets import get_dataset                      # noqa: E402
    from datasets.pl_data import FOLLOW_BATCH             # noqa: E402
    import utils.transforms as trans                      # noqa: E402
    from torch_geometric.loader import DataLoader         # noqa: E402
    from torch_geometric.transforms import Compose        # noqa: E402
    from easydict import EasyDict                         # noqa: E402

    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_featurizer = trans.FeaturizeLigandAtom("add_aromatic")
    transform = Compose([
        protein_featurizer,
        ligand_featurizer,
        trans.FeaturizeLigandBond(),
        trans.NormalizeVina("pl"),
        trans.FeaturizeVinaAtomTypes(),
    ])
    cfg = EasyDict({"name": "pl", "path": data_path, "split": split_path})
    _, subsets = get_dataset(config=cfg, transform=transform)
    loader = DataLoader(subsets["test"], batch_size=batch_size, shuffle=False,
                        num_workers=0, follow_batch=FOLLOW_BATCH,
                        exclude_keys=["ligand_nbh_list"])
    batch = next(iter(loader)).to(device)
    return batch, protein_featurizer.feature_dim, ligand_featurizer.feature_dim


def load_models(ref_repo: Path, ckpt_path: str, p_dim: int, l_dim: int, device: str):
    """Instantiate the reference and the pruned model from the same checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device)
    model_cfg = ckpt["config"].model

    # Reference. Its imports are bare ('from utils...', 'from models...') and resolve into
    # the original repository, which is already first on sys.path.
    sys.path.insert(0, str(ref_repo))
    from models.molopt_score_model2 import ScorePosNet3D as RefModel   # noqa: E402

    # Release. Its imports are all prefixed 'gated_energy_diffusion.', so there is no
    # collision with the reference's bare module names.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from gated_energy_diffusion.models.score_model import ScorePosNet3D as NewModel  # noqa: E402

    ref = RefModel(model_cfg, protein_atom_feature_dim=p_dim, ligand_atom_feature_dim=l_dim).to(device)
    new = NewModel(model_cfg, protein_atom_feature_dim=p_dim, ligand_atom_feature_dim=l_dim).to(device)

    r_missing, r_unexpected = ref.load_state_dict(ckpt["model"], strict=False)
    n_missing, n_unexpected = new.load_state_dict(ckpt["model"], strict=False)

    print("state_dict load:")
    print(f"  reference : {len(r_missing)} missing, {len(r_unexpected)} unexpected")
    print(f"  release   : {len(n_missing)} missing, {len(n_unexpected)} unexpected")
    if n_missing:
        print(f"  release missing keys: {sorted(n_missing)[:10]}")

    ref_keys, new_keys = set(ref.state_dict()), set(new.state_dict())
    only_ref = sorted(ref_keys - new_keys)
    only_new = sorted(new_keys - ref_keys)
    print(f"  parameters only in reference: {len(only_ref)}")
    if only_ref:
        # Expected: the dual head. Anything else means the prune removed live parameters.
        unexpected = [k for k in only_ref
                      if not any(m in k for m in ("head2", "pignet", "expert_pred_head2"))]
        print(f"    sample: {only_ref[:6]}")
        if unexpected:
            print(f"    UNEXPECTED (not head2/pignet): {unexpected}")
    print(f"  parameters only in release  : {len(only_new)}  {only_new[:6]}")

    ref.eval()
    new.eval()
    return ref, new, only_ref, only_new, n_missing


def compare_losses(ref, new, batch, t_value: int, protein_noise: torch.Tensor, device: str,
                   label: str = "ref vs release"):
    """One timestep. Returns [(what, max_abs_difference)] so the caller can compare the
    ref-vs-release spread against the same-model noise floor."""
    n_graphs = batch.num_graphs
    time_step = torch.full((n_graphs,), t_value, dtype=torch.long, device=device)
    gt_protein_pos = batch.protein_pos + protein_noise

    def run(model):
        # get_diffusion_loss draws the forward-process noise itself -- a Gaussian for the
        # coordinates and a categorical sample for the atom types. Pinning time_step is not
        # enough: whichever model runs second would otherwise see a different RNG state and
        # every term would differ, which is a measurement artifact, not a code difference.
        # Reseed immediately before each call so both consume the identical stream.
        torch.manual_seed(20260927 + t_value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(20260927 + t_value)
        model.zero_grad(set_to_none=True)
        results = model.get_diffusion_loss(
            protein_pos=gt_protein_pos,
            protein_v=batch.protein_atom_feature.float(),
            # .float() matches the training loop: batch.affinity is float64 and the
            # backward pass rejects a Double loss term.
            affinity=batch.affinity.float(),
            batch_protein=batch.protein_element_batch,
            ligand_pos=batch.ligand_pos,
            ligand_v=batch.ligand_atom_feature_full,
            batch_ligand=batch.ligand_element_batch,
            time_step=time_step,
            ligand_xs=batch.ligand_vina_xs,
            protein_xs=batch.protein_vina_xs,
        )
        loss = results["loss"]
        grads = torch.autograd.grad(
            loss, [p for _, p in sorted(model.named_parameters()) if p.requires_grad],
            allow_unused=True, retain_graph=False,
        )
        names = [n for n, p in sorted(model.named_parameters()) if p.requires_grad]
        return results, dict(zip(names, grads))

    r_res, r_grad = run(ref)
    n_res, n_grad = run(new)

    out: list[tuple[str, float]] = []

    for key in ("loss", "loss_pos", "loss_v", "loss_exp", "loss_vdw"):
        a, b = r_res.get(key), n_res.get(key)
        if a is None and b is None:
            continue
        if a is None or b is None:
            out.append((f"t={t_value} {key}: present in only one model", float("inf")))
            continue
        d = (a.detach() - b.detach()).abs().max().item()
        out.append((f"t={t_value} {key}", d))

    shared = sorted(set(r_grad) & set(n_grad))
    n_bad, worst = 0, (0.0, None)
    for name in shared:
        a, b = r_grad[name], n_grad[name]
        if a is None and b is None:
            continue
        if a is None or b is None:
            out.append((f"t={t_value} grad {name}: None in only one model", float("inf")))
            continue
        if not torch.equal(a, b):
            n_bad += 1
            d = (a - b).abs().max().item()
            if d > worst[0]:
                worst = (d, name)
    out.append((f"t={t_value} gradients ({n_bad}/{len(shared)} differ, worst at {worst[1]})",
                worst[0]))

    wl = max([v for _, v in out], default=0.0)
    vdw = r_res.get("loss_vdw")
    tail = f"  loss_vdw={vdw.item():.10g}" if vdw is not None else ""
    print(f"  {label:14s} t={t_value:4d}  max|diff|={wl:.3e}  "
          f"loss={r_res['loss'].item():.10g}{tail}")
    return out


def main() -> int:
    args = parse_args()
    ref_repo = Path(args.reference_repo).resolve()

    print("=" * 78)
    print("Control A: loss and gradient equivalence, reference vs pruned")
    print("=" * 78)
    print(f"reference repo : {ref_repo}")
    print(f"checkpoint     : {args.ckpt}")
    print(f"device         : {args.device}")
    if args.device.startswith("cuda") and torch.cuda.is_available():
        print(f"gpu            : {torch.cuda.get_device_name()}")
    print(f"torch          : {torch.__version__}")
    print()

    batch, p_dim, l_dim = build_batch(ref_repo, args.data_path, args.split_path,
                                      args.batch_size, args.device)
    print(f"batch: {batch.num_graphs} complexes, "
          f"{batch.ligand_pos.shape[0]} ligand atoms, {batch.protein_pos.shape[0]} pocket atoms")
    assert getattr(batch, "ligand_vina_xs", None) is not None, "XS flags missing from the batch"
    print()

    ref, new, only_ref, only_new, n_missing = load_models(
        ref_repo, args.ckpt, p_dim, l_dim, args.device)
    print()

    # The pocket perturbation the training loop applies, drawn ONCE and replayed.
    torch.manual_seed(2021)
    protein_noise = torch.randn_like(batch.protein_pos) * args.pos_noise_std

    # NOISE FLOOR. This architecture reduces with scatter_add / scatter_mean, which on CUDA
    # accumulate through atomics in nondeterministic order. Two runs of the SAME model
    # therefore disagree at float32 level, so "bitwise" is not a reachable criterion on GPU
    # and a raw ref-vs-release difference cannot be interpreted without knowing that floor.
    # Measure it first by comparing the reference against itself.
    print("noise floor: the reference model against itself (same inputs, same seed)")
    floor = compare_losses(ref, ref, batch, args.timesteps[0], protein_noise, args.device,
                           label="ref vs ref")
    floor_max = max([f[1] for f in floor], default=0.0)
    if floor_max == 0.0:
        print("  floor is exactly 0 -- this device/kernel set is deterministic, so "
              "bitwise equality is the correct criterion")
    else:
        print(f"  floor is {floor_max:.3e} -- nondeterministic reductions. ref-vs-release "
              f"must not exceed this")
    print()

    print(f"comparing at {len(args.timesteps)} timesteps (pos_noise_std={args.pos_noise_std}):")
    diffs: list[tuple[str, float]] = []
    for t in args.timesteps:
        diffs += compare_losses(ref, new, batch, t, protein_noise, args.device)

    worst = max([d[1] for d in diffs], default=0.0)
    failures: list[str] = []
    if floor_max == 0.0:
        failures = [f"{k}: {v:.6e}" for k, v in diffs if v != 0.0]
    elif worst > floor_max:
        failures = [f"{k}: {v:.6e} exceeds the {floor_max:.3e} floor"
                    for k, v in diffs if v > floor_max]
    print()
    print(f"worst ref-vs-release difference: {worst:.3e}   "
          f"same-model floor: {floor_max:.3e}")
    print()

    structural = []
    if n_missing:
        structural.append(f"release model has {len(n_missing)} missing state_dict keys")
    if only_new:
        structural.append(f"release model has {len(only_new)} parameters the reference lacks")
    bad_only_ref = [k for k in only_ref
                    if not any(m in k for m in ("head2", "pignet", "expert_pred_head2"))]
    if bad_only_ref:
        structural.append(f"reference-only parameters that are not head2/pignet: {bad_only_ref}")

    if failures or structural:
        print("RESULT: FAIL")
        for f in structural + failures[:20]:
            print(f"  {f}")
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more")
        print("\nThe difference exceeds what the same model produces against itself, so it is")
        print("a behavioural change rather than kernel nondeterminism. Do not widen the")
        print("criterion; find the pruned statement that mattered. Run with --device cpu for a")
        print("deterministic arm that isolates the code from the GPU reductions.")
        return 1

    if floor_max == 0.0:
        print("RESULT: PASS -- bitwise identical on a deterministic device")
    else:
        print(f"RESULT: PASS -- ref-vs-release ({worst:.3e}) is within the same-model floor "
              f"({floor_max:.3e})")
        print("  The floor is nonzero because this architecture reduces with scatter atomics,")
        print("  which are order-nondeterministic on CUDA: two runs of the SAME model differ by")
        print("  the same amount. Use --device cpu for a bitwise arm.")
    print(f"  timesteps checked: {args.timesteps}")
    print(f"  reference-only parameters: {len(only_ref)} (all dual-head/PIGNet, as expected)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
