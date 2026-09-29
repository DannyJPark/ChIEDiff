"""Sampling entry point: generate ligands for one pocket with affinity guidance.

Loads a training checkpoint, rebuilds the model from the config stored inside it, and runs
the reverse diffusion with single-head affinity guidance on both channels (coordinates and
atom types). Results are written as one `result_{i}.pt` per molecule under
`<result_path>/id{data_id}/`, plus a summary pickle.

Reproducibility notes:

  * The published sample sets are 100 molecules per pocket, produced by THREE chunked
    invocations per pocket -- 24, then 2, then 76 molecules -- all at `--seed 42`, each
    appending to the same pocket directory via `--start_index` (0, 24, 26). Each invocation
    re-seeds and then draws its own atom counts and initial noise, so a single
    `--num_samples 100` run at seed 42 yields a DIFFERENT set of 100 molecules. To reproduce
    the published set, reproduce the chunking.

  * Bitwise reproducibility additionally requires `OMP_NUM_THREADS=1`. At default GPU
    threading, two otherwise identical runs differ by 10-19 atom types; at
    `OMP_NUM_THREADS=8` they differ by 6-7. Only at 1 thread are they bitwise identical.

Usage:
    OMP_NUM_THREADS=1 python bin/sample.py --ckpt <checkpoint>.pt --data_id 0 \
        --num_samples 24 --start_index 0
"""

import argparse
import hashlib
import os
import sys
import time
import pickle
import gc

import numpy as np
import torch
from torch_geometric.data import Batch
from torch_geometric.transforms import Compose
from torch_scatter import scatter_mean
from tqdm.auto import tqdm

# Run from a clone without `pip install -e .`: put the repository root on sys.path so
# `gated_energy_diffusion` resolves. Harmless when the package IS installed.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from gated_energy_diffusion.utils import misc
from gated_energy_diffusion.utils import transforms as trans
from gated_energy_diffusion.datasets import get_dataset
from gated_energy_diffusion.datasets.pl_data import FOLLOW_BATCH
from gated_energy_diffusion.models.score_model import ScorePosNet3D, log_sample_categorical
from gated_energy_diffusion.utils.evaluation import atom_num


def sha256_file(path, chunk_size=1 << 20):
    """Hex sha256 of a file, read in chunks so a multi-GB checkpoint need not fit in RAM."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            h.update(chunk)
    return h.hexdigest()


def unbatch_v_traj(ligand_v_traj, n_data, ligand_cum_atoms):
    """Unbatch ligand atom type trajectory."""
    all_step_v = [[] for _ in range(n_data)]
    if not ligand_v_traj:
        return [np.array([]).reshape(0, 0) for _ in range(n_data)]

    for v in ligand_v_traj:
        v_array = v.detach().cpu().numpy()
        for k in range(n_data):
            all_step_v[k].append(v_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]])

    all_step_v = [np.stack(step_v) if step_v else np.array([]).reshape(0, 0) for step_v in all_step_v]
    return all_step_v


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--data_id', '-i', type=int, default=0, help='Test set data ID (pocket)')
    parser.add_argument('--data_path', type=str, default=None,
                       help='Override data path from checkpoint config (e.g., ./data/crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb)')
    parser.add_argument('--split_path', type=str, default=None,
                       help='Override split file path from checkpoint config')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42,
                        help='RNG seed. Seeded once, before the model is built; the published '
                             'samples used 42.')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_samples', type=int, default=4)
    parser.add_argument('--start_index', type=int, default=0,
                        help='Offset for result_{i}.pt numbering, to append to an existing pocket dir')
    parser.add_argument('--num_steps', type=int, default=None,
                        help='Diffusion steps; None uses the model\'s own number of timesteps (1000)')
    parser.add_argument('--sample_num_atoms', type=str, default='prior', choices=['prior', 'ref'])
    parser.add_argument('--result_path', type=str, default='./results/samples')
    parser.add_argument('--tag', type=str, default='')

    # Guidance options. Guidance is applied from the single affinity head to both channels.
    parser.add_argument('--head1_type_grad_weight', type=float, default=100.,
                       help='Atom-type guidance strength s_v')
    parser.add_argument('--head1_pos_grad_weight', type=float, default=25.,
                       help='Coordinate guidance strength s_x')
    parser.add_argument('--w_on', type=float, default=1.0,
                       help='Scale applied to both guidance channels')

    args = parser.parse_args()

    # Result directory. Downstream docking/export tools locate samples by this exact layout:
    # <result_path>/id{data_id}[_{tag}]/result_{i}.pt
    dir_name = f'id{args.data_id}'
    if args.tag:
        dir_name += f'_{args.tag}'
    result_path = os.path.join(args.result_path, dir_name)
    os.makedirs(result_path, exist_ok=True)

    logger = misc.get_logger('sampling', log_dir=result_path)
    logger.info(f'Arguments: {args}')

    # Load checkpoint
    logger.info(f'Loading checkpoint: {args.ckpt}')
    ckpt = torch.load(args.ckpt, map_location=args.device)
    ckpt_sha256 = sha256_file(args.ckpt)
    logger.info(f'Checkpoint sha256: {ckpt_sha256}')

    # Get config from checkpoint
    train_config = ckpt['config']
    logger.info(f'Training config: {train_config}')

    misc.seed_all(args.seed)

    # Setup transforms
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_atom_mode = train_config.data.transform.ligand_atom_mode
    ligand_featurizer = trans.FeaturizeLigandAtom(ligand_atom_mode)
    transform = Compose([
        protein_featurizer,
        ligand_featurizer,
        trans.FeaturizeLigandBond(),
    ])

    logger.info(f'Protein feature dim: {protein_featurizer.feature_dim}')
    logger.info(f'Ligand feature dim: {ligand_featurizer.feature_dim}')

    # Override data paths if provided (useful when checkpoint was trained with different paths)
    if args.data_path is not None:
        train_config.data.path = args.data_path
        logger.info(f'Overriding data path: {args.data_path}')
    if args.split_path is not None:
        train_config.data.split = args.split_path
        logger.info(f'Overriding split path: {args.split_path}')

    # Load dataset
    logger.info('Loading dataset...')
    dataset, subsets = get_dataset(
        config=train_config.data,
        transform=transform
    )

    test_set = subsets['test']

    logger.info(f'Test set size: {len(test_set)}')

    if args.data_id >= len(test_set):
        logger.error(f'data_id {args.data_id} out of range (test set has {len(test_set)} samples)')
        return

    # Get target data
    on_target_data = test_set[args.data_id]
    logger.info(f'data_id: {args.data_id}')
    if hasattr(on_target_data, 'protein_filename'):
        logger.info(f'Protein: {on_target_data.protein_filename}')

    # Build model
    logger.info('Building model...')
    model = ScorePosNet3D(
        train_config.model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim
    ).to(args.device)

    # Load weights
    model.load_state_dict(ckpt['model'])
    model.eval()
    logger.info(f'Model loaded from iteration {ckpt.get("iteration", "unknown")}')

    # Sampling parameters
    num_steps = args.num_steps
    center_pos_mode = getattr(train_config.model, 'center_pos_mode', 'protein')

    logger.info(f'Sampling with num_steps={num_steps}, center_pos_mode={center_pos_mode}')
    logger.info(f'Guidance weights: type={args.head1_type_grad_weight}, pos={args.head1_pos_grad_weight}')

    # Run sampling using model's method
    logger.info(f'Generating {args.num_samples} samples...')

    all_pred_pos, all_pred_v = [], []
    all_pred_exp_on = []
    all_pred_pos_traj, all_pred_v_traj = [], []
    all_pred_exp_on_traj = []
    time_list = []

    num_batch = int(np.ceil(args.num_samples / args.batch_size))

    for batch_idx in tqdm(range(num_batch), desc='Sampling batches'):
        n_data = args.batch_size if batch_idx < num_batch - 1 else args.num_samples - args.batch_size * (num_batch - 1)

        batch_on = Batch.from_data_list(
            [on_target_data.clone() for _ in range(n_data)],
            follow_batch=FOLLOW_BATCH
        ).to(args.device)

        t1 = time.time()

        # Memory optimization: Clear cache before processing
        if args.device == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()

        with torch.no_grad():
            batch_protein = batch_on.protein_element_batch

            # Determine number of atoms per sample
            if args.sample_num_atoms == 'prior':
                pocket_size = atom_num.get_space_size(batch_on.protein_pos.detach().cpu().numpy())
                ligand_num_atoms = [atom_num.sample_atom_num(pocket_size).astype(int) for _ in range(n_data)]
            elif args.sample_num_atoms == 'ref':
                batch_ligand_ref = batch_on.ligand_element_batch
                ligand_num_atoms = [(batch_ligand_ref == k).sum().item() for k in range(n_data)]
            else:
                raise ValueError(f"Unknown sample_num_atoms: {args.sample_num_atoms}")

            batch_ligand = torch.repeat_interleave(
                torch.arange(n_data, device=args.device),
                torch.tensor(ligand_num_atoms, device=args.device)
            )

            # Initialize ligand positions centered on protein
            center_pos_tensor = scatter_mean(batch_on.protein_pos, batch_protein, dim=0)
            batch_center_pos = center_pos_tensor[batch_ligand]
            init_ligand_pos = batch_center_pos + torch.randn_like(batch_center_pos)

            # Initialize ligand atom types (uniform distribution)
            uniform_logits = torch.zeros(len(batch_ligand), model.num_classes, device=args.device)
            init_ligand_v_prob = log_sample_categorical(uniform_logits)
            init_ligand_v = init_ligand_v_prob.argmax(dim=-1)

        # Call model's sampling method
        r = model.sample_diffusion_with_guidance(
            protein_pos=batch_on.protein_pos,
            protein_v=batch_on.protein_atom_feature.float(),
            batch_protein=batch_protein,
            init_ligand_pos=init_ligand_pos,
            init_ligand_v=init_ligand_v,
            batch_ligand=batch_ligand,
            num_steps=num_steps,
            center_pos_mode=center_pos_mode,
            guide_mode='head1_only',
            head1_type_grad_weight=args.head1_type_grad_weight,
            head1_pos_grad_weight=args.head1_pos_grad_weight,
            w_on=args.w_on,
        )

        t2 = time.time()
        time_list.append(t2 - t1)

        # Unbatch results
        ligand_cum_atoms = np.cumsum([0] + ligand_num_atoms)
        ligand_pos = r['pos'].detach().cpu().numpy()
        ligand_v = r['v'].detach().cpu().numpy()
        exp_on = r['exp_on'].detach().cpu() if r['exp_on'] is not None else torch.zeros(n_data)

        for k in range(n_data):
            all_pred_pos.append(ligand_pos[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]])
            all_pred_v.append(ligand_v[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]])
            all_pred_exp_on.append(exp_on[k].item())

        # Unbatch trajectories
        pos_traj = r['pos_traj']
        v_traj = r['v_traj']

        all_step_pos = [[] for _ in range(n_data)]
        for p in pos_traj:
            p_array = p.detach().cpu().numpy().astype(np.float64)
            for k in range(n_data):
                all_step_pos[k].append(p_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]])
        all_step_pos = [np.stack(step_pos) for step_pos in all_step_pos]
        all_pred_pos_traj += all_step_pos

        all_step_v = unbatch_v_traj(v_traj, n_data, ligand_cum_atoms)
        all_pred_v_traj += all_step_v

        if 'exp_on_traj' in r and len(r['exp_on_traj']) > 0:
            # Each element in exp_on_traj is [batch_size] tensor at that timestep
            exp_on_traj_batch = torch.stack([t.cpu() for t in r['exp_on_traj']], dim=0)  # [num_steps, batch_size]
            all_pred_exp_on_traj.append(exp_on_traj_batch)

        # Memory cleanup
        del batch_on, r
        if args.device == 'cuda':
            torch.cuda.empty_cache()

    # Unbatch exp trajectories for per-sample saving
    all_exp_on_traj_per_sample = []
    if all_pred_exp_on_traj:
        exp_on_traj_all = torch.cat(all_pred_exp_on_traj, dim=1).numpy()  # [num_steps, total_samples]
        for i in range(exp_on_traj_all.shape[1]):
            all_exp_on_traj_per_sample.append(exp_on_traj_all[:, i])

    # Save results (compatible with the docking/export tools). --start_index offsets the file
    # numbering so a second run can APPEND to an existing pocket dir (result_{start}..) instead of
    # overwriting result_0.. from scratch.
    for idx in range(len(all_pred_pos)):
        result_file = os.path.join(result_path, f'result_{idx + args.start_index}.pt')
        torch.save({
            'pos': all_pred_pos[idx],
            'v': all_pred_v[idx],
            'exp_on': all_pred_exp_on[idx],
            'pos_traj': all_pred_pos_traj[idx] if all_pred_pos_traj else None,
            'v_traj': all_pred_v_traj[idx] if all_pred_v_traj else None,
            'exp_on_traj': all_exp_on_traj_per_sample[idx] if all_exp_on_traj_per_sample else None,
            'seed': args.seed,
            'start_index': args.start_index,
            'ckpt_sha256': ckpt_sha256,
            'argv': list(sys.argv),
        }, result_file)

    # Save summary (suffix by start_index so an append run does not clobber the first run's summary)
    summary_name = 'samples_summary.pkl' if args.start_index == 0 else f'samples_summary_{args.start_index}.pkl'
    summary_file = os.path.join(result_path, summary_name)
    with open(summary_file, 'wb') as f:
        pickle.dump({
            'data_id': args.data_id,
            'num_samples': len(all_pred_pos),
            'exp_on': all_pred_exp_on,
            'time': time_list,
            'args': vars(args),
            'seed': args.seed,
            'start_index': args.start_index,
            'ckpt_sha256': ckpt_sha256,
            'argv': list(sys.argv),
        }, f)

    logger.info(f'Results saved to {result_path}')

    # Print summary
    logger.info('=== Sampling Summary ===')
    logger.info(f'Generated {len(all_pred_pos)} samples')
    logger.info(f'Average time per batch: {np.mean(time_list):.2f}s')

    exp_on_array = np.array(all_pred_exp_on)
    logger.info(f'Affinity - mean: {exp_on_array.mean():.4f}, std: {exp_on_array.std():.4f}')

    # ========================================================================
    # Affinity trajectory per timestep
    # ========================================================================
    if all_pred_exp_on_traj:
        try:
            logger.info('Generating affinity trajectory visualization...')
            from pathlib import Path
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            # [num_steps, num_samples]
            exp_on_traj = torch.cat(all_pred_exp_on_traj, dim=1).numpy()
            num_steps = exp_on_traj.shape[0]
            num_samples = exp_on_traj.shape[1]
            timesteps = np.arange(num_steps)

            vis_dir = Path(result_path) / 'visualizations'
            vis_dir.mkdir(exist_ok=True)

            max_vis_samples = min(num_samples, 10)
            if num_samples > max_vis_samples:
                logger.info(f'Limiting per-sample curves to first {max_vis_samples} (out of {num_samples})')

            fig, axes = plt.subplots(1, 2, figsize=(16, 5))

            # ---- Left: per-sample trajectories ----
            ax = axes[0]
            for i in range(max_vis_samples):
                ax.plot(timesteps, exp_on_traj[:, i], label=f'Sample {i+1}', alpha=0.7, linewidth=2)
            ax.set_xlabel('Iteration (Timestep 999->0)', fontsize=12)
            ax.set_ylabel('Normalized Affinity (Vina scale)', fontsize=12)
            ax.set_title('On-target BA Prediction During Sampling', fontsize=13, fontweight='bold')
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)

            # Secondary tick labels showing approximate Vina score (vina = -v * 16)
            y_lo, y_hi = ax.get_ylim()
            yticks = np.linspace(y_lo, y_hi, 6)
            ax.set_yticks(yticks)
            ax.set_yticklabels([f'{v:.3f} ({-v*16:.1f})' for v in yticks])

            # ---- Right: mean +/- std over all samples ----
            ax = axes[1]
            mean_on = exp_on_traj.mean(axis=1)
            std_on = exp_on_traj.std(axis=1)
            ax.plot(timesteps, mean_on, 'b-', linewidth=2.5, label='Mean')
            ax.fill_between(timesteps, mean_on - std_on, mean_on + std_on,
                            alpha=0.3, color='blue', label='+/- Std')
            ax.set_xlabel('Iteration (Timestep 999->0)', fontsize=12)
            ax.set_ylabel('Normalized Affinity (Vina scale)', fontsize=12)
            ax.set_title(f'On-target BA (Mean +/- Std, N={num_samples})', fontsize=13, fontweight='bold')
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            out_png = vis_dir / 'affinity_trajectory.png'
            plt.savefig(out_png, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f'Affinity trajectory saved to: {out_png}')
        except Exception as e:
            logger.warning(f'Failed to generate trajectory visualization: {e}')
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()
