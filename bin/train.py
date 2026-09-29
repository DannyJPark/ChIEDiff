"""Training entry point for the chemistry-gated interaction-energy diffusion model.

Trains the diffusion denoiser, the atom-type head and the affinity head jointly, and adds
the chemistry-gated Vina-derived intermolecular energy evaluated at the predicted clean
coordinates. The energy weight is ramped inside the training loop (see the ramp block in
`train()`).

Usage:
    python bin/train.py --config configs/train.yml
    python bin/train.py --config configs/train.yml --tag myrun --wandb
"""

import os
import sys
import argparse
import shutil
import signal
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
import torch.utils.tensorboard
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from scipy import stats
from torch.nn.utils import clip_grad_norm_
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from tqdm.auto import tqdm

# Run from a clone without `pip install -e .`: put the repository root on sys.path so
# `gated_energy_diffusion` resolves. Harmless when the package IS installed.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from gated_energy_diffusion.utils import misc
from gated_energy_diffusion.utils import train as utils_train
from gated_energy_diffusion.utils import transforms as trans
from gated_energy_diffusion.utils.path_fix import fix_checkpoint_data_paths

from gated_energy_diffusion import models as models_pkg
from gated_energy_diffusion.datasets import get_dataset
from gated_energy_diffusion.datasets.pl_data import FOLLOW_BATCH
from gated_energy_diffusion.models.score_model import ScorePosNet3D

# Optional experiment tracking; the trainer runs without it.
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def get_auroc(y_true, y_pred, feat_mode, logger=None):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    avg_auroc = 0.
    possible_classes = set(y_true)
    for c in possible_classes:
        auroc = roc_auc_score(y_true == c, y_pred[:, c])
        avg_auroc += auroc * np.sum(y_true == c)
        mapping = {
            'basic': trans.MAP_INDEX_TO_ATOM_TYPE_ONLY,
            'add_aromatic': trans.MAP_INDEX_TO_ATOM_TYPE_AROMATIC,
            'full': trans.MAP_INDEX_TO_ATOM_TYPE_FULL
        }
        if logger:
            logger.info(f'atom: {mapping[feat_mode][c]} \t auc roc: {auroc:.4f}')
    return avg_auroc / len(y_true)


def get_pearsonr(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return stats.pearsonr(y_true, y_pred)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='Training configuration, e.g. configs/train.yml')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--logdir', type=str, default='./logs_diffusion')
    parser.add_argument('--ckpt', type=str, default='',
                        help='Load model WEIGHTS only and start a fresh run at iteration 0 '
                             '(fine-tuning / warm start). Optimizer, LR schedule and the energy ramp '
                             'all restart. For continuing an interrupted run use --resume instead.')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--train_report_iter', type=int, default=200)
    # Resume options (used by the chained training launcher to survive the wall-time cap)
    parser.add_argument('--resume', type=str, default='',
                        help='Continue an interrupted run: restores model + optimizer + scheduler + '
                             'iteration counter and keeps writing into the SAME log dir. Accepts either '
                             'a run directory (uses its checkpoints/last.pt, falling back to best.pt) '
                             'or a direct path to a .pt file.')
    parser.add_argument('--save_freq', type=int, default=0,
                        help='Write a rolling checkpoints/last.pt every N iterations regardless of '
                             'validation loss, so a job killed at wall-time resumes from where it died '
                             'rather than from the last best. 0 (default) means use train.val_freq.')
    # Wandb options
    parser.add_argument('--wandb', action='store_true', help='Enable wandb logging')
    parser.add_argument('--wandb_project', type=str, default='gated_energy_diffusion',
                        help='Wandb project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='Wandb entity (team/username)')
    parser.add_argument('--wandb_name', type=str, default=None, help='Wandb run name (default: auto-generated)')

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        exit()
    args = parser.parse_args()

    if args.resume and args.ckpt:
        raise SystemExit('--resume and --ckpt are mutually exclusive: --resume continues a run, '
                         '--ckpt starts a new one from borrowed weights.')

    # Load checkpoint or config.
    # A checkpoint carries the data root it was trained under, and older ones point at roots
    # this repository never had, so normalize to ./data/ before the config is used
    # (gated_energy_diffusion/utils/path_fix.py). The run logger does not exist yet at this
    # point, hence the print-backed stand-in.
    boot_logger = SimpleNamespace(info=print)
    ckpt = None
    resume_ckpt_path = ''
    resume_log_dir = ''
    if args.resume:
        if os.path.isdir(args.resume):
            resume_log_dir = os.path.abspath(args.resume.rstrip('/'))
            found = [p for p in (os.path.join(resume_log_dir, 'checkpoints', n)
                                 for n in ('last.pt', 'best.pt')) if os.path.exists(p)]
            if not found:
                raise SystemExit(f'--resume {args.resume}: no checkpoints/last.pt or best.pt found')
            resume_ckpt_path = found[0]
        else:
            resume_ckpt_path = os.path.abspath(args.resume)
            if not os.path.exists(resume_ckpt_path):
                raise SystemExit(f'--resume {args.resume}: no such checkpoint file')
            # <run_dir>/checkpoints/<name>.pt  ->  <run_dir>
            resume_log_dir = os.path.dirname(os.path.dirname(resume_ckpt_path))
        print(f'Resuming run {resume_log_dir} from {resume_ckpt_path}...')
        ckpt = torch.load(resume_ckpt_path, map_location=args.device)
        config = fix_checkpoint_data_paths(ckpt, logger=boot_logger)['config']
    elif args.ckpt:
        print(f'Loading checkpoint: {args.ckpt}...')
        ckpt = torch.load(args.ckpt, map_location=args.device)
        config = fix_checkpoint_data_paths(ckpt, logger=boot_logger)['config']
    else:
        config = misc.load_config(args.config)

    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    # Iteration to continue from. Offset the seed by it on resume: the training iterator is rebuilt
    # from scratch on every launch, so reusing the bare seed would replay the exact batch sequence
    # the run already consumed before it was killed.
    start_it = int(ckpt['iteration']) + 1 if (args.resume and ckpt.get('iteration') is not None) else 0
    misc.seed_all(config.train.seed + start_it)

    # Logging. A resumed run keeps writing into its original directory, so a chained multi-day
    # training leaves ONE run dir with one continuous log/tensorboard/checkpoint history instead of
    # a fresh timestamped directory per chain link.
    log_dir = resume_log_dir if args.resume else misc.get_new_log_dir(args.logdir, prefix=config_name, tag=args.tag)
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    vis_dir = os.path.join(log_dir, 'vis')
    os.makedirs(vis_dir, exist_ok=True)
    logger = misc.get_logger('train', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)

    logger.info(args)
    logger.info(config)

    # Only snapshot config/model source when the run is first created — on resume these already
    # exist, and copytree onto an existing directory would abort the job.
    if not args.resume:
        shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
        shutil.copytree(os.path.dirname(models_pkg.__file__), os.path.join(log_dir, 'models'))

    # Initialize wandb
    use_wandb = args.wandb and WANDB_AVAILABLE
    if args.wandb and not WANDB_AVAILABLE:
        logger.warning('wandb not installed. Install with: pip install wandb')

    if use_wandb:
        wandb_run_name = args.wandb_name or f"{config_name}_{args.tag}" if args.tag else config_name
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=wandb_run_name,
            config={
                'model': dict(config.model),
                'train': dict(config.train),
                'data': dict(config.data),
            },
            dir=log_dir,
            reinit=True
        )
        logger.info(f'Wandb initialized: {wandb.run.url}')

    # Transforms
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_featurizer = trans.FeaturizeLigandAtom(config.data.transform.ligand_atom_mode)
    transform_list = [
        protein_featurizer,
        ligand_featurizer,
        trans.FeaturizeLigandBond(),
        trans.NormalizeVina(config.data.name)
    ]
    # The gated energy loss gates its hydrophobic / H-bond terms by real atom typing
    # (donor-acceptor H-bonds; hydrophobic carbons only), which needs the per-atom XS flags built
    # from the ground-truth topology (gated_energy_diffusion/utils/vina_types.py). Unconditional:
    # the shipped configuration always uses the gated energy.
    transform_list.append(trans.FeaturizeVinaAtomTypes())

    if config.data.transform.random_rot:
        transform_list.append(trans.RandomRotation())
    transform = Compose(transform_list)

    # Datasets and loaders
    logger.info('Loading dataset...')
    dataset, subsets = get_dataset(
        config=config.data,
        transform=transform,
    )

    if config.data.name != 'pl':
        raise ValueError(
            f"Unsupported dataset: {config.data.name!r}. This release ships the CrossDocked2020 "
            f"'pl' dataset only.")
    # The shipped split has val=0 and test=100, so the trainer validates on the TEST subset. This
    # is deliberate and is what produced the released weights: keeping it means checkpoint
    # selection is NOT independent of the evaluation pockets.
    train_set, val_set, test_set = subsets['train'], subsets['test'], []

    logger.info(f'Training: {len(train_set)} Validation: {len(val_set)} Test: {len(test_set)}')

    collate_exclude_keys = ['ligand_nbh_list']
    train_iterator = utils_train.inf_iterator(DataLoader(
        train_set,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        follow_batch=FOLLOW_BATCH,
        exclude_keys=collate_exclude_keys
    ))
    val_loader = DataLoader(
        val_set, config.train.batch_size, shuffle=False,
        follow_batch=FOLLOW_BATCH, exclude_keys=collate_exclude_keys
    )

    # Model
    logger.info('Building model...')
    model = ScorePosNet3D(
        config.model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim
    ).to(args.device)

    logger.info(f'Protein feature dim: {protein_featurizer.feature_dim}')
    logger.info(f'Ligand feature dim: {ligand_featurizer.feature_dim}')
    logger.info(f'# trainable parameters: {misc.count_parameters(model) / 1e6:.4f} M')

    # Optimizer and scheduler
    optimizer = utils_train.get_optimizer(config.train.optimizer, model)
    scheduler = utils_train.get_scheduler(config.train.scheduler, optimizer)

    # NOTE: start_it was computed above (before seed_all, which is offset by it). Do not reset it here.
    if args.resume:
        missing = [k for k in ('optimizer', 'scheduler') if ckpt.get(k) is None]
        if missing:
            raise SystemExit(
                f'--resume {resume_ckpt_path}: checkpoint has no {"/".join(missing)} state, so the run '
                f'cannot be continued faithfully. Use --ckpt to start a new run from these weights.')
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        logger.info(f'[Resume] {resume_ckpt_path} -> model + optimizer + scheduler restored; '
                    f'continuing at iteration {start_it} (lr={optimizer.param_groups[0]["lr"]:.3e})')
    elif args.ckpt:
        model.load_state_dict(ckpt['model'])
        logger.info('Model weights loaded from checkpoint '
                    '(new run: optimizer, LR schedule and energy ramp all start from 0)')

    def train(it):
        model.train()
        optimizer.zero_grad()

        for _ in range(config.train.n_acc_batch):
            batch = next(train_iterator).to(args.device)

            # A missing FeaturizeVinaAtomTypes transform must fail hard here: without the per-atom
            # XS flags the energy term would silently fall back to the element-level proxy, which
            # is a different objective and not the published one.
            assert hasattr(batch, 'ligand_vina_xs') and batch.ligand_vina_xs is not None, (
                'batch has no ligand_vina_xs: the FeaturizeVinaAtomTypes transform is missing')
            assert hasattr(batch, 'protein_vina_xs') and batch.protein_vina_xs is not None, (
                'batch has no protein_vina_xs: the FeaturizeVinaAtomTypes transform is missing')

            # Gaussian noise on the pocket coordinates (train.pos_noise_std = 0.1 Angstrom), seen by
            # BOTH the network input and the energy branch. The quantity minimised is therefore the
            # pair potential convolved with a 0.1 A Gaussian in the surface distance d; validation
            # evaluates the same loss on the unperturbed pocket.
            protein_noise = torch.randn_like(batch.protein_pos) * config.train.pos_noise_std
            gt_protein_pos = batch.protein_pos + protein_noise

            results = model.get_diffusion_loss(
                protein_pos=gt_protein_pos,
                protein_v=batch.protein_atom_feature.float(),
                affinity=batch.affinity.float(),
                batch_protein=batch.protein_element_batch,
                ligand_pos=batch.ligand_pos,
                ligand_v=batch.ligand_atom_feature_full,
                batch_ligand=batch.ligand_element_batch,
                ligand_xs=batch.ligand_vina_xs,
                protein_xs=batch.protein_vina_xs
            )
            loss = results['loss']
            loss_pos = results['loss_pos']
            loss_v = results['loss_v']
            loss_exp = results['loss_exp']

            # === Gated intermolecular energy term with a curriculum ramp ===
            # total energy weight = loss_vdw_weight * min(it / vdw_ramp_iters, vdw_ramp_cap)
            # Production: loss_vdw_weight 0.05, vdw_ramp_iters 200000, vdw_ramp_cap 1.0 -> omega_m = 0.05 * min(it/200000, 1); applied here, not in the model, because the model's loss method has no access to the optimizer iteration.
            # The ramp is capped (default 1.0) so the physics term does not keep
            # growing past full strength and overwhelm the diffusion loss.
            loss_vdw = results.get('loss_vdw', None)
            vdw_weight = getattr(config.model, 'loss_vdw_weight', 0.)
            if loss_vdw is not None and vdw_weight > 0:
                ramp_iters = getattr(config.train, 'vdw_ramp_iters', 200000)
                ramp_cap = getattr(config.train, 'vdw_ramp_cap', 1.0)
                vdw_ramp = min(it / ramp_iters, ramp_cap)
                loss = loss + vdw_weight * vdw_ramp * loss_vdw

            loss = loss / config.train.n_acc_batch
            loss.backward()

        orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
        optimizer.step()

        if it % args.train_report_iter == 0:
            logger.info(
                '[Train] Iter %d | Loss %.6f (pos %.6f | v %.6f | exp %.6f) | Lr: %.6f | Grad: %.4f' % (
                    it, loss * config.train.n_acc_batch, loss_pos, loss_v, loss_exp,
                    optimizer.param_groups[0]['lr'], orig_grad_norm
                )
            )

            for k, v in results.items():
                if torch.is_tensor(v) and v.squeeze().ndim == 0:
                    writer.add_scalar(f'train/{k}', v, it)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
            writer.add_scalar('train/grad', orig_grad_norm, it)
            writer.flush()

            # Wandb logging
            if use_wandb:
                wandb_log = {
                    'train/loss': float(loss * config.train.n_acc_batch),
                    'train/loss_pos': float(loss_pos),
                    'train/loss_v': float(loss_v),
                    'train/loss_exp': float(loss_exp),
                    'train/lr': optimizer.param_groups[0]['lr'],
                    'train/grad_norm': float(orig_grad_norm),
                    'iteration': it,
                }
                # Log the raw energy, the ramp factor, and the effective (weighted) term
                # actually added to the total loss.
                if loss_vdw is not None:
                    ramp_iters = getattr(config.train, 'vdw_ramp_iters', 200000)
                    ramp_cap = getattr(config.train, 'vdw_ramp_cap', 1.0)
                    vdw_ramp = min(it / ramp_iters, ramp_cap)
                    wandb_log['train/loss_vdw'] = float(loss_vdw)
                    wandb_log['train/vdw_ramp'] = vdw_ramp
                    wandb_log['train/loss_vdw_weighted'] = float(vdw_weight * vdw_ramp * loss_vdw)
                wandb.log(wandb_log, step=it)

    def validate(it):
        sum_loss, sum_loss_pos, sum_loss_v, sum_loss_exp, sum_n = 0, 0, 0, 0, 0
        all_pred_v, all_true_v = [], []
        all_pred_exp, all_true_exp = [], []

        with torch.no_grad():
            model.eval()
            for batch in tqdm(val_loader, desc='Validate'):
                batch = batch.to(args.device)
                batch_size = batch.num_graphs

                for t in np.linspace(0, model.num_timesteps - 1, 10).astype(int):
                    time_step = torch.tensor([t] * batch_size).to(args.device)

                    results = model.get_diffusion_loss(
                        protein_pos=batch.protein_pos,
                        protein_v=batch.protein_atom_feature.float(),
                        affinity=batch.affinity.float(),
                        batch_protein=batch.protein_element_batch,
                        ligand_pos=batch.ligand_pos,
                        ligand_v=batch.ligand_atom_feature_full,
                        batch_ligand=batch.ligand_element_batch,
                        time_step=time_step,
                        ligand_xs=batch.ligand_vina_xs,
                        protein_xs=batch.protein_vina_xs
                    )
                    pred_exp = results['pred_exp']
                    all_pred_exp.append(pred_exp.detach().cpu().numpy())

                    loss = results['loss']
                    loss_pos = results['loss_pos']
                    loss_v = results['loss_v']
                    loss_exp = results['loss_exp']

                    sum_loss += float(loss) * batch_size
                    sum_loss_pos += float(loss_pos) * batch_size
                    sum_loss_v += float(loss_v) * batch_size
                    sum_loss_exp += float(loss_exp) * batch_size
                    sum_n += batch_size

                    all_pred_v.append(results['ligand_v_recon'].detach().cpu().numpy())
                    all_true_v.append(batch.ligand_atom_feature_full.detach().cpu().numpy())
                    all_true_exp.append(batch.affinity.float().detach().cpu().numpy())

        avg_loss = sum_loss / sum_n
        avg_loss_pos = sum_loss_pos / sum_n
        avg_loss_v = sum_loss_v / sum_n
        avg_loss_exp = sum_loss_exp / sum_n

        atom_auroc = get_auroc(
            np.concatenate(all_true_v),
            np.concatenate(all_pred_v, axis=0),
            feat_mode=config.data.transform.ligand_atom_mode,
            logger=logger
        )

        # Pearson correlation between predicted and reference affinity
        pred_exp_all = np.concatenate(all_pred_exp, axis=0)
        true_exp_all = np.concatenate(all_true_exp, axis=0)
        exp_pearsonr = get_pearsonr(true_exp_all, pred_exp_all)

        if config.train.scheduler.type == 'plateau':
            scheduler.step(avg_loss)
        elif config.train.scheduler.type == 'warmup_plateau':
            scheduler.step_ReduceLROnPlateau(avg_loss)
        else:
            scheduler.step()

        logger.info(
            '[Validate] Iter %05d | Loss %.6f | pos %.6f | v %.6f | exp %.6f | pcc %.4f | auroc %.4f' % (
                it, avg_loss, avg_loss_pos, avg_loss_v, avg_loss_exp,
                exp_pearsonr[0], atom_auroc
            )
        )

        writer.add_scalar('val/loss', avg_loss, it)
        writer.add_scalar('val/loss_pos', avg_loss_pos, it)
        writer.add_scalar('val/loss_v', avg_loss_v, it)
        writer.add_scalar('val/atom_auroc', atom_auroc, it)
        writer.add_scalar('val/loss_exp', avg_loss_exp, it)
        writer.add_scalar('val/pcc', exp_pearsonr[0], it)

        fig = sns.lmplot(
            data=pd.DataFrame({'pred': pred_exp_all, 'true': true_exp_all}),
            x='pred', y='true'
        ).set(title=f'Affinity PCC: {exp_pearsonr[0]:.4f}').figure
        writer.add_figure('val/pcc_fig', fig, it)

        # Wandb validation logging
        if use_wandb:
            wandb_val_log = {
                'val/loss': avg_loss,
                'val/loss_pos': avg_loss_pos,
                'val/loss_v': avg_loss_v,
                'val/atom_auroc': atom_auroc,
                'val/loss_exp': avg_loss_exp,
                'val/pcc': exp_pearsonr[0],
            }
            wandb.log({'val/pcc_fig': wandb.Image(fig)}, step=it)
            wandb.log(wandb_val_log, step=it)

        plt.close('all')
        writer.flush()
        return avg_loss

    save_freq = args.save_freq if args.save_freq > 0 else config.train.val_freq

    def save_last(it, best_loss, best_iter, reason):
        """Rolling checkpoint that --resume picks up. Written atomically: a SIGKILL landing
        mid-write would otherwise leave a truncated last.pt and break the next chain link."""
        last_path = os.path.join(ckpt_dir, 'last.pt')
        tmp_path = last_path + '.tmp'
        torch.save({
            'config': config,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'iteration': it,
            'val_loss': best_loss,
            'best_loss': best_loss,
            'best_iter': best_iter,
        }, tmp_path)
        os.replace(tmp_path, last_path)
        logger.info(f'[Checkpoint] last.pt <- iteration {it} ({reason})')

    # The scheduler this model was trained under caps wall time at 3 days, so a full run is a chain
    # of jobs and every link must save on its way out. Slurm sends SIGTERM at wall-time, then
    # SIGKILL after KillWait: flip a flag and checkpoint at the next clean iteration boundary
    # rather than saving inside the handler itself.
    stop_requested = {'v': False}

    def _on_term(signum, _frame):
        logger.info(f'[Signal] {signal.Signals(signum).name} received — will checkpoint and exit '
                    f'at the next iteration boundary')
        stop_requested['v'] = True

    signal.signal(signal.SIGTERM, _on_term)
    signal.signal(signal.SIGUSR1, _on_term)

    try:
        best_loss, best_iter = None, None
        best_ckpt_path = None
        if args.resume:
            # Carry the historical best across the resume. Without this, the first validation of
            # every chain link looks like a new best and overwrites best.pt with a worse model.
            best_loss = ckpt.get('best_loss', ckpt.get('val_loss'))
            best_iter = ckpt.get('best_iter', ckpt.get('iteration'))
            if best_loss is not None:
                logger.info(f'[Resume] carrying best val loss {best_loss:.6f} from iteration {best_iter}')

        for it in range(start_it, config.train.max_iters):
            if stop_requested['v']:
                save_last(it - 1, best_loss, best_iter, 'signal')
                logger.info(f'[Signal] exiting cleanly at iteration {it - 1}; resume with '
                            f'--resume {log_dir}')
                break

            train(it)

            if it > start_it and it % save_freq == 0:
                save_last(it, best_loss, best_iter, f'every {save_freq} iters')

            if it % config.train.val_freq == 0 or it == config.train.max_iters:
                val_loss = validate(it)

                if best_loss is None or val_loss < best_loss:
                    logger.info(f'[Validate] Best val loss achieved: {val_loss:.6f}')

                    # Every new best is kept on disk (previous bests are NOT deleted), so the whole
                    # improvement trajectory stays available for later analysis / re-sampling.
                    best_loss, best_iter = val_loss, it
                    ckpt_path = os.path.join(ckpt_dir, '%d.pt' % it)
                    best_ckpt_path = ckpt_path
                    torch.save({
                        'config': config,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'iteration': it,
                        'val_loss': val_loss,
                    }, ckpt_path)
                    # Stable pointer to the current best, so downstream scripts never have to guess.
                    shutil.copyfile(ckpt_path, os.path.join(ckpt_dir, 'best.pt'))
                    logger.info(f'[Checkpoint] Saved new best checkpoint: {ckpt_path} (also -> best.pt)')

                    # Log best model info to wandb
                    if use_wandb:
                        wandb.run.summary['best_val_loss'] = best_loss
                        wandb.run.summary['best_iter'] = best_iter
                else:
                    logger.info(f'[Validate] Val loss not improved. Best: {best_loss:.6f} at iter {best_iter}')

    except KeyboardInterrupt:
        logger.info('Training interrupted by user.')

    finally:
        # Save final model artifact to wandb
        if use_wandb:
            if best_ckpt_path and os.path.exists(best_ckpt_path):
                artifact = wandb.Artifact(
                    name=f'model-{wandb.run.id}',
                    type='model',
                    description=f'Best model at iter {best_iter}, val_loss={best_loss:.6f}'
                )
                artifact.add_file(best_ckpt_path)
                wandb.log_artifact(artifact)
                logger.info(f'Model artifact saved to wandb')

            wandb.finish()
            logger.info('Wandb run finished')


if __name__ == '__main__':
    main()
