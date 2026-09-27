"""Score network for pocket-conditioned 3D molecule diffusion with an affinity head.

The denoiser refines ligand coordinates and atom types inside a fixed protein pocket.
The ligand atom embeddings of the protein-ligand complex graph feed two outputs: the
atom-type inference used for generation, and a per-atom affinity head pooled over each
ligand. The affinity head supplies the sampling-time guidance gradients, and the
chemistry-gated intermolecular interaction energy is added to the training loss.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean
from tqdm.auto import tqdm

from gated_energy_diffusion.models.common import compose_context, ShiftedSoftplus
from gated_energy_diffusion.models.uni_transformer import UniTransformerO2TwoUpdateGeneral


def get_refine_net(refine_net_type, config):
    if refine_net_type == 'uni_o2':
        refine_net = UniTransformerO2TwoUpdateGeneral(
            num_blocks=config.num_blocks,
            num_layers=config.num_layers,
            hidden_dim=config.hidden_dim,
            n_heads=config.n_heads,
            k=config.knn,
            edge_feat_dim=config.edge_feat_dim,
            num_r_gaussian=config.num_r_gaussian,
            num_node_types=config.num_node_types,
            act_fn=config.act_fn,
            norm=config.norm,
            cutoff_mode=config.cutoff_mode,
            ew_net_type=config.ew_net_type,
            num_x2h=config.num_x2h,
            num_h2x=config.num_h2x,
            r_max=config.r_max,
            x2h_out_fc=config.x2h_out_fc,
            sync_twoup=config.sync_twoup
        )
    else:
        raise ValueError(refine_net_type)
    return refine_net


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":
        betas = 1.0 / np.linspace( ## 1/T, 1/(T-1), 1/(T-2), ..., 1
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])
    alphas = np.clip(alphas, a_min=0.001, a_max=1.)
    alphas = np.sqrt(alphas)
    return alphas


def get_distance(pos, edge_index):
    return (pos[edge_index[0]] - pos[edge_index[1]]).norm(dim=-1)


def to_torch_const(x):
    x = torch.from_numpy(x).float()
    x = nn.Parameter(x, requires_grad=False)
    return x


def center_pos(protein_pos, ligand_pos, batch_protein, batch_ligand, mode='protein'):
    if mode == 'none':
        offset = 0.
    elif mode == 'protein':
        offset = scatter_mean(protein_pos, batch_protein, dim=0)
        protein_pos = protein_pos - offset[batch_protein]
        ligand_pos = ligand_pos - offset[batch_ligand]
    else:
        raise NotImplementedError
    return protein_pos, ligand_pos, offset


# Categorical diffusion utilities
def index_to_log_onehot(x, num_classes):
    assert x.max().item() < num_classes, f'Error: {x.max().item()} >= {num_classes}'
    x_onehot = F.one_hot(x, num_classes)
    log_x = torch.log(x_onehot.float().clamp(min=1e-30))
    return log_x


def log_onehot_to_index(log_x):
    return log_x.argmax(1)


def categorical_kl(log_prob1, log_prob2):
    kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
    return kl


def log_categorical(log_x_start, log_prob):
    return (log_x_start.exp() * log_prob).sum(dim=1)


def normal_kl(mean1, logvar1, mean2, logvar2):
    kl = 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2) + (mean1 - mean2) ** 2 * torch.exp(-logvar2))
    return kl.sum(-1)


def log_normal(values, means, log_scales):
    var = torch.exp(log_scales * 2)
    log_prob = -((values - means) ** 2) / (2 * var) - log_scales - np.log(np.sqrt(2 * np.pi))
    return log_prob.sum(-1)


def log_sample_categorical(logits):
    uniform = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
    return gumbel_noise + logits


def log_1_min_a(a):
    return np.log(1 - np.exp(a) + 1e-40)


def log_add_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))


# Time embedding
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ScorePosNet3D(nn.Module):
    """Diffusion score network with a single binding-affinity head.

    The protein-ligand complex graph (WITH interaction) is refined by the equivariant
    transformer; the ligand atom embeddings are read out by `v_inference` for the atom
    types and by `expert_pred_head1` for a per-atom affinity that is pooled with
    scatter_mean over each ligand.
    """

    def __init__(self, config, protein_atom_feature_dim, ligand_atom_feature_dim):
        super().__init__()
        # Config guards. Written as raises rather than asserts on purpose: `python -O`
        # strips asserts, and the failure these prevent is silent -- a stale config from the
        # development tree would train a DIFFERENT model while reporting success.
        _mode = getattr(config, 'vdw_loss_mode', 'vina')
        if _mode != 'vina':
            raise ValueError(
                "the only interaction loss implemented here is the chemistry-gated Vina "
                f"energy; vdw_loss_mode={_mode!r} selects a loss that is not part of this "
                "release")
        if getattr(config, 'use_dual_head_sam_pl', False):
            raise ValueError(
                "use_dual_head_sam_pl=True requests the dual-head architecture, which is "
                "not part of this release. Without this guard the run would silently train "
                "a single head instead, producing a model that is not what the config asks "
                "for. Use configs/train.yml, which omits the key.")
        if getattr(config, 'vina_type_gating', True) is not True:
            raise ValueError(
                "vina_type_gating=False makes the hydrophobic and hydrogen-bond terms fire "
                "on every pair instead of switching them off. That ungated form is not the "
                "published model. To ablate a channel, set vina_hydro_scale or "
                "vina_hbond_scale to 0 instead.")
        self.config = config

        # === Diffusion schedule ===
        self.model_mean_type = config.model_mean_type # ['noise', 'C0']
        self.loss_v_weight = config.loss_v_weight
        self.loss_exp_weight = config.loss_exp_weight
        self.sample_time_method = config.sample_time_method # ['importance', 'symmetric']
        self.use_classifier_guide = config.use_classifier_guide

        # === Chemistry-gated intermolecular interaction energy ===
        # The raw energy is computed in get_diffusion_loss; the base weight and the curriculum
        # ramp (it / vdw_ramp_iters) are applied in the train loop.
        # Gate the hydrophobic / H-bond terms by atom type (the real AutoDock Vina rule). The
        # steric terms (gauss1/gauss2/repulsion) stay ungated. False -> apply both terms to all
        # pairs, which is not what the released model was trained with.
        self.vina_type_gating = getattr(config, 'vina_type_gating', True)
        # Per-TERM ablation switches. They scale the three term GROUPS before the weighted sum, so
        # a group set to 0.0 is removed exactly (its energy is identically zero, not merely small):
        #   steric -> gauss1 + gauss2 + repulsion   (the "vdW" group: the gaussians + the quadratic)
        #   hydro  -> hydrophobic                   hbond -> hbonding
        # 1.0 everywhere == the unmodified 5-term Vina energy, i.e. these keys are a no-op for a
        # config that does not set them (the released model reproduces bit-identically).
        self.vina_steric_scale = float(getattr(config, 'vina_steric_scale', 1.0))
        self.vina_hydro_scale = float(getattr(config, 'vina_hydro_scale', 1.0))
        self.vina_hbond_scale = float(getattr(config, 'vina_hbond_scale', 1.0))

        # Beta schedule
        if config.beta_schedule == 'cosine':
            alphas = cosine_beta_schedule(config.num_diffusion_timesteps, config.pos_beta_s) ** 2
            print('cosine pos alpha schedule applied!')
            betas = 1. - alphas
        else:
            betas = get_beta_schedule(
                beta_schedule=config.beta_schedule,
                beta_start=config.beta_start,
                beta_end=config.beta_end,
                num_diffusion_timesteps=config.num_diffusion_timesteps,
            )
            alphas = 1. - betas

        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        self.betas = to_torch_const(betas)
        self.num_timesteps = self.betas.size(0)
        self.alphas_cumprod = to_torch_const(alphas_cumprod)
        self.alphas_cumprod_prev = to_torch_const(alphas_cumprod_prev)
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = to_torch_const(np.sqrt(alphas_cumprod))
        self.sqrt_one_minus_alphas_cumprod = to_torch_const(np.sqrt(1. - alphas_cumprod))
        self.sqrt_recip_alphas_cumprod = to_torch_const(np.sqrt(1. / alphas_cumprod))
        self.sqrt_recipm1_alphas_cumprod = to_torch_const(np.sqrt(1. / alphas_cumprod - 1))

        # Computed tensor (will be moved to device when sqrt_* tensors are moved)
        self._pos_classifier_grad_weight = None  # Lazy initialization

         # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_mean_c0_coef = to_torch_const(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.posterior_mean_ct_coef = to_torch_const(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))
        # log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_var = to_torch_const(posterior_variance)
        self.posterior_logvar = to_torch_const(np.log(np.append(self.posterior_var[1], self.posterior_var[1:])))

        # Atom type diffusion schedule in log space
        if config.v_beta_schedule == 'cosine':
            alphas_v = cosine_beta_schedule(self.num_timesteps, config.v_beta_s)
            print('cosine v alpha schedule applied!')
        else:
            raise NotImplementedError
        log_alphas_v = np.log(alphas_v)
        log_alphas_cumprod_v = np.cumsum(log_alphas_v)
        self.log_alphas_v = to_torch_const(log_alphas_v)
        self.log_one_minus_alphas_v = to_torch_const(log_1_min_a(log_alphas_v))
        self.log_alphas_cumprod_v = to_torch_const(log_alphas_cumprod_v)
        self.log_one_minus_alphas_cumprod_v = to_torch_const(log_1_min_a(log_alphas_cumprod_v))

        self.register_buffer('Lt_history', torch.zeros(self.num_timesteps))
        self.register_buffer('Lt_count', torch.zeros(self.num_timesteps))

        # === Model definition ===
        self.hidden_dim = config.hidden_dim
        self.num_classes = ligand_atom_feature_dim

        if self.config.node_indicator:
            emb_dim = self.hidden_dim - 1
        else:
            emb_dim = self.hidden_dim

        # Atom embeddings
        self.protein_atom_emb = nn.Linear(protein_atom_feature_dim, emb_dim)

        # center pos
        self.center_pos_mode = config.center_pos_mode # ['none', 'protein']

        # Time embedding
        self.time_emb_dim = config.time_emb_dim
        self.time_emb_mode = config.time_emb_mode # ['simple', 'sin']
        if self.time_emb_dim > 0:
            if self.time_emb_mode == 'simple':
                self.ligand_atom_emb = nn.Linear(ligand_atom_feature_dim + 1, emb_dim)
            elif self.time_emb_mode == 'sin':
                self.time_emb = nn.Sequential(
                    SinusoidalPosEmb(self.time_emb_dim),
                    nn.Linear(self.time_emb_dim, self.time_emb_dim * 4),
                    nn.GELU(),
                    nn.Linear(self.time_emb_dim * 4, self.time_emb_dim)
                )
                self.ligand_atom_emb = nn.Linear(ligand_atom_feature_dim + self.time_emb_dim, emb_dim)
            else:
                raise NotImplementedError
        else:
            self.ligand_atom_emb = nn.Linear(ligand_atom_feature_dim, emb_dim)

        # RefineNet
        self.refine_net_type = config.model_type
        self.refine_net = get_refine_net(self.refine_net_type, config)

        # Ligand type inference (for generation)
        self.v_inference = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            ShiftedSoftplus(),
            nn.Linear(self.hidden_dim, ligand_atom_feature_dim),
        )

        # Affinity head: per ligand atom (Input: hidden_dim, Output: 1), pooled by scatter_mean
        self.expert_pred_head1 = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            ShiftedSoftplus(),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid()
        )

        # Interaction energy: build vdW-radius table + index->Z LUTs (needs num_classes)
        self._init_vina_tables()

    @property
    def pos_classifier_grad_weight(self):
        """Compute grad weight on the same device as model parameters."""
        if self._pos_classifier_grad_weight is None or \
           self._pos_classifier_grad_weight.device != self.sqrt_one_minus_alphas_cumprod.device:
            self._pos_classifier_grad_weight = (
                self.sqrt_one_minus_alphas_cumprod / self.sqrt_alphas_cumprod
            )
        return self._pos_classifier_grad_weight

    def forward(self, protein_pos, protein_v, batch_protein, init_ligand_pos, init_ligand_v, batch_ligand,
                time_step=None, return_all=False, fix_x=False):
        """
        Forward pass: refine the complex graph, then read out atom types and affinity.
        """
        batch_size = batch_protein.max().item() + 1

        if len(init_ligand_v.shape) == 1:
            init_ligand_v = F.one_hot(init_ligand_v, self.num_classes).float()

        # Time embedding
        if self.time_emb_dim > 0:
            if self.time_emb_mode == 'simple':
                input_ligand_feat = torch.cat([
                    init_ligand_v,
                    (time_step / self.num_timesteps)[batch_ligand].unsqueeze(-1)
                ], -1)
            elif self.time_emb_mode == 'sin':
                time_feat = self.time_emb(time_step)
                input_ligand_feat = torch.cat([init_ligand_v, time_feat], -1)
            else:
                raise NotImplementedError
        else:
            input_ligand_feat = init_ligand_v

        h_protein = self.protein_atom_emb(protein_v)
        init_ligand_h = self.ligand_atom_emb(input_ligand_feat)

        if self.config.node_indicator:
            h_protein = torch.cat([h_protein, torch.zeros(len(h_protein), 1).to(h_protein)], -1)
            init_ligand_h = torch.cat([init_ligand_h, torch.ones(len(init_ligand_h), 1).to(h_protein)], -1)

        # Compose protein-ligand complex graph
        h_all, pos_all, batch_all, mask_ligand, mask_protein, _ = compose_context(
            h_protein=h_protein,
            h_ligand=init_ligand_h,
            pos_protein=protein_pos,
            pos_ligand=init_ligand_pos,
            batch_protein=batch_protein,
            batch_ligand=batch_ligand,
        )

        # RefineNet forward
        outputs = self.refine_net(h_all, pos_all, mask_ligand, batch_all, return_all=return_all, fix_x=fix_x)
        final_pos, final_h = outputs['x'], outputs['h']
        final_ligand_pos, final_ligand_h = final_pos[mask_ligand], final_h[mask_ligand]
        final_ligand_v = self.v_inference(final_ligand_h)

        # Affinity prediction: per ligand atom, then mean-pooled over each ligand
        atom_affinity = self.expert_pred_head1(final_ligand_h).squeeze(-1)  # (N_ligand,)
        final_exp_pred = scatter_mean(atom_affinity, batch_ligand)  # (batch_size,)

        preds = {
            'pred_ligand_pos': final_ligand_pos,
            'pred_ligand_v': final_ligand_v,
            'final_h': final_h,
            'final_ligand_h': final_ligand_h,
            'atom_affinity': atom_affinity,
            'final_exp_pred': final_exp_pred,
            'pred_affinity_head1': final_exp_pred,
            'atom_affinity_head1': atom_affinity,
            'batch_all': batch_all,
            'mask_ligand': mask_ligand
        }

        if return_all:
            final_all_pos, final_all_h = outputs['all_x'], outputs['all_h']
            final_all_ligand_pos = [pos[mask_ligand] for pos in final_all_pos]
            final_all_ligand_v = [self.v_inference(h[mask_ligand]) for h in final_all_h]
            preds.update({
                'layer_pred_ligand_pos': final_all_ligand_pos,
                'layer_pred_ligand_v': final_all_ligand_v
            })
        return preds

    # === Diffusion methods ===
    # Compute q(vt | v0 )
    def q_v_pred_one_timestep(self, log_vt_1, t, batch):
        log_alpha_t = extract(self.log_alphas_v, t, batch)
        log_1_min_alpha_t = extract(self.log_one_minus_alphas_v, t, batch)

        log_probs = log_add_exp(
            log_vt_1 + log_alpha_t,
            log_1_min_alpha_t - np.log(self.num_classes)
        )
        return log_probs

    def q_v_pred(self, log_v0, t, batch):
        log_cumprod_alpha_t = extract(self.log_alphas_cumprod_v, t, batch)
        log_1_min_cumprod_alpha = extract(self.log_one_minus_alphas_cumprod_v, t, batch)
        log_probs = log_add_exp(
            log_v0 + log_cumprod_alpha_t,
            log_1_min_cumprod_alpha - np.log(self.num_classes)
        )
        return log_probs

    def q_v_sample(self, log_v0, t, batch):
        log_qvt_v0 = self.q_v_pred(log_v0, t, batch)
        sample_prob = log_sample_categorical(log_qvt_v0)
        sample_index = sample_prob.argmax(dim=-1)
        log_sample = index_to_log_onehot(sample_index, self.num_classes)
        return sample_index, log_sample

    # atom type generative process
    def q_v_posterior(self, log_v0, log_vt, t, batch):
        # q(vt-1 | vt, v0) = q(vt | vt-1, v0) * q(vt-1 | v0) / q(vt | v0)
        t_minus_1 = t - 1
        # Remove negative values, will not be used anyway for final decoder
        t_minus_1 = torch.where(t_minus_1 < 0, torch.zeros_like(t_minus_1), t_minus_1)
        log_qvt1_v0 = self.q_v_pred(log_v0, t_minus_1, batch)
        unnormed_logprobs = log_qvt1_v0 + self.q_v_pred_one_timestep(log_vt, t, batch)
        log_vt1_given_vt_v0 = unnormed_logprobs - torch.logsumexp(unnormed_logprobs, dim=-1, keepdim=True)
        return log_vt1_given_vt_v0

    def kl_v_prior(self, log_x_start, batch):
        num_graphs = batch.max().item() + 1
        log_qxT_prob = self.q_v_pred(log_x_start, t=[self.num_timesteps - 1] * num_graphs, batch=batch)
        log_half_prob = -torch.log(self.num_classes * torch.ones_like(log_qxT_prob))
        kl_prior = categorical_kl(log_qxT_prob, log_half_prob)
        kl_prior = scatter_mean(kl_prior, batch, dim=0)
        return kl_prior

    def _predict_x0_from_eps(self, xt, eps, t, batch):
        pos0_from_e = extract(self.sqrt_recip_alphas_cumprod, t, batch) * xt - \
                      extract(self.sqrt_recipm1_alphas_cumprod, t, batch) * eps
        return pos0_from_e

    def q_pos_posterior(self, x0, xt, t, batch):
        # Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0)
        pos_model_mean = extract(self.posterior_mean_c0_coef, t, batch) * x0 + \
                         extract(self.posterior_mean_ct_coef, t, batch) * xt
        return pos_model_mean

    def sample_time(self, num_graphs, device, method):
        if method == 'importance':
            if not (self.Lt_count > 10).all():
                return self.sample_time(num_graphs, device, method='symmetric')

            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            Lt_sqrt[0] = Lt_sqrt[1]
            pt_all = Lt_sqrt / Lt_sqrt.sum()

            time_step = torch.multinomial(pt_all, num_samples=num_graphs, replacement=True)
            pt = pt_all.gather(dim=0, index=time_step)
            return time_step, pt
        # Uniform Time Sampling
        elif method == 'symmetric':
            time_step = torch.randint(0, self.num_timesteps, size=(num_graphs // 2 + 1,), device=device)
            time_step = torch.cat([time_step, self.num_timesteps - time_step - 1], dim=0)[:num_graphs]
            pt = torch.ones_like(time_step).float() / self.num_timesteps
            return time_step, pt
        else:
            raise ValueError

    def compute_pos_Lt(self, pos_model_mean, x0, xt, t, batch):
        # fixed pos variance
        pos_log_variance = extract(self.posterior_logvar, t, batch)
        pos_true_mean = self.q_pos_posterior(x0=x0, xt=xt, t=t, batch=batch)
        kl_pos = normal_kl(pos_true_mean, pos_log_variance, pos_model_mean, pos_log_variance)
        kl_pos = kl_pos / np.log(2.)
        decoder_nll_pos = -log_normal(x0, means=pos_model_mean, log_scales=0.5 * pos_log_variance)
        assert kl_pos.shape == decoder_nll_pos.shape
        mask = (t == 0).float()[batch]
        loss_pos = scatter_mean(mask * decoder_nll_pos + (1. - mask) * kl_pos, batch, dim=0)
        return loss_pos

    def compute_v_Lt(self, log_v_model_prob, log_v0, log_v_true_prob, t, batch):
        kl_v = categorical_kl(log_v_true_prob, log_v_model_prob)
        decoder_nll_v = -log_categorical(log_v0, log_v_model_prob)
        mask = (t == 0).float()[batch]
        loss_v = scatter_mean(mask * decoder_nll_v + (1. - mask) * kl_v, batch, dim=0)
        return loss_v

    # Constant lookup tables that used to be saved into the state_dict. Older checkpoints still
    # carry them (with the pre-fix Bondi radii, and with element-level gating tables this code no
    # longer builds at all), so drop them on load and let __init__ rebuild what it needs.
    _LEGACY_CONST_BUFFERS = ('_vina_radii', '_vina_ligand_Z', '_vina_protein_Z',
                             '_vina_is_hbond', '_vina_is_hydro')

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        for name in self._LEGACY_CONST_BUFFERS:
            state_dict.pop(prefix + name, None)
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def _init_vina_tables(self):
        """Build an atomic-number-indexed vdW-radius table + ligand/protein index->Z LUTs for the
        interaction energy.

        Radii are AutoDock Vina's `xs_radius` (src/lib/atom_constants.h: C 1.9, N 1.8, O 1.7,
        S 2.0, P 2.1, F 1.5, Cl 1.8, Br 2.0, I 2.2) -- NOT Bondi radii. Every Vina term is a
        function of the surface distance d = r - (R_i + R_j), so the radii set where the Gaussians
        and the H-bond/hydrophobic ramps sit: with Vina's radii an N...O pair has R_i+R_j = 3.5 A,
        putting the H-bond ramp (active for d in [-0.7, 0]) at r in [2.8, 3.5] A -- the real
        H-bond range."""
        from gated_energy_diffusion.utils import transforms as _t  # lazy import: avoids circular import at module load
        from gated_energy_diffusion.utils.vina_rules import VINA_WEIGHT
        from gated_energy_diffusion.utils.vina_types import XS_RADIUS_BY_Z, XS_RADIUS_DEFAULT

        ZMAX = 54
        radii = torch.full((ZMAX,), XS_RADIUS_DEFAULT)
        for z, r in XS_RADIUS_BY_Z.items():
            radii[z] = r
        # persistent=False: these are constants, not learned state. Keeping them out of the
        # state_dict stops a checkpoint saved before the Bondi->xs_radius fix from silently
        # restoring the old radii (see _load_from_state_dict below).
        self.register_buffer('_vina_radii', radii, persistent=False)

        nc = self.num_classes
        if nc == len(_t.MAP_ATOM_TYPE_ONLY_TO_INDEX):        # 'basic' (8)
            lut = [_t.MAP_INDEX_TO_ATOM_TYPE_ONLY[i] for i in range(nc)]
        elif nc == len(_t.MAP_ATOM_TYPE_AROMATIC_TO_INDEX):  # 'add_aromatic' (13)
            lut = [_t.MAP_INDEX_TO_ATOM_TYPE_AROMATIC[i][0] for i in range(nc)]
        elif nc == len(_t.MAP_ATOM_TYPE_FULL_TO_INDEX):      # 'full' (23)
            lut = [_t.MAP_INDEX_TO_ATOM_TYPE_FULL[i][0] for i in range(nc)]
        else:
            raise ValueError(f"cannot infer the ligand atom mode from num_classes={nc}")
        self.register_buffer('_vina_ligand_Z', torch.tensor(lut, dtype=torch.long), persistent=False)
        # protein element one-hot order in protein_v[:, :6] (FeaturizeProteinAtom): H,C,N,O,S,Se
        self.register_buffer('_vina_protein_Z', torch.tensor([1, 6, 7, 8, 16, 34], dtype=torch.long), persistent=False)
        # Per-term ablation scale, aligned with VINA_WEIGHT's term order
        # [gauss1, gauss2, repulsion, hydrophobic, hbonding]. persistent=False keeps it out of the
        # state_dict, so a checkpoint written before this patch still loads without a missing key
        # (and an ablation checkpoint never silently carries its scales into a different config).
        self.register_buffer('_vina_term_scale', torch.tensor(
            [self.vina_steric_scale, self.vina_steric_scale, self.vina_steric_scale,
             self.vina_hydro_scale, self.vina_hbond_scale], dtype=torch.float32), persistent=False)
        active = [n for n, s in (('steric', self.vina_steric_scale),
                                 ('hydrophobic', self.vina_hydro_scale),
                                 ('hbond', self.vina_hbond_scale)) if s != 0.]
        print(f"[INFO] Gated interaction energy enabled | VINA_WEIGHT={VINA_WEIGHT} | "
              f"type_gating={self.vina_type_gating} (surface-distance energy, minimized directly)")
        print(f"[INFO] Interaction term scales: steric={self.vina_steric_scale} "
              f"hydrophobic={self.vina_hydro_scale} hbond={self.vina_hbond_scale} "
              f"-> active terms: {'+'.join(active) if active else 'NONE (physics loss is identically 0)'}")

    def _compute_vina_energy(self, pred_ligand_pos, protein_pos, batch_ligand, batch_protein,
                             num_graphs, ligand_v, protein_v, ligand_xs=None, protein_xs=None):
        """AutoDock Vina empirical interaction ENERGY, reusing vina_rules.py's terms and weights
        verbatim (correct SIGNS and magnitudes): VINA_WEIGHT = [gauss1 -0.0356, gauss2 -0.00516,
        repulsion +0.840, hydrophobic -0.0351, hbonding -0.587], on the SURFACE distance
        d = r - (R_i + R_j) with Vina's xs_radius.

        This is minimized DIRECTLY as an energy (single backward), not as a sum of forces
        (sum dE/dd, double backward): the per-pair energy is bounded below (~ -0.66) so the
        objective is well-posed, and the piecewise-linear hydrophobic/hbonding terms DO
        contribute gradient (nonzero 1st derivative). Lower = more Vina-favorable pose; repulsion
        (positive weight) penalizes clashes.

        TYPE GATING (self.vina_type_gating) reproduces Vina's own rules from the per-atom XS flags
        `ligand_xs` / `protein_xs` (N,3) = [is_hydrophobic, is_donor, is_acceptor] supplied by
        transforms.FeaturizeVinaAtomTypes (see gated_energy_diffusion/utils/vina_types.py):

            hydrophobic fires iff xs_is_hydrophobic(t1) && xs_is_hydrophobic(t2)
                -> a carbon with NO heteroatom neighbour, or F/Cl/Br/I
            h-bond fires iff xs_h_bond_possible(t1, t2)
                -> (donor(t1) && acceptor(t2)) || (donor(t2) && acceptor(t1))   [never A-A or D-D]

        The steric gauss1/gauss2/repulsion terms apply to every pair. 8 A cutoff. Returns a scalar
        (sum over ligand-protein pairs, averaged over graphs)."""
        assert ligand_v is not None and protein_v is not None, \
            "the gated interaction energy requires ligand_v and protein_v"
        assert ligand_xs is not None and protein_xs is not None, (
            "chemistry-gated Vina energy requires per-atom XS flags; add "
            "transforms.FeaturizeVinaAtomTypes() to the transform list")
        from gated_energy_diffusion.utils.vina_rules import (
            VINA_WEIGHT, gauss1, gauss2, repulsion, hydrophobic, hbonding)
        w = torch.tensor(VINA_WEIGHT, device=pred_ligand_pos.device, dtype=pred_ligand_pos.dtype)
        # Per-term ablation: zero out whole term groups. The energy is LINEAR in these scales, so
        # E(steric only) + E(hydro only) + E(hbond only) == E(full) exactly.
        w = w * self._vina_term_scale.to(device=w.device, dtype=w.dtype)

        lig_idx = ligand_v.argmax(-1) if ligand_v.dim() > 1 else ligand_v
        lig_Z = self._vina_ligand_Z[lig_idx.long()]                                       # (N_lig,)
        pro_Z = self._vina_protein_Z[protein_v[:, :6].argmax(-1).long()]                  # (N_pro,)
        lig_r_all, pro_r_all = self._vina_radii[lig_Z], self._vina_radii[pro_Z]           # Vina xs_radius

        lig_hy, lig_do, lig_ac = ligand_xs.bool().unbind(-1)
        pro_hy, pro_do, pro_ac = protein_xs.bool().unbind(-1)

        total = pred_ligand_pos.new_zeros(())
        for i in range(num_graphs):
            lm, pm = batch_ligand == i, batch_protein == i
            lig, pro = pred_ligand_pos[lm], protein_pos[pm]
            if lig.shape[0] == 0 or pro.shape[0] == 0:
                continue
            # manual sqrt with +1e-10 -> finite gradient as the pair distance goes to 0
            r = torch.sqrt(((lig.unsqueeze(1) - pro.unsqueeze(0)) ** 2).sum(-1) + 1e-10)  # (L,P) interatomic
            d = r - (lig_r_all[lm].unsqueeze(1) + pro_r_all[pm].unsqueeze(0))             # surface distance
            hydro_t, hbond_t = hydrophobic(d), hbonding(d)
            if self.vina_type_gating:
                # Gates and radii come from ground-truth chemistry (the XS flags and ligand_v as
                # given by the data, as Vina does), so they are batch constants: this energy
                # backpropagates into the predicted COORDINATES only and sends no gradient into
                # the atom-type head.
                m_hy = lig_hy[lm].unsqueeze(1) & pro_hy[pm].unsqueeze(0)
                m_hb = (lig_do[lm].unsqueeze(1) & pro_ac[pm].unsqueeze(0)) | \
                       (lig_ac[lm].unsqueeze(1) & pro_do[pm].unsqueeze(0))    # xs_h_bond_possible
                hydro_t = hydro_t * m_hy.to(d.dtype)
                hbond_t = hbond_t * m_hb.to(d.dtype)
            terms = torch.stack([gauss1(d), gauss2(d), repulsion(d), hydro_t, hbond_t], dim=0)  # (5, L, P)
            e = (terms * w[:, None, None]).sum(0)                                         # (L, P) weighted Vina energy
            total = total + (e * (r < 8.0).to(e.dtype)).sum()                             # Vina 8 A cutoff
        return total / max(num_graphs, 1)

    def compute_vdw_loss(self, pred_ligand_pos, protein_pos, batch_ligand, batch_protein, num_graphs,
                         ligand_v=None, protein_v=None, ligand_xs=None, protein_xs=None):
        """
        Physics-informed interaction energy between the predicted ligand atom positions (x0) and
        the fixed protein atoms: all 5 AutoDock Vina terms (gauss1, gauss2, repulsion,
        hydrophobic, hbonding) on the SURFACE distance d = r - (R_i + R_j), weighted by
        VINA_WEIGHT (correct signs AND magnitudes) and minimized directly as an energy (single
        backward). Well-posed (bounded below); every term trains. Needs ligand_v/protein_v, and
        ligand_xs/protein_xs for Vina's donor-acceptor / hydrophobic type gating (see
        _compute_vina_energy). Returns a scalar (sum over all ligand-protein pairs, averaged over
        graphs in the batch).
        """
        return self._compute_vina_energy(
            pred_ligand_pos, protein_pos, batch_ligand, batch_protein, num_graphs, ligand_v, protein_v,
            ligand_xs=ligand_xs, protein_xs=protein_xs)

    def get_diffusion_loss(self, protein_pos, protein_v, affinity, batch_protein, ligand_pos, ligand_v, batch_ligand,
                           time_step=None, ligand_xs=None, protein_xs=None):
        """
        Diffusion training loss: coordinate + atom-type reconstruction, the affinity MSE, and the
        gated interaction energy (returned raw as 'loss_vdw'; its weight and curriculum ramp are
        applied in the train loop).

        ligand_xs / protein_xs: (N, 3) bool Vina XS flags [hydrophobic, donor, acceptor] from
        transforms.FeaturizeVinaAtomTypes, required by the gated interaction energy.
        """
        num_graphs = batch_protein.max().item() + 1
        protein_pos, ligand_pos, _ = center_pos(
            protein_pos, ligand_pos, batch_protein, batch_ligand, mode=self.center_pos_mode)

        # Sample noise levels
        if time_step is None:
            time_step, pt = self.sample_time(num_graphs, protein_pos.device, self.sample_time_method)
        else:
            pt = torch.ones_like(time_step).float() / self.num_timesteps
        a = self.alphas_cumprod.index_select(0, time_step)

        # Perturb pos and v
        a_pos = a[batch_ligand].unsqueeze(-1)
        pos_noise = torch.zeros_like(ligand_pos)
        pos_noise.normal_()
        ligand_pos_perturbed = a_pos.sqrt() * ligand_pos + (1.0 - a_pos).sqrt() * pos_noise

        log_ligand_v0 = index_to_log_onehot(ligand_v, self.num_classes)
        ligand_v_perturbed, log_ligand_vt = self.q_v_sample(log_ligand_v0, time_step, batch_ligand)

         # 3. forward-pass NN, feed perturbed pos and v, output noise
        preds = self(
            protein_pos=protein_pos,
            protein_v=protein_v,
            batch_protein=batch_protein,
            init_ligand_pos=ligand_pos_perturbed,
            init_ligand_v=ligand_v_perturbed,
            batch_ligand=batch_ligand,
            time_step=time_step
        )

        pred_ligand_pos, pred_ligand_v = preds['pred_ligand_pos'], preds['pred_ligand_v']
        pred_pos_noise = pred_ligand_pos - ligand_pos_perturbed

        # Atom Position loss
        if self.model_mean_type == 'C0':
            target, pred = ligand_pos, pred_ligand_pos
        elif self.model_mean_type == 'noise':
            target, pred = pos_noise, pred_pos_noise
        else:
            raise ValueError
        loss_pos = scatter_mean(((pred - target) ** 2).sum(-1), batch_ligand, dim=0)
        loss_pos = torch.mean(loss_pos)

        # Atom type loss
        log_ligand_v_recon = F.log_softmax(pred_ligand_v, dim=-1)
        log_v_model_prob = self.q_v_posterior(log_ligand_v_recon, log_ligand_vt, time_step, batch_ligand)
        log_v_true_prob = self.q_v_posterior(log_ligand_v0, log_ligand_vt, time_step, batch_ligand)
        kl_v = self.compute_v_Lt(log_v_model_prob=log_v_model_prob, log_v0=log_ligand_v0,
                                 log_v_true_prob=log_v_true_prob, t=time_step, batch=batch_ligand)
        loss_v = torch.mean(kl_v)

        # Affinity loss
        loss_exp = F.mse_loss(preds['final_exp_pred'], affinity)

        if self.use_classifier_guide:
            loss = loss_pos + loss_v * self.loss_v_weight + loss_exp * self.loss_exp_weight
        else:
            loss = loss_pos + loss_v * self.loss_v_weight

        # Interaction energy (raw energy; weight + curriculum ramp applied in train loop)
        loss_vdw = self.compute_vdw_loss(pred_ligand_pos, protein_pos, batch_ligand, batch_protein, num_graphs,
                                         ligand_v=ligand_v, protein_v=protein_v,
                                         ligand_xs=ligand_xs, protein_xs=protein_xs)

        return {
            'loss_pos': loss_pos,
            'loss_v': loss_v,
            'loss_exp': loss_exp,
            'loss_vdw': loss_vdw,
            'loss': loss,
            'x0': ligand_pos,
            'pred_ligand_pos': pred_ligand_pos,
            'pred_ligand_v': pred_ligand_v,
            'pred_exp': preds['final_exp_pred'],
            'pred_pos_noise': pred_pos_noise,
            'ligand_v_recon': F.softmax(pred_ligand_v, dim=-1),
            'final_ligand_h': preds['final_ligand_h']
        }

    @torch.no_grad()
    def sample_diffusion(self, protein_pos, protein_v, batch_protein,
                         init_ligand_pos, init_ligand_v, batch_ligand,
                         num_steps=None, center_pos_mode=None):
        """
        Sample from the diffusion model, without affinity guidance.
        """
        if num_steps is None:
            num_steps = self.num_timesteps
        num_graphs = batch_protein.max().item() + 1

        protein_pos, init_ligand_pos, offset = center_pos(
            protein_pos, init_ligand_pos, batch_protein, batch_ligand, mode=center_pos_mode)

        pos_traj, v_traj, exp_traj = [], [], []
        ligand_pos, ligand_v = init_ligand_pos, init_ligand_v

        time_seq = list(reversed(range(self.num_timesteps - num_steps, self.num_timesteps)))
        for i in tqdm(time_seq, desc='sampling', total=len(time_seq)):
            t = torch.full(size=(num_graphs,), fill_value=i, dtype=torch.long, device=protein_pos.device)

            preds = self(
                protein_pos=protein_pos,
                protein_v=protein_v,
                batch_protein=batch_protein,
                init_ligand_pos=ligand_pos,
                init_ligand_v=ligand_v,
                batch_ligand=batch_ligand,
                time_step=t
            )

            # Compute posterior
            if self.model_mean_type == 'noise':
                pred_pos_noise = preds['pred_ligand_pos'] - ligand_pos
                pos0_from_e = self._predict_x0_from_eps(xt=ligand_pos, eps=pred_pos_noise, t=t, batch=batch_ligand)
                v0_from_e = preds['pred_ligand_v']
            elif self.model_mean_type == 'C0':
                pos0_from_e = preds['pred_ligand_pos']
                v0_from_e = preds['pred_ligand_v']
            else:
                raise ValueError

            pos_model_mean = self.q_pos_posterior(x0=pos0_from_e, xt=ligand_pos, t=t, batch=batch_ligand)
            pos_log_variance = extract(self.posterior_logvar, t, batch_ligand)

            log_ligand_v_recon = F.log_softmax(v0_from_e, dim=-1)
            log_ligand_v = index_to_log_onehot(ligand_v, self.num_classes)

            nonzero_mask = (1 - (t == 0).float())[batch_ligand].unsqueeze(-1)

            ligand_pos_next = pos_model_mean + nonzero_mask * (0.5 * pos_log_variance).exp() * torch.randn_like(ligand_pos)
            ligand_pos = ligand_pos_next

            log_model_prob = self.q_v_posterior(log_ligand_v_recon, log_ligand_v, t, batch_ligand)
            ligand_v_next = log_sample_categorical(log_model_prob).argmax(dim=-1)
            ligand_v = ligand_v_next

            ori_ligand_pos = ligand_pos + offset[batch_ligand]
            pos_traj.append(ori_ligand_pos.clone().cpu())
            v_traj.append(ligand_v.clone().cpu())
            exp_traj.append(preds['final_exp_pred'].clone().cpu())

        ligand_pos = ligand_pos + offset[batch_ligand]
        return {
            'pos': ligand_pos,
            'v': ligand_v,
            'exp': exp_traj[-1] if exp_traj else None,
            'pos_traj': pos_traj,
            'v_traj': v_traj,
            'exp_traj': exp_traj,
        }

    def sample_diffusion_with_guidance(
        self, protein_pos, protein_v, batch_protein,
        init_ligand_pos, init_ligand_v, batch_ligand,
        num_steps=None, center_pos_mode=None,
        guide_mode='head1_only',
        head1_type_grad_weight=0., head1_pos_grad_weight=0.,
        w_on=1.0
    ):
        """
        Sample from the diffusion model with affinity guidance from the single head.

        Args:
            protein_pos: Target protein positions
            protein_v: Target protein features
            batch_protein: Protein batch indices
            init_ligand_pos: Initial ligand positions (noise)
            init_ligand_v: Initial ligand atom types
            batch_ligand: Ligand batch indices
            num_steps: Number of diffusion steps
            center_pos_mode: Position centering mode
            guide_mode: 'head1_only' (the only mode implemented here)
            head1_type_grad_weight: atom-type guidance strength s_v
            head1_pos_grad_weight: coordinate guidance strength s_x
            w_on: scale applied to both guidance channels

        Returns:
            Dictionary with sampled ligands and trajectories
        """
        import gc

        if num_steps is None:
            num_steps = self.num_timesteps
        num_graphs = batch_protein.max().item() + 1
        device = protein_pos.device

        # Center protein
        protein_pos_on, init_ligand_pos, offset = center_pos(
            protein_pos, init_ligand_pos, batch_protein, batch_ligand, mode=center_pos_mode)

        pos_traj, v_traj = [], []
        exp_on_traj = []
        ligand_pos, ligand_v = init_ligand_pos, init_ligand_v

        time_seq = list(reversed(range(self.num_timesteps - num_steps, self.num_timesteps)))

        for iter_idx, i in enumerate(tqdm(time_seq, desc='sampling', total=len(time_seq))):
            t = torch.full(size=(num_graphs,), fill_value=i, dtype=torch.long, device=device)

            if guide_mode == 'head1_only':
                # Compute guidance with standard forward
                if head1_type_grad_weight > 0 or head1_pos_grad_weight > 0:
                    with torch.enable_grad():
                        ligand_pos_h1 = ligand_pos.detach().requires_grad_(True)
                        ligand_v_onehot = F.one_hot(ligand_v, self.num_classes).float().requires_grad_(True)

                        preds = self(
                            protein_pos=protein_pos_on,
                            protein_v=protein_v,
                            batch_protein=batch_protein,
                            init_ligand_pos=ligand_pos_h1,
                            init_ligand_v=ligand_v_onehot,
                            batch_ligand=batch_ligand,
                            time_step=t
                        )
                        exp_on = preds['final_exp_pred']

                        # Compute log(affinity) for position gradient
                        exp_on_log = exp_on.log()

                        # TWO SEPARATE backward calls, on purpose. One tuple-backward over both
                        # inputs traverses the graph in a different order and lands ~1e-7 away in
                        # float, which the 1000-step stochastic sampler amplifies into a visibly
                        # different molecule.
                        # The two channels are deliberately asymmetric: the type gradient is
                        # grad A_hat, the position gradient grad log A_hat.
                        head1_v_grad = torch.autograd.grad(
                            exp_on.sum(), ligand_v_onehot,
                            grad_outputs=None,
                            retain_graph=True, create_graph=False, allow_unused=True
                        )[0]

                        head1_pos_grad = torch.autograd.grad(
                            exp_on_log.sum(), ligand_pos_h1,
                            grad_outputs=None,
                            retain_graph=False, create_graph=False, allow_unused=True
                        )[0]

                        if head1_v_grad is None:
                            head1_v_grad = torch.zeros(len(ligand_v), self.num_classes, device=device)
                        if head1_pos_grad is None:
                            head1_pos_grad = torch.zeros_like(ligand_pos)

                    del ligand_pos_h1, ligand_v_onehot
                    torch.cuda.empty_cache()
                else:
                    with torch.no_grad():
                        preds = self(
                            protein_pos=protein_pos_on,
                            protein_v=protein_v,
                            batch_protein=batch_protein,
                            init_ligand_pos=ligand_pos,
                            init_ligand_v=F.one_hot(ligand_v, self.num_classes).float(),
                            batch_ligand=batch_ligand,
                            time_step=t
                        )
                    exp_on = preds['final_exp_pred']
                    head1_pos_grad = torch.zeros_like(ligand_pos)
                    head1_v_grad = torch.zeros(len(ligand_v), self.num_classes, device=device)
            else:
                raise ValueError(f"guide_mode must be 'head1_only', got {guide_mode!r}")

            # === Compute posterior (always with no_grad) ===
            with torch.no_grad():
                # Get denoising prediction
                preds = self(
                    protein_pos=protein_pos_on,
                    protein_v=protein_v,
                    batch_protein=batch_protein,
                    init_ligand_pos=ligand_pos,
                    init_ligand_v=ligand_v,
                    batch_ligand=batch_ligand,
                    time_step=t
                )

                if self.model_mean_type == 'noise':
                    pred_pos_noise = preds['pred_ligand_pos'] - ligand_pos
                    pos0_from_e = self._predict_x0_from_eps(xt=ligand_pos, eps=pred_pos_noise, t=t, batch=batch_ligand)
                    v0_from_e = preds['pred_ligand_v']
                elif self.model_mean_type == 'C0':
                    pos0_from_e = preds['pred_ligand_pos']
                    v0_from_e = preds['pred_ligand_v']
                else:
                    raise ValueError

                # Compute posterior
                pos_model_mean = self.q_pos_posterior(x0=pos0_from_e, xt=ligand_pos, t=t, batch=batch_ligand)
                pos_log_variance = extract(self.posterior_logvar, t, batch_ligand)

                # type posterior
                log_ligand_v_recon = F.log_softmax(v0_from_e, dim=-1)
                log_ligand_v = index_to_log_onehot(ligand_v, self.num_classes)

                # Apply guidance. The type score is added to the log-onehot BEFORE q_v_posterior
                # and is NOT centred over classes.
                pos_model_mean_h1 = pos_model_mean + w_on * head1_pos_grad_weight * (0.5 * pos_log_variance).exp() * head1_pos_grad
                log_ligand_v_h1 = log_ligand_v + w_on * head1_type_grad_weight * head1_v_grad

                # no noise when t == 0
                nonzero_mask = (1 - (t == 0).float())[batch_ligand].unsqueeze(-1)

                # Sample next step
                ligand_pos_next = pos_model_mean_h1 + nonzero_mask * (0.5 * pos_log_variance).exp() * torch.randn_like(ligand_pos)
                ligand_pos = ligand_pos_next.detach()  # Break computation graph

                log_model_prob = self.q_v_posterior(log_ligand_v_recon, log_ligand_v_h1, t, batch_ligand)

                ligand_v_next = log_sample_categorical(log_model_prob).argmax(dim=-1)
                ligand_v = ligand_v_next.detach()  # Break computation graph

                # Store trajectory
                ori_ligand_pos = ligand_pos + offset[batch_ligand]
                pos_traj.append(ori_ligand_pos.clone().cpu())
                v_traj.append(ligand_v.clone().cpu())
                exp_on_traj.append(exp_on.detach().clone().cpu())

            # Clean up iteration variables
            del preds, head1_pos_grad, head1_v_grad

            # Clear CUDA cache periodically
            if iter_idx % 50 == 0 and device.type == 'cuda':
                torch.cuda.empty_cache()
                gc.collect()

        # Final positions with offset
        ligand_pos = ligand_pos + offset[batch_ligand]

        return {
            'pos': ligand_pos,
            'v': ligand_v,
            'exp_on': exp_on_traj[-1] if exp_on_traj else None,
            'pos_traj': pos_traj,
            'v_traj': v_traj,
            'exp_on_traj': exp_on_traj,
        }


def extract(coef, t, batch):
    out = coef[t][batch]
    return out.unsqueeze(-1)
