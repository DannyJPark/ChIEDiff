#!/bin/bash
#SBATCH --job-name=ged_train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs_slurm/%x_%j.out
#SBATCH --error=logs_slurm/%x_%j.err
# =============================================================================
# Train the chemistry-gated interaction-energy model.
#
#   TAG=my_run sbatch -p <partitions> -q <qos> slurm/train.sbatch
#
# Partition and QoS are site-specific and are not set in this file. Pass them at submit time.
#
# --time is set to 3 days. A site can cap wall time below what its partitions advertise, and a
# request above that cap is accepted and then pends with no error and nothing in the log, so
# check the local limit before raising it.
#
# The run keeps every new best-validation checkpoint (nothing is deleted) and mirrors the
# latest to <logdir>/<run>/checkpoints/best.pt.
# =============================================================================
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs_slurm

# Activate conda from the script. Where conda auto-initialisation is disabled in the shell
# profile, a fresh non-interactive shell has no `conda` on PATH, so `conda info --base` cannot
# locate the installation either, hence the literal default below.
# Do not run `conda init` here. It rewrites the shell profile and can re-enable that
# auto-initialisation.
CONDA_BASE="${CONDA_BASE:-$HOME/anaconda3}"
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV:-ged-train}"
else
    echo "no conda at $CONDA_BASE; set CONDA_BASE" >&2; exit 1
fi

CONFIG="${CONFIG:-./configs/train.yml}"
TAG="${TAG:?set TAG=<short run name>}"
LOGDIR="${LOGDIR:-./logs_diffusion}"

echo "=== train ${SLURM_JOB_ID:-local} on ${SLURMD_NODENAME:-$(hostname)} (${SLURM_JOB_PARTITION:-?}) ==="
echo "config=${CONFIG}  tag=${TAG}"
# Record the GPU model. Bitwise output depends on the device, and the device is not
# recoverable after the fact, so this line is part of the run record.
nvidia-smi -L 2>/dev/null | head -1 || true

python - "$CONFIG" <<'PY'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1]))
m, t = c["model"], c["train"]
omega = m["loss_vdw_weight"]
print(f"  loss_vdw_weight={omega}  ramp_iters={t['vdw_ramp_iters']}  ramp_cap={t['vdw_ramp_cap']}"
      f"  -> omega_m = {omega} * min(it/{t['vdw_ramp_iters']}, {t['vdw_ramp_cap']})")
print(f"  vina_type_gating={m['vina_type_gating']}  use_classifier_guide={m['use_classifier_guide']}")
print(f"  batch_size={t['batch_size']} x n_acc_batch={t['n_acc_batch']}  pos_noise_std={t['pos_noise_std']}")
PY

python bin/train.py --config "$CONFIG" --tag "$TAG" --logdir "$LOGDIR" "$@"

echo "=== train ${SLURM_JOB_ID:-local} finished ==="
