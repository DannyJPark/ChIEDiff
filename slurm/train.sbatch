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
# Partition and QoS are deliberately NOT in this file: they are site-specific, and a
# launcher that hardcodes them is wrong on every other cluster. Pass them at submit time.
#
# --time is capped at 3 days on purpose. The association this was developed under caps
# wall time at 4320 minutes regardless of what the partitions advertise, and a longer
# request is ACCEPTED and then pends forever with no error and nothing in the log. If your
# site allows longer, raise it; do not raise it blindly.
#
# The run keeps every new best-validation checkpoint (nothing is deleted) and mirrors the
# latest to <logdir>/<run>/checkpoints/best.pt.
# =============================================================================
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs_slurm

# Activate conda ourselves. A fresh non-interactive shell on this cluster has no `conda`
# on PATH (auto-initialisation is disabled in the shell profile), so `conda info --base`
# is not available to locate the installation -- hence the literal default below.
# Never run `conda init` here: it rewrites the shell profile and can re-enable the
# auto-initialisation that was switched off deliberately.
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
# Record the GPU model. The device is not recoverable after the fact, and the published
# sampling runs are missing exactly this line -- which is why bitwise reproduction of the
# published molecules cannot be claimed. Do not remove it.
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
