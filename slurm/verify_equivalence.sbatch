#!/bin/bash
#SBATCH --job-name=ged_verify
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs_slurm/%x_%j.out
#SBATCH --error=logs_slurm/%x_%j.err
# =============================================================================
# Controls A, B, C, D: the pruned release must match the original implementation.
#
#   REF_REPO=/path/to/original CKPT=/path/to/best.pt \
#     sbatch -p <partitions> -q <qos> slurm/verify_equivalence.sbatch
#
# The original implementation is read from REF_REPO and is never committed here.
# OMP_NUM_THREADS=1 throughout: Control B is meaningless without it.
# =============================================================================
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs_slurm

CONDA_BASE="${CONDA_BASE:-$HOME/anaconda3}"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-kgdiff}"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0

REF_REPO="${REF_REPO:?set REF_REPO=/path/to/the/original/repo}"
CKPT="${CKPT:?set CKPT=/path/to/best.pt}"
DATA_PATH="${DATA_PATH:-$REF_REPO/data/crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb}"
SPLIT_PATH="${SPLIT_PATH:-$REF_REPO/data/crossdocked_pocket10_pose_split.pt}"

echo "job ${SLURM_JOB_ID:-local} on ${SLURMD_NODENAME:-$(hostname)} (${SLURM_JOB_PARTITION:-?})"
nvidia-smi -L 2>/dev/null | head -1 || true
echo "ref=$REF_REPO"
echo "ckpt=$CKPT  sha256=$(sha256sum "$CKPT" | cut -c1-16)"
echo

rc=0

echo "###### Control A: loss and gradient equivalence ######"
python scripts/verify_loss_equivalence.py \
    --reference-repo "$REF_REPO" --ckpt "$CKPT" \
    --data-path "$DATA_PATH" --split-path "$SPLIT_PATH" || rc=1
echo

echo "###### Controls C and D: gate assertion, term linearity ######"
python scripts/verify_energy_invariants.py \
    --reference-repo "$REF_REPO" --ckpt "$CKPT" \
    --data-path "$DATA_PATH" --split-path "$SPLIT_PATH" || rc=1
echo

echo "###### Control B: sampling equivalence ######"
python scripts/verify_sampling_equivalence.py \
    --reference-repo "$REF_REPO" --ckpt "$CKPT" \
    --data-path "$DATA_PATH" --split-path "$SPLIT_PATH" \
    --data-ids 0 1 --num-samples 8 --batch-size 4 || rc=1
echo

if [ "$rc" = "0" ]; then
    echo "ALL CONTROLS PASSED"
else
    echo "AT LEAST ONE CONTROL FAILED -- do not trust the release"
fi
exit "$rc"
