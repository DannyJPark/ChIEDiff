#!/bin/bash
#SBATCH --job-name=ged_sample
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=logs_slurm/%x_%A_%a.out
#SBATCH --error=logs_slurm/%x_%A_%a.err
# =========================================================================
# Generate NUM_SAMPLES molecules for ONE test pocket per array task
# (data_id = SLURM_ARRAY_TASK_ID), so 100 pockets fan out across as many nodes as the
# partition and QoS allow.
#
#   sbatch -p <partitions> -q <qos> --array=0-99%24 \
#          --export=ALL,CKPT=$PWD/archive/weights/best.pt,RESULT=./results/generated \
#          slurm/sample_array.sbatch
#
# Partition and QoS are passed at submit time; listing several partitions lets the
# scheduler place tasks wherever GPUs are free.
#
# Re-runnable: a pocket whose id{N}/result_<first>.pt already exists is skipped, so a
# preempted or partially completed array resumes cleanly. FORCE=1 to redo.
#
# The published cohort was generated in three chunks (24, then 2, then 76) using
# START_INDEX. A single 100-molecule run at the same seed produces a DIFFERENT molecule
# set -- see README section 6.3.
# =========================================================================
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs_slurm

# See the note in train.sbatch: activate conda ourselves, never `conda init`.
CONDA_BASE="${CONDA_BASE:-$HOME/anaconda3}"
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV:-ged-train}"
else
    echo "no conda at $CONDA_BASE; set CONDA_BASE" >&2; exit 1
fi

# Bitwise reproducibility requires a single thread. At default GPU threading two identical
# runs of this script differ by 10-19 atom types; at OMP_NUM_THREADS=8 by 6-7; only at 1
# are they bit-identical. Set BITWISE=0 to trade that for throughput on an exploratory run.
if [ "${BITWISE:-1}" = "1" ]; then
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
fi

# Preflight. A few GPU nodes sit in a broken CUDA state (torch.cuda.is_available() is False,
# nvidia-smi reports "Unknown Error"). A task landing there fails instantly at checkpoint
# load and, because it fails in seconds, DRAINS the whole array through that one node.
# Detect it and requeue THIS task with the bad node excluded, so it reschedules elsewhere
# rather than silently burning the array. Keep this: the set of bad nodes changes, so a
# static exclude list is not a substitute.
if ! python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "[sample] CUDA unavailable on ${SLURMD_NODENAME} -> exclude node + requeue this task"
    scontrol update JobId="${SLURM_JOB_ID}" ExcludeNodes="${SLURMD_NODENAME}" 2>/dev/null || true
    scontrol requeue "${SLURM_JOB_ID}" 2>/dev/null || true
    sleep 20      # avoid a tight requeue loop if the scheduler re-lands us immediately
    exit 1
fi

TID="${SLURM_ARRAY_TASK_ID:?must be run as a job array (data_id = array index)}"
CKPT="${CKPT:?must pass CKPT=/abs/path/to/checkpoint.pt}"
RESULT="${RESULT:-./results/generated}"
DATA_PATH="${DATA_PATH:-./data/crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb}"
SPLIT_PATH="${SPLIT_PATH:-./data/crossdocked_pocket10_pose_split.pt}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
START_INDEX="${START_INDEX:-0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
SEED="${SEED:-42}"
FORCE="${FORCE:-0}"

echo "=== sample ${SLURM_ARRAY_JOB_ID:-local}:${TID} on ${SLURMD_NODENAME:-$(hostname)} (${SLURM_JOB_PARTITION:-?}) ==="
echo "ckpt=${CKPT}  result=${RESULT}  n=${NUM_SAMPLES} from ${START_INDEX}  batch=${BATCH_SIZE}  seed=${SEED}  OMP=${OMP_NUM_THREADS:-unset}"
# Same reason as in train.sbatch: the device is not recoverable after the fact.
nvidia-smi -L 2>/dev/null | head -1 || true

if [ "$FORCE" != "1" ] && [ -f "${RESULT}/id${TID}/result_${START_INDEX}.pt" ]; then
    echo "[sample] id${TID}: result_${START_INDEX}.pt exists -> skip"
    exit 0
fi

python bin/sample.py \
    --ckpt "$CKPT" \
    --data_path "$DATA_PATH" \
    --split_path "$SPLIT_PATH" \
    --data_id "$TID" \
    --num_samples "$NUM_SAMPLES" \
    --start_index "$START_INDEX" \
    --batch_size "$BATCH_SIZE" \
    --seed "$SEED" \
    --result_path "$RESULT"

echo "=== sample id${TID} done ==="
