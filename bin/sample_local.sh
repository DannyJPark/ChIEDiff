#!/usr/bin/env bash
# Sample molecules on a local GPU, one pocket after another. No Slurm.
#
#   bash bin/sample_local.sh --ckpt archive/<weights file> --pockets 0-4
#
# Defaults reproduce the published sampling settings: head1_only guidance at s_x=25 / s_v=100,
# 1000 diffusion steps, 100 molecules per pocket, seed 42.
#
# COST. One pocket at 100 molecules is roughly an hour on a 24 GB card, so the full 100-pocket
# benchmark is days sequentially. --pockets defaults to 0-4. The published molecules are in the
# data archive, and `make tables` reproduces every published number without sampling at all.
#
# Re-runnable: a pocket whose first result file exists is skipped.
#
# OMP_NUM_THREADS=1 is set because it is the only way the sampler is even run-to-run
# comparable. Note that it still does not reproduce individual molecules -- see README 6.4.
set -uo pipefail
cd "$(dirname "$0")/.."

CKPT=""; POCKETS="0-4"; N=100; BATCH=4; SEED=42
RESULT="${RESULT:-results/generated}"
DATA="${DATA:-data/crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb}"
SPLIT="${SPLIT:-data/crossdocked_pocket10_pose_split.pt}"

while [ $# -gt 0 ]; do
    case "$1" in
        --ckpt)    CKPT="$2"; shift 2 ;;
        --pockets) POCKETS="$2"; shift 2 ;;
        --num)     N="$2"; shift 2 ;;
        --batch)   BATCH="$2"; shift 2 ;;
        --seed)    SEED="$2"; shift 2 ;;
        --result)  RESULT="$2"; shift 2 ;;
        --data)    DATA="$2"; shift 2 ;;
        --split)   SPLIT="$2"; shift 2 ;;
        -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
[ -n "$CKPT" ] || { echo "--ckpt <file> is required; fetch it with tools/download_archive.py" >&2; exit 2; }
[ -f "$CKPT" ] || { echo "no such checkpoint: $CKPT" >&2; exit 2; }
[ -f "$DATA" ] || { echo "no such dataset: $DATA (see data/README.md)" >&2; exit 2; }

PY="${PY:-python}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

FIRST="${POCKETS%%-*}"; LAST="${POCKETS##*-}"
[ "$FIRST" = "$POCKETS" ] && LAST="$FIRST"

echo "ckpt=$CKPT"
echo "pockets=$FIRST..$LAST  molecules/pocket=$N  batch=$BATCH  seed=$SEED"
echo "result=$RESULT"
"$PY" -c "import torch; print(f'gpu: {torch.cuda.get_device_name() if torch.cuda.is_available() else \"none (CPU: very slow)\"}')" 2>/dev/null || true
echo

for i in $(seq "$FIRST" "$LAST"); do
    if [ -f "$RESULT/id$i/result_0.pt" ]; then
        echo "pocket $i: already sampled, skipping"
        continue
    fi
    echo "=== pocket $i ==="
    "$PY" bin/sample.py \
        --ckpt "$CKPT" --data_path "$DATA" --split_path "$SPLIT" \
        --data_id "$i" --num_samples "$N" --batch_size "$BATCH" --seed "$SEED" \
        --result_path "$RESULT" || {
            echo "pocket $i failed; continuing with the next one" >&2
        }
done

echo
echo "Sampling done. Next: bash bin/measure.sh --samples $RESULT --tag mymodel --pockets $POCKETS"
