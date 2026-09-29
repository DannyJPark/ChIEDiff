#!/usr/bin/env bash
# Generated molecules -> docking, interaction profiling and physical checks.
#
# Runs LOCALLY by default: no Slurm, no cluster, one GPU (or none -- only the sampler needs
# one; every measurement here is CPU work). Pass --slurm to fan the same steps out as job
# arrays instead.
#
#   bash bin/measure.sh --samples results/generated --tag mymodel          # local, sequential
#   bash bin/measure.sh --samples results/generated --tag mymodel --pockets 0-4   # a subset
#   bash bin/measure.sh --samples results/generated --tag mymodel --slurm   # cluster
#
# SCALE WARNING. The published benchmark is 9 models x 100 pockets x ~100 molecules, measured
# by four instruments. On one workstation that is weeks, and it is not what a reviewer needs
# to do: `make tables` already reproduces every published number from the aggregate CSVs in
# this repository. Use this script to confirm the measurement pipeline runs, on a handful of
# pockets, or to measure a model of your own. --pockets defaults to 0-4 for exactly that
# reason; pass 0-99 for the full set.
#
# Each step skips work whose output already exists, so an interrupted run resumes.
set -uo pipefail
cd "$(dirname "$0")/.."

CONDA_BASE="${CONDA_BASE:-$HOME/anaconda3}"
KG="${KG:-$CONDA_BASE/envs/ged-train/bin/python}"
PC="${PC:-$CONDA_BASE/envs/ged-posecheck/bin/python}"
PB="${PB:-$CONDA_BASE/envs/ged-posebusters/bin/python}"
VN="${VN:-$CONDA_BASE/envs/ged-vina/bin/python}"

SAMPLES=""; TAG=""; POCKETS="0-4"; USE_SLURM=0
RECEPTORS="${RECEPTORS:-data/test_set_chainfix}"

while [ $# -gt 0 ]; do
    case "$1" in
        --samples)   SAMPLES="$2"; shift 2 ;;
        --tag)       TAG="$2"; shift 2 ;;
        --pockets)   POCKETS="$2"; shift 2 ;;
        --receptors) RECEPTORS="$2"; shift 2 ;;
        --slurm)     USE_SLURM=1; shift ;;
        -h|--help)   sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
[ -n "$SAMPLES" ] || { echo "--samples <dir> is required (the sampler's output)" >&2; exit 2; }
[ -n "$TAG" ] || { echo "--tag <name> is required (names the output under eval_out/)" >&2; exit 2; }

OUT="eval_out/$TAG"
step() { echo; echo "=== $* ==="; }
need() { [ -x "$1" ] || { echo "not executable: $1" >&2; echo "create the environments in envs/ first" >&2; exit 1; }; }

echo "samples=$SAMPLES  tag=$TAG  pockets=$POCKETS  receptors=$RECEPTORS"
if [ "$USE_SLURM" = 1 ]; then
    command -v sbatch >/dev/null || { echo "--slurm given but sbatch is not on PATH" >&2; exit 1; }
    echo "mode=slurm"
else
    echo "mode=local (sequential; pass --slurm on a cluster)"
fi
[ -d "$RECEPTORS" ] || {
    echo
    echo "receptors not found: $RECEPTORS"
    echo "Fetch them from the data archive (README section 2), or regenerate the chain-remapped"
    echo "copies from your own CrossDocked2020 download:"
    echo "    python3 scripts/make_chainfix_receptors.py"
    exit 1
}

# ---- 1. consolidate + export -------------------------------------------------------
# Cheap and sequential either way, so it never goes to Slurm.
need "$KG"
step "consolidate the sampler output into one grouped file"
$KG scripts/consolidate_docking.py --base "$SAMPLES" --out "results/${TAG}_gen/${TAG}_vina_docked.pt" || exit 1

step "export per-pocket SDFs (assigns the p<NNN>_m<MMMM> names)"
$KG scripts/eval_export_sdf.py --pt "results/${TAG}_gen/${TAG}_vina_docked.pt" --out "$OUT" || exit 1

# ---- 2. the four instruments -------------------------------------------------------
# All CPU. Locally they run one after another; with --slurm each becomes an array.
if [ "$USE_SLURM" = 1 ]; then
    step "submitting measurement arrays"
    echo "  Each instrument has its own launcher under slurm/. They are independent, so submit"
    echo "  them together and aggregate afterwards with --dependency=afterok:"
    for s in slurm/eval_posecheck.sbatch slurm/eval_posebusters.sbatch slurm/eval_plip.sbatch slurm/dock_vina.sbatch; do
        [ -f "$s" ] && echo "    sbatch -p <partition> -q <qos> --export=ALL,TAG=$TAG $s"
    done
    echo
    echo "  Then: bash bin/remeasure.sh"
    exit 0
fi

step "docking (Vina, SMINA/Vinardo, GNINA)"
echo "  External binaries. smina and gnina must be on PATH; see envs/README.md."
if command -v smina >/dev/null && command -v gnina >/dev/null; then
    need "$KG"
    $KG scripts/dock_generated_ligands.py --sdf_dir "$OUT/sdf" --out "$OUT" --pockets "$POCKETS" \
        || echo "  docking reported a failure; continuing so the other instruments still run"
else
    echo "  SKIPPED: smina and/or gnina not found on PATH."
fi

step "PoseCheck strain and clashes, and ProLIF fingerprints"
echo "  Absolute interpreter path on purpose: this environment's RDKit must not shadow the"
echo "  training environment's. See envs/README.md."
if [ -x "$PC" ]; then
    $PC scripts/eval_posecheck.py --sdf_dir "$OUT/sdf" --receptors "$RECEPTORS" \
        --out "$OUT/posecheck" --pockets "$POCKETS" --workers "${WORKERS:-4}" \
        || echo "  PoseCheck reported a failure; continuing"
    $PC scripts/eval_prolif_counts.py --sdf_dir "$OUT/sdf" --receptors "$RECEPTORS" \
        --out "$OUT/prolif_counts" --pockets "$POCKETS" --workers "${WORKERS:-4}" \
        || echo "  ProLIF reported a failure; continuing"
else
    echo "  SKIPPED: $PC not found."
fi

step "PoseBusters validity checks"
echo "  One pathological molecule can stall a pocket for 15+ minutes in the energy check, so"
echo "  the worker parallelises per molecule with a timeout, single-threaded per molecule."
if [ -x "$PB" ]; then
    EVAL_ROOT=eval_out MODELS="$TAG" PB_THREADS=1 $PB scripts/pb_mol_worker.py \
        || echo "  PoseBusters reported a failure; continuing"
else
    echo "  SKIPPED: $PB not found."
fi

step "PLIP interaction profiling"
if [ -x "$KG" ]; then
    $KG scripts/plip_interactions.py --sdf_dir "$OUT/sdf" --receptors "$RECEPTORS" \
        --out "eval_plip/$TAG" --pockets "$POCKETS" \
        || echo "  PLIP reported a failure; continuing"
else
    echo "  SKIPPED."
fi

echo
echo "Measurement done for tag '$TAG', pockets $POCKETS."
echo "Aggregate with:  bash bin/remeasure.sh"
echo
echo "Note that bin/remeasure.sh aggregates the FULL nine-model roster and will report the"
echo "models it cannot find. To compare your own model against the published ones you also"
echo "need their measurement outputs; see configs/models.json and results/README.md."
