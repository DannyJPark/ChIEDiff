#!/usr/bin/env bash
# Generated molecules -> docking, interaction profiling and physical checks.
#
#   bash bin/measure.sh --pt archive/chiediff_generated_molecules.pt --tag chiediff
#   bash bin/measure.sh --pt mymodel.pt --tag mymodel --pockets 0-4
#   bash bin/measure.sh --base results/samples --tag mymodel     # group the sampler output first
#
# Everything here is CPU work and runs sequentially on one machine. Each step skips work whose
# output already exists, so an interrupted run resumes.
#
# Scale: the full benchmark is 100 pockets x ~100 molecules through five instruments, which is
# days on one workstation. --pockets defaults to 0-4 so that a run confirms the pipeline works;
# pass 0-99 for the whole test set.
#
# The instruments live in four conda environments that must not be merged, so each step calls
# its own interpreter by absolute path. See envs/README.md.
set -uo pipefail
cd "$(dirname "$0")/.."

CONDA_BASE="${CONDA_BASE:-$HOME/anaconda3}"
KG="${KG:-$CONDA_BASE/envs/ged-train/bin/python}"
PC="${PC:-$CONDA_BASE/envs/ged-posecheck/bin/python}"
PB="${PB:-$CONDA_BASE/envs/ged-posebusters/bin/python}"
VN="${VN:-$CONDA_BASE/envs/ged-vina/bin/python}"

PT=""; BASE=""; TAG=""; POCKETS="0-4"; WORKERS="${WORKERS:-4}"
RECEPTORS="${RECEPTORS:-data/test_set_chainfix}"

while [ $# -gt 0 ]; do
    case "$1" in
        --pt)        PT="$2"; shift 2 ;;
        --base)      BASE="$2"; shift 2 ;;
        --tag)       TAG="$2"; shift 2 ;;
        --pockets)   POCKETS="$2"; shift 2 ;;
        --receptors) RECEPTORS="$2"; shift 2 ;;
        --workers)   WORKERS="$2"; shift 2 ;;
        -h|--help)   sed -n '2,17p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
[ -n "$TAG" ] || { echo "--tag <name> is required (names the output under eval_out/)" >&2; exit 2; }
[ -n "$PT$BASE" ] || { echo "pass --pt <grouped .pt> or --base <sampler output dir>" >&2; exit 2; }

# --pockets accepts 0-4 or 3,7,11; the workers take a space-separated list.
case "$POCKETS" in
    *-*) PK_LIST=$(seq "${POCKETS%-*}" "${POCKETS#*-}" | tr '\n' ' ') ;;
    *)   PK_LIST=$(echo "$POCKETS" | tr ',' ' ') ;;
esac

OUT="eval_out/$TAG"
step() { echo; echo "=== $* ==="; }
have() { [ -x "$1" ] || { echo "  SKIPPED: no interpreter at $1 (create the environments in envs/)"; return 1; }; }

echo "tag=$TAG  pockets=$POCKETS  receptors=$RECEPTORS  workers=$WORKERS"
[ -d "$RECEPTORS" ] || {
    echo
    echo "receptors not found: $RECEPTORS"
    echo "Fetch them from the data archive (README section 2), or regenerate the chain-remapped"
    echo "copies from your own CrossDocked2020 download:"
    echo "    python scripts/make_chainfix_receptors.py --test_set data/test_set"
    exit 1
}

# ---- 1. the sampler's output -> one grouped .pt ------------------------------------
if [ -z "$PT" ]; then
    have "$KG" || exit 1
    PT="results/${TAG}_grouped.pt"
    step "group the sampler output into one file"
    $KG scripts/group_samples.py --base "$BASE" --out "$PT" || exit 1
fi

# ---- 2. per-pocket SDFs, which every instrument below reads ------------------------
have "$KG" || exit 1
step "export per-pocket SDFs (assigns the p<NNN>_m<MMMM> names)"
$KG scripts/eval_export_sdf.py --pt "$PT" --out "$OUT" || exit 1

# ---- 3. docking -------------------------------------------------------------------
step "AutoDock Vina: score_only, minimize and dock, with an exact meeko atom map"
if have "$VN"; then
    for p in $PK_LIST; do
        $VN scripts/vina_meeko_dock.py --dir "$OUT" --pocket "$p" --box ref \
            || echo "  pocket $p failed; continuing"
    done
fi

step "smina, Vinardo and gnina: re-dock and re-minimise"
echo "  External binaries, installed from their own upstreams; see envs/README.md."
for eng in smina vinardo gnina; do
    if command -v "$eng" >/dev/null || { [ "$eng" = vinardo ] && command -v smina >/dev/null; }; then
        for mode in dock min; do
            python3 scripts/dock_engines.py --dir "$OUT" --engine "$eng" --mode "$mode" \
                --pockets $PK_LIST --cpus "$WORKERS" \
                || echo "  $eng $mode reported a failure; continuing"
        done
    else
        echo "  SKIPPED $eng: not on PATH."
    fi
done

step "generated-pose to engine-pose RMSD"
have "$KG" && $KG scripts/pose_rmsd.py --dir "$OUT" --jobs "$WORKERS" \
    || echo "  RMSD step skipped or failed; continuing"

# ---- 4. interactions and physical validity ----------------------------------------
step "PoseCheck strain and clashes, and ProLIF fingerprints"
echo "  Absolute interpreter path on purpose: this environment's RDKit must not shadow the"
echo "  training environment's. ProLIF must be 2.2.0; see envs/README.md."
if have "$PC"; then
    for p in $PK_LIST; do
        $PC scripts/eval_posecheck.py --dir "$OUT" --pocket "$p" --workers "$WORKERS" \
            || echo "  PoseCheck pocket $p failed; continuing"
        $PC scripts/eval_prolif_counts.py --dir "$OUT" --pocket "$p" --workers "$WORKERS" \
            || echo "  ProLIF pocket $p failed; continuing"
    done
fi

step "PoseBusters validity, one process per molecule"
echo "  The energy check runs an ETKDG ensemble, and one pathological molecule can take"
echo "  hundreds of seconds, so the unit is a molecule and each one has a timeout."
if have "$PB"; then
    $PB scripts/eval_posebusters.py --dir "$OUT" --pockets $PK_LIST --workers "$WORKERS" \
        || echo "  PoseBusters reported a failure; continuing"
    $PB scripts/eval_pb_distance.py --dir "$OUT" --pockets $PK_LIST \
        || echo "  PoseBusters distance checks reported a failure; continuing"
fi

step "PLIP interaction profiling"
if have "$KG"; then
    $KG scripts/plip_interactions.py --pt "$PT" --tag "$TAG" --out "eval_plip/$TAG" \
        --test_set "$RECEPTORS" --pockets "$POCKETS" \
        || echo "  PLIP reported a failure; continuing"
fi

step "MMFF94s internal strain, and the chemical properties"
if have "$KG"; then
    $KG scripts/strain_energy.py --dir "$OUT" --jobs "$WORKERS" --pockets $PK_LIST \
        || echo "  strain step failed; continuing"
    $KG scripts/chem_props.py --dir "$OUT" --pockets $PK_LIST \
        || echo "  property step failed; continuing"
fi

echo
echo "Done: tag '$TAG', pockets $POCKETS."
echo "Per-molecule results are under $OUT/ and eval_plip/$TAG/."
echo "reproduction.ipynb reads the published aggregates; point its CSV paths at these"
echo "directories to compare your own model against them."
