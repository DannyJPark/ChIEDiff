#!/usr/bin/env bash
# Generated molecules -> docking, interaction profiling and physical checks.
#
#   bash bin/measure.sh --pt archive/chiediff_generated_molecules.pt --tag chiediff
#   bash bin/measure.sh --pt mymodel.pt --tag mymodel --pockets 0-4
#   bash bin/measure.sh --base results/samples --tag mymodel     # group the sampler output first
#
# Everything here is CPU work and runs sequentially on one machine. The SMINA, Vinardo, GNINA and
# PoseBusters steps skip a pocket whose output already exists; the other steps run again.
#
# Scale: the full benchmark is 100 pockets x ~100 molecules through five instruments, which is
# days on one workstation. --pockets defaults to 0-4 so that a run confirms the pipeline works;
# pass 0-99 for the whole test set.
#
# The instruments live in four conda environments that must not be merged, so each step calls
# its own interpreter by absolute path. See envs/README.md.
set -uo pipefail
cd "$(dirname "$0")/.."

CONDA_BASE="${CONDA_BASE:-$(conda info --base 2>/dev/null || echo "$HOME/anaconda3")}"
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
        -h|--help)   sed -n '2,16p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
[ -n "$TAG" ] || { echo "--tag <name> is required (names the output under eval_out/)" >&2; exit 2; }
[ -n "$PT$BASE" ] || { echo "pass --pt <grouped .pt> or --base <sampler output dir>" >&2; exit 2; }

# --pockets accepts any comma-separated mixture of single indices and A-B ranges, such as
# "0-4", "3,7,11" or "0-4,9". The workers take a space-separated list of integers, while
# PLIP takes the string form unchanged.
PK_LIST=$(echo "$POCKETS" | tr ',' '\n' | while read -r part; do
    case "$part" in
        '')  : ;;
        *-*) seq "${part%-*}" "${part#*-}" || exit 1 ;;
        *)   echo "$part" ;;
    esac
done | tr '\n' ' ')
[ -n "${PK_LIST// /}" ] || { echo "could not read --pockets '$POCKETS'" >&2; exit 2; }

OUT="eval_out/$TAG"
step() { echo; echo "=== $* ==="; }
# A missing interpreter is a setup error, not a step to skip: skipping every instrument
# would end in a clean exit with no measurements. $CONDA_BASE relocates the four paths.
have() {
    [ -x "$1" ] && return 0
    echo "no interpreter at $1" >&2
    echo "Create the environments in envs/ first, or point CONDA_BASE at your conda" >&2
    echo "installation, or set KG, PC, PB and VN to the four interpreters directly." >&2
    exit 1
}

echo "tag=$TAG  pockets=$POCKETS  receptors=$RECEPTORS  workers=$WORKERS"
[ -d "$RECEPTORS" ] || {
    echo
    echo "receptors not found: $RECEPTORS"
    echo "Fetch them from the data archive (see data/README.md), or regenerate the chain-remapped"
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
# --test_set is the root the manifest's receptor and ref_ligand paths are built from, so
# every later step inherits it. It has to be the chain-remapped copies, or PoseCheck and
# ProLIF measure twelve pockets against a partial receptor and report no error.
$KG scripts/eval_export_sdf.py --pt "$PT" --out "$OUT" --test_set "$RECEPTORS" || exit 1
[ "$(wc -l < "$OUT/manifest.csv" 2>/dev/null || echo 0)" -gt 1 ] || {
    echo "the export matched no pocket. Check that $RECEPTORS holds <pocket>/<pdb>_rec.pdb" >&2
    echo "and the reference ligands: scripts/make_chainfix_receptors.py writes both." >&2
    exit 1
}

# ---- 3. docking -------------------------------------------------------------------
step "AutoDock Vina: score_only, minimize and dock, with an exact meeko atom map"
if have "$VN"; then
    for p in $PK_LIST; do
        $VN scripts/vina_meeko_dock.py --dir "$OUT" --pocket "$p" --box ref \
            || echo "  pocket $p failed; continuing"
    done
fi

step "AutoDock Vina: score_only in a box fitted to each generated molecule"
have "$KG" && $KG scripts/vina_genbox_score.py --pt "$PT" --dir "$OUT" --pockets $PK_LIST \
    || echo "  Vina genbox step failed; continuing"

step "smina, Vinardo and gnina: score, re-dock and re-minimise"
echo "  External binaries, installed from their own upstreams; see envs/README.md."
have_engine() {
    # vinardo is smina with another scoring function, so both come from $SMINA.
    case "$1" in
        gnina) [ -x "${GNINA:-}" ] || command -v gnina >/dev/null ;;
        *)     [ -x "${SMINA:-}" ] || command -v smina >/dev/null ;;
    esac
}
for eng in smina vinardo gnina; do
    if have_engine "$eng"; then
        for mode in dock min score; do
            $KG scripts/dock_engines.py --dir "$OUT" --engine "$eng" --mode "$mode" \
                --pockets $PK_LIST --cpus "$WORKERS" \
                || echo "  $eng $mode reported a failure; continuing"
        done
    else
        echo "  SKIPPED $eng: not on PATH and not named by \$SMINA or \$GNINA."
    fi
done

step "generated-pose to engine-pose RMSD"
have "$KG" && $KG scripts/pose_rmsd.py --dir "$OUT" --jobs "$WORKERS" \
    || echo "  RMSD step skipped or failed; continuing"

# ---- 4. interactions and physical validity ----------------------------------------
step "PoseCheck clashes and UFF strain, and ProLIF fingerprints"
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
    # --out is the parent: plip_interactions appends --tag to it itself.
    $KG scripts/plip_interactions.py --pt "$PT" --tag "$TAG" --out "eval_plip" \
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
