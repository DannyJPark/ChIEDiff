#!/usr/bin/env bash
# Raw measurement outputs -> the aggregate CSVs -> the tables.
#
# Tier 2 of the three reproduction tiers. Needs the measurement outputs on disk, either from
# `make measure` or from the data archive; `make tables` alone does not need any of this.
#
#   bash bin/remeasure.sh
#
# Order matters in three places, and each is load-bearing:
#
#  * eval_pb_distance.py --summarize imports build_pose_table.populations(), so the pose
#    table must be built first.
#  * build_master_table.py reads the other builders' outputs WITH the non-manuscript rows
#    still present. Run it on pruned inputs and its self-check fails, so pruning comes after
#    every builder.
#  * make_round2_consistency.py prints the pose-table n, so it runs after pruning; and
#    make_tables.sh consumes its output, so it runs before that.
#
# Environments differ per step. PoseCheck and PoseBusters are called by ABSOLUTE interpreter
# path rather than `conda activate`, so their RDKit (2026.03.x) cannot shadow the training
# environment's 2024.03.6. See envs/README.md.
set -uo pipefail
cd "$(dirname "$0")/.."

CONDA_BASE="${CONDA_BASE:-$HOME/anaconda3}"
KG="${KG:-$CONDA_BASE/envs/ged-train/bin/python}"
PC="${PC:-$CONDA_BASE/envs/ged-posecheck/bin/python}"
FIG="${FIG:-$CONDA_BASE/envs/ged-figures/bin/python}"
for p in "$KG" "$PC"; do
    [ -x "$p" ] || { echo "not executable: $p  (create the environments in envs/ first)" >&2; exit 1; }
done

step() { echo; echo "=== $* ==="; }

# ---- Tier 0: gates that must hold before anything aggregates -------------------------
step "canonical pocket order"
$KG scripts/check_pockets_equivalence.py --shipped || exit 1
step "docked() reproduces the originals"
$KG scripts/check_docked_equivalence.py || exit 1

# ---- Tier 1: the per-molecule join every family builds on ----------------------------
step "pose-fidelity master (one row per model x molecule)"
$KG scripts/aggregate_pose_fidelity.py || exit 1

# ---- Tier 1b: family builders --------------------------------------------------------
step "pose aggregates (strict and RMSD-complete)"
$KG scripts/build_pose_table.py || exit 1
step "PoseBusters distance summary"      # imports build_pose_table.populations()
$PC scripts/eval_pb_distance.py --summarize || exit 1
step "scoring table"
$KG scripts/build_main_table.py || exit 1
step "interaction profiles (strict)"
$KG scripts/build_nci_strict.py || exit 1
step "interaction summary"
$KG scripts/build_nci_summary.py --out results/comparison/f3_nci/nci_summary.txt || exit 1
step "PoseCheck summary"
$KG scripts/aggregate_posecheck.py || exit 1
step "calculated properties"
$KG scripts/build_property_tables.py || exit 1
step "ring-size distributions"
$KG scripts/build_ring_size_table.py || exit 1
step "bond-type Jensen-Shannon distances"
$KG scripts/build_bond_jsd_table.py || exit 1
step "pocket-specificity per molecule"
$KG scripts/build_delta_score.py || exit 1
step "ablation PoseCheck aggregate"
bash scripts/aggregate_posecheck_abl.sh || exit 1
step "ablation interaction summary"
$KG scripts/build_nci_summary_abl.py --out results/diagnostics/ablation_nci/nci_summary_abl.txt || exit 1

# ---- Tier 2: the master table, last, on unpruned inputs ------------------------------
step "master table"
# This builder writes its outputs and then exits non-zero on one self-check item that
# predates the release: the "Ours (Vina Only)" row has had empty f4 cells in every copy of
# master_table.csv, and the specificity table does not print that row. Tolerate exactly that
# string and stop on anything else -- a blanket `|| true` here would hide a real regression.
MT_LOG=$(mktemp)
$KG scripts/build_master_table.py 2>&1 | tee "$MT_LOG" || true
if grep -q "SELF-CHECK FAILED" "$MT_LOG"; then
    fails=$(sed -n '/SELF-CHECK FAILED/,$p' "$MT_LOG" | grep -E "^\s+\S" || true)
    if [ "$fails" != "  Ours (Vina Only) / f4: status 'ok' but every cell is empty" ]; then
        echo "build_master_table.py self-check failed beyond the known item:" >&2
        echo "$fails" >&2
        rm -f "$MT_LOG"; exit 1
    fi
    echo "  (known pre-existing item tolerated: Ours (Vina Only) / f4 empty)"
fi
rm -f "$MT_LOG"

# ---- Tier 3: restrict to the manuscript roster --------------------------------------
step "prune non-manuscript rows"
$KG scripts/prune_nonmanuscript_rows.py || exit 1
step "roster fixture"
$KG scripts/check_rosters.py --shipped || exit 1

# ---- Tier 4: the paired cohort audit, which reads the pruned pose-table n ------------
step "paired Vinardo cohort audit"
if [ -x "$FIG" ]; then
    $FIG paper/figure_build/make_round2_consistency.py || exit 1
else
    echo "  ged-figures not found; skipping. Its output (paper/data/round2_cohort_audit.csv)"
    echo "  ships with the release, so the tables still build -- but it will not be refreshed."
fi

# ---- Tier 5: the tables ------------------------------------------------------------
step "tables"
bash bin/make_tables.sh || exit 1

echo
echo "Aggregation complete. Regenerate the figures with the scripts in paper/figure_build/;"
echo "see paper/figure_build/figures/PROVENANCE.md for which one makes which."
