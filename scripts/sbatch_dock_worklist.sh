#!/bin/bash
#SBATCH -J pf_dock
#SBATCH -o logs_slurm/pf_dock_%A_%a.out
#SBATCH -e logs_slurm/pf_dock_%A_%a.err
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -t 24:00:00
#
# Universal worker for the pose-fidelity docking runs. One array task chews through its stride
# of a flat (model, pocket) work list built by scripts/build_worklist.py.
#
# Engines:
#   smina       smina, Vina scoring,    --autobox_ligand <crystal ligand>
#   vinardo     smina, Vinardo scoring, same box            (cross-scoring-function control)
#   gnina       gnina, CNN rescoring,   same box            (needs a GPU: --gres=gpu:1)
#   vina_meeko  AutoDock Vina via the meeko-prepared PDBQT  (exact atom mapping, ref/genbbox box)
#
# Modes (MODE=, default dock -- an unset MODE reproduces the pre-2026-08 command line exactly,
# which is what keeps the already-measured smina/gnina outputs protocol-compatible):
#   dock    full re-search in the reference-ligand box  -> pocketNNN_dock.sdf
#   min     --minimize, relax in place, no box          -> pocketNNN_min.sdf
#   score   --score_only, no coordinate moved, no box   -> pocketNNN_score.sdf
# min/score take no box on purpose: that is what scripts/sbatch_eval_smina.sh has always done,
# and it is what makes them box-independent (the cleanest cross-scoring-function comparison).
# vina_meeko ignores MODE -- its worker runs score_only/minimize/dock itself in one pass.
#
# WHY -c 1: smina's --cpu parallelises over exhaustiveness runs and scales sublinearly, so the
# same pool of CPUs finishes more molecules as many 1-CPU tasks than as few 8-CPU tasks.
# dell_cpu has 208 CPUs total, which is the real ceiling on concurrency.
#
# Preemption safety: every engine writes <out>.tmp and only then renames onto <out>. A killed
# task therefore leaves either nothing or a complete file -- never a truncated one that the
# next build_worklist.py would mistake for finished work. That is what makes -q big_qos safe.
#
# Usage:
#   WL=results/pose_fidelity/worklist_smina_ref.tsv \
#     sbatch -p dell_cpu -q cpu_qos -a 0-199 scripts/sbatch_dock_worklist.sh
#   WL=results/pose_fidelity/worklist_gnina_ref.tsv ENGINE=gnina \
#     sbatch -p suma_rtx4090 -q base_qos --gres=gpu:1 -a 0-7 scripts/sbatch_dock_worklist.sh
#   WL=results/pose_fidelity/worklist_vinardo_min_ref.tsv ENGINE=vinardo MODE=min \
#     sbatch -p dell_cpu -q cpu_qos -a 0-99 scripts/sbatch_dock_worklist.sh
set -uo pipefail
cd "${REPO:-${SLURM_SUBMIT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}}"

WL=${WL:?set WL=<worklist.tsv>}
ENGINE=${ENGINE:-$(basename "$WL" | cut -d_ -f2)}
MODE=${MODE:-dock}
SMINA=${SMINA:-$(command -v smina || echo smina)}   # external binary; see envs/README.md
GNINA=${GNINA:-$(command -v gnina || echo gnina)}   # external binary; see envs/README.md
MEEKOPY=${MEEKOPY:-${CONDA_BASE:-$HOME/anaconda3}/envs/ged-vina/bin/python}
EXH=${EXH:-8}
VINA_EXH=${VINA_EXH:-16}
SEED=${SEED:-42}
CPUS=${SLURM_CPUS_PER_TASK:-1}
NTASKS=${SLURM_ARRAY_TASK_COUNT:-1}
TASKID=${SLURM_ARRAY_TASK_ID:-0}
mkdir -p logs_slurm

# validate up front: the per-row case below runs inside a pipeline subshell, where `exit 2` would
# only kill the subshell and the task would still report COMPLETE.
case "$MODE" in dock|min|score) ;; *) echo "unknown MODE=$MODE (dock|min|score)" >&2; exit 2 ;; esac

echo "worker: engine=$ENGINE mode=$MODE task=$TASKID/$NTASKS list=$WL"
n_done=0; n_skip=0; n_fail=0

# stride the list; NR==1 is the header
tail -n +2 "$WL" | tr -d '\r' | awk -v t="$TASKID" -v n="$NTASKS" 'NR % n == t' |
while IFS=$'\t' read -r tag dir pocket receptor ref out box sample lig; do
    [ -n "${out:-}" ] || continue
    mkdir -p "$(dirname "$out")"
    t=$(printf '%03d' "$pocket")
    # lig comes from the work list so a capped subset (sdfN/) is used identically by every engine
    sdf="${lig:-$dir/sdf/pocket$t.sdf}"
    [ -f "$sdf" ] || { echo "[skip] missing $sdf"; continue; }

    # Resume test MUST match build_worklist.py::_complete(), i.e. one result per submitted
    # molecule -- not merely "non-empty". With a bare `[ -s "$out" ]` the two scripts disagree:
    # the builder correctly queues a SHORT output for repair and the worker then skips it, so a
    # truncated pocket is queued forever and repaired never. That is how molcraft gnina
    # pocket063 sat at 58/100 molecules across several resubmissions (legacy file from
    # sbatch_eval_gnina.sh, which predates the promotion gate).
    if [ -s "$out" ]; then
        want_r=$(grep -c '^\$\$\$\$' "$sdf" 2>/dev/null || echo 0)
        if [ "$ENGINE" = "vina_meeko" ]; then
            # its output is a CSV, not an SDF: one data row per molecule, minus the header.
            # Counting '$$$$' here would always return 0 and re-run every finished pocket.
            have_r=$(($(wc -l < "$out") - 1))
        else
            have_r=$(grep -c '^\$\$\$\$' "$out" 2>/dev/null || echo 0)
        fi
        if [ "$have_r" -ge "$want_r" ]; then n_skip=$((n_skip+1)); continue; fi
        echo "[redo] $tag p$t $ENGINE $MODE: existing output has $have_r/$want_r -- regenerating"
        # vina_meeko writes its own file; for the SDF engines the stale short file would
        # otherwise survive the `mv` only if the new run also fails, which is what we want.
    fi
    # The scratch name MUST keep the .sdf extension: smina/gnina pick the output format from the
    # extension and abort with exit 255 on an unknown one, so "$out.tmp" silently produces
    # nothing. Leading dot keeps it out of the way of glob-based aggregation.
    # It is also tagged with job+task id: when capacity is added by submitting a SECOND array
    # over the same engine, two workers must never share a scratch path -- that would interleave
    # two SDF streams into one corrupt file that still looks non-empty.
    tmp="$(dirname "$out")/.$(basename "$out" .sdf).partial.${SLURM_JOB_ID:-0}_${TASKID}.sdf"
    # a preempted worker must not leave a half-written file behind that looks like a result
    trap 'rm -f "$tmp"' EXIT

    # MODE decides the search; ENGINE decides the scoring function. Keeping them orthogonal is
    # what makes smina-Vina vs smina-Vinardo a single-variable swap (same binary, same search,
    # same box, same seed) rather than two differently-configured runs.
    case "$MODE" in
      dock)  MODEARGS=(--autobox_ligand "$ref" --exhaustiveness "$EXH" --num_modes 1 --seed "$SEED") ;;
      min)   MODEARGS=(--minimize) ;;
      score) MODEARGS=(--score_only) ;;
      *)     echo "unknown MODE=$MODE" >&2; exit 2 ;;
    esac

    case "$ENGINE" in
      smina)
        "$SMINA" -r "$receptor" -l "$sdf" "${MODEARGS[@]}" --cpu "$CPUS" \
            -o "$tmp" > /dev/null 2>&1
        ;;
      vinardo)
        "$SMINA" -r "$receptor" -l "$sdf" --scoring vinardo "${MODEARGS[@]}" --cpu "$CPUS" \
            -o "$tmp" > /dev/null 2>&1
        ;;
      gnina)
        "$GNINA" -r "$receptor" -l "$sdf" "${MODEARGS[@]}" \
            -o "$tmp" > /dev/null 2>&1
        ;;
      vina_meeko)
        # this engine writes its own CSV + pose SDFs atomically, so it takes --out directly
        lim=""; [ "${sample:-0}" != "0" ] && lim="--limit $sample"
        # shellcheck disable=SC2086
        "$MEEKOPY" scripts/vina_meeko_dock.py --dir "$dir" --pocket "$pocket" \
            --box "$box" --exhaustiveness "$VINA_EXH" --seed "$SEED" --out "$out" $lim \
            > /dev/null 2>&1
        rm -f "$tmp"
        ;;
      *) echo "unknown ENGINE=$ENGINE" >&2; exit 2 ;;
    esac

    if [ "$ENGINE" != "vina_meeko" ]; then
        # `-s` only proves non-empty. gnina/smina can ABORT partway through a multi-molecule
        # ligand file (observed: gnina core-dumped after 23 of 30 poses), leaving a truncated
        # but non-empty scratch file. Promoting that marks the unit permanently "done" with
        # silently missing molecules. Require the SDF record terminator, and require at least
        # as many poses as molecules submitted, before committing.
        want=$(grep -c '^\$\$\$\$' "$sdf" 2>/dev/null || echo 0)
        got=$(grep -c '^\$\$\$\$' "$tmp" 2>/dev/null || echo 0)
        if [ -s "$tmp" ] && [ "$got" -ge "$want" ]; then
            mv -f "$tmp" "$out"
        else
            # keep the evidence next to the unit instead of deleting it silently
            [ -s "$tmp" ] && mv -f "$tmp" "$out.CRASHED_${got}of${want}" || rm -f "$tmp"
            echo "[CRASH] $tag p$t $ENGINE produced $got/$want poses -- NOT promoted"
        fi
    fi
    trap - EXIT

    if [ -s "$out" ]; then
        n_done=$((n_done+1)); echo "[ok]   $tag p$t $ENGINE"
    else
        n_fail=$((n_fail+1)); echo "[FAIL] $tag p$t $ENGINE (no output)"
    fi
done

echo "PF_DOCK_TASK_${TASKID}_COMPLETE engine=$ENGINE mode=$MODE"
