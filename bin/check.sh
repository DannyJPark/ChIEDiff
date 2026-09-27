#!/usr/bin/env bash
# Every gate, no regeneration.
#
#   bash bin/check.sh            development: unassigned DOIs are a note
#   bash bin/check.sh --release  submission: unassigned DOIs are a failure
set -uo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-python3}"
REL=""
[ "${1:-}" = "--release" ] && REL="--release"
rc=0

run() {
    local label="$1"; shift
    printf '  %-34s ' "$label"
    if out=$("$@" 2>&1); then
        echo "PASS"
    else
        echo "FAIL"
        echo "$out" | tail -4 | sed 's/^/      /'
        rc=1
    fi
}

echo "== content guards =="
run "no model name outside its sites" "$PY" tools/check_no_model_name.py
if [ -n "$REL" ]; then
    run "no placeholders (release)" "$PY" tools/check_placeholders.py --release
else
    run "no placeholders" "$PY" tools/check_placeholders.py
fi

echo
echo "== data integrity =="
printf '  %-34s ' "results/MANIFEST.sha256"
if ( cd results && sha256sum -c MANIFEST.sha256 >/dev/null 2>&1 ); then echo "PASS"; else echo "FAIL"; rc=1; fi

echo
echo "== pipeline gates =="
run "registry invariants" "$PY" scripts/check_registry.py --shipped
run "registry internally consistent" "$PY" scripts/check_registry_sync.py
run "table row order" "$PY" scripts/check_rosters.py --shipped
run "docked() reproduces originals" "$PY" scripts/check_docked_equivalence.py
run "table cells equal their CSVs" "$PY" scripts/check_paper_tables.py

echo
echo "== package imports cleanly =="
printf '  %-34s ' "gated_energy_diffusion"
if out=$(PYTHONPATH="$PWD" "$PY" -c "
import gated_energy_diffusion as g
from gated_energy_diffusion.models.score_model import ScorePosNet3D
from gated_energy_diffusion.utils.reconstruct import reconstruct_from_generated
print(g.MODEL_NAME)
" 2>&1); then echo "PASS"; else echo "FAIL"; echo "$out" | tail -3 | sed 's/^/      /'; rc=1; fi

echo
if [ "$rc" = 0 ]; then
    echo "All gates passed."
    [ -z "$REL" ] && echo "(development mode: run with --release before submitting)"
else
    echo "At least one gate failed."
fi
exit "$rc"
