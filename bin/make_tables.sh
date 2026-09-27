#!/usr/bin/env bash
# Shipped CSVs -> the 22 generated tables. About a minute, CPU only, no downloads.
#
#   bash bin/make_tables.sh          inject into the LaTeX targets
#   bash bin/make_tables.sh --dry    render to stdout, touch nothing
#
# The generator and the renderer use only the Python standard library, so this runs in any
# Python 3.9+; PY overrides the interpreter.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-python3}"
DRY=""
[ "${1:-}" = "--dry" ] && DRY="--dry"

echo "== integrity: the shipped inputs are the published bytes =="
( cd results && sha256sum -c MANIFEST.sha256 >/dev/null ) \
  && echo "   results/MANIFEST.sha256: all $(grep -c . results/MANIFEST.sha256) files match"

echo
echo "== registry invariants =="
"$PY" scripts/check_registry.py --shipped >/dev/null \
  && echo "   check_registry --shipped: OK"

echo
echo "== generate =="
if [ -n "$DRY" ]; then
    "$PY" paper/tables/build_revision_tables.py --dry
    echo
    echo "dry run: nothing written."
    exit 0
fi
"$PY" paper/tables/build_revision_tables.py >/dev/null
echo "   22 tables injected into paper/main.tex and paper/supplementary.tex"

echo
echo "== every printed cell equals its source CSV =="
"$PY" scripts/check_paper_tables.py | tail -1

echo
echo "== row order and registry membership =="
"$PY" scripts/check_rosters.py --shipped | tail -2

echo
echo "Done. The LaTeX targets under paper/ are harness documents, not the manuscript;"
echo "see paper/README.md."
