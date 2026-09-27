#!/usr/bin/env python3
"""Assert the provisional model name appears only where a rename expects to find it.

The human-readable model name is not final. The package name is function-based
(`gated_energy_diffusion`) and does not change, so renaming the model should touch a
bounded set of files. This script is what keeps that set bounded: without it, the name
leaks into docstrings, log messages and config comments, and the next rename becomes a
repository-wide search-and-replace with no way to tell a real occurrence from a stale one.

Allowed sites are listed in ALLOWED below. Everything else is a failure.

Also rejects names from the working repository this release was distilled from
(`TheSelective`, `papersubmissiononly-abc`), which must not survive anywhere.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# The provisional model name, and the only places it may appear.
MODEL_NAME = "ChIEDiff"

ALLOWED = {
    # The single in-code definition.
    "gated_energy_diffusion/__init__.py",
    # Human-facing documents: the name is the point.
    "README.md",
    "NOTICE.md",
    "CITATION.cff",
    # The project URL is the repository URL, which necessarily carries the repository
    # name. Renaming the model means renaming the repository, and this line follows.
    "pyproject.toml",
    # The LaTeX macro definitions the table harness expands.
    "tables/tables_main.tex",
    "tables/tables_supplementary.tex",
    # Provenance records describe history and may quote old paths.
    "docs/provenance.md",
    "docs/migration_manifest.tsv",
}

# Names that must not appear anywhere at all.
FORBIDDEN = {
    "TheSelective": "the working repository's old model name",
    "papersubmissiononly-abc": "a throwaway GitHub account from the working repository",
    "theselective": "a conda environment that never existed",
}

# This file necessarily spells out every name it searches for.
SELF = {"tools/check_no_model_name.py"}

SKIP_DIRS = {".git", "__pycache__", "archive", "node_modules", ".venv"}
TEXT_SUFFIXES = {
    ".py", ".sh", ".yml", ".yaml", ".json", ".tex", ".md", ".cff", ".toml",
    ".cfg", ".txt", ".tsv", ".csv", ".bib", ".sbatch", ".make", "",
}


def tracked_files() -> list[Path]:
    """Prefer git's index so untracked scratch files do not fail the hook."""
    try:
        out = subprocess.run(
            ["git", "-C", str(ROOT), "ls-files"],
            capture_output=True, text=True, check=True,
        ).stdout.split("\n")
        files = [ROOT / p for p in out if p]
        if files:
            return files
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    # Fresh repository with an empty index: walk the tree instead.
    return [
        p for p in ROOT.rglob("*")
        if p.is_file() and not any(part in SKIP_DIRS for part in p.parts)
    ]


def main() -> int:
    name_re = re.compile(re.escape(MODEL_NAME), re.IGNORECASE)
    failures: list[str] = []

    for path in tracked_files():
        if not path.is_file():
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        rel = path.relative_to(ROOT).as_posix()
        if rel in SELF:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue

        for lineno, line in enumerate(text.splitlines(), 1):
            for bad, why in FORBIDDEN.items():
                if bad in line:
                    failures.append(f"{rel}:{lineno}: forbidden name {bad!r} ({why})")
            if rel not in ALLOWED and name_re.search(line):
                failures.append(
                    f"{rel}:{lineno}: model name {MODEL_NAME!r} outside the allowed set"
                )

    if failures:
        print("Model-name guard failed:\n", file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        print(
            f"\nThe model name belongs only in:\n"
            + "".join(f"  {a}\n" for a in sorted(ALLOWED))
            + "\nIf a new site is genuinely needed, add it to ALLOWED in this script and "
              "say why in the commit message.",
            file=sys.stderr,
        )
        return 1

    print(f"model-name guard: ok ({MODEL_NAME!r} confined to {len(ALLOWED)} declared sites)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
