#!/usr/bin/env python3
"""Fail if an unreplaced placeholder would ship.

Two levels, because a placeholder that is fine during development is not fine at release:

  WARN  (exit 0)  -- placeholders that are expected until the archive exists:
                     the Zenodo DOIs. Reported every commit so they stay visible.
  FAIL  (exit 1)  -- TODO/FIXME/XXX markers and obviously fake identifiers.

`--release` promotes the WARN set to FAIL. `bin/check.sh --release` uses that, so the
final pre-submission check refuses to pass while a DOI is still `XXXXXXX`.

This exists because the manuscript that accompanies this repository once compiled a
placeholder NRF grant number (`RS-XXXXX`) into a PDF that read as a real citation. The
lesson generalises: a placeholder with no gate behind it eventually ships.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Expected until the Zenodo records exist. Visible every commit, fatal only at release.
DEFERRED = [
    (re.compile(r"zenodo\.X{4,}", re.IGNORECASE), "Zenodo DOI not yet assigned"),
    (re.compile(r"10\.5281/zenodo\.Y{4,}", re.IGNORECASE), "Zenodo DOI not yet assigned"),
    (re.compile(r"github\.com/<user>"), "GitHub owner not yet set"),
]

# Never acceptable.
FATAL = [
    (re.compile(r"\bTODO\b"), "TODO marker"),
    (re.compile(r"\bFIXME\b"), "FIXME marker"),
    (re.compile(r"\bXXX\b(?!X)"), "XXX marker"),
    (re.compile(r"\bRS-X{3,}\b"), "placeholder grant number"),
    (re.compile(r"ghp_[A-Za-z0-9]{20,}"), "GitHub personal access token"),
    (re.compile(r"github_pat_[A-Za-z0-9_]{20,}"), "GitHub personal access token"),
]

# This file necessarily contains the patterns it searches for.
SELF = {"tools/check_placeholders.py"}

# Copied verbatim from upstream, with their sha256 recorded in docs/migration_manifest.tsv.
# A work marker inside inherited third-party code is not our unresolved work, and editing it
# would break both the "copied unmodified" claim and the recorded hash. Excluded from the
# work-marker scan only -- the credential patterns still apply everywhere.
UPSTREAM_VERBATIM = {
    "gated_energy_diffusion/utils/reconstruct.py",   # liGAN, GPL-2.0
    "gated_energy_diffusion/utils/evaluation/sascorer.py",
}

SKIP_DIRS = {".git", "__pycache__", "archive", ".venv", "node_modules"}
TEXT_SUFFIXES = {
    ".py", ".sh", ".yml", ".yaml", ".json", ".tex", ".md", ".cff", ".toml",
    ".cfg", ".txt", ".tsv", ".bib", ".sbatch", "",
}


def tracked_files() -> list[Path]:
    try:
        out = subprocess.run(
            ["git", "-C", str(ROOT), "ls-files", "-c", "-o", "--exclude-standard"],
            capture_output=True, text=True, check=True,
        ).stdout.split("\n")
        files = [ROOT / p for p in out if p]
        if files:
            return files
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return [
        p for p in ROOT.rglob("*")
        if p.is_file() and not any(part in SKIP_DIRS for part in p.parts)
    ]


def scan(patterns):
    hits = []
    for path in tracked_files():
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        rel = path.relative_to(ROOT).as_posix()
        if rel in SELF:
            continue
        if rel in UPSTREAM_VERBATIM and patterns is FATAL:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            for rx, why in patterns:
                if rx.search(line):
                    hits.append((rel, lineno, why, line.strip()[:100]))
    return hits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--release", action="store_true",
        help="treat deferred placeholders (Zenodo DOIs, GitHub owner) as failures",
    )
    args = ap.parse_args()

    fatal = scan(FATAL)
    deferred = scan(DEFERRED)

    for rel, lineno, why, line in deferred:
        label = "FAIL" if args.release else "note"
        print(f"{label}: {rel}:{lineno}: {why}\n      {line}")

    for rel, lineno, why, line in fatal:
        print(f"FAIL: {rel}:{lineno}: {why}\n      {line}", file=sys.stderr)

    if fatal or (args.release and deferred):
        n = len(fatal) + (len(deferred) if args.release else 0)
        print(f"\nplaceholder guard: {n} blocking item(s)", file=sys.stderr)
        return 1

    if deferred:
        print(
            f"\nplaceholder guard: ok, {len(deferred)} deferred placeholder(s) remain "
            "(fatal under --release)"
        )
    else:
        print("placeholder guard: ok, nothing outstanding")
    return 0


if __name__ == "__main__":
    sys.exit(main())
