#!/usr/bin/env python3
"""Re-apply the 2026-09-15 manuscript-roster pruning after a builder restores the removed rows.

    python3 scripts/prune_nonmanuscript_rows.py [--dry]

On 2026-09-15 rows of models that the manuscript does not report were removed from the paper-read
comparison CSVs; the untouched copies and the id list are in results/comparison/_archive_extra_rows/.
Every builder under scripts/ regenerates those files from the sample .pt files and the registry,
so re-running one brings the rows back. This removes exactly those ids again, from the same files,
and nothing else: the header, the order of the kept rows, their cell text and the file's line
endings are left as the builder wrote them. Idempotent.
"""
import argparse
import csv
import io
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "results" / "comparison" / "_archive_extra_rows"
PRUNED = {"ar", "bind", "cvae", "decompdiff", "deepicl", "head1_dock_342k", "ligan", "phardiff",
          "pidiff_pub", "posonly", "typeonly"}


# The live file behind each archived copy. Explicit on purpose: results/comparison/ also holds
# dated backups (_archive/, _backup_*/, *.bak.*) with the same file names.
LIVE = {
    "bond_jsd_overall.csv": "f1_sbdd/bond_jsd/bond_jsd_overall.csv",
    "delta_score_overall.csv": "f4_delta_score/delta_score_overall.csv",
    "main_table.csv": "f1_sbdd/main_table/main_table.csv",
    "master_table.csv": "master_table.csv",
    "nci_summary_strict.csv": "f3_nci/nci_summary_strict.csv",
    "pose_stability_engineonly.csv": "f2_pose_fidelity/pose_stability/pose_stability_engineonly.csv",
    "pose_stability_strict.csv": "f2_pose_fidelity/pose_stability/pose_stability_strict.csv",
    "posecheck_summary.csv": "f5_pose_quality/posecheck_summary.csv",
    "property_overall.csv": "f1_sbdd/properties/property_overall.csv",
    "ring_size_overall.csv": "f1_sbdd/ring_size/ring_size_overall.csv",
}


def targets():
    """Each archived file mapped to its live path; the two lists must match exactly."""
    archived = {p.name for p in ARCHIVE.glob("*.csv")}
    if archived != set(LIVE):
        raise SystemExit(f"archive and LIVE map disagree: {sorted(archived ^ set(LIVE))}")
    live = {}
    for name, rel in LIVE.items():
        path = ROOT / "results" / "comparison" / rel
        if not path.is_file():
            raise SystemExit(f"{name}: live file missing at {path}")
        live[name] = path
    return live


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    for name, path in targets().items():
        raw = path.read_bytes()
        newline = "\r\n" if b"\r\n" in raw else "\n"
        reader = csv.reader(io.StringIO(raw.decode()))
        header = next(reader)
        if "id" not in header:
            print(f"  skip {path.relative_to(ROOT)}: no id column")
            continue
        col = header.index("id")
        rows = list(reader)
        keep = [r for r in rows if r[col] not in PRUNED]
        removed = sorted({r[col] for r in rows if r[col] in PRUNED})
        print(f"  {path.relative_to(ROOT)}: {len(rows)} -> {len(keep)} rows, removed {removed}")
        if removed and not a.dry:
            buf = io.StringIO()
            w = csv.writer(buf, lineterminator=newline)
            w.writerow(header)
            w.writerows(keep)
            path.write_text(buf.getvalue())


if __name__ == "__main__":
    main()
