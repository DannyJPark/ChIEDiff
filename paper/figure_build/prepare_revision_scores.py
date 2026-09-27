#!/usr/bin/env python3
"""Extract current PIDiff scores from existing saved poses; no docking is run.

Run with the kgdiff interpreter to preserve serialized RDKit compatibility.
The old shared cache is not overwritten.
"""
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
import build_main_table as bmt
import model_registry

cache = ROOT / "results/comparison/figures/rsg_ecdf_per_molecule.csv.gz"
with gzip.open(cache, "rt") as f:
    reader = csv.DictReader(f)
    fields = reader.fieldnames
    rows = [r for r in reader if r["id"] != "pidiff_pub"]
entry = model_registry.by_id("pidiff_retrain")
records, provenance = bmt.model_records(entry)
for key, record in records.items():
    row = dict(id="pidiff_retrain", key=key, pocket_idx=record["pk"])
    for c in ("vina", "vinardo", "gnina", "cnnaff"):
        for mode in ("score", "dock"):
            row[f"{c}_{mode}"] = f"{record[f'{c}_{mode}']:.4f}"
    rows.append(row)
out = ROOT / "paper/data/score_distributions.csv.gz"
with gzip.open(out, "wt") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
(out.parent / "score_provenance.json").write_text(json.dumps({
    "base_cache": str(cache.relative_to(ROOT)),
    "base_sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
    "replacement": "pidiff_pub removed; pidiff_retrain extracted via build_main_table.model_records",
    "source": model_registry.source_for(entry, "sbdd"),
    "provenance": provenance,
    "n_pidiff": len(records),
}, indent=2) + "\n")
print(f"Extracted {len(records)} PIDiff records from existing results to {out}")
