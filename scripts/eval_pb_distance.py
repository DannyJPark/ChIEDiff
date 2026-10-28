#!/usr/bin/env python3
"""PoseBusters intermolecular-distance checks per generated molecule, keeping the clash COUNT.

    PY=$(conda info --base)/envs/posebusters/bin/python
    $PY scripts/eval_pb_distance.py --dir eval_out/mymodel
    $PY scripts/eval_pb_distance.py --dir eval_out/mymodel --pockets 0 1 2

What runs is PoseBusters' own
posebusters.modules.intermolecular_distance.check_intermolecular_distance, once per
"Distance to ..." module of the config, with that module's parameters read from the yml. A
clash is a heavy-atom pair with d < clash_cutoff * (r_i + r_j) (RDKit vdW radii for the
protein, organic-cofactor and water checks, covalent radii for inorganic cofactors;
hydrogens ignored on both sides). PoseBusters' own CSVs keep only each module's boolean;
this keeps num_pairwise_clashes as well.

Receptor: posebusters.tools.loading.safe_load_mol with the yml's mol_cond loading options
(RDKit's PDB reader keeps every atom, including the segID-coded chain copies Biopython
drops). Ligand: the exported pose, record by record, sanitize=False / removeHs=False.

Input:  <dir>/manifest.csv and <dir>/sdf/pocket<NNN>.sdf, from scripts/eval_export_sdf.py
Output: <dir>/pb_distance/pocket<NNN>.csv
"""
import argparse
import csv
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CONFIG = ROOT / "configs" / "posebusters_dock_1thread.yml"
OUT_SUB = "pb_distance"
SUFFIXES = ("protein", "organic_cofactors", "inorganic_cofactors", "waters")
FIELDS = (["molecule", "pocket_idx", "n_lig_heavy", "smallest_distance_protein",
           "not_too_far_away_protein"]
          + [f"{k}_{s}" for s in SUFFIXES for k in ("num_pairwise_clashes", "no_clashes")]
          + ["error"])


def distance_modules(config):
    import yaml
    cfg = yaml.safe_load(Path(config).read_text())
    mods = {m["rename_suffix"].lstrip("_"): m["parameters"]
            for m in cfg["modules"] if m["function"] == "intermolecular_distance"}
    if set(mods) != set(SUFFIXES):
        sys.exit(f"{config}: unexpected intermolecular_distance modules {sorted(mods)}")
    return mods, cfg["loading"]["mol_cond"]


def evaluate(base, config, pockets=None):
    from rdkit import Chem, RDLogger
    from posebusters.modules.intermolecular_distance import check_intermolecular_distance
    from posebusters.tools.loading import safe_load_mol

    RDLogger.DisableLog("rdApp.*")
    mods, cond_opts = distance_modules(config)
    base = Path(base)
    outdir = base / OUT_SUB
    outdir.mkdir(exist_ok=True)
    t0, n_mol, n_err, n_pk = time.time(), 0, 0, 0
    receptor_path, cond = None, None
    with open(base / "manifest.csv") as fh:
        manifest = list(csv.DictReader(fh))
    for row in manifest:
        pk = int(row["pocket_idx"])
        if pockets is not None and pk not in pockets:
            continue
        tag = f"{pk:03d}"
        sdf = base / "sdf" / f"pocket{tag}.sdf"
        if not sdf.is_file():
            continue
        if row["receptor"] != receptor_path:
            receptor_path = row["receptor"]
            p = Path(receptor_path)
            cond = safe_load_mol(p if p.is_absolute() else ROOT / p, **cond_opts)
            if cond is None:
                sys.exit(f"PoseBusters could not load {receptor_path}")
        rows, names = [], []
        for m in Chem.SDMolSupplier(str(sdf), sanitize=False, removeHs=False):
            if m is None:
                continue
            name = m.GetProp("_Name") if m.HasProp("_Name") else f"p{tag}_m{len(names):04d}"
            names.append(name)
            rec = {"molecule": name, "pocket_idx": pk,
                   "n_lig_heavy": sum(a.GetAtomicNum() > 1 for a in m.GetAtoms()), "error": ""}
            try:
                for s, params in mods.items():
                    r = check_intermolecular_distance(m, cond, **params)["results"]
                    rec[f"num_pairwise_clashes_{s}"] = int(r["num_pairwise_clashes"])
                    rec[f"no_clashes_{s}"] = bool(r["no_clashes"])
                    if s == "protein":
                        sd = float(r["smallest_distance"])
                        rec["smallest_distance_protein"] = "" if math.isnan(sd) else round(sd, 4)
                        rec["not_too_far_away_protein"] = bool(r["not_too_far_away"])
            except Exception as e:                         # recorded, never silently dropped
                rec["error"] = f"{type(e).__name__}: {e}"[:200]
                n_err += 1
            rows.append(rec)
        with open(outdir / f"pocket{tag}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        n_mol += len(rows)
        n_pk += 1
    print(f"{base}: {n_pk} pockets, {n_mol} molecules, {n_err} errors, "
          f"{time.time() - t0:.1f} s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="eval_out/<model> (holds manifest.csv and sdf/)")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG),
                    help="PoseBusters config whose intermolecular_distance parameters are used")
    ap.add_argument("--pockets", type=int, nargs="+", help="restrict to these pocket_idx")
    a = ap.parse_args()
    evaluate(a.dir, a.config, set(a.pockets) if a.pockets else None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
