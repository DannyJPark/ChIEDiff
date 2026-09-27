#!/usr/bin/env python3
"""PoseBusters intermolecular-distance checks per generated molecule, with the clash COUNT kept.

    PY=~/anaconda3/envs/posebusters/bin/python
    $PY scripts/eval_pb_distance.py --list                     # model directories, indexed
    $PY scripts/eval_pb_distance.py --dirs native vina_fixed   # [--pockets 23] for a smoke test
    $PY scripts/eval_pb_distance.py --index 7                  # array task
    $PY scripts/eval_pb_distance.py --summarize                # gates + Table 2 cohort summary

What runs is PoseBusters' own posebusters.modules.intermolecular_distance.check_intermolecular_distance,
once per "Distance to ..." module of configs/posebusters_dock_1thread.yml, with that module's
parameters read from the yml -- the config every stored eval_out/<model>/posebusters/ result was
produced with. A clash is a heavy-atom pair with d < clash_cutoff * (r_i + r_j) (RDKit vdW radii for
the protein, organic-cofactor and water checks, covalent radii for inorganic cofactors; hydrogens
ignored on both sides). The stored PoseBusters CSVs keep only each module's boolean; this keeps
num_pairwise_clashes as well, and adds the 100 crystal ligands (eval_out/native), which were never
run through PoseBusters.

Receptor: posebusters.tools.loading.safe_load_mol with the yml's mol_cond loading options (RDKit's PDB
reader keeps every atom, including the segID-coded chain copies PoseCheck drops). Ligand: the exported
generated pose, record by record, sanitize=False / removeHs=False. The summary gates the recomputed
booleans against every stored PoseBusters row.

Outputs:
  eval_out/<model>/pb_distance/pocket<NNN>.csv
  results/comparison/f2_pose_fidelity/pb_distance/pb_distance_strict.csv    one row per model, Table 2 cohort
  results/comparison/f2_pose_fidelity/pb_distance/pb_distance_cohorts.csv   every model x cohort
  results/comparison/f2_pose_fidelity/pb_distance/provenance.json
"""
import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import eval_heavy_clash as EH                                    # noqa: E402  (model_dirs, registry)

CONFIG = ROOT / "configs" / "posebusters_dock_1thread.yml"
OUT_SUB = "pb_distance"
SUMMARY_DIR = ROOT / "results" / "comparison" / "f2_pose_fidelity" / "pb_distance"
F2 = ROOT / "results" / "comparison" / "f2_pose_fidelity" / "pose_stability" / "pose_stability_strict.csv"
SUFFIXES = ("protein", "organic_cofactors", "inorganic_cofactors", "waters")
FIELDS = (["molecule", "pocket_idx", "n_lig_heavy", "smallest_distance_protein",
           "not_too_far_away_protein"]
          + [f"{k}_{s}" for s in SUFFIXES for k in ("num_pairwise_clashes", "no_clashes")]
          + ["error"])
# the boolean each module wrote into the stored PoseBusters CSV (dock.yml rename_outputs)
STORED = {"no_clashes_protein": "minimum_distance_to_protein",
          "not_too_far_away_protein": "protein-ligand_maximum_distance",
          "no_clashes_organic_cofactors": "minimum_distance_to_organic_cofactors",
          "no_clashes_inorganic_cofactors": "minimum_distance_to_inorganic_cofactors",
          "no_clashes_waters": "minimum_distance_to_waters"}


def distance_modules():
    import yaml
    cfg = yaml.safe_load(CONFIG.read_text())
    mods = {m["rename_suffix"].lstrip("_"): m["parameters"]
            for m in cfg["modules"] if m["function"] == "intermolecular_distance"}
    if set(mods) != set(SUFFIXES):
        sys.exit(f"{CONFIG}: unexpected intermolecular_distance modules {sorted(mods)}")
    return mods, cfg["loading"]["mol_cond"]


def evaluate(d, pockets=None):
    from rdkit import Chem, RDLogger
    from posebusters.modules.intermolecular_distance import check_intermolecular_distance
    from posebusters.tools.loading import safe_load_mol

    RDLogger.DisableLog("rdApp.*")
    mods, cond_opts = distance_modules()
    base = ROOT / "eval_out" / d
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
            cond = safe_load_mol(ROOT / receptor_path, **cond_opts)
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
    print(f"{d}: {n_pk} pockets, {n_mol} molecules, {n_err} errors, {time.time() - t0:.1f} s",
          flush=True)


# --------------------------------------------------------------------------- summary
def summarize():
    import pandas as pd

    os.chdir(ROOT)                                  # build_pose_table reads repo-relative paths
    import build_pose_table as BPT
    import model_registry

    reg = EH.registry()
    gates = []

    def gate(ok, what):
        gates.append({"check": what, "pass": bool(ok)})
        print(f"  [{'PASS' if ok else 'FAIL'}] {what}")

    f2 = pd.read_csv(F2).set_index("id")
    by_tag = BPT.load_models()
    strict_rows, cohort_rows, per_model = [], [], {}
    for d in EH.model_dirs():
        e = reg.get(d, {})
        mid, tier = e.get("id"), e.get("tier")
        label = model_registry.label_for(e, "sbdd") if e else d
        df = pd.DataFrame(EH.read_pockets(ROOT / "eval_out" / d / OUT_SUB).values())
        if df.empty:
            gate(False, f"{d}: pb_distance CSVs exist")
            continue
        err = df.error.fillna("") != ""
        gate(not err.any(), f"{d}: PoseBusters distance modules ran on all {len(df)} molecules "
                            f"({int(err.sum())} errors)")
        df = df[~err].copy()
        df["pocket_idx"] = df["pocket_idx"].astype(int)          # the reference joins on it
        for s in SUFFIXES:
            df[f"num_pairwise_clashes_{s}"] = df[f"num_pairwise_clashes_{s}"].astype(int)
        for c in list(STORED):
            df[c] = df[c].map({"True": True, "False": False})

        stored = EH.read_pockets(ROOT / "eval_out" / d / "posebusters")
        if stored:
            st = pd.DataFrame(stored.values())
            j = df.merge(st[["molecule"] + list(STORED.values())], on="molecule")
            for mine, col in STORED.items():
                sub = j[j[col].isin(["True", "False"])]
                diff = int((sub[mine] != (sub[col] == "True")).sum())
                gate(diff == 0, f"{d}: recomputed {mine} == stored '{col}' on {len(sub)} "
                                f"molecules ({diff} differ)")

        if tier == "reference":
            _, strict = BPT.populations(BPT.load_reference())
            sj = df.merge(pd.DataFrame({"pocket_idx": [int(r["pk"]) for r in strict]}),
                          on="pocket_idx")
            want = len(strict)
        else:
            _, strict = BPT.populations(by_tag.get(d, []))
            sj = df.merge(pd.DataFrame({"molecule": [r["name"] for r in strict]}), on="molecule")
            want = len(strict)
        cohorts = {"exported": df}
        if want:
            gate(len(sj) == want, f"{d}: every Table 2 strict molecule has a PoseBusters distance "
                                  f"result ({len(sj)}/{want})")
            if mid in f2.index:
                gate(want == int(f2.loc[mid, "n_strict"]),
                     f"{d}: strict cohort size {want} == Table 2 n {f2.loc[mid, 'n_strict']}")
            cohorts["table2_strict"] = sj

        for cname, c in cohorts.items():
            row = {"id": mid or d, "label": label, "eval_out": d, "tier": tier, "cohort": cname,
                   "n": len(c),
                   "clash_free_protein_pct": round(100 * float(c.no_clashes_protein.mean()), 2),
                   "pb_clashes_protein_avg": round(float(c.num_pairwise_clashes_protein.mean()), 4),
                   "pb_clashes_protein_med": float(c.num_pairwise_clashes_protein.median()),
                   "clash_free_organic_cofactors_pct":
                       round(100 * float(c.no_clashes_organic_cofactors.mean()), 2),
                   "clash_free_inorganic_cofactors_pct":
                       round(100 * float(c.no_clashes_inorganic_cofactors.mean()), 2),
                   "clash_free_waters_pct": round(100 * float(c.no_clashes_waters.mean()), 2),
                   "clash_free_all_pct": round(100 * float(
                       (c.no_clashes_protein & c.no_clashes_organic_cofactors
                        & c.no_clashes_inorganic_cofactors & c.no_clashes_waters).mean()), 2),
                   "within_5A_pct": round(100 * float(c.not_too_far_away_protein.mean()), 2)}
            cohort_rows.append(row)
            if cname == "table2_strict":
                strict_rows.append(row)
        per_model[d] = {"id": mid, "n_exported": len(df), "n_table2_strict": len(sj) if want else 0,
                        "stored_posebusters_rows": len(stored)}

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(strict_rows).to_csv(SUMMARY_DIR / "pb_distance_strict.csv", index=False)
    pd.DataFrame(cohort_rows).to_csv(SUMMARY_DIR / "pb_distance_cohorts.csv", index=False)
    # Summaries run in the posecheck env (build_pose_table needs its torch), which has neither
    # posebusters nor yaml. The versions and module parameters that matter are those of the env
    # that computed the checks, so that interpreter reports them.
    import subprocess
    code = ("import json, sys, yaml, posebusters, rdkit\n"
            "cfg = yaml.safe_load(open(sys.argv[1]))\n"
            "mods = {m['rename_suffix'].lstrip('_'): m['parameters'] for m in cfg['modules']"
            " if m['function'] == 'intermolecular_distance'}\n"
            "print(json.dumps({'posebusters': posebusters.__version__, 'rdkit': rdkit.__version__,"
            " 'modules': mods, 'mol_cond': cfg['loading']['mol_cond']}))\n")
    pb_env = subprocess.run([str(Path.home() / "anaconda3/envs/posebusters/bin/python"), "-c",
                             code, str(CONFIG)], capture_output=True, text=True, check=True)
    info = json.loads(pb_env.stdout.strip().splitlines()[-1])
    versions = {"posebusters": info["posebusters"], "rdkit": info["rdkit"]}
    mods, cond_opts = info["modules"], info["mol_cond"]
    prov = {
        "function": "posebusters.modules.intermolecular_distance.check_intermolecular_distance",
        "config": str(CONFIG.relative_to(ROOT)),
        "config_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
        "modules": mods, "mol_cond_loading": cond_opts,
        "clash_definition": "heavy-atom pair with distance < clash_cutoff * (r_i + r_j), "
                            "radius_scale 1.0; clash-free = no such pair",
        "ligand_reading": "RDKit SDMolSupplier(sanitize=False, removeHs=False) on "
                          "eval_out/<model>/sdf/pocket<NNN>.sdf",
        "cohort_table2_strict": "build_pose_table.populations() strict set (Table 2 denominator); "
                                "reference = build_pose_table.load_reference()",
        "versions": versions,
        "model_dirs": per_model, "gates": gates,
    }
    (SUMMARY_DIR / "provenance.json").write_text(json.dumps(prov, indent=1) + "\n")
    print(f"\n{sum(g['pass'] for g in gates)}/{len(gates)} gates passed -> {SUMMARY_DIR}")
    return 0 if all(g["pass"] for g in gates) else 1


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--list", action="store_true")
    g.add_argument("--dirs", nargs="+")
    g.add_argument("--index", type=int)
    g.add_argument("--summarize", action="store_true")
    ap.add_argument("--pockets", type=int, nargs="+", help="restrict to these pocket_idx (testing)")
    a = ap.parse_args()
    dirs = EH.model_dirs()
    if a.list:
        for i, d in enumerate(dirs):
            print(i, d)
        return 0
    if a.summarize:
        return summarize()
    todo = [dirs[a.index]] if a.index is not None else a.dirs
    unknown = [d for d in todo if d not in dirs]
    if unknown:
        sys.exit(f"not a model directory: {unknown}")
    for d in todo:
        evaluate(d, set(a.pockets) if a.pockets else None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
