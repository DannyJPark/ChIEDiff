#!/usr/bin/env python3
"""Heavy-atom-only protein-ligand clash count for every generated molecule a PoseCheck run covers.

    PY=~/anaconda3/envs/posecheck/bin/python
    $PY scripts/eval_heavy_clash.py --list                    # the model directories, indexed
    $PY scripts/eval_heavy_clash.py --dirs vina_fixed         # one or more models
    $PY scripts/eval_heavy_clash.py --index 7                 # array task: the 7th directory
    $PY scripts/eval_heavy_clash.py --summarize               # summary + provenance + gates

Criterion: PoseCheck's clash inequality (scripts/clash_pairs.py: d + 0.5 A < sum of RDKit vdW
radii) restricted to heavy atoms on BOTH sides. The ligand is the exported generated pose
(eval_out/<model>/sdf/, read as eval_posecheck.py reads it); the receptor is the manifest's receptor
file, whole, with no Hydride step. Run with the posecheck env so the radius table is PoseCheck's.

Relation to the published PoseCheck count (checked by scripts/check_posecheck_receptor_heavy.py):
  * 88 of 100 receptors: PoseCheck's protein holds exactly the file's heavy atoms, so this count is
    exactly the heavy-heavy subset of the published PoseCheck pairs.
  * 12 receptors encode chain copies only in segID. PoseCheck's Biopython parser keys atoms on
    (chainID, resSeq+iCode, name, altLoc), so it keeps one copy per key and measures a truncated
    receptor. This count uses the whole receptor there, so it can exceed PoseCheck's all-atom count;
    the summary flags those pockets and adds cohorts that exclude them.

Nothing here touches results/pose_fidelity/master.csv or any paper table. Outputs:
  eval_out/<model>/heavy_clash/pocket<NNN>.csv
      molecule,pocket_idx,n_lig_heavy,heavy_clashes,heavy_clashes_ion
      (same layout and molecule keys as eval_out/<model>/posecheck/, so a later master.csv column
       is a join in scripts/aggregate_pose_fidelity.py)
  results/pose_fidelity/heavy_clash/heavy_clash_summary.csv   per model x cohort
  results/pose_fidelity/heavy_clash/provenance.json           criterion, versions, receptors, gates
"""
import argparse
import csv
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger, rdBase

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import clash_pairs as CP                                        # noqa: E402

SKIP = re.compile(r"(_gnina_dock|_vinardo_dock|BROKEN)")      # docked-pose copies, broken export
OUT_SUB = "heavy_clash"
SUMMARY_DIR = ROOT / "results" / "pose_fidelity" / "heavy_clash"
FIELDS = ["molecule", "pocket_idx", "n_lig_heavy", "heavy_clashes", "heavy_clashes_ion"]


def registry():
    reg = json.loads((ROOT / "configs" / "models.json").read_text())
    models = reg["models"] if isinstance(reg, dict) and "models" in reg else reg
    return {(m.get("ids") or {}).get("eval_out"): m for m in models
            if (m.get("ids") or {}).get("eval_out")}


def model_dirs():
    """eval_out/<d> holding a PoseCheck run on generated poses: a manifest and posecheck CSVs, not a
    docked-pose copy, not the broken PIDiff export, not a quarantined registry entry."""
    reg = registry()
    out = []
    for d in sorted(os.listdir(ROOT / "eval_out")):
        p = ROOT / "eval_out" / d
        if SKIP.search(d) or not (p / "manifest.csv").is_file() or not (p / "posecheck").is_dir():
            continue
        if not any((p / "posecheck").glob("pocket*.csv")):
            continue
        if reg.get(d, {}).get("tier") == "quarantine":
            continue
        out.append(d)
    return out


def evaluate(d, pockets=None):
    base = ROOT / "eval_out" / d
    outdir = base / OUT_SUB
    outdir.mkdir(exist_ok=True)
    t0, n_mol, n_noconf, n_pk = time.time(), 0, 0, 0
    receptor_path, receptor = None, None
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
            receptor_path, receptor = row["receptor"], CP.read_receptor(ROOT / row["receptor"])
        rows, names = [], []
        # read exactly as scripts/eval_posecheck.py does, including its title fallback
        for m in Chem.SDMolSupplier(str(sdf), sanitize=False, removeHs=False):
            if m is None:
                continue
            name = m.GetProp("_Name") if m.HasProp("_Name") else f"p{tag}_m{len(names):04d}"
            names.append(name)
            if m.GetNumConformers() == 0:
                n_noconf += 1
                continue
            nh, hc, hi = CP.heavy_clash_counts(receptor, m)
            rows.append({"molecule": name, "pocket_idx": pk, "n_lig_heavy": nh,
                         "heavy_clashes": hc, "heavy_clashes_ion": hi})
        with open(outdir / f"pocket{tag}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        n_mol += len(rows)
        n_pk += 1
    print(f"{d}: {n_pk} pockets, {n_mol} molecules, {n_noconf} without a conformer, "
          f"{time.time() - t0:.1f} s", flush=True)


# --------------------------------------------------------------------------- summary
def read_pockets(folder):
    out = {}
    for f in sorted(folder.glob("pocket*.csv")):
        with open(f) as fh:
            for r in csv.DictReader(fh):
                out[r["molecule"]] = r
    return out


def biopython_keys(path):
    """Distinct (chainID, resSeq+iCode, atom name, altLoc) keys over the file's heavy atoms --
    Biopython's PDBParser keeps one atom per key and never looks at segID."""
    keys, n = set(), 0
    with open(path) as fh:
        for line in fh:
            if line.startswith(("ATOM", "HETATM")):
                line = line.rstrip("\n").ljust(80)
                if line[76:78].strip().upper() in ("H", "D"):
                    continue
                n += 1
                keys.add((line[21], line[22:27], line[12:16], line[16]))
    return n, len(keys)


def summarize():
    import pandas as pd
    import posecheck

    os.chdir(ROOT)                                    # build_pose_table reads repo-relative paths
    import build_pose_table as BPT

    reg = registry()
    gates = []

    def gate(ok, what):
        gates.append({"check": what, "pass": bool(ok)})
        print(f"  [{'PASS' if ok else 'FAIL'}] {what}")

    # ---- receptors: identical to PoseCheck's protein, or truncated by the key collision
    dirs = model_dirs()
    receptors = sorted({r["receptor"] for d in dirs
                        for r in csv.DictReader(open(ROOT / "eval_out" / d / "manifest.csv"))})
    checked = {}
    for p in sorted((SUMMARY_DIR / "receptor_check").glob("part_*.json")):
        checked.update(json.loads(p.read_text())["receptors"])
    gate(set(receptors) <= set(checked),
         f"all {len(receptors)} receptors compared with PoseCheck's loaded protein "
         f"({len(set(receptors) & set(checked))} found)")
    truncated, unexplained, collide_in_identical = {}, [], []
    for rec in receptors:
        v = checked.get(rec)
        if v is None:
            continue
        n_heavy, n_keys = biopython_keys(ROOT / rec)
        segids = {l[72:76].strip() for l in open(ROOT / rec) if l.startswith(("ATOM", "HETATM"))}
        if v["identical"]:
            if n_keys != n_heavy:
                collide_in_identical.append(rec)
            continue
        explained = (v["n_heavy_posecheck"] == n_keys and v["element_mismatches"] == 0
                     and v["one_to_one"] and v["max_offset"] is not None
                     and v["max_offset"] <= 1e-3)
        if not explained:
            unexplained.append(rec)
        truncated[rec] = {"n_heavy_file": v["n_heavy_file"],
                          "n_heavy_posecheck": v["n_heavy_posecheck"],
                          "kept_fraction": round(v["n_heavy_posecheck"] / v["n_heavy_file"], 4),
                          "n_segids": len(segids), "n_biopython_keys": n_keys}
    gate(not collide_in_identical,
         f"receptors PoseCheck loads whole have no key collisions ({collide_in_identical})")
    gate(not unexplained,
         f"every receptor PoseCheck loads differently is explained by the (chainID, resSeq, name, "
         f"altLoc) key collision: PoseCheck heavy atoms == distinct keys, on file coordinates, "
         f"same elements ({len(truncated)} truncated, unexplained {unexplained})")
    print(f"  [INFO] PoseCheck measured a truncated receptor in {len(truncated)} pockets' files: "
          + ", ".join(f"{Path(k).parent.name} ({v['kept_fraction']:.2f})"
                      for k, v in truncated.items()))

    f2 = pd.read_csv(ROOT / "results" / "comparison" / "f2_pose_fidelity" / "pose_stability"
                     / "pose_stability_strict.csv").set_index("id")
    by_tag = BPT.load_models()

    summary, per_model = [], {}
    for d in dirs:
        m = reg.get(d, {})
        mid, label, tier = m.get("id"), m.get("label"), m.get("tier")
        heavy = pd.DataFrame(read_pockets(ROOT / "eval_out" / d / OUT_SUB).values())
        if heavy.empty:
            gate(False, f"{d}: heavy-clash CSVs exist")
            continue
        for c in FIELDS[1:]:
            heavy[c] = heavy[c].astype(int)
        rec_of = {int(r["pocket_idx"]): r["receptor"]
                  for r in csv.DictReader(open(ROOT / "eval_out" / d / "manifest.csv"))}
        heavy["truncated_receptor"] = heavy.pocket_idx.map(lambda p: rec_of[p] in truncated)

        pc = read_pockets(ROOT / "eval_out" / d / "posecheck")
        for name, r in read_pockets(ROOT / "eval_out" / d / "posecheck_clash").items():
            pc.setdefault(name, r)                    # timeout-recovered clash counts
        pcdf = pd.DataFrame([{"molecule": k, "pc_clashes": float(v["clashes"])}
                             for k, v in pc.items() if v.get("clashes") not in ("", None)])
        missing = sorted(set(pcdf.molecule) - set(heavy.molecule)) if len(pcdf) else []
        gate(not missing, f"{d}: every PoseCheck-measured molecule has a heavy count "
                          f"({len(missing)} missing)")
        j = heavy.merge(pcdf, on="molecule") if len(pcdf) else heavy.assign(pc_clashes=np.nan)
        exceed = j.heavy_clashes > j.pc_clashes
        gate(int((exceed & ~j.truncated_receptor).sum()) == 0,
             f"{d}: heavy-heavy <= PoseCheck count wherever PoseCheck had the whole receptor "
             f"({int((~j.truncated_receptor).sum())} molecules; "
             f"{int((exceed & j.truncated_receptor).sum())} exceed in truncated pockets)")

        cohorts = {"exported": heavy.assign(pc_clashes=np.nan), "posecheck_measured": j}
        if tier == "reference":
            _, strict = BPT.populations(BPT.load_reference())
            srec = pd.DataFrame([{"pocket_idx": int(r["pk"]), "pc_clashes": r["clash"]}
                                 for r in strict])
            sj = heavy.merge(srec, on="pocket_idx") if len(srec) else None
        else:
            _, strict = BPT.populations(by_tag.get(d, []))
            srec = pd.DataFrame([{"molecule": r["name"], "pc_clashes": r["clash"]} for r in strict])
            sj = heavy.merge(srec, on="molecule") if len(srec) else None
        if sj is not None and len(sj):
            if mid in f2.index:
                gate(len(sj) == int(f2.loc[mid, "n_strict"])
                     and abs(sj.pc_clashes.mean() - float(f2.loc[mid, "clash_avg"])) < 5e-4,
                     f"{d}: Table 2 strict cohort reproduced (n {len(sj)} vs "
                     f"{f2.loc[mid, 'n_strict']}, PoseCheck clash {sj.pc_clashes.mean():.3f} vs "
                     f"{f2.loc[mid, 'clash_avg']})")
            cohorts["table2_strict"] = sj
            cohorts["table2_strict_excl_truncated"] = sj[~sj.truncated_receptor]

        for cname, df in cohorts.items():
            has_pc = df.pc_clashes.notna().any()
            summary.append({
                "eval_out": d, "id": mid, "label": label, "tier": tier, "cohort": cname,
                "n": len(df), "n_in_truncated_receptor_pockets": int(df.truncated_receptor.sum()),
                "heavy_clash_mean": round(float(df.heavy_clashes.mean()), 4),
                "heavy_clash_median": float(df.heavy_clashes.median()),
                "heavy_clash_any_pct": round(100 * float((df.heavy_clashes > 0).mean()), 2),
                "heavy_clash_ion_mean": round(float(df.heavy_clashes_ion.mean()), 4),
                "posecheck_clash_mean": round(float(df.pc_clashes.mean()), 4) if has_pc else "",
                "posecheck_clash_median": float(df.pc_clashes.median()) if has_pc else ""})
        per_model[d] = {"id": mid, "n_exported": len(heavy), "n_posecheck": len(j),
                        "n_table2_strict": int(len(sj)) if sj is not None else 0,
                        "n_in_truncated_receptor_pockets": int(heavy.truncated_receptor.sum())}

    fig = ROOT / "paper" / "figure_build" / "problem_definition" / "measurements"
    for role, d in (("baseline", "targetdiff"), ("ours", "vina_fixed")):
        f = fig / f"{role}_posecheck.json"
        if not f.is_file():
            continue
        J = json.loads(f.read_text())
        want = sum(not p["ligand"]["is_h"] and not p["protein"]["is_h"] for p in J["clash_pairs"])
        got = read_pockets(ROOT / "eval_out" / d / OUT_SUB).get(J["sample_id"], {})
        gate(bool(got) and int(got["heavy_clashes"]) == want,
             f"{d} {J['sample_id']}: heavy count {got.get('heavy_clashes')} == heavy-heavy pairs "
             f"in the PoseCheck pair list extracted for the figure ({want})")

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(SUMMARY_DIR / "heavy_clash_summary.csv", index=False)
    prov = {
        "criterion": "PoseCheck clash inequality d + tolerance < r_vdw(i) + r_vdw(j), heavy atoms "
                     "only on both sides; ligand = exported generated pose; receptor = the whole "
                     "manifest receptor file (no Hydride, no Biopython re-parse)",
        "tolerance": CP.POSECHECK_TOLERANCE,
        "radii_source": "rdkit.Chem.GetPeriodicTable().GetRvdw",
        "radii": {s: float(CP.vdw_radii([CP.atomic_number(s)])[0])
                  for s in ("C", "N", "O", "F", "P", "S", "CL", "BR", "I", "SE", "ZN", "CO", "MG",
                            "CA", "CU", "K", "NA")},
        "ion_column": "heavy_clashes_ion counts pairs whose receptor atom is a monatomic HETATM ion "
                      "(residue name == element, clash_pairs.ION_ELEMENTS)",
        "cohorts": {
            "exported": "every molecule in eval_out/<model>/sdf/",
            "posecheck_measured": "molecules with a published PoseCheck clash (posecheck/ or the "
                                  "timeout-recovered posecheck_clash/)",
            "table2_strict": "build_pose_table.populations() strict set: docking-successful, all "
                             "three engine RMSDs, PoseCheck clash and strain (Table 2 denominator)",
            "table2_strict_excl_truncated": "table2_strict without the pockets whose receptor "
                                            "PoseCheck truncated"},
        "posecheck_truncated_receptors": {
            "mechanism": "posecheck.utils.loading.load_protein_from_pdb parses with Biopython "
                         "PDBParser, which keeps one atom per (chainID, resSeq+iCode, atom name, "
                         "altLoc) and ignores segID; these files encode chain copies only in segID",
            "receptors": truncated},
        "versions": {"rdkit": rdBase.rdkitVersion,
                     "posecheck": getattr(posecheck, "__version__", "1.3.1")},
        "model_dirs": per_model, "gates": gates,
    }
    (SUMMARY_DIR / "provenance.json").write_text(json.dumps(prov, indent=1) + "\n")
    print(f"\n{sum(g['pass'] for g in gates)}/{len(gates)} gates passed -> {SUMMARY_DIR}")
    return 0 if all(g["pass"] for g in gates) else 1


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--list", action="store_true")
    g.add_argument("--all", action="store_true")
    g.add_argument("--dirs", nargs="+")
    g.add_argument("--index", type=int)
    g.add_argument("--summarize", action="store_true")
    ap.add_argument("--pockets", type=int, nargs="+", help="restrict to these pocket_idx (testing)")
    a = ap.parse_args()
    dirs = model_dirs()
    if a.list:
        for i, d in enumerate(dirs):
            print(i, d)
        return 0
    if a.summarize:
        return summarize()
    todo = dirs if a.all else ([dirs[a.index]] if a.index is not None else a.dirs)
    unknown = [d for d in todo if d not in dirs]
    if unknown:
        sys.exit(f"not a PoseCheck model directory: {unknown}")
    for d in todo:
        evaluate(d, set(a.pockets) if a.pockets else None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
