#!/usr/bin/env python3
"""Regenerate manuscript tables from existing results; never run experiments.

Derived CSVs are presentation extracts, not replacement authoritative results.
Their source hashes and transformations are written to paper/data/manifest.json.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
from statistics import mean
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
# The display name is provisional and defined in exactly one place.
from gated_energy_diffusion import MODEL_NAME
OUT = ROOT / "paper/data"
ROSTER = "reference,pocket2mol,targetdiff,ipdiff,alidiff,molcraft,pidiff_retrain,kgdiff,ours_vina".split(",")
F1 = "results/comparison/f1_sbdd/main_table/main_table.csv"
F2 = "results/comparison/f2_pose_fidelity/pose_stability/pose_stability_strict.csv"
F2E = "results/comparison/f2_pose_fidelity/pose_stability/pose_stability_engineonly.csv"
F3 = "results/comparison/f3_nci/nci_summary_strict.csv"
F4 = "results/comparison/master_table.csv"
PROP = "results/comparison/f1_sbdd/properties/property_overall.csv"
RING = "results/comparison/f1_sbdd/ring_size/ring_size_overall.csv"
BOND = "results/comparison/f1_sbdd/bond_jsd/bond_jsd_overall.csv"
PB = "results/comparison/appendix/posebusters/pose_quality_summary_posebusters.csv"
PBD = "results/comparison/f2_pose_fidelity/pb_distance/pb_distance_strict.csv"
ABL = "results/diagnostics/ablation_nci/nci_summary_abl.txt"
MANIFEST = {}
# One spec per emitted table, dumped to paper/data/table_specs.json. This is the contract
# scripts/check_paper_tables.py verifies main.tex against, so the cell-level CSV check and
# the generator cannot drift apart.
SPECS = []
# Which document each table is injected into. Declared once, in paper/float_registry.json,
# and read by tools/check_manuscript.py as well: moving a table between the article and
# Additional file 1 then means flipping one "place" and moving the two marker lines.
REGISTRY = {f["label"]: f for f in
            json.loads((ROOT/"paper/float_registry.json").read_text())["floats"]}
TEX = {"body": "paper/main.tex", "appendix": "paper/main.tex",
       "supp": "paper/supplementary.tex"}


def target_tex(label):
    entry = REGISTRY.get(label)
    if entry is None:
        sys.exit(f"ERROR: {label} is not in paper/float_registry.json. Add it there (with its\n"
                 f"       place) before generating it, so the checker knows where it belongs.")
    return TEX[entry["place"]]


def read(path):
    p = ROOT / path
    MANIFEST[path] = hashlib.sha256(p.read_bytes()).hexdigest()
    with p.open(newline="") as f:
        return list(csv.DictReader(f))


def derived(name, rows):
    p = OUT / name
    with p.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return str(p.relative_to(ROOT))


def prepare():
    OUT.mkdir(exist_ok=True)
    tables = {p: {r["id"]: r for r in read(p)} for p in (F1, F2, F3, PROP, RING, BOND)}
    coverage = []
    distribution = []
    properties = []
    for mid in ROSTER:
        a, b, c = [tables[p][mid] for p in (F1, F2, F3)]
        coverage.append(dict(id=mid, label=a["label"], score_n=a["n_mols"],
                             score_p=a["n_pockets"], pose_n=b["n_mols"],
                             pose_p=b["n_pockets"], nci_n=c["n"]))
        d = dict(tables[RING][mid])
        d.update({k: v for k, v in tables[BOND][mid].items() if k != "source"})
        distribution.append(d)
        # n_mols counts every docking-eligible molecule, but one whose chemistry will not
        # sanitize has no property value and is dropped from each average -- so the printed
        # n was larger than the denominator behind it. The ring and bond tables already
        # carry that count (n_ring_measured == n_measured == this, every row), which is what
        # lets the three distribution/property tables print one consistent n.
        p = tables[PROP][mid]
        properties.append(dict(p, n_measured=int(p["n_mols"]) - int(p["n_unmeasurable"])))
    derived("coverage.csv", coverage)
    derived("distributions.csv", distribution)
    derived("properties.csv", properties)
    # Table 2 = the strict pose aggregate plus PoseBusters' heavy-atom clash count on the SAME
    # molecules (scripts/eval_pb_distance.py builds PBD from build_pose_table.populations()). The
    # join is by id and refuses to run unless both sides count the same strict cohort.
    pbd = {r["id"]: r for r in read(PBD)}
    pose_table = []
    for mid in ROSTER:
        f2, d = tables[F2][mid], pbd[mid]
        assert int(d["n"]) == int(f2["n_strict"]), (mid, d["n"], f2["n_strict"])
        pose_table.append(dict(f2, pb_clash_avg=d["pb_clashes_protein_avg"],
                               pb_clash_free_pct=d["clash_free_protein_pct"]))
    derived("pose_table.csv", pose_table)
    # Main-text summaries join existing aggregates without intersecting or remeasuring
    # their populations. Keep each block's denominator alongside its values.
    quality, chemistry = [], []
    for mid, pose, prop in zip(ROSTER, pose_table, properties):
        nci, score, bond = (tables[p][mid] for p in (F3, F1, BOND))
        assert prop["id"] == pose["id"] == mid
        assert int(prop["n_measured"]) == int(bond["n_measured"])
        assert int(pose["n_strain"]) == int(pose["n_strict"])
        quality.append(dict(
            id=mid, label=nci["label"], n_nci=nci["n"], n_pose=pose["n_strict"],
            n_strain=pose["n_strain"], prolif_HBAcceptor=nci["prolif_HBAcceptor"],
            prolif_HBDonor=nci["prolif_HBDonor"], prolif_Hydrophobic=nci["prolif_Hydrophobic"],
            prolif_total=nci["prolif_total"], prolif_per_heavy=nci["prolif_per_heavy"],
            pb_clash_avg=pose["pb_clash_avg"], pb_clash_free_pct=pose["pb_clash_free_pct"],
            # Printed as a failure rate so both PoseBusters columns read lower-is-better.
            pb_clash_fail_pct=f"{100 - float(pose['pb_clash_free_pct']):.2f}",
            strain_med=pose["strain_med"]))
        chemistry.append(dict(
            id=mid, label=prop["label"], n_property=prop["n_measured"],
            qed_avg=prop["qed_avg"], sa_avg=prop["sa_avg"], logp_avg=prop["logp_avg"],
            heavy_avg=prop["heavy_avg"], div_avg=score["div_avg"],
            n_pockets_div=score["n_pockets_div"], ejsd=bond["ejsd"]))
    derived("main_quality.csv", quality)
    derived("main_chemistry.csv", chemistry)
    pb = []
    for i, row in enumerate(read(PB)):
        row = dict(row, id=f"pb{i}", label=row["model"])
        row["label"] = row["label"].replace("Ours (Vina+Guide)", MODEL_NAME)
        row["label"] = row["label"].replace("Ours (Guide Only)", "No energy + guide")
        row["label"] = row["label"].replace("Ours (PIGNet+Guide)", "PIGNet loss + guide")
        row["label"] = row["label"].replace("Ours (PGDiff+Guide)", "PGDiff-style loss + guide")
        row["subset"] = {"docking_successful": "Dock eligible", "all_exported": "Exported"}[row["subset"]]
        pb.append(row)
    derived("posebusters.csv", pb)
    # One registry-rostered aggregate (scripts/aggregate_posecheck_abl.sh) replaces the hand-run
    # ON/OFF pair under ablation_nci/, whose invocation was never recorded; it reproduces all 315
    # of their cells. It is split and re-keyed here so the derived CSVs stay byte-identical to the
    # ones already checked: `model` keeps the legacy arm string make_nci_distributions.ABL_MODEL
    # looks up (and nci_distribution_manifest.json pins by hash), and rows keep the table order.
    terms = {"vdw": ("vdW", "Ste"), "hbond": ("hbond", "Hb"), "hydro": ("hydro", "Hp"),
             "vdw_hbond": ("vdW+hbond", "Ste+Hb"), "vdw_hydro": ("vdW+hydro", "Ste+Hp"),
             "hbond_hydro": ("hbond+hydro", "Hb+Hp")}
    # The OFF list mirrors ON: the 0-term corner first, then the 3-term anchor, then the six
    # subsets. abl_none_off closes the lattice (generated 2026-09-15; RUN_RECORD.md next to it).
    arms = {"ON": [("novdw", "arm1_no-physics", "No energy"),
                   ("ours_vina", "arm7_S+H+P_full", "Ste+Hb+Hp")],
            "OFF": [("abl_none_off", "arm1_no-physics", "No energy"),
                    ("ours_noguide", "arm7_S+H+P_full", "Ste+Hb+Hp")]}
    for state in arms:
        arms[state] += [(f"abl_{t}_{state.lower()}", *terms[t]) for t in terms]
    abl = {r["id"]: r for r in read("results/diagnostics/ablation_nci_v2/posecheck_ablation_all.csv")}
    assert set(abl) == {mid for rows in arms.values() for mid, _, _ in rows}, sorted(abl)
    for state, roster in arms.items():
        rows = []
        for mid, legacy, label in roster:
            r = {k: v for k, v in abl[mid].items() if k != "id"}
            rows.append(dict(r, model=legacy, id=legacy, label=label))
        derived(f"ablation_{state}.csv", rows)
    main_arms = [("abl_none_off", "No energy", "OFF"),
                 ("novdw", "No energy", "ON"),
                 ("ours_noguide", "Full", "OFF"),
                 ("ours_vina", "Full", "ON"),
                 ("abl_vdw_on", "Ste", "ON"),
                 ("abl_vdw_hbond_on", "Ste+Hb", "ON"),
                 ("abl_vdw_hydro_on", "Ste+Hp", "ON")]
    main_ablation = []
    for mid, label, guidance in main_arms:
        row = abl[mid]
        assert int(row["n_inter"]) == int(row["n_clash"]) == int(row["n"])
        main_ablation.append(dict(row, label=label, guidance=guidance))
    derived("main_ablation.csv", main_ablation)
    # Read the existing PLIP aggregate report, not the manuscript table.
    report = (ROOT / ABL).read_text()
    MANIFEST[ABL] = hashlib.sha256((ROOT / ABL).read_bytes()).hexdigest()
    block = report.split("1. PLIP")[1].split("2. PoseCheck")[0]
    rows = []
    for line in block.splitlines():
        m = re.match(r"(arm\d+.*?)\s+(ON|OFF)\s+(\d+)\s+(.+)$", line)
        if not m:
            continue
        name, state, n, data = m.groups()
        vals = data.split()
        assert len(vals) == 9, line
        label = {"arm1": "No energy", "arm2": "Ste", "arm3": "Hb", "arm4": "Hp",
                 "arm5": "Ste+Hb", "arm6": "Ste+Hp", "arm8": "Hb+Hp",
                 "arm7": "Ste+Hb+Hp"}[name.split()[0]]
        rows.append(dict(id=f"{name.split()[0]}_{state}", label=label, guidance=state, n=n,
                         hbond=vals[0], hydrophobic=vals[1], pistack=vals[2],
                         salt=vals[3], halogen=vals[4], water=vals[5],
                         total=vals[6], heavy=vals[7], per_heavy=vals[8]))
    assert len(rows) == 16
    derived("ablation_plip.csv", rows)
    # Audit the same 99-pocket specificity cohort without replacing its results.
    f4raw = read("results/comparison/f4_delta_score/delta_score_per_mol.csv")
    tags = {mid: ("vina_fixed_on" if mid == "ours_vina" else mid) for mid in ROSTER}
    grouped = {tag: defaultdict(list) for tag in tags.values()}
    for r in f4raw:
        if r["model"] not in grouped:
            continue
        try:
            pair = float(r["on_dock"]), float(r["off_dock"])
        except ValueError:
            continue
        if all(math.isfinite(x) for x in pair):
            grouped[r["model"]][r["pocket_idx"]].append(pair)
    common = set.intersection(*(set(g) for g in grouped.values()))
    assert common == set(map(str, range(100))) - {"46"}
    clip = lambda x: max(-20, min(20, x))
    audit = []
    for mid, tag in tags.items():
        g = {p: grouped[tag][p] for p in common}
        screened = {p: [(a,b) for a,b in pairs if max(abs(a),abs(b)) < 1000]
                    for p,pairs in g.items()}
        assert all(screened.values()), "Penalty screen emptied a pocket"
        n = sum(map(len, g.values()))
        n_screen = sum(map(len, screened.values()))
        audit.append(dict(id=mid,label=tables[F1][mid]["label"],n=n,
                          clipped_pct=100*sum(max(abs(a),abs(b))>20 for ps in g.values() for a,b in ps)/n,
                          penalty_n=n-n_screen,
                          raw_mean=f"{mean(mean(b-a for a,b in ps) for ps in g.values()):.6f}",
                          clipped_mean=f"{mean(mean(clip(b)-clip(a) for a,b in ps) for ps in g.values()):.6f}",
                          screened_raw=f"{mean(mean(b-a for a,b in ps) for ps in screened.values()):.6f}",
                          screened_clipped=f"{mean(mean(clip(b)-clip(a) for a,b in ps) for ps in screened.values()):.6f}"))
    derived("specificity_audit.csv", audit)
    (OUT / "manifest.json").write_text(json.dumps({
        "description": "Presentation extracts; original aggregates retained unchanged.",
        "transformations": {
            "coverage": "Join F1/F2/F3 by registry id; preserve per-table counts.",
            "distributions": "Join ring and bond aggregates by id; no reaggregation.",
            "properties": "Add n_measured = n_mols - n_unmeasurable, the denominator of every property mean; no reaggregation.",
            "pose_table": "F2 strict rows plus pb_clash_avg / pb_clash_free_pct from PBD, joined by id with an n_strict equality assert; no reaggregation.",
            "main_quality": "F3 ProLIF means and n; F2 strict strain_med, n_strict and n_strain; PBD pb_clashes_protein_avg and clash_free_protein_pct, plus pb_clash_fail_pct = 100 - clash_free_protein_pct. Join by id, preserving the separate NCI and pose cohorts; no reaggregation.",
            "main_chemistry": "PROP pooled property means and measured n, BOND ejsd (assert matching measured n), and F1 div_avg/n_pockets_div. Diversity retains its original pocket-macro cohort; no reaggregation.",
            "main_ablation": "Select seven recorded rows of ablation_nci_v2/posecheck_ablation_all.csv, retain original interaction/clash/finite-strain counts and values, add energy/guidance labels; no reaggregation.",
            "posebusters": "Preserve all measured subsets and rates; shorten labels.",
            "ablation": "Split the registry-rostered PoseCheck aggregate into its ON/OFF cohorts and preserve PLIP report precision; no reaggregation."
        }, "source_sha256": MANIFEST}, indent=2) + "\n")


def table(label, source, columns, headers, caption, note, groups=(), precision="", best=(),
          wide=False, ids=ROSTER, align=None, tabcolsep=None, rank_within=None, rule_after=()):
    """Render one table through csv_to_latex.py and record its spec.

    align/tabcolsep exist for the two main results tables, which use vertical rules
    between scorer blocks and two levels of grouped header. Every other table keeps the
    plain all-right default, so the layout stays a per-table choice rather than a fork.
    """
    column_list = [c.strip() for c in columns.split(",")]
    cmd = [sys.executable, "scripts/csv_to_latex.py", "--csv", source,
           "--columns", columns, "--headers", headers, "--label", label,
           "--caption", caption,
           "--align", align or ("@{}l" + "r" * (len(column_list) - 1) + "@{}")]
    if tabcolsep is not None:
        cmd += ["--tabcolsep", str(tabcolsep)]
    relabel, rules, best_exclude = {}, list(rule_after), []
    if ids:
        cmd += ["--ids", ",".join(ids)]
        if "reference" in ids:
            # "Test set" rather than "Reference": those rows ARE the CrossDocked test
            # split's crystal ligands, and the row must read the same in all 20 tables.
            # Changing it here is what keeps that wording regenerable.
            relabel["reference"] = "Test set"
            rules, best_exclude = ["reference", "kgdiff"], ["reference"]
            cmd += ["--best-exclude", "reference"]
        if "ours_vina" in ids and label != "tab:ablation-main":
            relabel["ours_vina"] = r"\textbf{\method}"
        if "pidiff_retrain" in ids:
            relabel["pidiff_retrain"] = "PIDiff"
        for rid, text in relabel.items():
            cmd += ["--relabel", f"{rid}={text}"]
    # Row groups a table separates with a \midrule. Roster tables fence the Test set and the
    # external baselines; any other table (e.g. guidance ON/OFF in tab:abl-plip) names its own.
    if rules:
        cmd += ["--rule-after", ",".join(rules)]
    if note:
        cmd += ["--note", note]
    for g in groups:
        cmd += ["--group-header", g]
    if precision:
        cmd += ["--precision-map", precision]
    for b in best:
        cmd += ["--best", b]
    if best:
        cmd += ["--second"]
    if rank_within:
        # Rows from different cohorts (PoseBusters subsets, guidance ON/OFF) are ranked
        # only against their own group.
        cmd += ["--rank-within", rank_within]
    if wide:
        cmd += ["--env", "sidewaystable"]
    if not ARGS.dry:
        cmd += ["--inject", target_tex(label)]
    subprocess.run(cmd, cwd=ROOT, check=True)
    SPECS.append(dict(
        label=label, source=source, columns=column_list, headers=headers,
        ids=list(ids) if ids else None, relabel=relabel, rule_after=rules,
        best_exclude=best_exclude, best=list(best), second=bool(best), rank_within=rank_within,
        precision={s.split(":")[0]: int(s.split(":")[1])
                   for s in filter(None, (t.strip() for t in precision.split(",")))},
        default_precision=2, na="--", env="sidewaystable" if wide else "table"))


def main():
    prepare()
    rank = " Bold and underline identify the best and second-best generator within a column; reference ligands are excluded."
    # Every table that ranks a column states its direction in the note, because only
    # tab:f1/tab:f2 carry arrows in their headers.
    # Header columns are named after the SCORING FUNCTION that produced the number, not the
    # program that ran it: "SMINA" alone reads as smina's default Vina-like scoring, a
    # different scale. "CNNaff" is CNNaffinity shortened for Table 1's 372pt budget; tab:a1
    # repeats Table 1's headers exactly (author's request, 2026-09-14).
    F1_GROUPS = [r"\textbf{Vina (kcal)} $\downarrow$:2-5|\textbf{Vinardo (kcal)} $\downarrow$:6-9|\textbf{CNNaff (pK)} $\uparrow$:10-13",
                 "Gen:2-3|Dock:4-5|Gen:6-7|Dock:8-9|Gen:10-11|Dock:12-13"]
    score_best = [f"{e}_{m}_{s}:{'max' if e == 'cnnaff' else 'min'}"
                  for e in ("vina", "vinardo", "cnnaff")
                  for m in ("score", "dock") for s in ("avg", "med")]
    # Author-approved restructure (2026-09-21): means in the main score table;
    # all original medians remain in tab:a1.
    quality = read("paper/data/main_quality.csv")
    nci_counts = ", ".join(r["n_nci"] for r in quality)
    pose_counts = ", ".join(r["n_pose"] for r in quality)
    table("tab:quality", "paper/data/main_quality.csv",
          "label,prolif_HBAcceptor,prolif_HBDonor,prolif_Hydrophobic,prolif_total,"
          "prolif_per_heavy,pb_clash_avg,pb_clash_fail_pct",
          r"Model\footnotemark[1],HB acc.,HB don.,Hyd.,Total,Total/HA,\#Clashes,Clash Fail\%",
          "Intermolecular interactions and physical quality of generated poses",
          r"ProLIF columns are pooled means on the PLIP--ProLIF intersection; HB roles refer to the ligand. Total/HA is the mean of each molecule's total count divided by its heavy-atom count. Physical checks retain the separate strict pose cohort. A PoseBusters clash is a ligand--protein heavy-atom pair closer than $0.75(r_i+r_j)$ in van der Waals radii; \#Clashes is the mean count per molecule and Clash Fail\% the percentage of molecules with at least one. Neither is an all-check pass rate. In row order, $n_{\rm NCI}$: "
          + nci_counts + r"; $n_{\rm pose}$: " + pose_counts
          + r". Larger contact counts do not establish better binding. Bold and underline identify the best and second-best generator in each column, in the direction its group header marks; Test set rows are excluded. Median PoseCheck strain for the same cohort appears in \afref{tab:pose-extended}. Definitions and full profiles appear in Tables~\afn{tab:f2}, \afn{tab:f3}, and~\afn{tab:nci-normalized}.",
          [r"ProLIF interactions $\uparrow$:2-6|PoseBusters $\downarrow$:7-8"],
          precision="prolif_per_heavy:4,pb_clash_fail_pct:1",
          best=["prolif_HBAcceptor:max", "prolif_HBDonor:max", "prolif_Hydrophobic:max",
                "prolif_total:max", "prolif_per_heavy:max", "pb_clash_avg:min",
                "pb_clash_fail_pct:min"], tabcolsep=4)
    table("tab:chemistry", "paper/data/main_chemistry.csv",
          "label,n_property,qed_avg,sa_avg,logp_avg,heavy_avg,div_avg,ejsd",
          r"Model\footnotemark[1],$n_{\rm prop}$,QED,SA,logP,Heavy,Div.,Bond JS",
          "Molecular properties and structural fidelity",
          r"Property columns are pooled means over docking-eligible molecules passing sanitization; $n_{\rm prop}$ is their denominator. SA is normalized synthetic accessibility (higher is favorable), not the raw SA score. Heavy is heavy-atom count. Div. retains the original pocket-macro mean of pairwise fingerprint Tanimoto distance from \afref{tab:gaps}, on docking-eligible molecules with fingerprints: 100 contributing pockets per comparator, 99 for \method{}, and undefined for one Test set ligand per pocket. Thus $n_{\rm prop}$ is not its denominator. Bond JS is the bond-count-weighted Jensen--Shannon distance to Test set ligands (lower is favorable), with the same measured-molecule count as the property block; it is not divergence. Extended results appear in Tables~\afn{tab:properties}--\afn{tab:bond-types}.",
          precision="qed_avg:3,sa_avg:3,heavy_avg:1,div_avg:3,ejsd:4", tabcolsep=3)
    ablation = read("paper/data/main_ablation.csv")
    strain_counts = ", ".join(r["n_strain"] for r in ablation)
    table("tab:ablation-main", "paper/data/main_ablation.csv",
          "label,guidance,n_inter,int_HBAcceptor_mean,int_HBDonor_mean,int_Hydrophobic_mean,inter_mean,clash_mean,strain_median",
          r"Energy\footnotemark[1],Guide,$n_{\rm NCI}$,Acc.,Don.,Hyd.,Total,Clash,Strain",
          "Contributions of energy learning and affinity guidance",
          r"Full denotes $\mathrm{Ste}+\mathrm{Hb}+\mathrm{Hp}$: steric, hydrogen-bond, and hydrophobic channels. ProLIF counts and PoseCheck clashes are pooled means on each arm's recorded measurable docking-eligible cohort, with the displayed $n_{\rm NCI}$ also applying to clashes. Strain is the median finite UFF energy difference in kcal\,mol$^{-1}$; finite counts in row order are "
          + strain_counts + r". Interaction counts are not quality rankings. The full-energy ON anchor uses the larger benchmark sample budget; selected checkpoints and coverage differ across arms. Full ON/OFF tables and the separate PLIP cohorts appear in \afref{secAbl}.",
          ["ProLIF interactions:4-7|PoseCheck:8-9"],
          precision="strain_median:1", ids=[r["id"] for r in ablation],
          rule_after=["ours_vina"], tabcolsep=3)
    # Table 2 (author decision 2026-09-15): GNINA CNNscore and PoseCheck strain moved out to the
    # appendix pose tables; PoseBusters' heavy-atom clash count sits beside PoseCheck's clash.
    table("tab:f2", "paper/data/pose_table.csv",
          "label,vinardo_avg,vinardo_med,vinardo_pct_lt2,gnina_avg,gnina_med,gnina_pct_lt2,"
          "clash_avg,pb_clash_avg",
          r"Model\footnotemark[1],Avg,Med,\%,Avg,Med,\%,Avg,Avg",
          "Redocking consistency and physical validity of generated poses",
          r"Pooled measurements on the strict pose cohort: all three engine RMSDs, PoseCheck clash, and strain must be available. RMSD is non-superposed, symmetry-aware heavy-atom displacement in the receptor frame, in \AA; it is not distance to an experimental pose. PoseCheck Clash counts ligand--protein atom pairs, hydrogens included, closer than the van der Waals radius sum minus 0.5\,\AA. PoseBusters Clash counts ligand--protein heavy-atom pairs closer than 0.75 of the van der Waals radius sum; cofactors and waters are checked separately. Generated-pose CNNscore, strain, full statistics, measurement counts, and RMSD-complete sensitivity appear in Section~\ref{secA2}." + rank,
          [r"\textbf{Vinardo}:2-4|\textbf{GNINA}:5-7|\textbf{PoseCheck}:8-8|\textbf{PoseBusters}:9-9",
           r"RMSD (\AA{}) $\downarrow$:2-3|$<$2\AA{} $\uparrow$:4-4|RMSD (\AA{}) $\downarrow$:5-6|"
           r"$<$2\AA{} $\uparrow$:7-7|Clash $\downarrow$:8-8|Clash $\downarrow$:9-9"],
          "vinardo_pct_lt2:1,gnina_pct_lt2:1",
          best=["vinardo_avg:min", "vinardo_med:min", "vinardo_pct_lt2:max",
                "gnina_avg:min", "gnina_med:min", "gnina_pct_lt2:max",
                "clash_avg:min", "pb_clash_avg:min"],
          align=r"@{}l|rrc|rrc|r|r@{}", tabcolsep=4)
    table("tab:f3", F3,
          "label,heavy_mean,plip_n_hbond,plip_n_hydrophobic,plip_total,prolif_HBAcceptor,prolif_HBDonor,prolif_Hydrophobic,prolif_VdWContact,prolif_total",
          r"Model\footnotemark[1],Heavy,H-bond,Hyd.,Total,Acc.,Don.,Hyd.,VdW,Total",
          "Interaction profiles measured on generated poses",
          r"Pooled per-molecule means on the PLIP--ProLIF intersection. Heavy denotes mean heavy-atom count; Hyd. denotes hydrophobic contacts, and Acc./Don. denote hydrogen-bond acceptor/donor cells. PLIP counts interaction objects; ProLIF counts Boolean residue--type cells, so totals are on different scales. PLIP total includes categories retained in Table~\ref{tab:a2}. Reference ligands provide a profile, not a reference-contact recovery denominator. All measurements use generated coordinates and the recorded versions in Table~\ref{tab:instruments}; cohort sizes and per-molecule normalized totals are given in Section~\ref{secA2}. Bold and underline mark the highest and second-highest generator in each interaction column; Heavy is not ranked, and reference ligands are excluded.",
          ["PLIP:3-5|ProLIF:6-10"], "heavy_mean:1",
          best=[f"{c}:max" for c in ("plip_n_hbond", "plip_n_hydrophobic", "plip_total",
                                     "prolif_HBAcceptor", "prolif_HBDonor",
                                     "prolif_Hydrophobic", "prolif_VdWContact", "prolif_total")])
    # Body tables stay portrait (paper/README.md §1). Thirteen columns fit 372pt only at
    # tabcolsep 2pt and without the $n$ column, which \mtref{tab:coverage} already prints:
    # measured 361.4pt with emphasis, against 385.4pt at tabcolsep 3 and 527.8pt at the default.
    table("tab:a1", F1,
          "label,vina_score_avg,vina_score_med,vina_dock_avg,vina_dock_med,vinardo_score_avg,vinardo_score_med,vinardo_dock_avg,vinardo_dock_med,cnnaff_score_avg,cnnaff_score_med,cnnaff_dock_avg,cnnaff_dock_med",
          r"Model\footnotemark[1],Avg,Med,Avg,Med,Avg,Med,Avg,Med,Avg,Med,Avg,Med",
          "Complete generated-pose and redocked scoring statistics",
          r"Pooled means (Avg) and medians (Med) on docking-eligible molecules with complete scoring measurements (\afref{secEval}); all twelve columns share the per-model complete-score denominator listed in Table~\ref{tab:coverage}. Gen evaluates generated coordinates; Dock evaluates redocked poses. Vina and Vinardo are in kcal\,mol$^{-1}$ (lower is favorable); CNNaff denotes GNINA CNNaffinity in pK (higher is favorable). Vina uses a generated-ligand box at exhaustiveness 16; Vinardo (SMINA) and GNINA use a Test set ligand autobox at exhaustiveness 8, so models are compared within columns. The \method{} row uses affinity guidance. Bold and underline identify the best and second-best generator within a column; reference ligands are excluded.",
          F1_GROUPS,
          best=score_best, align=r"@{}lrrrrrrrrrrrr@{}", tabcolsep=2)
    # QED and normalized SA used to sit here too, at 2 dp on the complete-score cohort.
    # tab:properties carries the same quantities at 3 dp on its own (larger) population, so
    # the pair was a duplicate that differed only in n and precision -- dropped 2026-09-14.
    table("tab:gaps", F1,
          "label,n_mols,vina_gap_avg,vina_gap_med,vinardo_gap_avg,vinardo_gap_med,cnnaff_gap_avg,cnnaff_gap_med,ha_avg,div_avg",
          r"Model\footnotemark[1],$n$,Mean,Median,Mean,Median,Mean,Median,High-aff. \%,Diversity",
          "Absolute score gaps, high-affinity rates, and diversity",
          r"Gaps are computed as $|s_{\mathrm{gen},i}-s_{\mathrm{dock},i}|$ for each molecule before pooling. Vina/Vinardo gaps use kcal\,mol$^{-1}$; CNNaff (GNINA CNNaffinity) gaps use pK. High-affinity percentage and diversity are pocket-macro means. High-affinity denotes improvement on the reference Vina Dock score, not a heavy-atom count. Calculated properties are reported in Table~\ref{tab:properties}. Bold and underline identify the best and second-best generator within a column (lower for gaps; higher for high-affinity percentage and diversity); reference ligands are excluded.",
          ["Vina gap:3-4|Vinardo gap:5-6|CNNaff gap:7-8"], "ha_avg:1",
          best=[f"{e}_gap_{s}:min" for e in ("vina", "vinardo", "cnnaff") for s in ("avg", "med")]
               + ["ha_avg:max", "div_avg:max"], wide=True)
    table("tab:a2", F3,
          "label,n,plip_n_hbond,plip_n_hydrophobic,plip_n_pistack,plip_n_saltbridge,plip_n_halogen,plip_n_waterbridge,plip_total,prolif_HBAcceptor,prolif_HBDonor,prolif_Hydrophobic,prolif_VdWContact,prolif_total",
          r"Model\footnotemark[1],$n$,HB,Hyd.,Stack,Salt,Halogen,Water,Total,Acc.,Don.,Hyd.,VdW,Total",
          "Expanded generated-pose interaction profiles",
          r"Same intersected cohort and pooled means as Table~\ref{tab:f3}. PLIP total includes all six listed categories. ProLIF total is the sum of its four listed categories. Instrument totals describe different objects and are not combined. Bold and underline mark the highest and second-highest generator in each column; water bridges are zero for every model and are not ranked, and reference ligands are excluded.",
          ["PLIP:3-9|ProLIF:10-14"],
          best=[f"{c}:max" for c in ("plip_n_hbond", "plip_n_hydrophobic", "plip_n_pistack",
                                     "plip_n_saltbridge", "plip_n_halogen", "plip_total",
                                     "prolif_HBAcceptor", "prolif_HBDonor",
                                     "prolif_Hydrophobic", "prolif_VdWContact", "prolif_total")],
          wide=True)
    table("tab:coverage", "paper/data/coverage.csv",
          "label,score_n,score_p,pose_n,pose_p,nci_n",
          r"Model\footnotemark[1],Molecules,Pockets,Molecules,Pockets,Molecules",
          "Metric-specific benchmark cohort sizes",
          r"Score: complete scoring intersection. Pose: strict three-engine RMSD plus clash/strain intersection. NCI: PLIP--ProLIF intersection. These counts describe measurable populations, not attempted-generation success. NCI pocket coverage is audited separately from the retained molecular counts.",
          ["Score:2-3|Pose:4-5|NCI:6-6"])
    for label, source, title in (
            ("tab:pose-extended", F2, "Extended strict-cohort pose measurements"),
            ("tab:pose-engineonly", F2E, "RMSD-complete cohort without requiring physical-check completion")):
        table(label, source,
              "label,n_mols,vina_avg,vina_med,vina_pct_lt2,vinardo_avg,vinardo_med,vinardo_pct_lt2,gnina_avg,gnina_med,gnina_pct_lt2,gnina_cnnscore_med,strain_med,n_clash,n_strain",
              r"Model\footnotemark[1],$n$,Mean,Med.,$<2$\%,Mean,Med.,$<2$\%,Mean,Med.,$<2$\%,CNN med.,Strain med.,Clash $n$,Strain $n$",
              title,
              r"RMSD is in \AA. Strain is the PoseCheck UFF relaxation-based energy difference in kcal\,mol$^{-1}$, reported as a median over molecules with a strain value. Strict eligibility is defined in Section~\ref{secEval}; the RMSD-complete cohort requires all three RMSDs but permits missing physical-check results. It is not a per-engine eligible population. The listed counts retain the source denominators. Meeko/Vina uses a different box and search setting from the other two engines. Bold and underline identify the best and second-best generator within a column (lower RMSD and strain; higher $<2$\,\AA{} percentage and CNN score); counts are not ranked, and reference ligands are excluded.",
              ["Vina RMSD:3-5|Vinardo RMSD:6-8|GNINA RMSD:9-11"],
              "vina_pct_lt2:1,vinardo_pct_lt2:1,gnina_pct_lt2:1,gnina_cnnscore_med:3,strain_med:1",
              best=[f"{e}_{s}:min" for e in ("vina", "vinardo", "gnina") for s in ("avg", "med")]
                   + [f"{e}_pct_lt2:max" for e in ("vina", "vinardo", "gnina")]
                   + ["gnina_cnnscore_med:max", "strain_med:min"], wide=True)
    table("tab:consistency-paired", "paper/data/round2_cohort_audit.csv",
          "label,paired_n,score_table_n,pose_table_n,paired_rmsd_median,table_rmsd_median,paired_gap_median,table_gap_median",
          r"Model\footnotemark[1],Plot,Score,Pose,Plot,Pose,Plot,Score",
          "Paired Vinardo plot cohorts compared with benchmark tables",
          r"Plot denotes the Vinardo-specific paired cohort in \mtref{fig:consistency}. Score and Pose refer to \mts{tab:a1} and Table~\ref{tab:f2}. All medians are pooled over molecules. RMSD is in \AA{} and absolute gap in kcal\,mol$^{-1}$. Pocket counts and exclusions are documented in Section~\ref{subsec:paired-consistency}; all generators cover 100 pockets except \method{} (99). The reference has one ligand per pocket.",
          [r"Molecule $n$:2-4|RMSD median:5-6|Gap median:7-8"],
          precision="paired_rmsd_median:3,table_rmsd_median:3,paired_gap_median:3,table_gap_median:3")
    table("tab:nci-normalized", F3, "label,n,heavy_mean,plip_per_heavy,prolif_per_heavy",
          r"Model\footnotemark[1],$n$,Heavy,PLIP/heavy,ProLIF/heavy",
          "Per-molecule size-normalized interaction totals",
          r"Normalization is the pooled mean of each molecule's count divided by its own heavy-atom count, $\operatorname{mean}_i(c_i/N_i)$; it is not the ratio of two means. Cohorts match Table~\ref{tab:f3}. This adjustment does not establish a fully size-controlled causal effect. Bold and underline mark the highest and second-highest generator in each normalized column; Heavy is not ranked, and reference ligands are excluded.",
          precision="heavy_mean:1,plip_per_heavy:4,prolif_per_heavy:4",
          best=["plip_per_heavy:max", "prolif_per_heavy:max"])
    table("tab:properties", "paper/data/properties.csv", "label,n_measured,qed_avg,sa_avg,logp_avg,tpsa_avg,mw_avg,lipinski_pass5_pct,heavy_avg",
          r"Model\footnotemark[1],$n$,QED,SA,logP,TPSA,MW,5-rule \%,Heavy",
          "Additional calculated properties of docking-eligible molecules",
          r"Pooled means except the percentage meeting all five implemented drug-likeness rules. $n$ counts docking-eligible molecules whose properties could be computed; molecules that fail RDKit sanitization are excluded from $n$ and from every mean. TPSA is in \AA$^2$, MW in g\,mol$^{-1}$; SA is the normalized score with higher values favorable. The population need not equal the complete-score cohort. The five-rule definition is given in Section~\ref{secEval}. Bold and underline identify the best and second-best generator for QED, SA, and the five-rule percentage (higher is better); logP, TPSA, MW, and heavy-atom count have no preferred direction and are not ranked. Reference ligands are excluded.",
          precision="qed_avg:3,sa_avg:3,tpsa_avg:1,mw_avg:1,lipinski_pass5_pct:1,heavy_avg:1",
          best=["qed_avg:max", "sa_avg:max", "lipinski_pass5_pct:max"])
    table("tab:distributions", "paper/data/distributions.csv",
          "label,n_measured,ejsd,jsd_atom,mean_rings_per_mol,acyclic_pct,ring_ge10_share_pct",
          r"Model\footnotemark[1],$n$,Bond JS,Atom JS,Rings/mol,Acyclic \%,Ring $\geq$10 \%",
          "Bond, atom-type, and ring-distribution summaries",
          r"$n$ counts docking-eligible molecules that pass RDKit sanitization; the bond, atom-type, and ring statistics share this denominator. Bond JS is a bond-count-weighted mean of per-bond-type Jensen--Shannon distances to reference ligands. Atom JS compares the element distribution with the fixed empirical reference encoded in the evaluation library. Ring counts use RDKit-perceived rings; the last column is the share of all ring occurrences with size at least ten. The reference bond profile is its own target, so its distance is omitted. Bold and underline identify the lowest and second-lowest generator for both distances and the ring-size-$\geq$10 share; ring count and acyclic share are not ranked, and reference ligands are excluded.",
          precision="ejsd:4,jsd_atom:4,mean_rings_per_mol:3,acyclic_pct:1,ring_ge10_share_pct:1",
          best=["ejsd:min", "jsd_atom:min", "ring_ge10_share_pct:min"])
    table("tab:bond-types", BOND,
          "label,n_measured,jsd_C-C,jsd_C=C,jsd_C:C,jsd_C-N,jsd_C=N,jsd_C:N,jsd_C-O,jsd_C=O,jsd_C:O,ejsd_ci_lo,ejsd_ci_hi",
          r"Model\footnotemark[1],$n$,C--C,C=C,C:C,C--N,C=N,C:N,C--O,C=O,C:O,CI low,CI high",
          "Bond-type distribution distances and existing pocket-bootstrap intervals",
          r"Distances are square roots of Jensen--Shannon divergence with natural logarithms. Intervals describe the bond-weighted aggregate and resample generated pockets while holding the 100-ligand reference fixed; they omit reference-sampling and training-seed uncertainty. Missing categories remain blank. Bold and underline identify the lowest and second-lowest generator distance in each bond column; interval bounds are not ranked.",
          best=[f"jsd_{b}:min" for b in ("C-C", "C=C", "C:C", "C-N", "C=N", "C:N", "C-O", "C=O", "C:O")],
          wide=True)
    table("tab:posebusters", "paper/data/posebusters.csv",
          "label,subset,n_mol,pass_all_pct,bond_lengths,bond_angles,internal_steric_clash,internal_energy",
          r"Configuration\footnotemark[1],Subset,$n$,All \%,Lengths \%,Angles \%,Internal clash \%,Energy \%",
          "PoseBusters pass rates in measured exported and docking-eligible cohorts",
          r"All cells after $n$ are percentages passing the indicated checks. Exported denotes measured exported structures, not all attempted generations. PIGNet-loss and PGDiff-style-loss rows are internal configurations, not reproductions of published model scores. Coverage differs across rows; no universal generation-success denominator is available. Bold and underline identify the highest and second-highest pass rate within each subset.",
          best=[f"{c}:max" for c in ("pass_all_pct", "bond_lengths", "bond_angles",
                                     "internal_steric_clash", "internal_energy")],
          ids=None, wide=True, rank_within="subset")
    table("tab:abl-plip", "paper/data/ablation_plip.csv",
          "label,guidance,n,hbond,hydrophobic,total,heavy,per_heavy",
          r"Terms\footnotemark[1],Guidance,$n$,H-bond,Hydrophobic,Total,Heavy,Total/heavy",
          "PLIP profiles for available energy-term and guidance configurations",
          r"Pooled means transcribed programmatically from the existing aggregate report, retaining its printed precision. PLIP cohorts differ from the ProLIF cohorts below. All four factorial corners are measured; the full-energy ON anchor is the only one with a larger sampling budget. These are descriptive contrasts between selected checkpoints. Bold and underline identify the highest and second-highest configuration in each interaction column within each guidance setting; $n$ and Heavy are not ranked.",
          best=["hbond:max", "hydrophobic:max", "total:max", "per_heavy:max"],
          ids=None, rank_within="guidance", rule_after=["arm7_ON"])
    for state in ("ON", "OFF"):
        table(f"tab:abl-{state.lower()}", f"paper/data/ablation_{state}.csv",
              "label,n,n_strain,int_HBAcceptor_mean,int_HBDonor_mean,int_Hydrophobic_mean,int_VdWContact_mean,inter_mean,clash_mean,strain_median,heavy_mean",
              r"Terms\footnotemark[1],$n$,$n_{\rm strain}$,Acc.,Don.,Hyd.,VdW,Total,Clash,Strain median,Heavy",
              f"ProLIF and physical-check summaries with guidance {state}",
              r"Pooled measurements on the source's eligible PoseCheck population. Strain uses its own finite-value count and is in kcal\,mol$^{-1}$; interactions and clashes are counts. Full-energy ON is the larger benchmark run. These populations do not match the PLIP denominators exactly. Bold and underline identify the best and second-best configuration in each ranked column (higher interaction counts; lower clash and strain); $n$, $n_{\rm strain}$, and Heavy are not ranked.",
              precision="strain_median:1,heavy_mean:1", ids=None, wide=True,
              best=[f"int_{t}_mean:max" for t in ("HBAcceptor", "HBDonor", "Hydrophobic", "VdWContact")]
                   + ["inter_mean:max", "clash_mean:min", "strain_median:min"])
    table("tab:f4", F4,
          "label,f4_n_mol,f4_delta_p_avg,f4_delta_p_med,f4_ratio_pct_avg,f4_win_pct_avg,f4_on_med,f4_off_med",
          r"Model\footnotemark[1],$n$,Mean,Median,Ratio \%,Win \%,On,Off",
          "Exploratory preference under a fixed random off-target assignment",
          r"Values retain the 99 pockets covered by every reported model, with seed 2027. Mean and median aggregate the per-pocket mean differences. Value-based energies are winsorized at $\pm20$ kcal\,mol$^{-1}$; Win and Ratio are pocket-macro percentages using raw comparisons. Win measures $P(D_{\rm off}>D_{\rm on})$ and is defined for reference ligands. Ratio requires improvement over the reference in both on-target energy and difference, so its reference value is undefined. On/off settings are asymmetric; computational failures require a separate audit. Bold and underline identify the best and second-best generator within a column (higher $\Delta$, Ratio, Win, and Off; lower On); reference ligands are excluded.",
          [r"$\Delta$ (kcal\,mol$^{-1}$):3-4|Component medians:7-8"],
          "f4_ratio_pct_avg:1,f4_win_pct_avg:1",
          best=["f4_delta_p_avg:max", "f4_delta_p_med:max", "f4_ratio_pct_avg:max",
                "f4_win_pct_avg:max", "f4_on_med:min", "f4_off_med:max"])
    table("tab:specificity-audit", "paper/data/specificity_audit.csv",
          "label,n,clipped_pct,penalty_n,raw_mean,clipped_mean,screened_raw,screened_clipped",
          r"Model\footnotemark[1],$n$,Clip \%,Penalty $n$,Raw,Clipped,Raw,Clipped",
          "Sensitivity of random-pocket differences to clipping and penalty screening",
          r"All means are means of pocket means on the same 99 pockets, in kcal\,mol$^{-1}$. Clip \% counts molecule pairs with either raw energy outside $\pm20$. The sensitivity screen removes pairs with either energy magnitude at least $10^3$; every pocket remains represented. This numerical screen identifies suspected grid penalties, not all confirmed computational failures or a validated biological subset. Original results in Table~\ref{tab:f4} are unchanged.",
          ["Original population:5-6|Penalty-screened:7-8"], "clipped_pct:2")
    (OUT / "table_specs.json").write_text(json.dumps(SPECS, indent=2) + "\n")
    print(f"wrote {len(SPECS)} table specs to {(OUT / 'table_specs.json').relative_to(ROOT)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry", action="store_true")
    ARGS = parser.parse_args()
    main()
