#!/usr/bin/env python3
"""Publication figures from recorded results and analytic potentials.

No training, docking, or molecular relaxation is performed. Summary checks reject
stale cohort caches. All graphics are vector PDF/SVG with raster previews.
"""
import csv
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FixedLocator, ScalarFormatter

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper"
sys.path.insert(0, str(PAPER / "tables"))
from build_revision_tables import ROSTER, F1, F2, RING
sys.path.insert(0, str(Path(__file__).parent))
from make_sf_plot import curves
from gated_energy_diffusion import MODEL_NAME

IDS = ROSTER[1:]
MAIN = ["targetdiff", "kgdiff", "molcraft", "pidiff_retrain", "ours_vina"]
LABELS = dict(zip(IDS, ["Pocket2Mol", "TargetDiff", "IPDiff", "AliDiff",
                        "MolCRAFT", "PIDiff", "KGDiff", MODEL_NAME]))
COLORS = dict(zip(IDS, ["#8061aa", "#2878b5", "#e69f00", "#56b4e9",
                        "#009e73", "#9b5932", "#3e4a89", "#d43f3a"]))
STYLES = dict(zip(IDS, ["--", "-", ":", "-.", "-", "--", "-.", "-"]))
SOURCES = {}
CHECKS = {}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "svg.fonttype": "none"})


def read(path):
    p = ROOT / path
    SOURCES[path] = hashlib.sha256(p.read_bytes()).hexdigest()
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt") as f:
        return list(csv.DictReader(x for x in f if not x.startswith("#")))


def save(fig, name, unused=False):
    # dpi affects the PNG only (pdf/svg are vector). The journal wants >=300 dpi for a
    # raster figure measured at PRINT size, and main.tex shows these at \textwidth =
    # 372pt = 5.15 in -- so a 7.4 in canvas at dpi=180 gave just 1352 px, i.e. 263
    # effective dpi, under the requirement. 300 here yields ~2250 px = ~430 effective
    # dpi. Do not lower it again without redoing that arithmetic.
    # A figure the manuscript does not name goes to figures/extra/, so figures/ holds only
    # the manuscript's figures and their .svg/.png siblings.
    folder = PAPER / "figure_build" / "figures" / "extra" if unused else PAPER / "figure_build" / "figures"
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(folder / f"{name}.{suffix}", dpi=300, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)


def overview():
    fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.set(xlim=(0, 10), ylim=(0, 6.2))
    ax.axis("off")

    def box(x, y, w, h, text, color="#eaf1f8"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04",
                                   facecolor=color, edgecolor="#44566c", linewidth=1))
        ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=10)

    def arrow(a, b, color="#44566c", style="-"):
        ax.annotate("", xy=b, xytext=a, arrowprops=dict(
            arrowstyle="-|>", color=color, lw=1.4, linestyle=style,
            connectionstyle="arc3"))

    ax.text(.1, 6.0, "(a) Training", fontsize=12, weight="bold")
    box(.1, 4.55, 1.65, .75, "Noisy ligand\n+ fixed pocket")
    box(2.25, 4.55, 1.6, .75, "Equivariant\ndenoiser")
    arrow((1.8, 4.93), (2.2, 4.93))
    for y, label in [(5.3, r"Clean coordinates $\hat X_0$"),
                     (4.25, r"Atom-type logits $Z_0$"),
                     (3.2, r"Affinity head $\hat A$")]:
        box(4.45, y, 2.05, .65, label)
        arrow((3.9, 4.93), (4.4, y+.32))
    box(7.3, 5.3, 2.5, .65, "Coordinate + type loss")
    box(7.3, 4.1, 2.5, .8, "Interaction-energy loss", "#fff0d5")
    box(7.3, 3.2, 2.5, .65, "Affinity loss")
    arrow((6.55, 5.63), (7.25, 5.63))
    arrow((6.55, 4.58), (7.25, 5.43))
    arrow((6.55, 5.43), (7.25, 4.65), "#b97a13")
    arrow((6.55, 3.53), (7.25, 3.53))
    box(.1, 3.25, 2.85, .75, "GT chemistry gates\n+ pocket coordinates", "#fff0d5")
    # Route the fixed chemistry input below prediction boxes to avoid crossing.
    ax.plot([2.99, 3.6, 6.9, 6.9], [3.6, 2.95, 2.95, 4.45],
            color="#b97a13", lw=1.4)
    arrow((6.9, 4.45), (7.25, 4.45), "#b97a13")
    ax.text(8.55, 2.88, r"Sum losses $\rightarrow$ update $\theta$",
            ha="center", fontsize=10, weight="bold")
    ax.axhline(2.62, color="#ccd3db", lw=1)
    ax.text(.1, 2.35, "(b) Generation with learned parameters", fontsize=12, weight="bold")
    box(.1, 1.2, 1.55, .75, "Noisy ligand\n+ fixed pocket")
    box(2.15, 1.2, 1.65, .75, "Trained\ndenoiser")
    box(4.5, 1.2, 2.6, .75, "Coordinate and type\nreverse posteriors")
    box(7.75, 1.2, 2.05, .75, "Next ligand state")
    arrow((1.7, 1.57), (2.1, 1.57))
    arrow((3.85, 1.57), (4.45, 1.57))
    arrow((7.15, 1.57), (7.7, 1.57))
    box(2.15, .05, 1.65, .6, "Affinity head", "#f3eafa")
    arrow((3., 1.15), (3., .7), "#8452a1")
    ax.text(5.6, .58, "Affinity gradients", color="#8452a1", ha="center")
    ax.plot([3.85, 6.7, 6.7], [.35, .35, 1.05], color="#8452a1", lw=1.4)
    arrow((6.7, 1.05), (6.7, 1.15), "#8452a1")
    ax.text(8.7, .3, "Repeat reverse steps\nthen reconstruct bonds",
            ha="center", fontsize=9)
    save(fig, "model_overview")


def consistency():
    score = read("paper/data/score_distributions.csv.gz")
    poses = read("results/pose_fidelity/master.csv")
    registry = json.loads((ROOT / "configs/models.json").read_text())
    entries = registry["models"]
    tags = {e["id"]: e["ids"]["eval_out"] for e in entries if e["id"] in IDS}
    score_summary = {r["id"]: r for r in read(F1)}
    pose_summary = {r["id"]: r for r in read(F2)}
    data = {}
    for mid in IDS:
        sr = [r for r in score if r["id"] == mid]
        assert len(sr) == int(score_summary[mid]["n_mols"]), (mid, "score count", len(sr))
        for engine in ("vina", "vinardo"):
            for mode in ("score", "dock"):
                vals = np.array([float(r[f"{engine}_{mode}"]) for r in sr])
                expected = float(score_summary[mid][f"{engine}_{mode}_avg"])
                assert abs(vals.mean()-expected) < .001, (mid, engine, mode, vals.mean(), expected)
            data[mid, engine] = np.abs([float(r[f"{engine}_score"])-float(r[f"{engine}_dock"]) for r in sr])
        cols = ["vm_rmsd_gen_dock", "rmsd_dock_vinardo", "rmsd_dock_gnina", "pc_clashes", "pc_strain"]
        pr = []
        for row in poses:
            if row["model"] != tags[mid] or row["orig_docked"] != "1":
                continue
            try:
                good = all(np.isfinite(float(row[c])) for c in cols)
            except ValueError:
                good = False
            if good:
                pr.append(row)
        assert len(pr) == int(pose_summary[mid]["n_mols"]), (mid, "pose count", len(pr))
        for engine, col in (("smina", "rmsd_dock_vinardo"), ("gnina", "rmsd_dock_gnina")):
            vals = np.array([float(r[col]) for r in pr])
            target = "vinardo" if engine == "smina" else "gnina"
            assert abs(np.median(vals)-float(pose_summary[mid][f"{target}_med"])) < .001
            data[mid, engine] = vals
        CHECKS[mid] = {"score_n": len(sr), "pose_n": len(pr)}
    for ids, filename in ((MAIN, "redocking_consistency"), (IDS, "redocking_consistency_all")):
        fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.4), layout="constrained")
        for ax, engine, title, unit in zip(axes.flat, ["vina","vinardo","smina","gnina"],
                ["(a) Vina score gap", "(b) Vinardo score gap", "(c) Vinardo RMSD", "(d) GNINA RMSD"],
                [r"kcal mol$^{-1}$",r"kcal mol$^{-1}$","Å","Å"]):
            maximum = 0
            for mid in ids:
                vals = np.sort(data[mid, engine])
                maximum = max(maximum, vals[-1])
                ax.step(np.r_[0, vals], np.r_[0, np.arange(1, len(vals)+1)/len(vals)],
                        where="post", label=LABELS[mid], color=COLORS[mid],
                        ls=STYLES[mid], lw=1.8 if mid == "ours_vina" else 1.2)
            ax.set_xscale("symlog", linthresh=2, linscale=1.5)
            ax.set(xlim=(0, maximum*1.03), ylim=(0,1.01), xlabel=unit,
                   ylabel="Cumulative fraction", title=title)
            ticks = [v for v in [0,1,2,10,100] if v <= maximum*1.03]
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.grid(axis="y", alpha=.2)
            if engine in ("smina","gnina"):
                ax.axvline(2, color="#555555", ls="--", lw=.9)
        handles, labels = axes[0,0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
        save(fig, filename, unused=filename == "redocking_consistency")


def ablation():
    rows = {r["id"]: r for r in read("paper/data/ablation_plip.csv")}
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), layout="constrained")
    d = np.linspace(-1, 3, 4001)
    for y, label, color in zip(curves(d), ["Ste","Ste+Hp","Ste+Hb"], ["#444444","#009e73","#ae4b93"]):
        axes[0].plot(d, y, label=label, color=color, lw=1.8)
    axes[0].set(xlabel="Surface distance (Å)", ylabel="Pair energy",
                title="(a) Isolated pair potentials", ylim=(-.25,.15))
    axes[0].legend(frameon=False)
    ids = ["arm2_ON","arm5_ON","arm6_ON"]
    x = np.arange(3)
    for shift, key, label, color in [(-.18,"hbond","H-bond","#ae4b93"),(.18,"hydrophobic","Hydrophobic","#009e73")]:
        vals = [float(rows[i][key]) for i in ids]
        axes[1].bar(x+shift, vals, .35, label=label, color=color)
    axes[1].set(xticks=x, xticklabels=[f"{rows[i]['label']}\nn={rows[i]['n']}" for i in ids],
                ylabel="Mean PLIP count", title="(b) Guidance ON: observed profiles", ylim=(0,9))
    axes[1].legend(frameon=False, fontsize=8)
    save(fig,"energy_ablation")


def properties():
    rows = read("results/comparison/f1_sbdd/properties/property_per_molecule.csv.gz")
    fig, axes = plt.subplots(2,2,figsize=(7.4,5.4),layout="constrained")
    for ax, key, title in zip(axes.flat, ["heavy","qed","sa","logp"],
                              ["(a) Heavy-atom count","(b) QED","(c) Normalized SA","(d) Calculated logP"]):
        allvals = [float(r[key]) for r in rows if r["id"] in IDS and r[key]]
        bins = np.linspace(min(allvals), max(allvals), 45)
        for mid in IDS:
            vals = [float(r[key]) for r in rows if r["id"] == mid and r[key]]
            ax.hist(vals, bins=bins, weights=np.full(len(vals), 1/len(vals)),
                    histtype="step", color=COLORS[mid], ls=STYLES[mid], label=LABELS[mid],
                    lw=1.5 if mid=="ours_vina" else 1.0)
        ax.set(title=title, ylabel="Fraction per bin", xlim=(bins[0],bins[-1]))
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside lower center",ncol=4,frameon=False)
    save(fig,"chemical_properties")
    rings = {r["id"]: r for r in read(RING)}
    array = np.array([[float(rings[i][f"ring{s}_occ_pct"]) for s in range(3,10)] for i in ROSTER])
    fig, ax = plt.subplots(figsize=(7.4,4.2),layout="constrained")
    im = ax.imshow(array,cmap="Blues",vmin=0,vmax=np.max(array),aspect="auto")
    ax.set(xticks=np.arange(7), xticklabels=list(range(3,10)), xlabel="Perceived ring size",
           yticks=np.arange(len(ROSTER)), yticklabels=["Reference"]+[LABELS[i] for i in IDS])
    for i in range(array.shape[0]):
        for j in range(7):
            ax.text(j,i,f"{array[i,j]:.1f}",ha="center",va="center",
                    color="white" if array[i,j]>array.max()*.6 else "#222222",fontsize=9)
    fig.colorbar(im,ax=ax,label="Ring occurrences (%)")
    save(fig,"ring_distributions")


def diagnostics():
    by_pocket = {}
    for p in sorted((ROOT/"results/diagnostics/type_cf/type_cf").glob("pocket*.csv")):
        by_pocket[p.stem] = read(str(p.relative_to(ROOT)))
    times = sorted({int(r["t"]) for rows in by_pocket.values() for r in rows})
    means = np.array([[np.mean([float(r["disp_max"]) for r in rows if int(r["t"]) == t])
                       for t in times] for rows in by_pocket.values()])
    fig, ax = plt.subplots(figsize=(7.4,3.5),layout="constrained")
    for y, name in zip(means, by_pocket):
        ax.plot(times,y,lw=.9,alpha=.65,label=name.replace("pocket","Pocket "))
    ax.plot(times,means.mean(0),color="#d43f3a",lw=2,label="Five-pocket mean")
    ax.axhline(.469,color="#555555",ls="--",lw=.8)
    ax.text(990,.474,"Isolated-pair minimum separation",ha="right",fontsize=8)
    ax.set(xlabel="Diffusion index (sampling proceeds right to left)",
           ylabel="Token-induced displacement (Å)",xlim=(0,1000),
           ylim=(0,max(.62,means.max()*1.2)))
    ax.legend(frameon=False,ncol=3,loc="upper left",fontsize=8)
    save(fig,"type_counterfactual")
    gate = read("results/diagnostics/gate_within/gate_within/gate_within_per_mol.csv")
    group = defaultdict(lambda: {"hb": defaultdict(list), "hyd": defaultdict(list)})
    for r in gate:
        for metric,a,b,na,nb in [("hb","fopt_A","fopt_B","n_A","n_B"),
                                 ("hyd","fsat_C","fsat_D","n_C","n_D")]:
            d = float(r[a])-float(r[b])
            if int(r[na])>=1 and int(r[nb])>=1 and np.isfinite(d):
                group[r["tag"]][metric][r["pocket_idx"]].append(d)
    names = {"NATIVE":"Reference","novdw_ON":"None / ON","targetdiff":"TargetDiff",
             "vina_fixed_OFF":"S+H+P / OFF","abl_vdw_hbond_OFF":"S+H / OFF",
             "abl_hbond_OFF":"H / OFF","abl_hbond_hydro_OFF":"H+P / OFF",
             "abl_vdw_OFF":"S / OFF","abl_vdw_hydro_OFF":"S+P / OFF",
             "abl_hydro_OFF":"P / OFF"}
    rows = []
    for tag,label in names.items():
        r = dict(id=tag,label=label)
        for metric in ("hb","hyd"):
            values = group[tag][metric]
            r[f"{metric}_n"] = sum(len(v) for v in values.values())
            r[f"{metric}_pockets"] = len(values)
            r[f"{metric}_mean"] = float(np.mean([np.mean(v) for v in values.values()]))
        rows.append(r)
    path = PAPER/"data/gate_contrasts.csv"
    with path.open("w",newline="") as f:
        w = csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    fig, axes = plt.subplots(1,2,figsize=(7.4,4.3),layout="constrained",sharey=True)
    for ax,metric,title in zip(axes,["hb","hyd"],["H-bond band","Hydrophobic saturation"]):
        ax.scatter([r[f"{metric}_mean"] for r in rows],range(len(rows)),color="#2878b5",s=30)
        ax.axvline(0,color="#666666",lw=.8,ls="--")
        ax.set(yticks=range(len(rows)),yticklabels=[r["label"] for r in rows],
               xlabel="Mean within-molecule fraction difference",title=title)
        ax.grid(axis="x",alpha=.2)
    axes[0].invert_yaxis()
    save(fig,"gate_contrasts")


if __name__ == "__main__":
    # Round 2 Figure 1 is an immutable user PNG, not a generated asset.
    # Keep overview() available only as the archived Round 1 alternative.
    consistency()
    ablation()
    properties()
    diagnostics()
    (PAPER/"data/figure_manifest.json").write_text(json.dumps({
        "source_sha256": SOURCES, "cohort_checks": CHECKS,
        "archived_round1_selection": MAIN, "all_generators": IDS,
        "current_main_figures": ["main_figure_fixed2.png", "redocking_consistency_main_v2.pdf", "nci_dist_prolif_baselines_boxen.pdf"],
        "main_figure2_generator": "paper/figure_build/make_round2_consistency.py",
        "user_asset_not_generated": "main_figure_fixed2.png",
        "ecdf_weighting": "one equal weight per eligible molecule",
        "ecdf_axis": "symlog; linear threshold 2, linear scale 1.5; complete tails",
        "palette": COLORS, "line_styles": STYLES,
    },indent=2)+"\n")
    print("Built 7 supporting/archived figures; user Figure 1 untouched. Use make_round2_consistency.py for current Figure 2.")
