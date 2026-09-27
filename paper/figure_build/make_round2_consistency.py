#!/usr/bin/env python3
"""ID-paired Vinardo figures from existing score/dock SDFs and RMSD records.

No molecular calculation is executed. Scores use the same minimizedAffinity
field and file precedence as build_redock_comparison.read_dock_tags.
"""
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).parent))
from make_revision_figures import ROOT, PAPER, IDS, COLORS, LABELS, STYLES, save

OUT = PAPER / "data"
ORDER = ["reference", *IDS]
PALETTE = {**COLORS, "reference": "#666666"}
NAMES = {**LABELS, "reference": "Reference ligand"}
LINES = {**STYLES, "reference": "--"}
MARKERS = dict(zip(ORDER, ["s", "v", "^", "P", "X", "D", "<", ">", "o"]))
SOURCES = {}


def tracked(path):
    path = ROOT / path
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def rows(path):
    path = tracked(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        return list(csv.DictReader(f))


def sdf_scores(path):
    """Read saved SD properties, never regenerate poses or infer atom order."""
    text = tracked(path).read_text()
    for block in text.split("$$$$"):
        lines = block.lstrip("\r\n").splitlines()
        if not lines:
            continue
        name = lines[0].strip()
        for i, line in enumerate(lines[:-1]):
            if line.startswith(">") and "<minimizedAffinity>" in line:
                yield name, float(lines[i+1].strip())
                break


def extract():
    registry = {r["id"]: r for r in json.loads(tracked("configs/models.json").read_text())["models"]}
    master = rows("results/pose_fidelity/master.csv")
    by_tag = {}
    for r in master:
        key = (r["model"], r["name"])
        assert key not in by_tag, ("duplicate master key", key)
        by_tag[key] = r
    score_table = {r["id"]: r for r in rows("results/comparison/f1_sbdd/main_table/main_table.csv")}
    pose_table = {r["id"]: r for r in rows("results/comparison/f2_pose_fidelity/pose_stability/pose_stability_strict.csv")}
    cache = rows("paper/data/score_distributions.csv.gz")
    cached = {(r["id"], r["key"]): r for r in cache}
    assert len(cached) == len(cache)
    data, audit, paired = {}, [], []
    for mid in ORDER:
        scores, dock, rmsd, pockets = {}, {}, {}, {}
        if mid == "reference":
            for p in sorted((ROOT/"results/reference_protocol/self_reference_affinity_v2").glob("pocket*.csv")):
                for r in rows(p):
                    k = f"p{int(r['pocket_idx']):03d}_m0000"
                    scores[k] = float(r["vinardo_score_only"])
                    pockets[k] = int(r["pocket_idx"])
            for p in sorted((ROOT/"results/reference_protocol/self_redock_v4").glob("pocket*.csv")):
                for r in rows(p):
                    if r.get("rmsd_vinardo"):
                        rmsd[f"p{int(r['pocket_idx']):03d}_m0000"] = float(r["rmsd_vinardo"])
            for p in sorted((ROOT/"results/reference_protocol/self_redock_v4/poses").glob("pocket*_vinardo.sdf")):
                k = f"p{int(p.name[6:9]):03d}_m0000"
                vals = list(sdf_scores(p))
                if vals:
                    assert len(vals) == 1
                    dock[k] = vals[0][1]
        else:
            tag = registry[mid]["ids"]["eval_out"]
            base = ROOT/"eval_out"/tag
            if (base/"docked_names.txt").is_file():
                allowed = set(tracked(base/"docked_names.txt").read_text().split())
            else:
                allowed = {k for (t,k), r in by_tag.items() if t == tag and r["orig_docked"] == "1"}
            for mode, target in [("score", scores), ("dock", dock)]:
                d = base/"smina_vinardo"
                paths = sorted(d.glob(f"pocket*_{mode}.sdf")) + sorted(d.glob(f"pocket*_{mode}_*.sdf"))
                for p in paths:
                    for k, value in sdf_scores(p):
                        if k in allowed:
                            target.setdefault(k, value)
            for k in sorted(allowed):
                r = by_tag.get((tag, k))
                if r and r["orig_docked"] == "1" and r.get("rmsd_dock_vinardo"):
                    assert re.fullmatch(r"p\d{3}_m\d{4}", k), k
                    assert int(k[1:4]) == int(r["pocket_idx"]), (mid, k)
                    rmsd[k] = float(r["rmsd_dock_vinardo"])
                    pockets[k] = int(r["pocket_idx"])
        keys = sorted(set(scores) & set(dock) & set(rmsd))
        finite = [k for k in keys if np.isfinite([scores[k], dock[k], rmsd[k]]).all() and rmsd[k] >= 0]
        # Preserve the established energy-magnitude eligibility screen, not a plot-limit screen.
        valid = [k for k in finite if max(abs(scores[k]), abs(dock[k])) < 1000]
        table_keys = {k for (m, k) in cached if m == mid}
        assert len(table_keys) == int(score_table[mid]["n_mols"])
        for k in table_keys:
            c = cached[mid, k]
            assert k in scores and k in dock, (mid, k, "missing cached score")
            assert abs(scores[k] - float(c["vinardo_score"])) < .0002
            assert abs(dock[k] - float(c["vinardo_dock"])) < .0002
        x = np.array([rmsd[k] for k in valid])
        y = np.array([abs(scores[k]-dock[k]) for k in valid])
        assert len(valid) and len(valid) == len(set(valid))
        data[mid] = (x, y)
        for k, a, b in zip(valid, x, y):
            paired.append(dict(id=mid, key=k, pocket_idx=pockets[k], rmsd=a, gap=b,
                               vinardo_score=scores[k], vinardo_dock=dock[k]))
        audit.append(dict(id=mid, label=NAMES[mid], paired_n=len(valid), pockets=len({pockets[k] for k in valid}),
                          score_table_n=len(table_keys), pose_table_n=int(pose_table[mid]["n_mols"]),
                          paired_rmsd_median=float(np.median(x)), paired_gap_median=float(np.median(y)),
                          table_rmsd_median=float(pose_table[mid]["vinardo_med"]),
                          table_gap_median=float(score_table[mid]["vinardo_gap_med"]),
                          score_table_missing_rmsd=len(table_keys-set(valid)),
                          paired_outside_score_table=len(set(valid)-table_keys),
                          nonfinite_or_negative=len(keys)-len(finite), penalty_excluded=len(finite)-len(valid),
                          fraction_rmsd_above6=float(np.mean(x>6)), fraction_gap_above5=float(np.mean(y>5))))
    for filename, records in [("round2_vinardo_pairs.csv.gz", paired), ("round2_cohort_audit.csv", audit)]:
        p = OUT/filename
        opener = gzip.open if p.suffix == ".gz" else open
        with opener(p, "wt") as f:
            w = csv.DictWriter(f, fieldnames=list(records[0]))
            w.writeheader(); w.writerows(records)
    return data, audit


def density(data, step=.1):
    """Reflected binned Gaussian KDE in original nonnegative physical units.

    Same robust pooled diagonal bandwidth for every model, with pooled Scott
    n^(-1/6) factor. Full-data domain plus 4 bandwidths at upper boundaries;
    display limits never enter estimation. No log transform or tail truncation.
    """
    pool = np.concatenate([np.column_stack(data[m]) for m in IDS])
    bandwidth = (np.percentile(pool, 75, axis=0)-np.percentile(pool, 25, axis=0))/1.349 * len(pool)**(-1/6)
    bandwidth = np.maximum(bandwidth, .15)
    upper = np.ceil((np.max(pool, axis=0)+4*bandwidth)/step)*step
    bins = [np.arange(0, u+step*.5, step) for u in upper]
    result, checks = {}, {}
    for mid in IDS:
        hist, _, _ = np.histogram2d(*data[mid], bins=bins)
        assert hist.sum() == len(data[mid][0]), (mid, "density domain lost observations")
        smooth = gaussian_filter(hist/len(data[mid][0]), bandwidth/step, mode="reflect", truncate=4)
        assert abs(smooth.sum()-1) < 1e-10
        flat = np.sort(smooth.ravel())[::-1]
        i = np.searchsorted(np.cumsum(flat), .5)
        level = flat[i]
        mass = float(smooth[smooth >= level].sum())
        assert .5 <= mass < .51, (mid, mass)
        result[mid] = (bins[0][:-1]+step/2, bins[1][:-1]+step/2, smooth.T, level)
        checks[mid] = dict(mass=mass, peak_fraction=float(level/smooth.max()),
                           histogram_n=int(hist.sum()))
    return result, dict(method="reflected binned Gaussian KDE, original units", grid_step=step,
                        bandwidth=bandwidth.tolist(), upper=upper.tolist(),
                        rule="pooled generator IQR/1.349 * pooled_n^(-1/6), minimum 0.15 per axis",
                        lower_boundary="reflect at zero", upper_padding="4 bandwidths beyond maxima",
                        display_independent=True, checks=checks)


def draw(data, hdr):
    fig = plt.figure(figsize=(7.8, 7.8), layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2])
    axes = [fig.add_subplot(gs[0,0]), fig.add_subplot(gs[0,1]), fig.add_subplot(gs[1,:])]
    for mid in ORDER:
        x, y = data[mid]
        for ax, values in zip(axes[:2], [y, x]):
            v = np.sort(values)
            ax.step(np.r_[0,v], np.r_[0,np.arange(1,len(v)+1)/len(v)], where="post",
                    c=PALETTE[mid], ls=LINES[mid], lw=2 if mid=="ours_vina" else 1.05)
        axes[2].plot(np.median(x), np.median(y), marker=MARKERS[mid], color=PALETTE[mid],
                     ms=7, mec="white", mew=.65, zorder=5)
    for mid in ["kgdiff", "molcraft", "ours_vina"]:
        a,b,z,level = hdr[mid]
        axes[2].contour(a,b,z,levels=[level],colors=[PALETTE[mid]],linewidths=1.4,
                        linestyles=LINES[mid])
    for ax in axes[:2]:
        ax.set(ylim=(0,1.01), ylabel="Cumulative fraction")
        ax.axhline(.5, c="#dddddd", lw=.7, zorder=0)
    axes[0].set(xlim=(0,5), xlabel="Absolute redocking score gap\n(kcal mol$^{-1}$)", title="(a) Vinardo score gap")
    axes[1].set(xlim=(0,6), xlabel="Generated-to-redocked RMSD (Å)", title="(b) Vinardo displacement")
    axes[2].set(xlim=(0,7.5), ylim=(0,4.5), xlabel="Generated-to-redocked RMSD (Å)",
                ylabel="Absolute redocking score gap\n(kcal mol$^{-1}$)", title="(c) Joint pose–score consistency")
    for ax in axes[1:]:
        ax.axvline(2,c="#777777",ls="--",lw=.8)
    for ax in axes:
        ax.grid(alpha=.17)
    handles = [Line2D([],[],color=PALETTE[m],ls=LINES[m],marker=MARKERS[m],label=NAMES[m],ms=5)
               for m in ORDER]
    fig.legend(handles=handles,loc="outside lower center",ncol=3,frameon=False,fontsize=9)
    save(fig, "redocking_consistency_main_v2")

    # Same paired population, complete marginal tails, including Reference.
    fig, axes = plt.subplots(1,2,figsize=(7.8,3.7),layout="constrained")
    for ax, idx, title, label in [(axes[0],1,"(a) Full Vinardo gap", "Absolute score gap (kcal mol$^{-1}$)"),
                                  (axes[1],0,"(b) Full Vinardo RMSD", "Generated-to-redocked RMSD (Å)")]:
        for m in ORDER:
            v=np.sort(data[m][idx])
            ax.step(np.r_[0,v],np.r_[0,np.arange(1,len(v)+1)/len(v)],where="post",
                    c=PALETTE[m],ls=LINES[m],lw=1.8 if m=="ours_vina" else 1)
        ax.set_xscale("symlog",linthresh=2)
        ax.set(xlim=(0,max(data[m][idx].max() for m in ORDER)*1.03),ylim=(0,1.01),
               xlabel=label,ylabel="Cumulative fraction",title=title)
    fig.legend(handles=handles,loc="outside lower center",ncol=3,frameon=False,fontsize=8)
    save(fig,"redocking_consistency_paired_full_v2")

    fig, axes=plt.subplots(3,3,figsize=(7.8,8),layout="constrained",sharex=True,sharey=True)
    for ax,m in zip(axes.flat,ORDER):
        if m in hdr:
            a,b,z,level=hdr[m]
            ax.contour(a,b,z,levels=[level],colors=[PALETTE[m]],linewidths=1.2)
        ax.plot(np.median(data[m][0]),np.median(data[m][1]),marker=MARKERS[m],c=PALETTE[m],ms=6)
        ax.set(xlim=(0,7.5),ylim=(0,4.5),title=NAMES[m])
        ax.axvline(2,c="#aaaaaa",ls="--",lw=.6)
    fig.supxlabel("Generated-to-redocked RMSD (Å)")
    fig.supylabel("Absolute Vinardo score gap (kcal mol$^{-1}$)")
    save(fig,"redocking_joint_all_v2")


def main():
    for path in ["paper/figure_build/make_round2_consistency.py", "scripts/build_redock_comparison.py",
                 "scripts/sbatch_dock_worklist.sh", "scripts/self_redock.py", "scripts/pose_rmsd.py"]:
        tracked(path)
    data,audit=extract()
    hdr,method=density(data)
    # Recheck mass and central-region endpoints at double resolution for the
    # three main contours. This is numerical sensitivity, not uncertainty.
    finer,fine_method=density(data,.05)
    sensitivity={}
    for m in ["kgdiff","molcraft","ours_vina"]:
        extents=[]
        for a,b,z,level in [hdr[m],finer[m]]:
            yy,xx=np.where(z>=level)
            extents.append([float(a[xx].max()),float(b[yy].max())])
        difference=float(np.max(np.abs(np.diff(extents,axis=0))))
        assert difference < .3, (m,"HDR grid sensitivity",difference)
        sensitivity[m]=dict(extents=extents,max_difference=difference)
    draw(data,hdr)
    manifest=dict(source_sha256=SOURCES,cohort=audit,join="registry id + evaluation molecule name + pocket id; reference pocket id",
                  eligibility="docking-eligible, finite nonnegative Vinardo RMSD, finite score/dock minimizedAffinity, abs(energy)<1000",
                  cache_check="all current complete-score cohort Vinardo S/D values matched raw SD tags to 0.0002",
                  protocol="smina --scoring vinardo; score_only; dock reference autobox, exhaustiveness 8, seed 42, num_modes 1",
                  weighting="equal per eligible molecule; all three main panels share identical keys",density=method,
                  hdr_grid_sensitivity=sensitivity,palette=PALETTE,order=ORDER,
                  main_display=dict(gap=[0,5],rmsd=[0,6],joint=[[0,7.5],[0,4.5]]),
                  user_figure1_sha256=hashlib.sha256((PAPER/"figures/main_figure_fixed2.png").read_bytes()).hexdigest(),
                  artifacts_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in [OUT/"round2_vinardo_pairs.csv.gz", OUT/"round2_cohort_audit.csv",
                                              PAPER/"figures/redocking_consistency_main_v2.pdf",
                                              PAPER/"figures/redocking_consistency_paired_full_v2.pdf", PAPER/"figures/redocking_joint_all_v2.pdf"]})
    (OUT/"round2_figure_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(dict(cohort=audit,density=method,hdr_grid_sensitivity=sensitivity),indent=2))


if __name__=="__main__":
    main()
