#!/usr/bin/env python3
"""Per-molecule PLIP / ProLIF interaction distributions (boxen and violin figures).

Population, for every row: generated pose, docking-successful, measured by BOTH PLIP and
ProLIF -- the tab:f3 cohort of scripts/build_nci_strict.py. Nothing is re-measured: the script
reads eval_plip/ and eval_out/, and refuses to draw when a row's n or pooled means drift from
the tables the figures sit beside (nci_summary_strict.csv, ablation_{ON,OFF}.csv).

PLIP's H-bond count is split by ligand role. An hbond entry with protisdon=True has the protein
as donor, so the ligand is the acceptor; acceptor + donor == n_hbond for every molecule. ProLIF
counts are Boolean residue cells (PoseCheck int_*), as in tab:f3.

Two join rules, both inherited rather than re-derived:
  registry rows  -- build_nci_strict.measure(): .pt docking filter, PLIP re-keyed through the
                    canonical pocket order (PIDiff's flat .pt), then intersect with ProLIF.
  ablation arms  -- build_nci_summary_abl.py: PLIP ran via --samples, so the set is already
                    connected == docked (keep=None). Its index is NOT the SDF-title index ProLIF
                    used, so PLIP rows are re-keyed per pocket by molecule identity before the
                    join, and pairing is gated against the exported SDF's heavy-atom counts.

Usage:
    ~/anaconda3/envs/kgdiff/bin/python paper/figure_build/make_nci_distributions.py
    ~/anaconda3/envs/kgdiff/bin/python paper/figure_build/make_nci_distributions.py --refigure
"""
import argparse
import csv
import difflib
import glob
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection, PolyCollection
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)                       # the eval_plip/ and eval_out/ loaders take relative paths
sys.path.insert(0, str(ROOT / "scripts"))
from make_revision_figures import COLORS, LABELS, save    # noqa: E402  shared palette + rcParams

STRICT = "results/comparison/f3_nci/nci_summary_strict.csv"
ABL_CSV = {"ON": "paper/data/ablation_ON.csv", "OFF": "paper/data/ablation_OFF.csv"}
PER_MOL = "results/comparison/f3_nci/nci_per_molecule_strict.csv.gz"
MANIFEST = "paper/data/nci_distribution_manifest.json"

BASELINES = ["reference", "pocket2mol", "targetdiff", "ipdiff", "alidiff", "molcraft",
             "pidiff_retrain", "kgdiff", "ours_vina"]
REGISTRY_ROWS = BASELINES + ["ours_noguide"]          # ours_noguide = S+H+P, guidance OFF
ABL_TAGS = ["abl_vdw_on", "abl_vdw_off", "abl_vdw_hbond_on", "abl_vdw_hbond_off",
            "abl_vdw_hydro_on", "abl_vdw_hydro_off"]
DISPLAY = dict(LABELS, reference="Test set", ours_noguide="Ste+Hb+Hp OFF",
               abl_vdw_on="Ste ON", abl_vdw_off="Ste OFF", abl_vdw_hbond_on="Ste+Hb ON",
               abl_vdw_hbond_off="Ste+Hb OFF", abl_vdw_hydro_on="Ste+Hp ON",
               abl_vdw_hydro_off="Ste+Hp OFF")
# ablation_{ON,OFF}.csv `model` -> the row id that must reproduce it
ABL_MODEL = {"ON": {"vdW": "abl_vdw_on", "vdW+hbond": "abl_vdw_hbond_on",
                    "vdW+hydro": "abl_vdw_hydro_on", "arm7_S+H+P_full": "ours_vina"},
             "OFF": {"vdW": "abl_vdw_off", "vdW+hbond": "abl_vdw_hbond_off",
                     "vdW+hydro": "abl_vdw_hydro_off", "arm7_S+H+P_full": "ours_noguide"}}

# PLIP records one hydrogen-bond object per bond plus a protisdon flag; the acceptor/donor
# split is ours, not PLIP's, and Table tab:f3 reports the single undivided count. Splitting it
# here only to mirror ProLIF's ligand-centred cells invited the two to be read as the same
# measurement, so the figure shows the category PLIP actually reports.
PLIP_PANELS = [("plip_hbond", "H-bond"), ("plip_hydrophobic", "Hydrophobic"),
               ("plip_pistack", r"$\pi$-stacking"), ("plip_saltbridge", "Salt bridge")]
PROLIF_PANELS = [("prolif_HBAcceptor", "H-bond acceptor"), ("prolif_HBDonor", "H-bond donor"),
                 ("prolif_Hydrophobic", "Hydrophobic"), ("prolif_VdWContact", "van der Waals contact")]
PROLIF_KEYS = ["HBAcceptor", "HBDonor", "Hydrophobic", "VdWContact"]

REF_COLOR = "#bdbdbd"
TERM_COLOR = {"Ste": "#444444", "Ste+Hb": "#ae4b93", "Ste+Hp": "#009e73", "Ste+Hb+Hp": "#d43f3a"}  # energy_ablation
INK = "#222222"
HATCH = "////"
plt.rcParams.update({"hatch.linewidth": 0.5, "hatch.color": "#555555"})


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plip_fields(r):
    hb = r.get("hbond") or []
    assert len(hb) == r.get("n_hbond", 0), ("hbond list != n_hbond", r.get("pk"), r.get("mi"))
    acc = sum(1 for h in hb if h["protisdon"])
    # `.get(c, 0)`: a molecule with none of a type carries no key; that is zero, not missing.
    return {"plip_hb_acc": acc, "plip_hb_don": len(hb) - acc,
            "plip_hydrophobic": r.get("n_hydrophobic", 0), "plip_pistack": r.get("n_pistack", 0),
            "plip_saltbridge": r.get("n_saltbridge", 0)}


def molecule_rows(mid, P, R, both):
    return [dict(id=mid, label=DISPLAY[mid], mol=k, pocket_idx=int(k[1:4]), heavy=P[k].get("heavy"),
                 **plip_fields(P[k]),
                 **{f"prolif_{c}": R[k].get(f"int_{c}", 0.0) for c in PROLIF_KEYS})
            for k in both]


def join_registry(mid, maps):
    import build_nci_summary as bns
    import build_nci_strict as strict
    import model_registry
    m = model_registry.by_id(mid)
    ptag, pdir = m["ids"]["plip_tag"], m["ids"]["eval_out"]
    pt = model_registry.source_for(m, "nci")
    keep, heavy = bns.docked_keys(pt)
    if ptag in bns.CONNECTED_IS_DOCKED and heavy:
        keep = set(heavy)
    P = {}
    for r in bns.load_plip(ptag, None):               # unfiltered, re-key, THEN filter (PIDiff)
        k = strict.plip_key(r, *maps)
        if k is not None and (keep is None or k in keep):
            P[k] = r
    R = {r["mol"]: r for r in bns.load_prolif(pdir, keep, heavy)}
    both = sorted(set(P) & set(R))
    mism = sum(1 for k in both if P[k].get("heavy") is not None and R[k].get("heavy") is not None
               and P[k]["heavy"] != R[k]["heavy"])
    assert mism == 0, f"{mid}: {mism} molecules differ in heavy-atom count between PLIP and .pt"
    info = dict(label=DISPLAY[mid], join="registry", source_pt=pt, plip_tag=ptag, eval_out=pdir,
                n_plip=len(P), n_prolif=len(R), n=len(both), heavy_gate=".pt", heavy_mismatch=0)
    return molecule_rows(mid, P, R, both), info


def flat_smiles(smiles=None, mol=None):
    """Stereo-free canonical SMILES. PLIP's SMILES carry no stereo; the SDF's 3D mol does."""
    from rdkit import Chem
    try:
        if mol is None:
            mol = Chem.MolFromSmiles(smiles)
        else:
            mol = Chem.Mol(mol)
            Chem.SanitizeMol(mol)
            mol = Chem.RemoveHs(mol)
        return None if mol is None else Chem.MolToSmiles(mol, isomericSmiles=False)
    except Exception:
        return None


def align_pocket(pkeys, skeys, psmi, ssmi):
    """Order-preserving PLIP key -> SDF title map for one pocket.

    A same-length `replace` block is the same slot written two ways -- PLIP's SMILES pass through
    OpenBabel perception, the SDF's through RDKit, and ~1-5% of molecules differ even without
    stereo. It is paired positionally; the heavy-atom gate in join_ablation() then decides.
    """
    sm = difflib.SequenceMatcher(a=[psmi[k] for k in pkeys], b=[ssmi[k] for k in skeys],
                                 autojunk=False)
    pairs = {}
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal" or (op == "replace" and i2 - i1 == j2 - j1):
            pairs.update(zip(pkeys[i1:i2], skeys[j1:j2]))
    return pairs


def join_ablation(tag):
    """PLIP (--samples index) re-keyed into the SDF-title space ProLIF measured, then intersected.

    The two indices are NOT guaranteed to agree. abl_vdw_hbond_on pocket 26 has 21 PLIP rows but
    20 exported SDF molecules: PLIP's m0007 has no SDF counterpart, so a raw p<NNN>_m<MMMM> join
    pairs PLIP m0008..m0020 with SDF m0007..m0019 -- 13 different molecules. Aligning each pocket
    by molecule identity removes that; it is a no-op wherever the indices already agree.
    """
    import build_nci_summary as bns
    from rdkit import Chem
    raw = {}
    for r in bns.load_plip(tag, None):
        k = f"p{int(r['pocket_idx']):03d}_m{int(r['mi']):04d}"
        assert k not in raw, f"{tag}: duplicate PLIP key {k}"
        raw[k] = r
    sdf_heavy, sdf_smi = {}, {}
    for path in sorted(glob.glob(f"eval_out/{tag}/sdf/pocket*.sdf")):
        for mol in Chem.SDMolSupplier(path, sanitize=False, removeHs=False):
            if mol is not None:
                name = mol.GetProp("_Name")
                sdf_heavy[name] = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
                sdf_smi[name] = flat_smiles(mol=mol)
    psmi = {k: flat_smiles(smiles=r.get("smiles")) for k, r in raw.items()}
    by_pocket = lambda keys: {pk: sorted(k for k in keys if k[:4] == pk) for pk in {k[:4] for k in keys}}
    pk_plip, pk_sdf = by_pocket(raw), by_pocket(sdf_heavy)
    pairs = {}
    for pk in sorted(pk_plip):
        pairs.update(align_pocket(pk_plip[pk], pk_sdf.get(pk, []), psmi, sdf_smi))
    P = {title: raw[k] for k, title in pairs.items()}

    prolif = bns.load_prolif(tag, None, {})
    R = {r["mol"]: r for r in prolif}
    assert len(R) == len(prolif), f"{tag}: duplicate ProLIF molecule names"
    both = sorted(set(P) & set(R))
    mism = [k for k in both if sdf_heavy[k] != P[k].get("heavy")]
    assert not mism, f"{tag}: {len(mism)} molecules differ in heavy-atom count, PLIP vs SDF"
    info = dict(label=DISPLAY[tag], join="ablation_identity_aligned", plip_tag=tag, eval_out=tag,
                n_plip=len(raw), n_plip_aligned=len(P), n_prolif=len(R), n=len(both),
                n_rekeyed=sum(1 for k, t in pairs.items() if k != t),
                n_smiles_representation_differs=sum(1 for k, t in pairs.items() if psmi[k] != sdf_smi[t]),
                plip_without_sdf=sorted(set(raw) - set(pairs)),
                heavy_gate="sdf", heavy_mismatch=0)
    return molecule_rows(tag, P, R, both), info


def extract():
    import build_nci_strict as strict
    maps = strict.canonical_maps()
    records, rows = [], {}
    for mid in REGISTRY_ROWS:
        recs, rows[mid] = join_registry(mid, maps)
        records += recs
        print(f"  {mid:16s} plip {rows[mid]['n_plip']:5d}  prolif {rows[mid]['n_prolif']:5d}  -> n {rows[mid]['n']:5d}")
    for tag in ABL_TAGS:
        recs, rows[tag] = join_ablation(tag)
        records += recs
        print(f"  {tag:16s} plip {rows[tag]['n_plip']:5d}  prolif {rows[tag]['n_prolif']:5d}  -> n {rows[tag]['n']:5d}")
    return pd.DataFrame(records), rows


def check_tables(df):
    """The figures must show the tables' populations: same n, same pooled means."""
    checks = []
    strict = {r["id"]: r for r in csv.DictReader(open(STRICT))}
    for mid in REGISTRY_ROWS:
        sub, s = df[df.id == mid], strict[mid]
        assert len(sub) == int(s["n"]), (mid, "n", len(sub), s["n"])
        cols = {"plip_n_hbond": sub.plip_hb_acc + sub.plip_hb_don,
                "plip_n_hydrophobic": sub.plip_hydrophobic, "plip_n_pistack": sub.plip_pistack,
                "plip_n_saltbridge": sub.plip_saltbridge,
                **{f"prolif_{c}": sub[f"prolif_{c}"] for c in PROLIF_KEYS}}
        for col, vals in cols.items():
            assert abs(vals.mean() - float(s[col])) < 5e-4, (mid, col, vals.mean(), s[col])
        checks.append(f"{mid}: n={len(sub)} and {len(cols)} means == {STRICT}")
    for g, path in ABL_CSV.items():
        for r in csv.DictReader(open(path)):
            mid = ABL_MODEL[g].get(r["model"])
            if mid is None:
                continue
            sub = df[df.id == mid]
            assert len(sub) == int(r["n"]), (g, mid, "n", len(sub), r["n"])
            for c in PROLIF_KEYS:
                got, want = sub[f"prolif_{c}"].mean(), float(r[f"int_{c}_mean"])
                assert abs(got - want) < 1e-6, (g, mid, c, got, want)
            checks.append(f"{mid}: n={len(sub)} and 4 ProLIF means == {path} ({r['model']})")
    return checks


def tint(color, amount=0.6):
    rgb = np.array(to_rgb(color))
    return tuple(rgb + (1 - rgb) * amount)


def draw(ax, kind, vals, x, color, width, hatch=None):
    before = {id(c) for c in ax.collections}
    frame = pd.DataFrame({"x": np.full(len(vals), float(x)), "y": vals.astype(float)})
    common = dict(data=frame, x="x", y="y", native_scale=True, orient="x", width=width,
                  color=color, saturation=1, linewidth=0.5, ax=ax)
    if kind == "boxen":
        sns.boxenplot(k_depth="trustworthy", showfliers=True,
                      # explicit facecolor: seaborn's default flier face is unfilled, and with
                      # linewidth=0 the tails silently vanished
                      flier_kws=dict(s=3, marker="o", facecolor="#333333", edgecolor="none",
                                     linewidth=0),
                      line_kws=dict(linewidth=1.0, color=INK),
                      **common)
    else:
        sns.violinplot(cut=0, density_norm="width", inner="quart",
                       inner_kws=dict(linewidth=0.6, color=INK), **common)
    if hatch:
        for c in ax.collections:
            if id(c) not in before and isinstance(c, (PatchCollection, PolyCollection)):
                c.set_hatch(hatch)
    ax.scatter([x], [vals.mean()], marker="D", s=11, facecolor="white", edgecolor=INK,
               linewidth=0.7, zorder=6)


def figure(df, panels, spec, kind, name, handles, xticks, ylabel, width, annotate_mean=False):
    """spec: [(row id, x, colour, hatch)]; xticks: (positions, labels); width in x units.

    annotate_mean prints each pooled mean in bold above its body. It is only safe where the
    bodies sit one x unit apart: the ablation layout packs an ON/OFF pair 0.75 apart, which is
    narrower than the widest label, so those figures keep the diamond alone.
    """
    ncols = 3 if len(panels) == 5 else 2
    fig, axes = plt.subplots(2, ncols, figsize=(7.4, 5.3 if ncols == 3 else 5.6),
                             layout="constrained")
    flat = axes.flat
    for i, (col, title) in enumerate(panels):
        ax = next(flat)
        top, means = 0, []
        for mid, x, color, hatch in spec:
            vals = df.loc[df.id == mid, col].to_numpy()
            draw(ax, kind, vals, x, color, width, hatch)
            top = max(top, vals.max())
            means.append((x, float(vals.mean())))
        hi = max(top, 1)
        ax.set(xlim=(min(s[1] for s in spec) - 0.6, max(s[1] for s in spec) + 0.6),
               ylim=(-0.04 * hi, hi * (1.15 if annotate_mean else 1.04)),
               xlabel="", ylabel=ylabel if i % ncols == 0 else "")
        if annotate_mean:
            # Two decimals throughout. One decimal is narrower but prints the same label for
            # models the tables separate (1.51 and 1.47 both become "1.5"), and a label that
            # hides a difference the figure exists to show is worse than a wider label.
            fmt = "%.2f"
            for x, m in means:
                ax.text(x, hi * 1.05, fmt % m, ha="center", va="bottom", fontsize=6,
                        fontweight="bold", color=INK)
        ax.set_title(f"({chr(97 + i)}) {title}")
        ax.set_xticks(xticks[0], xticks[1], rotation=45, ha="right", rotation_mode="anchor",
                      fontsize=7.5)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
        ax.set_axisbelow(True)
    rest = list(flat)
    if rest:
        for ax in rest:
            ax.axis("off")
        rest[0].legend(handles=handles, loc="center", frameon=False, fontsize=8)
    else:
        fig.legend(handles=handles, loc="outside lower center", ncol=5, frameon=False, fontsize=8)
    save(fig, name, unused=kind == "violin")      # the violins are not in the manuscript
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refigure", action="store_true",
                    help="redraw from the per-molecule CSV without reloading any .pt")
    args = ap.parse_args()

    if args.refigure:
        df = pd.read_csv(PER_MOL)
        rows = json.loads(Path(MANIFEST).read_text())["rows"]
    else:
        df, rows = extract()
        df.to_csv(PER_MOL, index=False, compression="gzip")
    checks = check_tables(df)
    print(f"  table checks passed: {len(checks)}")
    # Derived after the gate, so the gate keeps checking the stored primitives.
    df["plip_hbond"] = df.plip_hb_acc + df.plip_hb_don

    mean_handle = Line2D([], [], marker="D", ls="none", markerfacecolor="white",
                         markeredgecolor=INK, markersize=4, label="Mean")
    base_spec = [(mid, i, REF_COLOR if mid == "reference" else COLORS[mid], None)
                 for i, mid in enumerate(BASELINES)]
    base_handles = [Patch(facecolor=c, label=DISPLAY[m]) for m, _, c, _ in base_spec] + [mean_handle]
    base_ticks = (list(range(len(BASELINES))), [DISPLAY[m] for m in BASELINES])

    abl_spec = [("reference", 0, REF_COLOR, None)]
    for j, (term, on, off) in enumerate([("Ste", "abl_vdw_on", "abl_vdw_off"),
                                         ("Ste+Hb", "abl_vdw_hbond_on", "abl_vdw_hbond_off"),
                                         ("Ste+Hp", "abl_vdw_hydro_on", "abl_vdw_hydro_off"),
                                         ("Ste+Hb+Hp", "ours_vina", "ours_noguide")]):
        x0 = 1.25 + 2 * j
        abl_spec += [(on, x0, TERM_COLOR[term], None), (off, x0 + 0.75, tint(TERM_COLOR[term]), HATCH)]
    abl_handles = [Patch(facecolor=REF_COLOR, label="Test set"),
                   Patch(facecolor="#777777", label="Guidance ON"),
                   Patch(facecolor=tint("#777777"), hatch=HATCH, label="Guidance OFF"), mean_handle]
    abl_ticks = ([0] + [1.625 + 2 * j for j in range(4)], ["Test set", "Ste", "Ste+Hb", "Ste+Hp", "Ste+Hb+Hp"])

    built = []
    for kind in ("boxen", "violin"):
        for inst, panels, ylabel in (("plip", PLIP_PANELS, "Count per molecule"),
                                     ("prolif", PROLIF_PANELS, "Residues per molecule")):
            built.append(figure(df, panels, base_spec, kind, f"nci_dist_{inst}_baselines_{kind}",
                                base_handles, base_ticks, ylabel, width=0.75,
                                annotate_mean=True))
            # ON/OFF sit 0.75 apart inside a term pair, so narrower bodies keep a visible gap.
            built.append(figure(df, panels, abl_spec, kind, f"nci_dist_{inst}_ablation_{kind}",
                                abl_handles, abl_ticks, ylabel, width=0.62))

    Path(MANIFEST).write_text(json.dumps({
        "population": "generated pose; docking-successful; PLIP and ProLIF both measured (tab:f3 cohort)",
        "plip_hbond_split": "hbond entry protisdon=True -> ligand acceptor; acceptor + donor == n_hbond",
        "prolif_unit": "Boolean residue-type cells (PoseCheck int_*), as in tab:f3",
        "mean_marker": "white diamond = pooled per-molecule mean; the baselines figures also print it in bold above each body",
        "rows": rows, "table_checks": checks,
        "source_sha256": {p: sha(p) for p in [STRICT, *ABL_CSV.values(), PER_MOL]},
        "figures": [f"paper/figures/{'unused_figures/' if n.endswith('_violin') else ''}{n}.{s}"
                    for n in built for s in ("pdf", "svg", "png")],
        "palette": {"baselines": {m: c for m, _, c, _ in base_spec},
                    "ablation_terms": TERM_COLOR, "reference": REF_COLOR,
                    "off": "same hue tinted 60% toward white, hatched " + HATCH},
    }, indent=2) + "\n")
    print(f"  wrote {PER_MOL}, {MANIFEST} and {len(built)} figures")


if __name__ == "__main__":
    main()
