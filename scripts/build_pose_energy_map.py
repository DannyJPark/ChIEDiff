#!/usr/bin/env python3
"""Fig. 2c -- Pose-Energy Consistency Map.

Two corrections a re-dock applies to a generated molecule, on one plane, per molecule:

    x_i = RMSD_i(generated pose -> re-docked pose)          the GEOMETRIC correction, Angstrom
    y_i = | S_i(generated pose) - S_i(re-docked pose) |     the ENERGETIC correction, kcal/mol

Bottom-left is a model whose pose the docking search neither moves nor re-scores.

WHY VINARDO AND NOT VINA. The two axes must come from ONE engine under ONE protocol, or the map
mixes experiments. Vina cannot supply both: its affinities come from each model's own sampling .pt
(box = the generated ligand's own bounding box + 5 A, exhaustiveness 16) while every RMSD in
results/pose_fidelity/master.csv comes from eval_out/ (box = --autobox_ligand <CRYSTAL ligand>,
exhaustiveness 8). That is the same reason the pose-stability tables drop the Vina RMSD column and
compare only smina-Vinardo and gnina. Vinardo has both arms under the crystal-ligand autobox, so
x and y here are the same binary, the same box, the same search depth and the same seed -- the pose
and the energy of ONE re-docking run. `--engine gnina` draws the other self-consistent pair; its
dock pose is the CNN's pick rather than the best-affinity pose, so read that panel knowing the
y axis carries that artifact.

Vinardo is not on the Vina scale (no gauss2, refit terms, rotatable-bond penalty weighted 0), so
this y axis is not the y axis of the Vina panel of Fig. 2a. Rank models within the figure.

POPULATION. Identical to Fig. 2a (scripts/build_rsg_ecdf.py): all six Vina/Vinardo/gnina affinities
present, no out-of-the-box grid penalty. Every molecule in it also has a Vinardo dock RMSD, so the
join is 100% and this figure and Fig. 2a stand on exactly the same molecules. Which also means it
inherits Fig. 2a's asymmetry: ours is 8338 of 9857 generated (the 1519 that failed docking have
neither coordinate), while the baselines' .pt files arrived already filtered.

CONTOURS are highest-density regions: the smallest area containing 50% / 90% of that model's
molecules. Density is a 2-D histogram (0.1 A x 0.1 kcal/mol bins) smoothed with a Scott-rule
Gaussian, reflected at x=0 and y=0 because both quantities are non-negative. The grid extends far
past the drawn window (see GRID_MAX), so a contour's 50% / 90% is a share of the WHOLE population,
not of the visible part.

Reads:  results/comparison/figures/rsg_ecdf_per_molecule.csv.gz   (y; from build_rsg_ecdf.py)
        results/pose_fidelity/master.csv                          (x, model rows)
        results/reference_protocol/self_redock_v4/pocket*.csv     (x, Reference row)
Writes: paper/figure_build/figures/panels/fig2c_pose_energy_map{,_gnina,_vina,_purevina}{,_facets}.{png,pdf}

Usage:
    python scripts/build_pose_energy_map.py
    python scripts/build_pose_energy_map.py --engine gnina
"""
import argparse
import csv
import glob
import gzip
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                          # noqa: E402
import numpy as np                                                       # noqa: E402
from scipy.ndimage import gaussian_filter                                # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT_DIR = 'paper/figure_build/figures/panels'   # Main Fig. 3 panel (b); assembled by paper/figure_build/consistency_combined/combine.py
GAP_CACHE = 'results/comparison/figures/rsg_ecdf_per_molecule.csv.gz'
MASTER = 'results/pose_fidelity/master.csv'
REF_DIR = 'results/reference_protocol/self_redock_v4'

# engine -> (display name, master.csv RMSD column, reference CSV RMSD column, file suffix)
#
# 'vina' is the odd one and takes a SECOND code path: for Vinardo and gnina the y axis is read from
# the Fig. 2a cache (eval_out/, --autobox_ligand <crystal>, exhaustiveness 8), while for Vina it is
# read from master.csv's vm_ columns, because that is the only place a Vina affinity and a Vina RMSD
# come from the SAME run. It is NOT the Vina of Fig. 2a: that panel's affinities are each paper's own
# docking in the generated ligand's bounding box, and no RMSD exists on that protocol at all.
ENGINES = {
    'vinardo': ('Vinardo', 'rmsd_dock_vinardo', 'rmsd_vinardo', ''),
    'gnina': ('gnina', 'rmsd_dock_gnina', 'rmsd_gnina', '_gnina'),
    'vina': ('AutoDock Vina', 'vm_rmsd_gen_dock', 'rmsd_vina_meeko', '_vina'),
    # PURE VINA: neither axis comes from this repo's re-docking. Both are read out of the SAME row
    # of the SAME .pt each paper (or the upstream protocol) shipped -- score_only, dock and the
    # generated->docked RMSD of that one docking run, box = the generated ligand's own bounding box.
    # It is the only variant whose y axis IS the |dV| column of the F1 table.
    'purevina': ('AutoDock Vina, as published', None, None, '_purevina'),
}
PUREVINA_DIR = 'results/diagnostics/gen_vs_docked/purevina'
VM = {'score': 'vm_score_only', 'dock': 'vm_dock'}      # master.csv, the meeko Vina 1.2.2 arm
REF_AFF_DIR = 'results/reference_protocol/self_reference_affinity_v2'
PENALTY = 1e3           # |E| this large is a grid penalty, not an affinity (repo-wide threshold)

# Same palette, same slots, same order as Fig. 2a -- a model must not change colour between the
# two figures. Ours is red; see scripts/build_rsg_ecdf.py for why the documented order is rotated
# and for the validator run.
SLOTS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7']
OURS = '#e34948'
REF = '#111111'
# Figure-only axis/legend names. NOT written back to configs/models.json -- that `label` feeds every
# table in results/comparison/, so overriding there would rename the tables as a side effect. Same
# mechanism as scripts/plot_pose_rmsd.py; the CrossDocked reference ligand is "Test set", never
# "Reference".
DISPLAY = {'reference': 'Test set', 'ours_vina': 'Ours', 'pidiff_retrain': 'PIDiff'}

ORDER =['ours_vina', 'reference', 'kgdiff', 'molcraft', 'alidiff', 'pidiff_retrain', 'ipdiff',
         'targetdiff', 'pocket2mol']

# Drawn window, per layout. The facet window holds every model's whole 90% region (widest 10.4 A,
# 6.7 kcal/mol) so each contour closes inside its panel; the combined panel draws 50% only, whose
# widest is 6.7 A / 3.7 kcal/mol, and a window sized for the 90% regions would leave it two-thirds
# empty. A window TIGHTER than the contours it holds is the thing to avoid: it cuts them into open
# arcs that read as stray lines.
WINDOW = {'combined': (7.5, 4.3), 'facets': (11.0, 7.5)}
# The Vina arm searches a 20 A cube instead of the crystal-ligand autobox and moves molecules
# further, so its regions need a wider window than the Vinardo/gnina ones (90%: 12.2 A / 6.2).
WINDOW_BY_ENGINE = {('vina', 'combined'): (9.5, 5.0), ('vina', 'facets'): (13.0, 7.5)}
XMAX, YMAX = WINDOW['facets']    # module default; figure() sets the pair it needs
GRID_MAX = (30.0, 40.0)          # density grid, well past the window so the HDR mass is the total
BIN = 0.1                        # A and kcal/mol


def _sd_tag(path, tag):
    """One SD tag out of a single-molecule SDF, by text.

    These files are written by scripts/self_redock.py two lines at a time, so a four-line parse is
    exact and saves importing RDKit for one float.
    """
    lines = open(path).read().splitlines()
    for i, ln in enumerate(lines):
        # The writer emits '>  <tag>  (1)' with doubled spaces, so match the bracketed name
        # rather than a fixed prefix.
        if ln.startswith('>') and f'<{tag}>' in ln and i + 1 < len(lines):
            try:
                return float(lines[i + 1].strip())
            except ValueError:
                return None
    return None


def load_purevina():
    """-> ({id: (x, y)}, {id: (kept, rows, penalty)}) from the per-model gen-vs-docked CSVs.

    Self-contained on purpose: x and y are two columns of ONE row, so there is no join and no
    key-space risk. (The p<NNN>_m<MMMM> spaces of a model's sbdd .pt and its pose .pt genuinely
    differ -- MolCRAFT is the live case -- and every cross-file join has to remap for it.)

    A molecule is kept when the obabel skeleton match produced an RMSD and neither affinity is a
    grid penalty. The RMSD is the binding constraint and it is NOT missing at random: obabel
    re-perceives the docked PDBQT and fails on exactly the distorted and fragmented poses, so the
    coverage column below is part of the result, not bookkeeping.
    """
    out, cover = {}, {}
    for mid in ORDER:
        xs, ys, pen, n = [], [], 0, 0
        with open(os.path.join(PUREVINA_DIR, f'{mid}.csv')) as f:
            for r in csv.DictReader(f):
                # Denominator = DOCKING-SUCCESSFUL, the repo's population rule. Counting undocked
                # molecules here would report our 9857 against a baseline's pre-filtered 9036 and
                # make one obabel failure rate look like two different things.
                if r.get('dock'):
                    n += 1
                if not (r.get('score_only') and r.get('dock') and r.get('rmsd_gen_dock')):
                    continue
                sc, dk = float(r['score_only']), float(r['dock'])
                if abs(sc) >= PENALTY or abs(dk) >= PENALTY:
                    pen += 1
                    continue
                xs.append(float(r['rmsd_gen_dock']))
                ys.append(abs(sc - dk))
        out[mid] = (np.array(xs), np.array(ys))
        cover[mid] = (len(xs), n, pen)
    return out, cover


def load(engine):
    """-> {model_id: (x, y)} on the Fig. 2a population, plus per-model (kept, population, penalty)."""
    if engine == 'purevina':
        return load_purevina()
    _name, master_col, ref_col, _sfx = ENGINES[engine]
    from_master = engine == 'vina'

    # The Fig. 2a population, as a key set. For Vinardo/gnina it also carries y.
    gap = {}
    with gzip.open(GAP_CACHE, 'rt') as f:
        for r in csv.DictReader(f):
            v = None if from_master else abs(float(r[f'{engine}_score'])
                                             - float(r[f'{engine}_dock']))
            gap.setdefault(r['id'], {})[r['key']] = v

    tag = {m['id']: (m.get('ids') or {}).get('eval_out')
           for m in json.load(open('configs/models.json'))['models']}

    rmsd, ygap, npen = {}, {}, {}
    with open(MASTER) as f:
        for r in csv.DictReader(f):
            m, n = r['model'], r['name']
            if r.get(master_col):
                rmsd.setdefault(m, {})[n] = float(r[master_col])
            if from_master and r.get(VM['score']) and r.get(VM['dock']):
                sc, dk = float(r[VM['score']]), float(r[VM['dock']])
                if abs(sc) >= PENALTY or abs(dk) >= PENALTY:
                    npen[m] = npen.get(m, 0) + 1     # out-of-the-box grid penalty, not an affinity
                else:
                    ygap.setdefault(m, {})[n] = abs(sc - dk)

    # The reference ligand re-docked against its own crystal pose: one molecule per pocket, so its
    # key is p<pocket>_m0000 -- the same key reference_records wrote into the gap cache.
    ref = {}
    for p in sorted(glob.glob(f'{REF_DIR}/pocket*.csv')):
        for r in csv.DictReader(open(p)):
            if r.get(ref_col):
                ref[f"p{int(r['pocket_idx']):03d}_m0000"] = float(r[ref_col])
    rmsd['native'] = ref
    tag['reference'] = 'native'
    if from_master:
        # Same instrument as the model rows -- self_redock.py:59 runs Vina 1.2.2 in a 20 A cube on
        # the crystal centroid at exhaustiveness 16, seed 42, which is vina_meeko_dock.py's `ref`
        # mode. score_only takes no box, so the two halves are comparable by construction.
        sc = {}
        for p in sorted(glob.glob(f'{REF_AFF_DIR}/pocket*.csv')):
            for r in csv.DictReader(open(p)):
                if r.get('vina_score_only'):
                    sc[int(r['pocket_idx'])] = float(r['vina_score_only'])
        rn = {}
        for p in sorted(glob.glob(f'{REF_DIR}/poses/pocket*_vmk.sdf')):
            pk = int(os.path.basename(p)[len('pocket'):].split('_')[0])
            d = _sd_tag(p, 'minimizedAffinity')
            if d is None or pk not in sc or abs(d) >= PENALTY or abs(sc[pk]) >= PENALTY:
                continue
            rn[f'p{pk:03d}_m0000'] = abs(sc[pk] - d)
        ygap['native'] = rn

    out, cover = {}, {}
    for mid in ORDER:
        t = tag[mid]
        pop, rm = gap[mid], rmsd.get(t, {})
        yy = ygap.get(t, {}) if from_master else pop
        keys = [k for k in pop if k in rm and yy.get(k) is not None]
        out[mid] = (np.array([rm[k] for k in keys]), np.array([yy[k] for k in keys]))
        cover[mid] = (len(keys), len(pop), npen.get(t, 0))
    return out, cover


def hdr_contours(x, y, levels=(0.5, 0.9)):
    """-> (X, Y, density, [level values]) for the smallest regions holding `levels` of the mass.

    Histogram-and-smooth rather than gaussian_kde: it is O(n + grid) instead of O(n * grid), and
    reflecting the smoothing kernel at the axes keeps a distribution that piles up against RMSD 0
    from leaking its mass into negative RMSD, which an unbounded KDE would do.
    """
    nx, ny = int(GRID_MAX[0] / BIN), int(GRID_MAX[1] / BIN)
    h, xe, ye = np.histogram2d(np.clip(x, 0, GRID_MAX[0] - 1e-9), np.clip(y, 0, GRID_MAX[1] - 1e-9),
                               bins=[nx, ny], range=[[0, GRID_MAX[0]], [0, GRID_MAX[1]]])
    # Scott's rule per axis, in bins. n^(-1/6) is the 2-D exponent.
    n = max(len(x), 2)
    sx = max(x.std() * n ** (-1 / 6) / BIN, 1.0)
    sy = max(y.std() * n ** (-1 / 6) / BIN, 1.0)
    d = gaussian_filter(h, (sx, sy), mode='reflect')
    d /= d.sum()

    flat = np.sort(d.ravel())[::-1]
    cum = np.cumsum(flat)
    vals = [flat[min(np.searchsorted(cum, q), len(flat) - 1)] for q in levels]
    xc, yc = (xe[:-1] + xe[1:]) / 2, (ye[:-1] + ye[1:]) / 2
    return xc, yc, d.T, vals


DRAW_TITLE = False          # the combined figure supplies the panel title
# This panel is scaled to about 0.35 in the combined figure, so 10 pt here prints at 3.5 pt.
# Chosen so the axis text lands near 7.5 pt (labels) and 6.5 pt (ticks).
AXIS_LABEL_SIZE = 22
TICK_LABEL_SIZE = 19
XLABEL = 'RMSD   ($\\mathrm{\\AA}$)'
YLABEL = 'Score Gap   (kcal/mol)'


def _frame(ax, xlabel=True, ylabel=True):
    ax.set_xlim(0, XMAX)
    ax.set_ylim(0, YMAX)
    ax.grid(True, color='#e9e8e4', linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'):
        ax.spines[sp].set_color('#9b9a95')
        ax.spines[sp].set_linewidth(0.8)
    ax.tick_params(colors='#52514e', length=3, width=0.8, labelsize=TICK_LABEL_SIZE)
    if xlabel:
        ax.set_xlabel(XLABEL, fontsize=AXIS_LABEL_SIZE)
    if ylabel:
        ax.set_ylabel(YLABEL, fontsize=AXIS_LABEL_SIZE)


def colours():
    slot, si = {}, 0
    for k in ORDER:
        if k == 'ours_vina':
            slot[k] = OURS
        elif k == 'reference':
            slot[k] = REF
        else:
            slot[k] = SLOTS[si]
            si += 1
    return slot


def extent90(x, y):
    """-> (x, y) the 90% region actually reaches. Needed because a window sized for the readable
    part of the plane cuts some gnina contours open, and an open contour must say so in words."""
    xc, yc, d, lv = hdr_contours(x, y)
    ys, xs = np.nonzero(d >= lv[1])
    return float(xc[xs.max()]), float(yc[ys.max()])


def draw_model(ax, x, y, c, fill=True, lw50=1.6, anchor_only=False, ls='solid', outer=True):
    """50% region (filled + edge) and 90% region (light line) for one model."""
    xc, yc, d, lv = hdr_contours(x, y)
    if anchor_only:
        ax.contour(xc, yc, d, levels=[lv[0]], colors=[c], linewidths=1.1, alpha=0.9,
                   linestyles='dashed', zorder=2)
        return
    if outer:
        ax.contour(xc, yc, d, levels=[lv[1]], colors=[c], linewidths=1.0, alpha=0.45,
                   linestyles=ls, zorder=2)
    if fill:
        ax.contourf(xc, yc, d, levels=[lv[0], d.max()], colors=[c], alpha=0.16, zorder=1)
    ax.contour(xc, yc, d, levels=[lv[0]], colors=[c], linewidths=lw50, linestyles=ls, zorder=3)


def figure(engine='vinardo', out_stem=None, layout='combined'):
    global XMAX, YMAX
    XMAX, YMAX = WINDOW_BY_ENGINE.get((engine, layout), WINDOW[layout])
    name, _mc, _rc, suffix = ENGINES[engine]
    data, cover = load(engine)
    label = {m['id']: DISPLAY.get(m['id'], m['label'])
             for m in json.load(open('configs/models.json'))['models']}
    slot = colours()

    if layout == 'facets':
        # Nine models will not carry eighteen overlapping contours on one axes -- the 90% regions
        # alone cover most of the plane. Small multiples keep the same encoding per model and put
        # OUR 50% outline in every panel as the common anchor, so each comparison is direct.
        fig, axes = plt.subplots(3, 3, figsize=(10.6, 9.4), dpi=200, sharex=True, sharey=True)
        ax_ours = None
        for i, k in enumerate(ORDER):
            ax = axes[i // 3][i % 3]
            x, y = data[k]
            draw_model(ax, x, y, slot[k], lw50=1.7)
            if k != 'ours_vina':
                draw_model(ax, *data['ours_vina'], OURS, anchor_only=True)
            else:
                ax_ours = ax
            # Ours is a star: a '*' fills about a third of its bounding circle, so it needs roughly
            # twice the dot's markersize to read as the larger mark.
            ours = k == 'ours_vina'
            ax.plot(np.median(x), np.median(y), '*' if ours else 'o', color=slot[k],
                    markersize=19 if ours else 8, markeredgecolor='white',
                    markeredgewidth=1.2 if ours else 1.6, zorder=6 if ours else 5)
            _frame(ax, xlabel=False, ylabel=False)
            # Ink, not the series colour: three of the eight slots sit below 3:1 on white and a
            # yellow facet title is unreadable. The fill and the median dot carry identity.
            ax.set_title(label[k], loc='left', fontsize=11, color='#0b0b0b', pad=6)
            ax.text(0.975, 0.94, f'median  {np.median(x):.2f} $\\mathrm{{\\AA}}$  /  '
                    f'{np.median(y):.2f} kcal', transform=ax.transAxes, ha='right', va='top',
                    fontsize=8.4, color='#52514e')
            # An open 90% contour reads as a stray line unless the panel says where it goes. The
            # gnina y axis is where this bites: its dock pose is the CNN's pick, so a minority of
            # molecules carry a Vina energy tens of kcal/mol off and the region runs off the top.
            ex, ey = extent90(x, y)
            if ex > XMAX or ey > YMAX:
                over = []
                if ex > XMAX:
                    over.append(f'{ex:.1f} $\\mathrm{{\\AA}}$')
                if ey > YMAX:
                    over.append(f'{ey:.1f} kcal')
                ax.text(0.975, 0.855, '90% reaches ' + ' / '.join(over), transform=ax.transAxes,
                        ha='right', va='top', fontsize=8.0, color='#9b9a95')
        del ax_ours
        fig.supxlabel(XLABEL, fontsize=11, color='#0b0b0b')
        fig.supylabel(YLABEL, fontsize=11, color='#0b0b0b')
        if DRAW_TITLE:
            fig.suptitle(f'Pose-Energy Consistency Map ({name})', x=0.008, y=0.995, ha='left',
                         fontsize=13.5, color='#0b0b0b')
            fig.text(0.008, 0.972, 'filled 50% and outlined 90% highest-density regions; the red '
                     'dashed outline repeats our own 50% region in every panel',
                     fontsize=9, color='#52514e', va='top')
        fig.tight_layout(rect=(0.012, 0.012, 1, 0.962))
    else:
        # 50% ONLY. Nine 90% regions cover most of the plane and turn this panel into eighteen
        # crossing arcs with no readable structure -- the facet layout is where both levels fit.
        fig, ax = plt.subplots(figsize=(7.8, 6.4), dpi=200)
        for k in ORDER:
            x, y = data[k]
            ours = k == 'ours_vina'
            draw_model(ax, x, y, slot[k], fill=ours, lw50=2.2 if ours else 1.5, outer=False,
                       ls='dotted' if k == 'reference' else 'solid')
            # 2 px surface ring: the medians overlap, and a white edge keeps each readable on top
            # of whichever contour it lands in. Ours is a star, sized as in the facet layout.
            ax.plot(np.median(x), np.median(y), '*' if ours else 'o', color=slot[k],
                    markersize=19 if ours else 7.5, markeredgecolor='white',
                    markeredgewidth=1.2 if ours else 1.6, zorder=6 if ours else 5, label=label[k])
        _frame(ax)
        if DRAW_TITLE:
            ax.set_title(f'Pose-Energy Consistency Map ({name})', loc='left', fontsize=13.5,
                         color='#0b0b0b', pad=26)
            ax.text(0, 1.018, '50% highest-density region and median per model', fontsize=8.6,
                    transform=ax.transAxes, color='#52514e', va='bottom')
        leg = ax.legend(loc='upper right', fontsize=9.5, framealpha=0.94, edgecolor='#dcdbd6',
                        borderpad=0.7, labelspacing=0.45, handlelength=1.2, numpoints=1)
        leg.get_frame().set_linewidth(0.8)
        for t in leg.get_texts():
            t.set_color('#0b0b0b')

    os.makedirs(OUT_DIR, exist_ok=True)
    stem = out_stem or (f'fig2c_pose_energy_map{suffix}'
                        + ('_facets' if layout == 'facets' else ''))
    png = os.path.join(OUT_DIR, f'{stem}.png')
    fig.savefig(png, bbox_inches='tight', facecolor='white')
    fig.savefig(os.path.join(OUT_DIR, f'{stem}.pdf'), bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f'wrote {png}   [{layout}]')
    print(f'  {"model":<20}{"n":>7}{"of pop":>10}{"penalty":>9}{"RMSD med":>10}{"gap med":>9}'
          f'{"in window":>11}{"90% extent":>16}')
    for k in ORDER:
        x, y = data[k]
        n, ntot, pen = cover[k]
        inw = 100 * float(((x <= XMAX) & (y <= YMAX)).mean())
        # Where the 90% region actually reaches. A '!' marks a contour the window cuts open -- in
        # the facet layout that is legible (one curve per panel) and says "the tail leaves the
        # panel", but it must never be mistaken for a closed region.
        xc, yc, d, lv = hdr_contours(x, y)
        ys, xs = np.nonzero(d >= lv[1])
        ex, ey = xc[xs.max()], yc[ys.max()]
        cut = '!' if (ex > XMAX or ey > YMAX) else ' '
        print(f'  {label[k]:<20}{n:>7}{100 * n / ntot:>9.1f}%{pen:>9}{np.median(x):>10.2f}'
              f'{np.median(y):>9.2f}{inw:>10.1f}%{ex:>9.1f}/{ey:<5.1f}{cut}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--engine', default='vinardo', choices=sorted(ENGINES))
    ap.add_argument('--out', default=None)
    ap.add_argument('--layout', default=None, choices=('combined', 'facets'),
                    help='default: draw both')
    a = ap.parse_args()
    for lay in ([a.layout] if a.layout else ('combined', 'facets')):
        figure(a.engine, a.out, lay)
