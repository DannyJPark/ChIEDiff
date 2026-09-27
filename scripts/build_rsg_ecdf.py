#!/usr/bin/env python3
"""Fig. 2a -- ECDF of the per-molecule Redocking Score Gap (RSG), one panel per scoring function.

    RSG_i = | S_i(generated pose, rescored in place) - S_i(re-docked pose) |

A model that already places its molecule where the scoring function would put it has a small RSG; a
model whose reported affinity depends on the docking search having moved the ligand has a large
one. The ECDF answers "what fraction of this model's molecules have a gap <= x" for every x at
once, so no threshold has to be chosen and the mean/median split (Ours, Vina: 1.19 / 0.24) is
visible rather than hidden.

POPULATION. The population of results/comparison/f1_sbdd/main_table/ -- a molecule counts only if
all six Vina/Vinardo/gnina affinities exist and none is an out-of-the-box grid penalty. Every med
and mean this script prints reproduces that table's published |dV| / |dS| / gnina_gap_aff columns
exactly at 3 dp, all 10 rows, all three engines. The looser "Vina score + dock present, nothing
else required" population was measured too and is NOT kept: it adds exactly one molecule to Ours
(Vina+Guide) and one to IPDiff and moves no median or mean at 3 dp.

The filter only bites on two rows. Ours goes 9857 -> 8338 (15.4%) and IPDiff 9797 -> 9011 (8.0%);
every other model's .pt was shipped by its authors already filtered to docking-successful molecules,
so nothing is removed. Since `docked <=> connected` holds in this repo, our missing 15.4% are the
fragmented molecules -- and they have no dock score at all, so their RSG is UNDEFINED, not merely
unmeasured. The comparison is "our surviving 84.6%" against "their shipped 100%".

THREE PANELS, THREE PROTOCOLS -- read models within a panel, never across panels:
  * Vina comes from each model's own sampling .pt: box = the generated ligand's own bounding box
    + 5 A, exhaustiveness 16.
  * Vinardo and gnina come from eval_out/: box = --autobox_ligand <CRYSTAL ligand>, exhaustiveness
    8. A different box and search depth find different minima.
  * Vinardo is not on the Vina scale (no gauss2, refit terms, rotatable-bond penalty weighted 0).
    Its gap is Vinardo-minus-Vinardo, so the panel is internally consistent -- but the x axes of
    the three panels are not one axis.
  * gnina's Dock pose is the CNN's favourite, not the best-affinity pose: sbatch_eval_gnina.sh runs
    --dock with no --cnn_scoring, so gnina v1.1 searches on the Vina function and then re-ranks by
    CNNscore, and --num_modes 1 keeps the CNN's pick. Its minimizedAffinity can therefore be tens
    of kcal/mol worse than the pose the search actually found, which inflates the gnina gap's upper
    tail for every model. See analysis notes / the gnina-dock-reports-cnn-pose finding.

CNNaffinity is cached but deliberately not plotted: gnina selects the dock pose BY CNNscore, so
scoring it with the CNN again asks the selector to grade its own choice. A small CNN gap means
"the CNN kept a pose it liked", not "re-docking could not improve this molecule".

Reads:  the same sources as scripts/build_main_table.py (registry configs/models.json)
Writes: results/comparison/figures/rsg_ecdf_per_molecule.csv.gz   (per-molecule cache)
        paper/figures/unused_figures/fig2a_rsg_ecdf{,_vinardo,_gnina}.{png,pdf}

Usage:
    python scripts/build_rsg_ecdf.py                 # re-measure, then draw all three panels
    python scripts/build_rsg_ecdf.py --refigure      # redraw from the cache in seconds
    python scripts/build_rsg_ecdf.py --refigure --engine gnina
"""
import argparse
import gzip
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DRAW_TITLE = False          # the combined figure supplies the panel title
# This panel is scaled to about 0.40 when it goes into the combined figure, so 10 pt here prints
# at 4 pt. These are chosen so the axis text lands near 7.5 pt (labels) and 6.5 pt (ticks).
AXIS_LABEL_SIZE = 19
TICK_LABEL_SIZE = 16
OUT_DIR = 'paper/figures/unused_figures'   # not in the manuscript; results/ keeps only data
CACHE = 'results/comparison/figures/rsg_ecdf_per_molecule.csv.gz'

# cache column stem -> (display name, output file suffix, x-axis upper limit)
PANELS = {
    'vina': ('Vina', '', 5.0),
    'vinardo': ('Vinardo', '_vinardo', 5.0),
    'gnina': ('gnina', '_gnina', 5.0),
}
COLS = ['vina', 'vinardo', 'gnina', 'cnnaff']


def measure():
    import model_registry
    import build_main_table as bmt

    rows = []
    for mid in bmt.ROW_IDS:
        entry = model_registry.by_id(mid)
        fn = bmt.reference_records if mid == 'reference' else bmt.model_records
        recs, prov = fn(entry)
        for key, r in recs.items():
            rows.append([mid, key, r['pk']]
                        + [f'{r[f"{c}_{m}"]:.4f}' for c in COLS for m in ('score', 'dock')])
        print(f'{mid:<14} n={len(recs):>6}  ({prov["n_keys"]} keys, {prov["n_vina"]} with vina, '
              f'{prov["n_pen"]} penalties)', flush=True)

    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    head = ['id', 'key', 'pocket_idx'] + [f'{c}_{m}' for c in COLS for m in ('score', 'dock')]
    with gzip.open(CACHE, 'wt') as f:
        f.write(','.join(head) + '\n')
        for r in rows:
            f.write(','.join(str(x) for x in r) + '\n')
    print(f'wrote {CACHE} ({len(rows)} rows)')


# ---------------------------------------------------------------------------- figure

# dataviz reference palette. OURS IS RED, by request (2026-09-08), so red cannot also be a baseline:
# the palette's documented order is rotated by one -- red moves from slot 8 to slot 1 and everything
# else shifts up, which leaves Pocket2Mol on violet and puts blue beside our red. That rotation
# preserves every documented adjacent pair and introduces exactly one new one, red<->blue, which is
# the palette's own diverging pair and the best-separated pair in it. The baselines then take the
# rest IN THE ORDER THEY APPEAR IN THE LEGEND, so the adjacent pairs the validator checked are the
# adjacent pairs the reader actually sees. Reference is not a category and takes no slot, which is
# what keeps the nine curves inside the documented eight.
#   node scripts/validate_palette.js "#e34948,#2a78d6,#eb6834,#1baf7a,#eda100,#e87ba4,#008300,#4a3aa7" --mode light
#   lightness PASS, chroma PASS, CVD PASS (worst adjacent dE 9.1 protan), normal-vision PASS (19.6),
#   contrast WARN on aqua/yellow/magenta -- relieved by the always-visible legend labels.
SLOTS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7']
OURS = '#e34948'
REF = '#111111'          # the native ligand is an anchor, not a category: neutral ink, dotted

# Figure-only axis/legend names. NOT written back to configs/models.json -- that `label` feeds every
# table in results/comparison/, so overriding there would rename the tables as a side effect. Same
# mechanism as scripts/plot_pose_rmsd.py; the CrossDocked reference ligand is "Test set", never
# "Reference".
DISPLAY = {'reference': 'Test set', 'ours_vina': 'Ours', 'pidiff_retrain': 'PIDiff'}


# Colour and legend order are PINNED to the Vina ranking and are identical in all three panels --
# "colour follows the entity, never its rank". Re-sorting each panel by its own median would repaint
# the models between panels, which is exactly the comparison the three figures exist to support.
ORDER =['ours_vina', 'reference', 'kgdiff', 'molcraft', 'alidiff', 'pidiff_retrain', 'ipdiff',
         'targetdiff', 'pocket2mol']

# Measured and cached, deliberately not drawn. `ours_noguide` is the Vina-loss-only ablation arm,
# and this panel is the main comparison -- its curve sits inside our own arm's (Vina med 0.301 vs
# 0.244) and says nothing a baseline row does not. `--keep-ablation` puts it back, at the end of the
# roster so no other model's colour moves.
EXCLUDE = ['ours_noguide']


def load_cache(engine):
    import numpy as np
    vals = {}
    with gzip.open(CACHE, 'rt') as f:
        head = f.readline().strip().split(',')
        ci = {c: i for i, c in enumerate(head)}
        for ln in f:
            r = ln.rstrip('\n').split(',')
            vals.setdefault(r[ci['id']], []).append(
                abs(float(r[ci[f'{engine}_score']]) - float(r[ci[f'{engine}_dock']])))
    return {k: np.sort(np.asarray(v)) for k, v in vals.items()}


def ecdf(v):
    """Step coordinates for P(RSG <= x). Starts at (0,0) so the curve leaves the origin."""
    import numpy as np
    x = np.concatenate([[0.0], v])
    y = np.concatenate([[0.0], np.arange(1, len(v) + 1) / len(v)])
    return x, y


def figure(engine='vina', xmax=None, out_stem=None, keep_ablation=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import model_registry

    name, suffix, default_xmax = PANELS[engine]
    vals = load_cache(engine)
    label = {m['id']: DISPLAY.get(m['id'], m['label']) for m in model_registry.models()}
    order = ORDER + (EXCLUDE if keep_ablation else [])

    slot, style, si = {}, {}, 0
    for k in order:
        if k == 'ours_vina':
            slot[k], style[k] = OURS, ('-', 2.8)
        elif k == 'ours_noguide':
            slot[k], style[k] = OURS, ((0, (5, 2)), 2.2)
        elif k == 'reference':
            slot[k], style[k] = REF, ((0, (1, 1.6)), 2.0)
        else:
            slot[k], style[k] = SLOTS[si], ('-', 1.7)
            si += 1

    # Every curve is flat past ~4 kcal/mol, so a data-driven 99th percentile would spend half the
    # panel on the shared tail and squeeze the region the models actually differ in. One limit for
    # all three panels, and the clipped share goes to stdout.
    xmax = default_xmax if xmax is None else xmax

    fig, ax = plt.subplots(figsize=(7.6, 5.4), dpi=200)
    for k in order:
        x, y = ecdf(vals[k])
        ls, lw = style[k]
        ax.plot(x, y, drawstyle='steps-post', color=slot[k], linestyle=ls, linewidth=lw,
                solid_capstyle='round', zorder=3 if k.startswith('ours') else 2, label=label[k])

    ax.set_xlim(0, xmax)
    ax.set_ylim(0, 1.004)
    ax.set_xlabel('Score Gap   (kcal/mol)', fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel(r'$P(\mathrm{Score\ Gap}\leq x)$', fontsize=AXIS_LABEL_SIZE)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(True, color='#e9e8e4', linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    # The median is where a bar chart would have stopped; drawn once, so every curve's crossing of
    # it is readable without turning the panel into a grid of thresholds.
    ax.axhline(0.5, color='#c9c8c3', linewidth=0.9, linestyle=(0, (4, 3)), zorder=1)
    ax.text(xmax * 0.995, 0.508, 'median', ha='right', va='bottom', fontsize=7.5, color='#9b9a95')
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color('#9b9a95')
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors='#52514e', length=3, width=0.8, labelsize=TICK_LABEL_SIZE)
    # The title lives in the combined figure's panel label, not in the panel itself
    # (paper/figure_build/consistency_combined/). DRAW_TITLE restores it for standalone use.
    if DRAW_TITLE:
        ax.set_title(f'{name} Redocking Score Gap', loc='left', fontsize=13.5,
                     color='#0b0b0b', pad=12)

    leg = ax.legend(loc='lower right', fontsize=9.5, framealpha=0.94, edgecolor='#dcdbd6',
                    borderpad=0.7, labelspacing=0.45, handlelength=2.6)
    leg.get_frame().set_linewidth(0.8)
    for t in leg.get_texts():
        t.set_color('#0b0b0b')                  # text wears ink, never the series colour

    os.makedirs(OUT_DIR, exist_ok=True)
    stem = out_stem or f'fig2a_rsg_ecdf{suffix}'
    png = os.path.join(OUT_DIR, f'{stem}.png')
    fig.savefig(png, bbox_inches='tight', facecolor='white')
    fig.savefig(os.path.join(OUT_DIR, f'{stem}.pdf'), bbox_inches='tight', facecolor='white')
    plt.close(fig)

    # The panel carries no numbers by design, so every number that could have sat in its legend is
    # printed here instead -- including the share clipped off the right edge.
    print(f'wrote {png}')
    for k in sorted(order, key=lambda k: float(np.median(vals[k]))):
        v = vals[k]
        print(f'  {label[k]:<20s} n={len(v):>6}  med={np.median(v):.3f}  mean={v.mean():.3f}  '
              f'P(<=1)={100 * (v <= 1).mean():.1f}%  P(<=2)={100 * (v <= 2).mean():.1f}%  '
              f'>{xmax:g}={100 * (v > xmax).mean():.1f}%')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--refigure', action='store_true', help='redraw from the cache')
    ap.add_argument('--engine', default=None, choices=sorted(PANELS), help='default: all three')
    ap.add_argument('--xmax', type=float, default=None)
    ap.add_argument('--out', default=None)
    ap.add_argument('--keep-ablation', action='store_true',
                    help='also draw Ours (Vina Only), the Vina-loss-only ablation arm')
    a = ap.parse_args()
    if not a.refigure:
        measure()
    for eng in ([a.engine] if a.engine else sorted(PANELS, key=list(PANELS).index)):
        figure(eng, a.xmax, a.out, a.keep_ablation)
