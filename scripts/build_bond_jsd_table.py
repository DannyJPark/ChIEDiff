#!/usr/bin/env python3
"""Bond-length / pair-distance / atom-type Jensen-Shannon panel for the F1 SBDD roster.

WHAT IT REPORTS. The three distribution-matching metrics the TargetDiff/KGDiff code base ships,
computed the way that code base computes them, over the same molecules as F1:

  EJSD + 9 per-bond JSDs  PRIMARY. KGDiff_supplementary.pdf Table S7 (and TargetDiff Table 2).
        Reference = the bond-length profile of the 100 test-set crystal ligands
        (crossdocked_test_vina_docked.pt), bonds C-C / C=C / C:C / C-N / C=N / C:N / C-O / C=O /
        C:O. EJSD = sum_b (N_b / N_B) JSD(b), where N_b counts the GENERATED set's bonds of type
        b and N_B is their sum over the 9 types.
        Source: github.com/CMACH508/KGDiff reproduction.ipynb, cells 36-37, reproduced here as
        notebook_bond_jsd(). NOT utils/evaluation/eval_bond_length.eval_bond_length_profile,
        which compares against the hard-coded EMPIRICAL_DISTRIBUTIONS (8 types, no C:O). Gate G3
        reports how far those constants sit from the test-set profile.
  JSD CC_2A / All_12A     SECONDARY. scripts/evaluate_diffusion.py's pair-distance JSDs: C-C
        pairs under 2 A, and all heavy-atom pairs under 12 A, against the hard-coded
        PAIR_EMPIRICAL_DISTRIBUTIONS. Bonded or not, every pair counts.
  JSD atom type           SECONDARY. evaluate_diffusion.py's element JSD over C/N/O/F/P/S/Cl
        against the hard-coded ATOM_TYPE_DISTRIBUTION (eval_atom_type.py).

JSD HERE IS scipy.spatial.distance.jensenshannon, i.e. the JS DISTANCE (the square root of the
divergence, natural log), which is what every one of those papers prints under "JSD". Values are
therefore not comparable with a paper that reports the divergence itself.

BINS ARE THE REPO'S: bond length 1.1-1.7 A in 0.005 A steps (eval_bond_length_config
.DISTANCE_BINS, 119 edges -> 120 bins incl. the two overflow bins); pairs 100 edges over 0-2 A and
0-12 A. All three use np.searchsorted(bins, d) with side='left', exactly as get_distribution does.

MOLECULE HANDLING matches the notebook: Chem.RemoveAllHs(mol) with its default sanitize=True,
bond type from utils.data.BOND_TYPES, distance from the stored conformer. A molecule whose
RemoveAllHs raises is unsanitizable. The notebook would have crashed on it, and here it is excluded
and counted, the same rule as build_ring_size_table.py. Pair distances and atom types in
evaluate_diffusion.py come from the pre-reconstruction (pos, element) arrays, and the heavy atoms
of the stored mol are those same atoms at the same coordinates.

READ THE REFERENCE SIZE BEFORE THE DECIMALS. The bond reference is 100 molecules. Its C=N, C:O
and C=C profiles rest on a few dozen bonds spread over 120 bins of 0.005 A, so a large part of
every bond JSD is reference-side sampling noise that no model can remove. The per-type reference
bond counts are printed with the table. The pocket bootstrap CI resamples the GENERATED set only,
with the reference held fixed, so it understates the total uncertainty. Read it as "how stable is
this model's number", not as a test between two models.

POPULATION. Docking-successful molecules, as F1 (gate G2 checks n against the F1 cell). For OUR
rows that is a survivor population (see F1's HA% caveat).

ENVIRONMENT. kgdiff (RDKit 2024.03.6), same as build_ring_size_table.py.

Usage:
    conda run -n kgdiff python scripts/build_bond_jsd_table.py --jobs 8
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
from scipy import spatial as sci_spatial
from scipy.special import rel_entr

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)

from rdkit import Chem, RDLogger                                         # noqa: E402
import rdkit                                                             # noqa: E402

RDLogger.DisableLog('rdApp.*')

import build_comparison_tables as bct                                    # noqa: E402
import model_registry                                                    # noqa: E402
from build_property_tables import (build_roster, read_f1, F1_CSV,        # noqa: E402
                                   PAPER_ROSTER, PAPER_RELABEL)
from utils.data import BOND_TYPES                                        # noqa: E402
from utils.evaluation import eval_bond_length_config as cfg              # noqa: E402
from utils.evaluation.eval_atom_type import ATOM_TYPE_DISTRIBUTION       # noqa: E402

# reproduction.ipynb cell 37 REPORT_TYPE, in its order. (atomic_num_lo, atomic_num_hi, bond_type)
REPORT_TYPES = ((6, 6, 1), (6, 6, 2), (6, 6, 4), (6, 7, 1), (6, 7, 2), (6, 7, 4),
                (6, 8, 1), (6, 8, 2), (6, 8, 4))
TYPE_INDEX = {t: i for i, t in enumerate(REPORT_TYPES)}
BOND_LABEL = {(6, 6, 1): 'C-C', (6, 6, 2): 'C=C', (6, 6, 4): 'C:C',
              (6, 7, 1): 'C-N', (6, 7, 2): 'C=N', (6, 7, 4): 'C:N',
              (6, 8, 1): 'C-O', (6, 8, 2): 'C=O', (6, 8, 4): 'C:O'}
DIST_BINS = cfg.DISTANCE_BINS
N_DIST = len(DIST_BINS) + 1
PAIR_BINS = cfg.PAIR_EMPIRICAL_BINS
N_PAIR = {k: len(v) + 1 for k, v in PAIR_BINS.items()}
ELEMENTS = list(ATOM_TYPE_DISTRIBUTION)                                  # 6,7,8,9,15,16,17
ELEM_INDEX = {z: i for i, z in enumerate(ELEMENTS)}

# reproduction.ipynb printed outputs (cells 37, 39, 40, 41), full precision. Each is
# eval_bond_length_profile(<model>_profile, <model>_instance) against the test-set reference, on
# the same .pt the registry id reads: cvae = benchmark/CVAE_test_docked_sf1.5.pt (the notebook's
# cvae_path; KGDiff Table S7 prints it as "liGAN"), kgdiff = benchmark/our_vina_score_docked.pt.
# The notebook's TargetDiff column is NOT here: cell 38 evaluates rep_targetdiff (a re-run we do
# not have) AND weights it by the REFERENCE's bond counts instead of its own, so Table S7's
# TargetDiff EJSD 0.348 is neither our file nor the stated formula.
KGDIFF_NOTEBOOK = {
    'kgdiff': ([0.3786626288068219, 0.5585490587757274, 0.20483423922313168,
                0.37976682238404713, 0.58266826947758, 0.3266088852681303,
                0.44330452514409463, 0.43452520604803785, 0.7620637059553215],
               0.3871865256566862),
    'ar': ([0.6090847252592992, 0.6210207716132661, 0.4504618126310079,
            0.4730538614136365, 0.6356224885267152, 0.5514678253305576,
            0.49176438120726135, 0.5588741374756954, 0.810528953192532],
           0.5443094350322022),
    'pocket2mol': ([0.4959307399546982, 0.5614630763511905, 0.4158556788727053,
                    0.42529234081136563, 0.6285164082861112, 0.4873078388654103,
                    0.4540257178262056, 0.5160702401133311, 0.7828900804172967],
                   0.4611709930756237),
    'cvae': ([0.6011662253516732, 0.6649823526244647, 0.49068619494120264,
              0.6336144840358152, 0.7489228631212249, 0.638033543660596,
              0.6559325575706507, 0.6609779429428257, 0.8325546111576978],
             0.617453735248754),
}
G0_TOLERANCE = 1e-9            # same file, same formula: anything above float noise is a bug
REF_ID = 'reference'
BOOT_B = 1000
BOOT_SEED = 2027


# ------------------------------------------------------------------ per-molecule extraction
def extract_geom(mol, need_fp=False):
    """Extractor for build_comparison_tables.normalize_model: pocket alignment and the docked-only
    fields from extract_mol, plus this molecule's binned bond lengths, pair distances and element
    counts. Only bin INDICES / small count vectors are kept, so a 10k-molecule model pickles
    cheaply back from the worker."""
    md = bct.extract_mol(mol, need_fp)
    md.update({'bond_t': None, 'bond_b': None, 'pair_cc': None, 'pair_all': None,
               'elem': None, 'unsanitizable': False, 'n_bond_unknown': 0, 'n_bond_other': 0,
               'n_bond_all': 0})
    # Measured whether or not it docked: the reported rows filter to docked afterwards, but G0
    # needs the whole file, because the notebook read every entry (CVAE's file holds 9911, of
    # which 9783 docked, and only the 9911 reproduce the notebook).
    rdmol = mol.get('mol')
    if rdmol is None:
        return md
    try:
        m = Chem.RemoveAllHs(Chem.Mol(rdmol))                 # sanitize=True, as the notebook
    except Exception:
        md['unsanitizable'] = True
        return md

    pos = m.GetConformer().GetPositions()
    pdist = np.sqrt(np.sum((pos[None, :] - pos[:, None]) ** 2, axis=-1))

    t_idx, b_idx = [], []
    for bond in m.GetBonds():
        md['n_bond_all'] += 1
        bt = BOND_TYPES.get(bond.GetBondType())
        if bt is None:                                        # notebook would KeyError here
            md['n_bond_unknown'] += 1
            continue
        a1, a2 = bond.GetBeginAtom().GetAtomicNum(), bond.GetEndAtom().GetAtomicNum()
        k = TYPE_INDEX.get((min(a1, a2), max(a1, a2), bt))
        if k is None:
            md['n_bond_other'] += 1
            continue
        d = pdist[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
        t_idx.append(k)
        b_idx.append(int(np.searchsorted(DIST_BINS, d)))
    md['bond_t'] = np.asarray(t_idx, dtype=np.int16)
    md['bond_b'] = np.asarray(b_idx, dtype=np.int16)

    z = np.array([a.GetAtomicNum() for a in m.GetAtoms()])
    iu = np.triu_indices(len(z), k=1)
    d = pdist[iu]
    cc = d[(z[iu[0]] == 6) & (z[iu[1]] == 6) & (d < 2)]
    md['pair_cc'] = np.bincount(np.searchsorted(PAIR_BINS['CC_2A'], cc),
                                minlength=N_PAIR['CC_2A']).astype(np.int32)
    md['pair_all'] = np.bincount(np.searchsorted(PAIR_BINS['All_12A'], d[d < 12]),
                                 minlength=N_PAIR['All_12A']).astype(np.int32)
    md['elem'] = np.bincount([ELEM_INDEX[x] for x in z if x in ELEM_INDEX],
                             minlength=len(ELEMENTS)).astype(np.int32)
    return md


# ------------------------------------------------------------------ pooled counts
def pooled_counts(mds):
    """dict of pooled count arrays over molecules that were measured."""
    bond = np.zeros((len(REPORT_TYPES), N_DIST), dtype=np.int64)
    cc = np.zeros(N_PAIR['CC_2A'], dtype=np.int64)
    al = np.zeros(N_PAIR['All_12A'], dtype=np.int64)
    el = np.zeros(len(ELEMENTS), dtype=np.int64)
    for md in mds:
        if md['bond_t'] is None:
            continue
        np.add.at(bond, (md['bond_t'], md['bond_b']), 1)
        cc += md['pair_cc']
        al += md['pair_all']
        el += md['elem']
    return {'bond': bond, 'cc': cc, 'all': al, 'elem': el}


# ------------------------------------------------------------------ JSD
def js_distance(p, q):
    """scipy.spatial.distance.jensenshannon along the last axis (natural log), vectorised so the
    bootstrap can evaluate B resamples at once. Both inputs are normalised to sum 1, as scipy
    does. Rows with no mass return nan. main() asserts it equals scipy on every point estimate."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    with np.errstate(invalid='ignore', divide='ignore'):
        p = p / p.sum(axis=-1, keepdims=True)
        q = q / q.sum(axis=-1, keepdims=True)
        m = (p + q) / 2.0
        js = (rel_entr(p, m).sum(axis=-1) + rel_entr(q, m).sum(axis=-1)) / 2.0
    return np.sqrt(np.maximum(js, 0.0))


def notebook_bond_jsd(bond_counts, ref_counts):
    """reproduction.ipynb cell 37 eval_bond_length_profile: per-type JSD vs the reference profile,
    and EJSD weighted by THIS set's per-type bond counts. A type the generated set never forms
    has no distribution (the notebook would KeyError), so its JSD is None and its weight 0."""
    n_b = bond_counts.sum(axis=1)
    jsd = [None if n_b[i] == 0 else
           float(sci_spatial.distance.jensenshannon(ref_counts[i] / ref_counts[i].sum(),
                                                    bond_counts[i] / n_b[i]))
           for i in range(len(REPORT_TYPES))]
    tot = n_b.sum()
    ejsd = (sum(j * n for j, n in zip(jsd, n_b) if j is not None) / tot) if tot else None
    return jsd, ejsd


def ejsd_vectorised(bond_counts, ref_counts):
    """EJSD for a stack of resampled count tensors [..., 9, N_DIST]."""
    jsd = js_distance(bond_counts, ref_counts)                       # [..., 9]
    n_b = bond_counts.sum(axis=-1)
    jsd = np.where(n_b > 0, jsd, 0.0)
    return (jsd * n_b).sum(axis=-1) / n_b.sum(axis=-1)


def pair_and_atom_jsd(c):
    return {
        'jsd_cc_2a': float(sci_spatial.distance.jensenshannon(
            cfg.PAIR_EMPIRICAL_DISTRIBUTIONS['CC_2A'], c['cc'] / c['cc'].sum())),
        'jsd_all_12a': float(sci_spatial.distance.jensenshannon(
            cfg.PAIR_EMPIRICAL_DISTRIBUTIONS['All_12A'], c['all'] / c['all'].sum())),
        'jsd_atom': float(sci_spatial.distance.jensenshannon(
            np.array(list(ATOM_TYPE_DISTRIBUTION.values())), c['elem'] / c['elem'].sum())),
    }


# ------------------------------------------------------------------ per model
def measure_one(job):
    entry, src, full2idx, dir2idx = job
    t0 = time.time()
    obj = bct.load_pt(src)
    per_pocket, flat = bct.normalize_model(obj, full2idx, dir2idx,
                                           need_fp=False, extract=extract_geom)
    del obj
    valid = bct.docked_only(flat)
    pockets = sorted(per_pocket)
    per_pk = [pooled_counts(bct.docked_only(per_pocket[i])) for i in pockets]
    return {
        'id': entry['id'], 'label': entry['label'], 'source': src,
        'n_pockets': len(per_pocket), 'n_mols': len(valid),
        'pooled': pooled_counts(valid),
        'file_bond': pooled_counts(flat)['bond'],       # G0 only: the notebook's population
        'n_file': len(flat),
        'pk_bond': np.stack([c['bond'] for c in per_pk]),
        'pk_cc': np.stack([c['cc'] for c in per_pk]),
        'pk_all': np.stack([c['all'] for c in per_pk]),
        'pk_elem': np.stack([c['elem'] for c in per_pk]),
        'n_measured': sum(1 for md in valid if md['bond_t'] is not None),
        'n_unsanitizable': sum(1 for md in valid if md['unsanitizable']),
        'n_bond_unknown': sum(md['n_bond_unknown'] for md in valid),
        'n_bond_other': sum(md['n_bond_other'] for md in valid),
        'n_bond_all': sum(md['n_bond_all'] for md in valid),
        'secs': time.time() - t0,
    }


def score(r, ref_counts, rng):
    """Point estimates + pocket-bootstrap 95% CIs (generated side only, reference fixed)."""
    c = r['pooled']
    jsd, ejsd = notebook_bond_jsd(c['bond'], ref_counts)
    r.update({'bond_jsd': jsd, 'ejsd': ejsd, 'bond_n': c['bond'].sum(axis=1)})
    r.update(pair_and_atom_jsd(c))

    # the vectorised JSD must equal scipy exactly on the point estimate, or the CIs are of
    # something else
    assert abs(float(ejsd_vectorised(c['bond'], ref_counts)) - ejsd) < 1e-12, r['id']
    assert abs(float(js_distance(c['cc'], np.asarray(cfg.PAIR_EMPIRICAL_DISTRIBUTIONS['CC_2A'])))
               - r['jsd_cc_2a']) < 1e-12, r['id']

    P = len(r['pk_bond'])
    idx = rng.integers(0, P, size=(BOOT_B, P))
    counts = np.bincount(np.repeat(np.arange(BOOT_B), P) * P + idx.ravel(),
                         minlength=BOOT_B * P).reshape(BOOT_B, P)          # resample weights
    boot = {
        'ejsd': ejsd_vectorised(np.einsum('bp,ptk->btk', counts, r['pk_bond']), ref_counts),
        'jsd_cc_2a': js_distance(counts @ r['pk_cc'],
                                 np.asarray(cfg.PAIR_EMPIRICAL_DISTRIBUTIONS['CC_2A'])),
        'jsd_all_12a': js_distance(counts @ r['pk_all'],
                                   np.asarray(cfg.PAIR_EMPIRICAL_DISTRIBUTIONS['All_12A'])),
        'jsd_atom': js_distance(counts @ r['pk_elem'],
                                np.array(list(ATOM_TYPE_DISTRIBUTION.values()))),
    }
    r['ci'] = {k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5)))
               for k, v in boot.items()}
    for k in ('pk_bond', 'pk_cc', 'pk_all', 'pk_elem'):
        del r[k]
    return r


# ------------------------------------------------------------------ gates
def run_gates(results, f1, f1_path, ref_counts):
    L = ['GATE REPORT', '=' * 110, '',
         'Reference (G0): KGDiff reproduction.ipynb cells 37/39/40/41 printed output (same .pt files)',
         f'Reference (G2): {f1_path}',
         f'  G0  the 9 per-bond JSDs and EJSD reproduce the notebook to max |diff| <= {G0_TOLERANCE:g},',
         '      computed over EVERY entry of the file as the notebook did (n_file), not the docked',
         '      subset the tables report (among the G0 models only CVAE differs: 9911 vs 9783).',
         '  G1  no bond has a type outside utils.data.BOND_TYPES (the notebook would KeyError)',
         '  G2  n_mols / n_pockets equal the F1 cell exactly (same population, same alignment)',
         '']
    hdr = ['Model', 'G0 max|d|', 'G1 bond', 'G2 n', 'n_file', 'measured', 'unsan', 'bonds',
           'in 9 types', 'unknown']
    rows, ok = [], True
    for r in results:
        nb = KGDIFF_NOTEBOOK.get(r['id'])
        # G0 runs on every entry of the file (the notebook's population), not the docked subset
        # the table reports; see extract_geom.
        fj, fe = notebook_bond_jsd(r['file_bond'], ref_counts)
        if nb is None:
            g0 = 'n/a'
        elif any(j is None for j in fj):
            g0 = 'FAIL missing type'
        else:
            d = max([abs(a - b) for a, b in zip(fj, nb[0])] + [abs(fe - nb[1])])
            g0 = f'{d:.1e}' if d <= G0_TOLERANCE else f'FAIL {d:.2e}'
        g1 = 'PASS' if r['n_bond_unknown'] == 0 else f'FAIL {r["n_bond_unknown"]}'
        ref = f1.get(r['id'])
        if ref is None:
            g2 = 'n/a'
        else:
            bad = [f'{k} {r[k]} vs {str(ref.get(k, "")).strip()}' for k in ('n_mols', 'n_pockets')
                   if str(r[k]) != str(ref.get(k, '')).strip()]
            g2 = 'PASS' if not bad else 'FAIL ' + '; '.join(bad)
        cells = [r['id'], g0, g1, g2, r['n_file'], r['n_measured'], r['n_unsanitizable'],
                 r['n_bond_all'],
                 int(r['bond_n'].sum()), r['n_bond_unknown']]
        if any(str(c).startswith('FAIL') for c in cells):
            ok = False
        rows.append(cells)
    L += bct.render_table(hdr, rows, ['l'] * len(hdr))
    L += ['', f'GATES: {"ALL PASS" if ok else "FAILED"}', '']

    # G3 (informational): are the repo's hard-coded bond constants the test-set profile?
    L += ['G3 (informational) -- utils/evaluation/eval_bond_length_config.EMPIRICAL_DISTRIBUTIONS',
          'vs the test-set reference profile used for EJSD. JS distance per bond type:', '']
    g3 = []
    for t in REPORT_TYPES:
        const = cfg.EMPIRICAL_DISTRIBUTIONS.get(t)
        i = TYPE_INDEX[t]
        g3.append([BOND_LABEL[t], int(ref_counts[i].sum()),
                   'absent' if const is None else
                   f'{sci_spatial.distance.jensenshannon(const, ref_counts[i] / ref_counts[i].sum()):.3f}'])
    L += bct.render_table(['Bond', 'ref bonds', 'JSD(const, test set)'], g3, ['l', 'r', 'r'])
    L += ['', 'Non-zero means the constants are NOT the test-set ligands, so a number computed with',
          'evaluate_diffusion.py\'s bond JSD is not the Table S7 number. This builder uses the',
          'test-set profile, as Table S7 does.']
    return L, ok


# ------------------------------------------------------------------ writers
def _n(x, nd=3):
    return '' if x is None else f'{x:.{nd}f}'


def overall_cols():
    return (['id', 'label', 'n_pockets', 'n_mols', 'n_measured', 'n_unsanitizable',
             'ejsd', 'ejsd_ci_lo', 'ejsd_ci_hi']
            + [f'jsd_{BOND_LABEL[t]}' for t in REPORT_TYPES]
            + [f'n_{BOND_LABEL[t]}' for t in REPORT_TYPES]
            + ['jsd_cc_2a', 'jsd_cc_2a_ci_lo', 'jsd_cc_2a_ci_hi',
               'jsd_all_12a', 'jsd_all_12a_ci_lo', 'jsd_all_12a_ci_hi',
               'jsd_atom', 'jsd_atom_ci_lo', 'jsd_atom_ci_hi', 'source'])


def write_overall_csv(path, results):
    cols = overall_cols()
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in results:
            is_ref = r['id'] == REF_ID
            row = {'id': r['id'], 'label': r['label'], 'n_pockets': r['n_pockets'],
                   'n_mols': r['n_mols'], 'n_measured': r['n_measured'],
                   'n_unsanitizable': r['n_unsanitizable'], 'source': r['source'],
                   # the reference IS the bond target: its bond JSDs are 0 by construction
                   'ejsd': '' if is_ref else _n(r['ejsd'], 4),
                   'ejsd_ci_lo': '' if is_ref else _n(r['ci']['ejsd'][0], 4),
                   'ejsd_ci_hi': '' if is_ref else _n(r['ci']['ejsd'][1], 4)}
            for t, j, n in zip(REPORT_TYPES, r['bond_jsd'], r['bond_n']):
                row[f'jsd_{BOND_LABEL[t]}'] = '' if is_ref else _n(j, 4)
                row[f'n_{BOND_LABEL[t]}'] = int(n)
            for k in ('jsd_cc_2a', 'jsd_all_12a', 'jsd_atom'):
                row[k] = _n(r[k], 4)
                row[f'{k}_ci_lo'], row[f'{k}_ci_hi'] = (_n(v, 4) for v in r['ci'][k])
            w.writerow([row[c] for c in cols])


def txt_table(results, ids=None, relabel=None, title='', note=''):
    keep = [r for r in results if ids is None or r['id'] in ids]
    if ids:
        keep.sort(key=lambda r: ids.index(r['id']))
    relabel = relabel or {}
    L = ['-' * 150, title, '-' * 150, '']
    if note:
        L += [note, '']
    hdr = (['Model', 'n_mol', 'EJSD', 'EJSD 95% CI'] + [BOND_LABEL[t] for t in REPORT_TYPES]
           + ['CC_2A', 'All_12A', 'Atom'])
    rows = []
    for r in keep:
        is_ref = r['id'] == REF_ID
        lo, hi = r['ci']['ejsd']
        rows.append([relabel.get(r['id'], r['label']), r['n_mols'],
                     '(target)' if is_ref else _n(r['ejsd']),
                     '' if is_ref else f'[{lo:.3f}, {hi:.3f}]']
                    + ['' if is_ref else _n(j) for j in r['bond_jsd']]
                    + [_n(r['jsd_cc_2a']), _n(r['jsd_all_12a']), _n(r['jsd_atom'])])
    L += bct.render_table(hdr, rows, ['l'] + ['r'] * (len(hdr) - 1))
    L += ['']
    return L


def bond_count_table(results, ids, relabel):
    keep = sorted([r for r in results if r['id'] in ids], key=lambda r: ids.index(r['id']))
    hdr = ['Model'] + [BOND_LABEL[t] for t in REPORT_TYPES] + ['in 9 types', 'all bonds']
    rows = [[relabel.get(r['id'], r['label'])] + [int(n) for n in r['bond_n']]
            + [int(r['bond_n'].sum()), r['n_bond_all']] for r in keep]
    return (['-' * 150, 'BOND COUNTS per type (the EJSD weights; the test-set row is the size of the'
             ' reference each JSD is measured against)', '-' * 150, '']
            + bct.render_table(hdr, rows, ['l'] + ['r'] * (len(hdr) - 1)) + [''])


def txt_header(stamp, f1_path):
    return [
        '=' * 150,
        'BOND-LENGTH / PAIR-DISTANCE / ATOM-TYPE JSD -- KGDiff Table S7 + evaluate_diffusion.py',
        '=' * 150, '',
        stamp, '',
        'A SUB-TABLE of F1: same molecules, same docking-successful population (gate G2).',
        '',
        'EJSD + 9 bond columns  PRIMARY. KGDiff Table S7 / TargetDiff Table 2. JS distance between',
        '        the bond-length histogram of each bond type and the test-set crystal ligands\' one;',
        '        EJSD weights the 9 by the generated set\'s own bond counts. Lower is closer.',
        '        The test-set row is the TARGET of these columns, so they are blank for it.',
        'CC_2A / All_12A / Atom  SECONDARY. scripts/evaluate_diffusion.py\'s pair-distance and',
        '        element JSDs, against the constants hard-coded in utils/evaluation. The test-set',
        '        row DOES carry these: it shows how far the crystal ligands themselves sit from',
        '        those constants.',
        '',
        'JSD = scipy jensenshannon = JS DISTANCE (sqrt of the divergence, natural log), as every',
        'paper in this lineage prints it.',
        '',
        'THE REFERENCE IS 100 MOLECULES (see BOND COUNTS). Its sparse types -- C=N, C:O, C=C --',
        'carry reference-side sampling noise that sets a floor under every model. The 95% CI',
        'resamples POCKETS of the generated set with the reference fixed (B=1000, seed 2027), so',
        'it is a stability interval for one row, not a test between rows.',
        '',
        f'Gates below check the bond JSD against KGDiff\'s own notebook output (G0) and the',
        f'population against {f1_path} (G2).',
        '',
    ]


def write_txt(path, results, stamp, gate_lines, f1_path):
    L = txt_header(stamp, f1_path)
    L += txt_table(results, title='OVERALL (all registry rows)',
                   note='Row order is the registry `order`, never a sort on the numbers.')
    have = {r['id'] for r in results}
    paper = [i for i in PAPER_ROSTER if i in have]
    if len(paper) == len(PAPER_ROSTER):
        L += txt_table(results, ids=paper, relabel=PAPER_RELABEL,
                       title='PAPER ROSTER -- the nine rows paper/make_tables.sh prints')
        L += bond_count_table(results, paper, PAPER_RELABEL)
    L += gate_lines
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(L) + '\n')


# ------------------------------------------------------------------ driver
def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--out_dir', default='results/comparison/f1_sbdd/bond_jsd')
    p.add_argument('--registry', default='configs/models.json')
    p.add_argument('--sampling_dir', default='results/sampling_results')
    p.add_argument('--canonical', default='targetdiff_vina_docked.pt')
    p.add_argument('--f1_csv', default=F1_CSV)
    p.add_argument('--ids', default=None, help='comma-separated registry ids (reference is added)')
    p.add_argument('--jobs', type=int, default=1)
    p.add_argument('--allow_gate_failure', action='store_true')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ids = [s.strip() for s in args.ids.split(',')] if args.ids else None
    if ids is not None and REF_ID not in ids:
        ids = [REF_ID] + ids                  # every bond JSD is measured against it
    roster = build_roster(args.registry, ids)

    jobs = []
    for m in roster:
        src = model_registry.source_for(m, 'sbdd')
        if src and os.path.isfile(src):
            jobs.append((m, src))
        else:
            print(f'[warn] {m["id"]}: no source file on disk ({src})')
    if not any(m['id'] == REF_ID for m, _ in jobs):
        sys.exit('ABORT: the reference row is the bond-JSD target and is missing.')

    canon = bct.load_pt(os.path.join(args.sampling_dir, args.canonical))
    _, full2idx, dir2idx = bct.build_pocket_maps(bct.canonical_index(canon))
    del canon

    payload = [(m, src, full2idx, dir2idx) for m, src in jobs]
    t0 = time.time()
    if args.jobs > 1:
        import multiprocessing as mp
        with mp.get_context('fork').Pool(args.jobs) as pool:
            results = pool.map(measure_one, payload, chunksize=1)
    else:
        results = [measure_one(j) for j in payload]
    order = {m['id']: i for i, (m, _) in enumerate(jobs)}
    results.sort(key=lambda r: order[r['id']])

    ref_counts = next(r for r in results if r['id'] == REF_ID)['pooled']['bond']
    if (ref_counts.sum(axis=1) == 0).any():
        sys.exit('ABORT: the test-set reference has no bonds of some report type.')
    rng = np.random.default_rng(BOOT_SEED)
    results = [score(r, ref_counts, rng) for r in results]
    for r in results:
        print(f'  {r["id"]:16s} pockets={r["n_pockets"]:3d} mols={r["n_mols"]:6d} '
              f'EJSD={_n(r["ejsd"])} ({r["secs"]:.0f}s)')
    print(f'measured in {time.time() - t0:.0f}s')

    stamp = (f'Built {time.strftime("%Y-%m-%d %H:%M")}; RDKit {rdkit.__version__}; '
             f'registry {args.registry}; {len(results)} models, '
             f'{sum(r["n_mols"] for r in results)} molecules')
    gate_lines, ok = run_gates(results, read_f1(args.f1_csv), args.f1_csv, ref_counts)
    if not ok and not args.allow_gate_failure:
        print('\n'.join(gate_lines))
        sys.exit('ABORT: a validation gate failed; nothing was written.')

    o = lambda n: os.path.join(args.out_dir, n)                          # noqa: E731
    write_overall_csv(o('bond_jsd_overall.csv'), results)
    write_txt(o('bond_jsd_tables.txt'), results, stamp, gate_lines, args.f1_csv)
    print('\n'.join(gate_lines))
    print(f'\nWrote:\n  {o("bond_jsd_overall.csv")}\n  {o("bond_jsd_tables.txt")}')


if __name__ == '__main__':
    main()
