#!/usr/bin/env python3
"""PoseBusters physical-validity checks, one process per molecule.

    PY=$(conda info --base)/envs/posebusters/bin/python
    $PY scripts/eval_posebusters.py --dir eval_out/mymodel [--workers 8]
    $PY scripts/eval_posebusters.py --dir eval_out/mymodel --pockets 0 1 2

Each molecule is written to its own SDF and passed to `bust` in dock mode under a timeout.
The unit is one molecule rather than one pocket because the internal-energy check runs an
ETKDG ensemble, and a single pathological molecule can take hundreds of seconds. With a
whole pocket per process that molecule stalls the other hundred; with one molecule per
process a pocket's wall time is its slowest molecule rather than the sum of them.

The config is posebusters' own dock.yml with num_threads: 1 in the energy module, so the
work parallelises across molecules instead of oversubscribing the machine inside one. ETKDG
is seeded (randomSeed=42), so the thread count does not change what the check returns.

A molecule that times out or exits non-zero gets a row whose checks are empty and whose
`error` column says why. Those rows are kept: the failure rate is itself a property of the
model's output, and dropping them would make the cohort depend on the quantity measured.

Input:  <dir>/manifest.csv and <dir>/sdf/pocket<NNN>.sdf, from scripts/eval_export_sdf.py
Output: <dir>/posebusters/pocket<NNN>.csv   (one row per molecule; existing files are kept)
"""
import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / 'configs' / 'posebusters_dock_1thread.yml'
OUT_SUB = 'posebusters'


def resolve(path):
    """manifest.csv stores receptor paths relative to wherever the export ran, so accept
    both the current directory and the repository root. Returns None if neither exists, so
    one unreadable pocket does not abandon the rest."""
    p = Path(path.strip())
    for cand in (p, ROOT / p):
        if cand.exists():
            return cand
    return None


def one_molecule(args):
    """-> (name, {column: value}) for a single molecule, never raising."""
    key, sdf_path, receptor, bust, config, timeout = args
    _, name = key
    cmd = ['timeout', str(timeout), bust, sdf_path, '-p', receptor,
           '--outfmt', 'csv', '--config', config]
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    t0 = time.time()
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       text=True, env=env)
    secs = time.time() - t0
    lines = r.stdout.strip().splitlines()
    if r.returncode != 0 or len(lines) < 2:
        if r.returncode == 124:
            why = f'timeout after {timeout}s'
        else:
            tail = (r.stderr or '').strip().splitlines()
            why = f'rc={r.returncode}: {tail[-1][:160]}' if tail else f'rc={r.returncode}'
        return key, {'molecule': name, 'error': why, 'seconds': round(secs, 1)}
    rows = list(csv.DictReader(lines))
    rec = dict(rows[0])
    rec.pop('file', None)                       # a path in a temporary directory
    rec['molecule'] = name
    rec['error'] = ''
    rec['seconds'] = round(secs, 1)
    return key, rec


def molecules_of(sdf):
    """(name, single-molecule SDF text) per record, read as text so nothing is re-perceived.

    A record is kept even when the file ends without a $$$$ terminator, because dropping it
    would leave a molecule out of the CSV with nothing to show that it was ever there.
    """
    out, block, idx = [], [], 0
    for line in open(sdf):
        block.append(line)
        if line.startswith('$$$$'):
            out.append((block[0].strip() or f'm{idx:04d}', ''.join(block)))
            block, idx = [], idx + 1
    if any(line.strip() for line in block):
        out.append((block[0].strip() or f'm{idx:04d}', ''.join(block) + '$$$$\n'))
    return out


def evaluate(base, bust, config, workers, timeout, pockets=None):
    base = Path(base)
    outdir = base / OUT_SUB
    outdir.mkdir(exist_ok=True)
    with open(base / 'manifest.csv') as fh:
        manifest = list(csv.DictReader(fh))

    for row in manifest:
        pk = int(row['pocket_idx'])
        if pockets is not None and pk not in pockets:
            continue
        tag = f'{pk:03d}'
        sdf = base / 'sdf' / f'pocket{tag}.sdf'
        dest = outdir / f'pocket{tag}.csv'
        if not sdf.is_file() or dest.is_file():
            continue
        rec = resolve(row['receptor'])
        if rec is None:
            print(f'pocket{tag}: receptor {row["receptor"]} not found, skipping', flush=True)
            continue
        receptor = str(rec)
        mols = molecules_of(sdf)
        if not mols:
            print(f'pocket{tag}: the SDF holds no records', flush=True)
            with open(dest, 'w', newline='') as fh:
                fh.write('molecule,error,seconds\n')
            continue
        work = tempfile.mkdtemp(prefix=f'pb_{tag}_')
        try:
            tasks = []
            for i, (name, text) in enumerate(mols):
                # Index the scratch file rather than naming it after the molecule: an SDF
                # title may contain a slash or repeat, which would respectively fail to open
                # and silently overwrite another molecule's input.
                path = os.path.join(work, f'{i:05d}.sdf')
                with open(path, 'w') as fh:
                    fh.write(text)
                tasks.append(((i, name), path, receptor, bust, config, timeout))
            t0 = time.time()
            with ThreadPoolExecutor(workers) as ex:
                results = dict(ex.map(one_molecule, tasks))
        finally:
            shutil.rmtree(work, ignore_errors=True)

        rows = [results[(i, name)] for i, (name, _) in enumerate(mols)]
        # Collect the check columns over every row, because a row that failed carries only
        # molecule/error/seconds. Sorting them keeps the header identical across pockets
        # whatever order the rows happen to be in.
        checks = sorted({k for rec in rows for k in rec
                         if k not in ('molecule', 'error', 'seconds')})
        fields = ['molecule'] + checks + ['error', 'seconds']
        tmp = dest.with_suffix('.csv.tmp')
        with open(tmp, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=fields, restval='')
            w.writeheader()
            w.writerows(rows)
        os.replace(tmp, dest)
        bad = sum(1 for r in rows if r['error'])
        print(f'pocket{tag}: {len(rows)} molecules, {bad} failed, '
              f'{time.time() - t0:.0f} s -> {dest}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True, help='eval_out/<model> (holds manifest.csv and sdf/)')
    ap.add_argument('--bust', default=os.environ.get('BUST') or shutil.which('bust') or 'bust',
                    help='the posebusters CLI; must come from the posebusters environment')
    ap.add_argument('--config', default=str(DEFAULT_CONFIG))
    ap.add_argument('--workers', type=int,
                    default=int(os.environ.get('SLURM_CPUS_PER_TASK', 4)))
    ap.add_argument('--timeout', type=int, default=900, help='seconds per molecule')
    ap.add_argument('--pockets', type=int, nargs='+', help='restrict to these pocket_idx')
    a = ap.parse_args()
    if not shutil.which(a.bust) and not os.path.exists(a.bust):
        sys.exit(f'no such bust executable: {a.bust} (see envs/environment-posebusters.yml)')
    evaluate(a.dir, a.bust, a.config, a.workers, a.timeout,
             set(a.pockets) if a.pockets else None)
    return 0


if __name__ == '__main__':
    sys.exit(main())
