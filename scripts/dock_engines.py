#!/usr/bin/env python3
"""Re-dock and re-minimise the generated poses with smina, Vinardo and gnina.

    python scripts/dock_engines.py --dir eval_out/mymodel --engine smina  --mode dock
    python scripts/dock_engines.py --dir eval_out/mymodel --engine vinardo --mode min
    python scripts/dock_engines.py --dir eval_out/mymodel --engine gnina  --mode dock --pockets 0 1

The mode decides the search and the engine decides the scoring function, and the two stay
orthogonal so that smina-Vina against smina-Vinardo is a single-variable swap: the same
binary, the same search, the same box and the same seed.

    dock   --autobox_ligand <reference ligand> --exhaustiveness 8 --num_modes 1 --seed 42
    min    --minimize, which relaxes in place and takes no box

`vinardo` is smina with --scoring vinardo. gnina adds CNN rescoring and reports the CNN-ranked
pose, which is why its dock and min arms are not interchangeable with smina's.

The AutoDock Vina arm is not here: scripts/vina_meeko_dock.py runs score_only, minimize and
dock in one pass with an exact meeko atom map, and writes its own output.

One CPU per invocation by default, because smina's --cpu parallelises over exhaustiveness runs
and scales sublinearly, so running several pockets at once beats giving one pocket more cores.

Input:  <dir>/manifest.csv and <dir>/sdf/pocket<NNN>.sdf, from scripts/eval_export_sdf.py
Output: <dir>/{smina,smina_vinardo,gnina}/pocket<NNN>_{dock,min}.sdf
        which is the layout scripts/pose_rmsd.py reads.
"""
import argparse
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# engine -> (binary env var, output subdirectory, extra scoring arguments)
ENGINES = {
    'smina':   ('SMINA', 'smina',         []),
    'vinardo': ('SMINA', 'smina_vinardo', ['--scoring', 'vinardo']),
    'gnina':   ('GNINA', 'gnina',         []),
}


def binary_for(engine, override=None):
    if override:
        return override
    var, _, _ = ENGINES[engine]
    name = 'smina' if var == 'SMINA' else 'gnina'
    path = os.environ.get(var) or shutil.which(name) or os.path.expanduser(f'~/bin/{name}')
    if not (shutil.which(path) or os.path.exists(path)):
        sys.exit(f'{name} not found. Install it from its own upstream and put it on PATH, '
                 f'or set ${var}. See envs/README.md.')
    return path


def resolve(path):
    p = Path(path.strip())
    for cand in (p, ROOT / p):
        if cand.exists():
            return cand
    sys.exit(f'not found: {p}')


def count_records(path):
    if not os.path.isfile(path):
        return 0
    with open(path, errors='replace') as fh:
        return sum(1 for line in fh if line.startswith('$$$$'))


def run_pocket(base, row, engine, mode, binary, cpus, exhaustiveness, seed, timeout):
    pk = int(row['pocket_idx'])
    tag = f'{pk:03d}'
    sdf = base / 'sdf' / f'pocket{tag}.sdf'
    if not sdf.is_file():
        return None
    _, sub, scoring = ENGINES[engine]
    outdir = base / sub
    outdir.mkdir(exist_ok=True)
    out = outdir / f'pocket{tag}_{mode}.sdf'

    want = count_records(sdf)
    if count_records(out) >= want and want > 0:
        print(f'pocket{tag} {engine} {mode}: {want} poses already present', flush=True)
        return out

    if mode == 'dock':
        modeargs = ['--autobox_ligand', str(resolve(row['ref_ligand'])),
                    '--exhaustiveness', str(exhaustiveness),
                    '--num_modes', '1', '--seed', str(seed)]
    else:
        modeargs = ['--minimize']

    cmd = [binary, '-r', str(resolve(row['receptor'])), '-l', str(sdf)] + scoring + modeargs
    if engine != 'gnina':                       # gnina has no --cpu
        cmd += ['--cpu', str(cpus)]
    # The scratch file must keep the .sdf extension: smina and gnina pick the output format
    # from it and exit 255 on anything else, writing nothing.
    tmp = outdir / f'.pocket{tag}_{mode}.partial.{os.getpid()}.sdf'
    cmd += ['-o', str(tmp)]
    try:
        r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                           text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        tmp.unlink(missing_ok=True)
        print(f'pocket{tag} {engine} {mode}: TIMEOUT after {timeout}s', flush=True)
        return None
    got = count_records(tmp)
    if r.returncode != 0 or got == 0:
        tail = (r.stderr or '').strip().splitlines()
        why = tail[-1][:160] if tail else f'rc={r.returncode}'
        tmp.unlink(missing_ok=True)
        print(f'pocket{tag} {engine} {mode}: FAILED ({why})', flush=True)
        return None
    os.replace(tmp, out)                        # only a complete run replaces the output
    print(f'pocket{tag} {engine} {mode}: {got}/{want} poses -> {out}', flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True, help='eval_out/<model> (holds manifest.csv and sdf/)')
    ap.add_argument('--engine', required=True, choices=sorted(ENGINES))
    ap.add_argument('--mode', default='dock', choices=('dock', 'min'))
    ap.add_argument('--binary', default='', help='override the engine executable')
    ap.add_argument('--cpus', type=int, default=1)
    ap.add_argument('--exhaustiveness', type=int, default=8)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--timeout', type=int, default=3600, help='seconds per pocket')
    ap.add_argument('--pockets', type=int, nargs='+', help='restrict to these pocket_idx')
    a = ap.parse_args()

    base = Path(a.dir)
    binary = binary_for(a.engine, a.binary)
    with open(base / 'manifest.csv') as fh:
        manifest = list(csv.DictReader(fh))
    todo = [r for r in manifest
            if a.pockets is None or int(r['pocket_idx']) in set(a.pockets)]
    if not todo:
        sys.exit('no pockets selected')
    for row in todo:
        run_pocket(base, row, a.engine, a.mode, binary, a.cpus,
                   a.exhaustiveness, a.seed, a.timeout)
    return 0


if __name__ == '__main__':
    sys.exit(main())
