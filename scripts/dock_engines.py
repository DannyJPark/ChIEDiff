#!/usr/bin/env python3
"""Score, re-dock and re-minimise the generated poses with smina, Vinardo and gnina.

The mode decides the search and the engine decides the scoring function, and the two stay
orthogonal, so smina-Vina against smina-Vinardo is a single-variable swap with the same binary,
the same search, the same box and the same seed. Dock mode boxes on the reference ligand,
min mode relaxes in place and takes no box, and score mode scores the pose as generated and
takes no box.

vinardo is smina with --scoring vinardo. gnina adds CNN rescoring and reports the CNN-ranked
pose, which is why its dock and min arms are not interchangeable with smina's.

The AutoDock Vina arm lives in scripts/vina_meeko_dock.py, which runs score_only, minimize and
dock in one pass with an exact meeko atom map and writes its own output.

One CPU per invocation by default, because smina's --cpu parallelises over exhaustiveness runs
and scales sublinearly, so running several pockets at once beats giving one pocket more cores.

The output layout is the one scripts/pose_rmsd.py reads.
"""
import argparse
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Values are (binary env var, output subdirectory, extra scoring arguments).
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
    """Accept a path relative to the current directory or to the repository root.

    An empty field returns None rather than Path('.'), which exists and would hand the engine a
    directory as its box. An unresolvable path skips one pocket rather than ending the run.
    """
    path = (path or '').strip()
    if not path:
        return None
    p = Path(path)
    for cand in (p, ROOT / p):
        if cand.exists():
            return cand
    return None


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
    have = count_records(out)
    # An engine legitimately returns fewer poses than it was given, because a molecule it cannot
    # parse is dropped, so any non-empty output counts as done. Delete the file to force a re-run.
    if have > 0:
        print(f'pocket{tag} {engine} {mode}: {have} poses already present '
              f'(from {want} inputs)', flush=True)
        return out

    receptor = resolve(row['receptor'])
    if receptor is None:
        print(f'pocket{tag} {engine} {mode}: receptor {row["receptor"]} not found, skipping',
              flush=True)
        return None
    if mode == 'dock':
        ref = resolve(row.get('ref_ligand', ''))
        if ref is None:
            # The export writes an empty ref_ligand when the crystal ligand is missing.
            print(f'pocket{tag} {engine} dock: no reference ligand for the box, skipping',
                  flush=True)
            return None
        modeargs = ['--autobox_ligand', str(ref),
                    '--exhaustiveness', str(exhaustiveness),
                    '--num_modes', '1', '--seed', str(seed)]
    elif mode == 'min':
        modeargs = ['--minimize']
    else:
        modeargs = ['--score_only']

    cmd = [binary, '-r', str(receptor), '-l', str(sdf)] + scoring + modeargs
    cmd += ['--cpu', str(cpus)]
    # The scratch file must keep the .sdf extension, because smina and gnina pick the output
    # format from it and exit 255 on anything else, writing nothing.
    tmp = outdir / f'.pocket{tag}_{mode}.partial.{os.getpid()}.sdf'
    cmd += ['-o', str(tmp)]
    # The budget is per molecule, because one engine call takes the whole pocket and gnina's CNN
    # on a CPU costs minutes per docked molecule.
    budget = timeout * max(want, 1) if timeout > 0 else None
    try:
        r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                           text=True, timeout=budget)
    except subprocess.TimeoutExpired:
        tmp.unlink(missing_ok=True)
        print(f'pocket{tag} {engine} {mode}: TIMEOUT after {budget}s', flush=True)
        return False
    got = count_records(tmp)
    if r.returncode != 0 or got == 0:
        tail = (r.stderr or '').strip().splitlines()
        why = tail[-1][:160] if tail else f'rc={r.returncode}'
        tmp.unlink(missing_ok=True)
        print(f'pocket{tag} {engine} {mode}: FAILED ({why})', flush=True)
        return False
    os.replace(tmp, out)                        # only a complete run replaces the output
    print(f'pocket{tag} {engine} {mode}: {got}/{want} poses -> {out}', flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True, help='eval_out/<model> (holds manifest.csv and sdf/)')
    ap.add_argument('--engine', required=True, choices=sorted(ENGINES))
    ap.add_argument('--mode', default='dock', choices=('dock', 'min', 'score'))
    ap.add_argument('--binary', default='', help='override the engine executable')
    ap.add_argument('--cpus', type=int, default=1)
    ap.add_argument('--exhaustiveness', type=int, default=8)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--timeout', type=int, default=900,
                    help='seconds per input molecule, summed over the pocket; 0 for none')
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
    results = [run_pocket(base, row, a.engine, a.mode, binary, a.cpus,
                          a.exhaustiveness, a.seed, a.timeout) for row in todo]
    # Non-zero when a pocket timed out or the engine failed, so bin/measure.sh reports it.
    return 1 if any(r is False for r in results) else 0


if __name__ == '__main__':
    sys.exit(main())
