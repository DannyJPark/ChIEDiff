#!/usr/bin/env python3
"""Self-redock each CrossDocked crystal ligand into its own receptor and measure
RMSD(crystal pose, redocked pose). This is the achievable ceiling of the pipeline, because if
an engine cannot reproduce the known crystal pose, a generated pose cannot beat that.

This is an optional control. No step of bin/measure.sh runs it.

    python scripts/self_redock.py --manifest eval_out/mymodel/manifest.csv --pocket 0

Same protocol as the generated-molecule redock:
  smina/vinardo/gnina : --autobox_ligand <crystal> --exhaustiveness 8 --num_modes 1 --seed 42
  vina_meeko          : crystal-ligand-centred 20A cube, exhaustiveness 16, meeko atom map
`vinardo` is smina with --scoring vinardo, so it is the same binary, search, box and seed, and
the ceiling row isolates the scoring function exactly as the model rows do.

RMSD is rdMolAlign.CalcRMS, non-superposed and symmetry-aware, as everywhere else.

Sanitisation has to match the model rows arm by arm. CalcRMS matches on element and bond
order, so whether RDKit perceived aromaticity decides whether an exact graph match exists at
all. The conventions are:
    smina/gnina  (pose_rmsd.py)       reference sanitize=False, docked SDF sanitize=False
    Vina         (vina_meeko_dock.py) reference sanitize=True, meeko mol in memory
Reading the reference sanitised and the smina or gnina output unsanitised would be a third
convention. Aromatic against Kekule then blocks the exact match and forces the
skeleton-isomorphism fallback, which biases RMSD downward and measures the ceiling on a more
permissive instrument than the models it bounds. So crystal_raw, read with sanitize=False, is
the reference for smina and gnina, and crystal, read sanitised, is the reference for meeko.
sanitize=True cannot be dropped everywhere, because meeko needs it. Unsanitised input gives
non-finite Gasteiger charges and writes no PDBQT.

Docked poses are kept under --pose_dir so RMSD can be recomputed without re-docking.

One pocket per invocation. Writes or updates <out>/pocketNNN.csv.
"""
import argparse, csv, os, shutil, subprocess, sys, tempfile
import numpy as np
from rdkit import Chem, RDLogger
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rmsd_core import rmsd_best_effort, heavy, coords
RDLogger.DisableLog('rdApp.*')
# smina and gnina are installed from their own projects. Override with $SMINA and $GNINA.
SMINA = os.environ.get('SMINA') or shutil.which('smina') or os.path.expanduser('~/bin/smina')
GNINA = os.environ.get('GNINA') or shutil.which('gnina') or os.path.expanduser('~/bin/gnina')

def manifest_row(manifest, pocket):
    for r in csv.DictReader(open(manifest)):
        if int(r['pocket_idx'])==pocket: return r
    return None

def rmsd_to_crystal(crystal_raw, docked_sdf):
    """smina/gnina arm. Both sides are unsanitised, which is pose_rmsd.py's convention, so the
    reference row is measured on the same instrument as the model rows. crystal_raw must be the
    sanitize=False read of the reference ligand."""
    d=next(iter(Chem.SDMolSupplier(docked_sdf, sanitize=False, removeHs=False)), None)
    if d is None: return None,'noread'
    r,meth,_=rmsd_best_effort(heavy(d), heavy(crystal_raw))
    return r,meth

def smina_like(binary, crystal_sdf, receptor, ref, out, scoring=''):
    scoreargs = ['--scoring', scoring] if scoring else []
    subprocess.run([binary,'-r',receptor,'-l',crystal_sdf]+scoreargs+['--autobox_ligand',ref,
                    '--exhaustiveness','8','--num_modes','1','--seed','42','-o',out],
                   capture_output=True, timeout=900)

def vina_meeko_self(crystal, receptor_pdbqt, ref_centroid, out_sdf):
    from meeko import MoleculePreparation, PDBQTMolecule, PDBQTWriterLegacy, RDKitMolCreate
    from vina import Vina
    mh=Chem.AddHs(crystal, addCoords=True)
    for cm in (None,'zero'):
        try:
            kw={} if cm is None else {'charge_model':cm}
            s=MoleculePreparation(**kw).prepare(mh)[0]
            pdbqt,ok,err=PDBQTWriterLegacy.write_string(s)
            if ok: break
        except Exception: return None
    else: return None
    v=Vina(sf_name='vina',seed=42,verbosity=0)
    v.set_receptor(receptor_pdbqt); v.set_ligand_from_string(pdbqt)
    v.compute_vina_maps(center=list(ref_centroid), box_size=[20.,20.,20.])
    v.dock(exhaustiveness=16,n_poses=1)
    aff=float(v.energies(n_poses=1)[0][0])
    back=RDKitMolCreate.from_pdbqt_mol(PDBQTMolecule(v.poses(n_poses=1),skip_typing=True))[0]
    # Record the affinity on the pose, so the reference row can carry an AutoDock Vina column in
    # the redocking comparison exactly as it carries smina-Vinardo and gnina.
    # The SD tag name matches smina and gnina ('minimizedAffinity'), so one reader handles all three.
    back.SetProp('minimizedAffinity', f'{aff:.5f}')
    w=Chem.SDWriter(out_sdf); w.write(back); w.close()
    # Return the in-memory mol. vina_meeko_dock.py compares against this object rather than
    # against a re-read of the SDF, so returning it keeps the reference row on the same
    # instrument.
    return back

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--pocket',type=int,required=True)
    ap.add_argument('--manifest',default='eval_out/native/manifest.csv',
                    help='any manifest.csv from eval_export_sdf.py, read only for receptor and ref_ligand')
    ap.add_argument('--engines',default='smina,vinardo,gnina,vina_meeko')
    ap.add_argument('--out',default='self_redock')
    ap.add_argument('--pose_dir',default='',help='keep the docked poses here (default <out>/poses)')
    a=ap.parse_args()
    r=manifest_row(a.manifest,a.pocket)
    if r is None: print(f'pocket {a.pocket} not in manifest'); return 0
    receptor=r['receptor'].strip(); ref=r['ref_ligand'].strip()
    receptor_pdbqt=receptor[:-4]+'.pdbqt'
    # Two reads of the same reference ligand, as the module docstring explains. `crystal` is the
    # sanitised one, which meeko needs. `crystal_raw` is the RMSD reference for smina and gnina,
    # so that arm has both sides unsanitised exactly as pose_rmsd.py does.
    crystal=next(iter(Chem.SDMolSupplier(ref, removeHs=False)), None)
    crystal_raw=next(iter(Chem.SDMolSupplier(ref, sanitize=False, removeHs=False)), None)
    if crystal is None or crystal_raw is None:
        print(f'pocket {a.pocket}: crystal unreadable'); return 0
    row={'pocket_idx':a.pocket,'pocket':r['pocket'],'heavy':crystal.GetNumHeavyAtoms()}
    pose_dir=a.pose_dir or os.path.join(a.out,'poses')
    os.makedirs(pose_dir, exist_ok=True)
    tmp=tempfile.mkdtemp(prefix='selfdock_')
    engines=a.engines.split(',')
    if 'smina' in engines:
        o=f'{pose_dir}/pocket{a.pocket:03d}_smina.sdf'; smina_like(SMINA,ref,receptor,ref,o)
        rm,meth=(rmsd_to_crystal(crystal_raw,o) if os.path.exists(o) and os.path.getsize(o) else (None,'fail'))
        row['rmsd_smina']='' if rm is None else round(rm,4); row['method_smina']=meth
    if 'vinardo' in engines:
        o=f'{pose_dir}/pocket{a.pocket:03d}_vinardo.sdf'
        smina_like(SMINA,ref,receptor,ref,o,scoring='vinardo')
        rm,meth=(rmsd_to_crystal(crystal_raw,o) if os.path.exists(o) and os.path.getsize(o) else (None,'fail'))
        row['rmsd_vinardo']='' if rm is None else round(rm,4); row['method_vinardo']=meth
    if 'gnina' in engines:
        o=f'{pose_dir}/pocket{a.pocket:03d}_gnina.sdf'; smina_like(GNINA,ref,receptor,ref,o)
        rm,meth=(rmsd_to_crystal(crystal_raw,o) if os.path.exists(o) and os.path.getsize(o) else (None,'fail'))
        row['rmsd_gnina']='' if rm is None else round(rm,4); row['method_gnina']=meth
    if 'vina_meeko' in engines:
        o=f'{pose_dir}/pocket{a.pocket:03d}_vmk.sdf'
        try:
            back=vina_meeko_self(crystal, receptor_pdbqt, coords(heavy(crystal)).mean(0), o)
            if back is None:
                rm,meth=None,'fail'
            else:
                # both sides sanitised and in memory, as in vina_meeko_dock.py
                rm,meth,_=rmsd_best_effort(heavy(back), heavy(crystal))
        except Exception as e:
            rm,meth=None,f'exc:{type(e).__name__}'
        row['rmsd_vina_meeko']='' if rm is None else round(rm,4); row['method_vina_meeko']=meth
    import shutil; shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(a.out, exist_ok=True)
    outp=f'{a.out}/pocket{a.pocket:03d}.csv'
    with open(outp,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(row.keys()),lineterminator='\n'); w.writeheader(); w.writerow(row)
    print(f'  pocket{a.pocket:03d}: '+' '.join(f'{k}={row.get(k)}' for k in row if k.startswith('rmsd')))
    return 0

if __name__=='__main__': sys.exit(main())
