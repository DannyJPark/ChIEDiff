

import os
import sys
sys.path.append(os.path.abspath('./'))

import pickle
import lmdb
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm
import logging

from gated_energy_diffusion.utils.data import PDBProtein, parse_sdf_file
from gated_energy_diffusion.datasets.pl_data import ProteinLigandData, torchify_dict
# CrossDocked2020's own types index, read only when an LMDB record lacks a cached affinity.
# scripts/data_preparation/clean_crossdocked.py defines the same constant. It is repeated
# here so that the package does not import from the scripts directory.
TYPES_FILENAME = 'types/it2_tt_v1.1_completeset_train0.types'


class PocketLigandPairDataset(Dataset):

    def __init__(self, raw_path, transform=None, version='final'):
        super().__init__()
        self.raw_path = raw_path.rstrip('/')

        # Check if raw_path is already a processed LMDB file
        if self.raw_path.endswith('.lmdb') and os.path.exists(self.raw_path):
            # Use the provided LMDB directly
            self.processed_path = self.raw_path
        else:
            # Generate processed path from raw path
            self.index_path = os.path.join(self.raw_path, 'index.pkl')
            self.processed_path = os.path.join(os.path.dirname(self.raw_path),
                                               os.path.basename(self.raw_path) + f'_processed_{version}.lmdb')

        self.raw_affinity_path = os.path.join(os.path.dirname(self.raw_path), TYPES_FILENAME)
        self.affinity_path = os.path.join('data', 'affinity_info_complete.pkl')
        self.transform = transform
        self.db = None
        self.keys = None
        self.affinity_info = None

        if not os.path.exists(self.processed_path):
            print(f'{self.processed_path} does not exist, begin processing data')
            self._process()
            
    def _load_affinity_info(self):
        if self.affinity_info is not None:
            return
        if os.path.exists(self.affinity_path):
            with open(self.affinity_path, 'rb') as f:
                affinity_info = pickle.load(f)
        else:
            affinity_info = {}
            with open(self.raw_affinity_path, 'r') as f:
                for ln in tqdm(f.readlines()):
                    # <label> <pK> <RMSD to crystal> <Receptor> <Ligand> # <Autodock Vina score>
                    label, pk, rmsd, protein_fn, ligand_fn, vina = ln.split()
                    ligand_raw_fn = ligand_fn[:ligand_fn.rfind('.')]
                    affinity_info[ligand_raw_fn] = {
                        'label': float(label),
                        'rmsd': float(rmsd),
                        'pk': float(pk),
                        'vina': float(vina[1:])
                    }
            # save affinity info
            with open(self.affinity_path, 'wb') as f:
                pickle.dump(affinity_info, f)
        
        self.affinity_info = affinity_info

    def _connect_db(self):
        """Open the LMDB.

        Writable, not read-only: a record without an `affinity` entry has one injected and
        written back by _inject_affinity, so the database is modified in place on first use.
        """
        assert self.db is None, 'A connection has already been opened.'
        self.db = lmdb.open(
            self.processed_path,
            map_size=10*(1024*1024*1024),   # 10GB
            create=False,
            subdir=False,
            readonly=False,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.db.begin() as txn:
            self.keys = list(txn.cursor().iternext(values=False))

        # Build filename to key mapping for tuple-based indexing (with caching)
        cache_path = self.processed_path + '_filename_mapping.pkl'
        if os.path.exists(cache_path):
            # Load cached mapping
            with open(cache_path, 'rb') as f:
                self._filename_to_key = pickle.load(f)
        else:
            # Build mapping and cache it
            self._filename_to_key = {}
            with self.db.begin() as txn:
                for key in tqdm(self.keys, desc='Building filename mapping'):
                    try:
                        data = pickle.loads(txn.get(key))
                        if 'protein_filename' in data and 'ligand_filename' in data:
                            filename_tuple = (data['protein_filename'], data['ligand_filename'])
                            self._filename_to_key[filename_tuple] = key
                    except:
                        continue
            # Save to cache
            with open(cache_path, 'wb') as f:
                pickle.dump(self._filename_to_key, f)

    def _close_db(self):
        self.db.close()
        self.db = None
        self.keys = None
        
    def _process(self):
        db = lmdb.open(
            self.processed_path,
            map_size=10*(1024*1024*1024),   # 10GB
            create=True,
            subdir=False,
            readonly=False,  # Writable
        )
        with open(self.index_path, 'rb') as f:
            index = pickle.load(f)

        num_skipped = 0
        with db.begin(write=True, buffers=True) as txn:
            for i, (pocket_fn, ligand_fn, *_) in enumerate(tqdm(index)):
                if pocket_fn is None: continue
                try:
                    data_prefix = self.raw_path
                    pocket_dict = PDBProtein(os.path.join(data_prefix, pocket_fn)).to_dict_atom()
                    ligand_dict = parse_sdf_file(os.path.join(data_prefix, ligand_fn))
                    data = ProteinLigandData.from_protein_ligand_dicts(
                        protein_dict=torchify_dict(pocket_dict),
                        ligand_dict=torchify_dict(ligand_dict),
                    )
                    data.protein_filename = pocket_fn
                    data.ligand_filename = ligand_fn
                    data = data.to_dict()  # avoid torch_geometric version issue
                    txn.put(
                        key=str(i).encode(),
                        value=pickle.dumps(data)
                    )
                except Exception as e:
                    num_skipped += 1
                    print('Skipping (%d) %s - Error: %s' % (num_skipped, ligand_fn, str(e)))
                    if num_skipped < 5:  # Only print full traceback for first few errors
                        import traceback
                        traceback.print_exc()
                    continue
        db.close()
    
    def __len__(self):
        if self.db is None:
            self._connect_db()
        return len(self.keys)

    def __getitem__(self, idx):
        data = self.get_ori_data(idx)
        if self.transform is not None:
            data = self.transform(data)
        return data
    
    def _update(self, sid, affinity):
        if self.db is None:
            self._connect_db()
        txn = self.db.begin(write=True)
        data = pickle.loads(txn.get(sid))
        data.update({
            'affinity': affinity['vina'],
            'rmsd': affinity['rmsd'],
            'pk': affinity['pk'],
            'rmsd<2': affinity['label']
        })
        txn.put(
            key=sid,
            value=pickle.dumps(data)
        )
        txn.commit()

    def _inject_affinity(self, sid, ligand_path):
        if ligand_path[:-4] in self.affinity_info:
            affinity = self.affinity_info[ligand_path[:-4]]
            self._update(sid, affinity)
        else:
            raise AttributeError(f'affinity_info has no {ligand_path[:-4]}')

    def get_ori_data(self, idx):
        if self.db is None:
            self._connect_db()

        # Handle case where idx is a tuple of (protein_filename, ligand_filename)
        if isinstance(idx, tuple) and len(idx) == 2 and isinstance(idx[0], str):
            # This is a filename tuple from split file
            if idx in self._filename_to_key:
                key = self._filename_to_key[idx]
                original_idx = self.keys.index(key) if key in self.keys else 0
            else:
                # Substituting another record here would train or evaluate on a different
                # pocket than the split asked for, and the only signal would be a warning.
                raise KeyError(
                    f"{idx[0]} is in the split but not in the LMDB at {self.processed_path}. "
                    f"The split indexes one exact build of the database, so a rebuilt or "
                    f"partial LMDB silently shifts every pocket. See data/README.md.")
        # Handle case where idx is a tuple (extract first element)
        elif isinstance(idx, tuple):
            idx = idx[0]
            # Recursively handle the extracted element
            return self.get_ori_data(idx)
        # Handle case where idx is a string - try to convert to int first
        elif isinstance(idx, str):
            try:
                # A decimal string is an index, so resolve it through the integer branch.
                return self.get_ori_data(int(idx))
            except ValueError:
                # Anything else is an LMDB key.
                key = idx.encode()
                original_idx = self.keys.index(key) if key in self.keys else 0
        # Handle case where idx is bytes (direct key)
        elif isinstance(idx, bytes):
            key = idx
            original_idx = self.keys.index(key) if key in self.keys else 0
        # Handle case where idx is an integer
        elif isinstance(idx, int):
            if idx >= len(self.keys):
                raise IndexError(f"Index {idx} is out of range for dataset with {len(self.keys)} samples")
            key = self.keys[idx]
            original_idx = idx
        else:
            raise TypeError(f"Unsupported index type: {type(idx)}")

        data = pickle.loads(self.db.begin().get(key))
        if 'affinity' not in data:
            self._load_affinity_info()
            self._inject_affinity(key, data['ligand_filename'])
            data = pickle.loads(self.db.begin().get(key))

        data = ProteinLigandData(**data)
        data.id = original_idx
        assert data.protein_pos.size(0) > 0
        return data


