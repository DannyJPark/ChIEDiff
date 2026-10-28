import torch
from torch.utils.data import Subset

from .pl_pair_dataset import PocketLigandPairDataset


def get_dataset(config, *args, **kwargs):
    """The CrossDocked pocket-ligand dataset, plus the train/val/test subsets if the
    configuration names a split file."""
    if config.name != 'pl':
        raise NotImplementedError(f'Unknown dataset: {config.name}')
    dataset = PocketLigandPairDataset(config.path, *args, **kwargs)

    if 'split' in config:
        split = torch.load(config.split)
        subsets = {k: Subset(dataset, indices=v) for k, v in split.items()}
        return dataset, subsets
    return dataset
