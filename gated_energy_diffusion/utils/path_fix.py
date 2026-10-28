"""Rewrite the dataset paths stored in a checkpoint config to ./data/ so a checkpoint trained
on another machine still loads."""

import os


def fix_checkpoint_data_paths(ckpt, logger=None):
    """Normalize config.data.path and config.data.split in place and return the checkpoint."""
    if not hasattr(ckpt, 'get'):
        return ckpt

    if 'config' not in ckpt:
        return ckpt

    config = ckpt['config']

    if not hasattr(config, 'data'):
        return ckpt

    def normalize_path(p):
        if p is None:
            return p
        # Ordered longest first, so a lustre or scratch2 path is matched before the bare
        # ./data/ entry.
        for prefix in ['./lustre/data/', 'lustre/data/',
                       './scratch2/data/', 'scratch2/data/', './data/']:
            if prefix in p:
                return './data/' + p.split(prefix)[-1]
        if '/data/' in p:
            return './data/' + p.split('/data/')[-1]
        return p

    if hasattr(config.data, 'path'):
        old_path = config.data.path
        config.data.path = normalize_path(old_path)
        if config.data.path != old_path and logger:
            logger.info(f"Updated data path: {old_path} -> {config.data.path}")

    if hasattr(config.data, 'split'):
        old_split = config.data.split
        config.data.split = normalize_path(old_split)
        if config.data.split != old_split and logger:
            logger.info(f"Updated split path: {old_split} -> {config.data.split}")

    return ckpt
