import torch
import numpy as np
import pandas as pd
from copy import copy
from pathlib import Path
import pytorch_lightning as pl
from argparse import ArgumentParser
from data_utils.dataset import CMambaDataset, DataConverter


def prepend_validation_history(train, validation, window_size, base_window=14):
    """Align a longer-window validation set to the paper's target dates."""
    prepend = max(0, window_size - base_window)
    if prepend == 0:
        return validation.copy()
    return pd.concat(
        [train.tail(prepend), validation], ignore_index=True
    )


def limit_validation_targets(validation, window_size, target_count):
    """Keep the earliest checkpoint-selection targets and their causal history."""
    if target_count is None:
        return validation
    if not isinstance(target_count, int) or target_count <= 0:
        raise ValueError('validation target count must be a positive integer')
    required_rows = window_size + 1 + target_count
    if required_rows > len(validation):
        raise ValueError(
            f'validation needs {required_rows} rows for {target_count} targets, '
            f'got {len(validation)}'
        )
    if hasattr(validation, 'iloc'):
        return validation.iloc[:required_rows].copy()
    return validation[:required_rows].copy()

def worker_init_fn(worker_id):
    """
    Handle random seeding.
    """
    worker_info = torch.utils.data.get_worker_info()
    data = worker_info.dataset  # pylint: disable=no-member

    # Check if we are using DDP
    is_ddp = False
    if torch.distributed.is_available():
        if torch.distributed.is_initialized():
            is_ddp = True

    # for NumPy random seed we need it to be in this range
    base_seed = worker_info.seed  # pylint: disable=no-member

    if is_ddp:  # DDP training: unique seed is determined by worker and device
        seed = base_seed + torch.distributed.get_rank() * worker_info.num_workers
    else:
        seed = base_seed

class CMambaDataModule(pl.LightningDataModule):

    def __init__(
        self, 
        data_config, 
        train_transform, 
        val_transform, 
        test_transform, 
        batch_size, 
        distributed_sampler,
        num_workers=4,
        normalize=False,
        window_size=14,
        exogenous_features=None,
        cross_boundary_validation=False,
        validation_base_window=14,
        validation_selection_rows=None,
    ):

        super().__init__()

        if exogenous_features and normalize:
            raise ValueError(
                'exogenous_features cannot be combined with normalize=True: the '
                'legacy normalize() is train-wide min-max (leaky) and would sweep '
                'the exogenous columns. Track-3 configs must use window_norm only.')

        self.data_config = data_config
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.distributed_sampler = distributed_sampler
        self.window_size = window_size
        self.factors = None
        self.exogenous_features = list(exogenous_features or [])

        self.converter = DataConverter(data_config)
        train, val, test = self.converter.get_data()
        self.data_dict = {
            'train': train,
            'val': val,
            'test': test,
        }

        if self.exogenous_features:
            # Track 3: join frozen exogenous columns by calendar day (in memory
            # only — the cached split CSVs on disk stay pristine OHLCV).
            from data_utils.exogenous import load_exogenous_frame, merge_exogenous
            exo = load_exogenous_frame(self.exogenous_features)
            self.data_dict = {
                split: merge_exogenous(df, exo, self.exogenous_features)
                for split, df in self.data_dict.items()
            }

        if cross_boundary_validation:
            self.data_dict['val'] = prepend_validation_history(
                self.data_dict['train'],
                self.data_dict['val'],
                window_size=self.window_size,
                base_window=validation_base_window,
            )

        if validation_selection_rows is not None:
            if cross_boundary_validation:
                raise ValueError(
                    'validation_selection_rows and cross_boundary_validation '
                    'cannot be combined'
                )
            self.data_dict['val'] = limit_validation_targets(
                self.data_dict['val'],
                window_size=self.window_size,
                target_count=validation_selection_rows,
            )

        if normalize:
            self.normalize()


    def normalize(self):
        tmp = {}
        for split in self.data_dict.keys():
            data = self.data_dict.get(split)
            for key in data.keys():
                max_val = max(data.get(key))
                min_val = min(data.get(key))
                if not key in tmp.keys():
                    tmp[key] = {
                        'min': min_val,
                        'max': max_val
                    }
                else:
                    tmp.get(key)['max'] = max(max_val, tmp.get(key).get('max'))
                    tmp.get(key)['min'] = min(min_val, tmp.get(key).get('min'))
        for data in self.data_dict.values():
            for key in data.keys():
                if key == 'Timestamp':
                    data['Timestamp_orig'] = data.get(key)
                data[key] = (data.get(key) - tmp.get(key).get('min')) / (tmp.get(key).get('max') - tmp.get(key).get('min'))
        self.factors = tmp

    def _create_data_loader(
        self,
        data_split,
        data_transform,
        batch_size=None
    ) :
        dataset = CMambaDataset(
            data=self.data_dict.get(data_split),
            split=data_split,
            window_size=self.window_size,
            transform=data_transform,
        )

        batch_size = self.batch_size if batch_size is None else batch_size
        sampler = torch.utils.data.DistributedSampler(dataset) if self.distributed_sampler else None
        
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_fn,
            sampler=sampler,
            drop_last=False
        )
        return dataloader

    def train_dataloader(self):
        return self._create_data_loader(data_split='train', data_transform=self.train_transform)

    def val_dataloader(self):
        return self._create_data_loader(data_split='val', data_transform=self.val_transform)

    def test_dataloader(self):
        return self._create_data_loader(data_split='test', data_transform=self.test_transform)
