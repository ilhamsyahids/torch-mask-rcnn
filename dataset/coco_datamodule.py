import os
import pytorch_lightning as pl
import torch
import torch.utils.data.dataloader
import torch.utils.data.distributed

from .utils import get_datasets, collate_fn, copypaste_collate_fn
from .group_by_aspect_ratio import create_aspect_ratio_groups, GroupedBatchSampler

class COCODataModule(pl.LightningDataModule):
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.class_names = get_datasets(self.cfg.DATASET.NAME, self.cfg.DATASET, mode="class_names")

        # not sure why I should define this
        self.prepare_data_per_node = False
        self._log_hyperparams = False
        self.allow_zero_length_dataloader_with_multiple_devices = False
    
    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str) -> None:
        print("Loading dataset...")

        self.train_dataset = get_datasets(self.cfg.DATASET.NAME, self.cfg.DATASET, mode="train")
        self.val_dataset = get_datasets(self.cfg.DATASET.NAME, self.cfg.DATASET, mode="val")

    def val_dataloader(self):
        print("Building validation dataloader...")

        num_workers = os.cpu_count() # self.cfg.DATALOADER.WORKERS

        val_dataloader_params = {
            'dataset': self.val_dataset,
            'collate_fn': collate_fn,
            'num_workers': num_workers,
            'batch_size': self.cfg.DATALOADER.VAL_BATCH_SIZE,
        }
        self.v_dataloader = torch.utils.data.DataLoader(**val_dataloader_params)
        return self.v_dataloader

    def train_dataloader(self):
        print("Building training dataloader...")

        train_collate_fn = collate_fn
        if self.cfg.USE_SIMPLE_COPY_PASTE:
            if self.cfg.DATASET.DATA_AUGMENTATION != "lsj":
                raise RuntimeError("SimpleCopyPaste algorithm currently only supports the 'lsj' data augmentation policies")

            train_collate_fn = copypaste_collate_fn

        num_workers = self.cfg.DATALOADER.WORKERS # os.cpu_count()

        train_dataloader_params = {
            'dataset': self.train_dataset,
            'collate_fn': train_collate_fn,
            'num_workers': num_workers,
        }

        train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset)

        # required for TPU support
        if self.cfg.ACCELERATOR.NAME == 'tpu':
            import torch_xla.core.xla_model as xm

            train_sampler = torch.utils.data.distributed.DistributedSampler(
                self.train_dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal(), shuffle=True
            )

            train_dataloader_params['sampler'] = train_sampler

        if self.cfg.DATASET.ASPECT_RATIO_GROUP_FACTOR >= 0:
            group_ids = create_aspect_ratio_groups(self.train_dataset, k=self.cfg.DATASET.ASPECT_RATIO_GROUP_FACTOR)
            train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, self.cfg.DATALOADER.TRAIN_BATCH_SIZE)
        else:
            train_dataloader_params['batch_size'] = self.cfg.DATALOADER.TRAIN_BATCH_SIZE
        
        train_dataloader_params['batch_sampler'] = train_batch_sampler

        self.t_dataloader = torch.utils.data.DataLoader(**train_dataloader_params)

        return self.t_dataloader

