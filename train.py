import argparse

import torch
import torch.utils.data.distributed

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger, CSVLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar

import utils
import dataset
import models
import optimizers

from config import cfg

def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description="PyTorch Distributed Training", formatter_class=argparse.ArgumentDefaultsHelpFormatter, add_help=add_help)

    parser.add_argument('--config-file', help="""Configuration file name in config/ folder.""", type=str)
    return parser

def main(cfg):
    if cfg.OUTPUT_DIR:
        utils.mkdir(cfg.OUTPUT_DIR)
    
    utils.init_random_seed()

    print("Loading dataset...")

    train_dataset, val_dataset = dataset.get_datasets(cfg.DATASET.NAME, cfg.DATASET)

    train_collate_fn = dataset.collate_fn
    if cfg.USE_SIMPLE_COPY_PASTE:
        if args.data_augmentation != "lsj":
            raise RuntimeError("SimpleCopyPaste algorithm currently only supports the 'lsj' data augmentation policies")

        train_collate_fn = dataset.copypaste_collate_fn

    # required for TPU support
    train_sampler = None
    if cfg.ACCELERATOR.NAME == 'tpu':
        import torch_xla.core.xla_model as xm

        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal(), shuffle=True
        )
    
    train_dataloader_params = {
        'dataset': train_dataset,
        'collate_fn': train_collate_fn,
        'num_workers': cfg.DATALOADER.WORKERS,
        'batch_size': cfg.DATALOADER.TRAIN_BATCH_SIZE,
        'sampler': train_sampler,
    }

    val_dataloader_params = {
        'dataset': val_dataset,
        'collate_fn': dataset.collate_fn,
        'num_workers': cfg.DATALOADER.WORKERS,
        'batch_size': cfg.DATALOADER.VAL_BATCH_SIZE,
    }

    train_dataloader = torch.utils.data.DataLoader(**train_dataloader_params)
    val_dataloader = torch.utils.data.DataLoader(**val_dataloader_params)

    print("Building logger...")
    wandb_logger_params = {
        'name': cfg.CONFIG_NAME,
        'project': cfg.PROJECT_NAME,
        'offline': True
    }
    wandb_logger = WandbLogger(**wandb_logger_params)

    csv_logger_params = {
        'save_dir': cfg.LOGGER.OUTPUT_DIR,
        'name': cfg.CONFIG_NAME,
    }
    csv_logger = CSVLogger(**csv_logger_params)

    tb_logger_params = {
        'save_dir': cfg.LOGGER.OUTPUT_DIR,
        'name': cfg.CONFIG_NAME,
    }
    tb_logger = TensorBoardLogger(**tb_logger_params)


    print("Building callback...")
    checkpoint_params = {
        'monitor': "val_loss",
        'mode': 'min',
        'every_n_train_steps': 0,
        'every_n_epochs': 1,
        'dirpath': cfg.OUTPUT_DIR,
    }
    checkpoint_callback = ModelCheckpoint(**checkpoint_params)  # Model check

    tqdm_params = {
        'refresh_rate': cfg.PRINT_FREQ,
    }
    tqdm_callback = TQDMProgressBar(**tqdm_params)


    print("Building model...")
    model_params = {
        'num_classes': cfg.DATASET.NUM_CLASSES,
        'version': cfg.MODEL.VERSION,
        'pretrained': cfg.MODEL.PRETRAINED,
        'pretrained_backbone': cfg.MODEL.PRETRAINED_BACKBONE,
    }
    model = models.get_maskrcnn(**model_params)

    print("Building optimizer...")
    optimizer_params = {
        'opt': cfg.OPTIMIZER.NAME,
        'parameters': model.parameters(),
        'lr': cfg.OPTIMIZER.LR,
        'weight_decay': cfg.OPTIMIZER.WEIGHT_DECAY,
        'nesterov': cfg.OPTIMIZER.NESTEROV,
        'momentum': cfg.OPTIMIZER.MOMENTUM,
    }
    optimizer = optimizers.get_optimizer(**optimizer_params)

    print("Building lightning model...")
    module_params = {
        'cfg': cfg,
        'model': model,
        'optimizer': optimizer,
    }
    model_lightning = models.Mask_RCNN_Lightning(**module_params)

    print("Training model...")
    training_params = {
        # 'enable_progress_bar': False,
        'profiler': "simple",
        "logger": [wandb_logger, csv_logger, tb_logger],
        'callbacks': [tqdm_callback, checkpoint_callback],
        'accelerator': cfg.ACCELERATOR.NAME,
        'devices': cfg.ACCELERATOR.DEVICES,
        'max_epochs': cfg.EPOCHS,
    }
    fit_params = {
        'model': model_lightning,
        'train_dataloaders': train_dataloader,
        'val_dataloaders': val_dataloader,
    }

    trainer = Trainer(**training_params)
    trainer.fit(**fit_params)


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    if args.config_file:
        cfg = cfg.merge_from_file(args.config_file)

    main(cfg)
