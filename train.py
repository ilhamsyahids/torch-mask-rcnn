import argparse

import torch
import torch.utils.data.distributed

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger, CSVLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, LearningRateMonitor

import utils
import dataset
import models

from callbacks import LogPredictionsCallback, COCOEvaluator
from config import cfg

def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description="PyTorch Distributed Training", formatter_class=argparse.ArgumentDefaultsHelpFormatter, add_help=add_help)

    parser.add_argument('--config-file', help="""Configuration file name in config/ folder.""", type=str)
    return parser

def main(cfg):
    if cfg.OUTPUT_DIR:
        utils.mkdir(cfg.OUTPUT_DIR + '/' + cfg.CONFIG_NAME)
    
    utils.init_random_seed()

    print("Building logger...")
    utils.mkdir(cfg.LOGGER.OUTPUT_DIR + "/wandb")

    wandb_logger_params = {
        'name': cfg.CONFIG_NAME,
        'project': cfg.PROJECT_NAME,
        'save_dir': cfg.LOGGER.OUTPUT_DIR,
        # 'log_model': True # log model at the end of training
        # 'offline': True
    }
    wandb_logger = WandbLogger(**wandb_logger_params)
    wandb_logger.log_hyperparams(cfg)

    csv_logger_params = {
        'name': cfg.CONFIG_NAME,
        'save_dir': cfg.LOGGER.OUTPUT_DIR,
        'version': cfg.LOGGER.VERSION,
    }
    csv_logger = CSVLogger(**csv_logger_params)

    tb_logger_params = {
        'name': cfg.CONFIG_NAME,
        'save_dir': cfg.LOGGER.OUTPUT_DIR,
        'version': cfg.LOGGER.VERSION,
    }
    tb_logger = TensorBoardLogger(**tb_logger_params)

    print("Building DataModule...")

    datamodule = dataset.COCODataModule(cfg)

    print("Building callback...")
    checkpoint_params = {
        # 'monitor': 'map_bbox',
        # 'every_n_epochs': 1,
        # 'mode': 'max',
        'save_top_k': 0, # no save checkpoint
        # 'save_top_k': -1, # save all
        # 'dirpath': cfg.OUTPUT_DIR + '/' + cfg.CONFIG_NAME,
    }
    checkpoint_callback = ModelCheckpoint(**checkpoint_params)

    tqdm_params = {
        'refresh_rate': cfg.PRINT_FREQ,
    }
    tqdm_callback = TQDMProgressBar(**tqdm_params)

    lr_monitor_params = {
        'logging_interval': 'step',
        'log_momentum': True,
    }
    lr_monitor_callback = LearningRateMonitor(**lr_monitor_params)

    pred_params = {
        'wandb_logger': wandb_logger,
        'class_names': datamodule.class_names,
    }
    pred_callback = LogPredictionsCallback(**pred_params)

    coco_evaluator = COCOEvaluator(datamodule=datamodule)

    print("Building model...")
    module_params = {
        'cfg': cfg,
    }
    model = models.Mask_RCNN(**module_params)

    wandb_logger.watch(model.model, log="all")

    print("Training model...")
    training_params = {
        # 'enable_progress_bar': False,
        'profiler': "simple",
        "logger": [wandb_logger, csv_logger, tb_logger],
        'callbacks': [pred_callback, tqdm_callback, checkpoint_callback, lr_monitor_callback, coco_evaluator],
        'precision': cfg.ACCELERATOR.PRECISION,
        'accelerator': cfg.ACCELERATOR.NAME,
        'devices': cfg.ACCELERATOR.DEVICES,
        'max_epochs': cfg.EPOCHS,
        'strategy': cfg.STRATEGY,
        'num_sanity_val_steps': 0,
    }
    fit_params = {
        'model': model,
        'datamodule': datamodule,
    }
    if cfg.MODEL.USE_SYNC_BATCH_NORM:
        training_params['sync_batchnorm'] = True
    if cfg.MODEL.DETERMINISTIC:
        training_params['deterministic'] = True

    trainer = Trainer(**training_params)
    trainer.fit(**fit_params)


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
        cfg.merge_from_list(["CONFIG_FILE", args.config_file])

    main(cfg)
