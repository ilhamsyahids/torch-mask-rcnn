import time
import pytorch_lightning as pl
import torch
import torchvision.models.detection

from dataset.coco import get_coco_api_from_dataset
from dataset.coco_eval import CocoEvaluator
from models.mask_rcnn import Mask_RCNN

class COCOEvaluator(pl.Callback):

    def __init__(self, datamodule) -> None:
        super().__init__()
        self.datamodule = datamodule

    def _get_iou_types(self, model):
        model_without_ddp = model
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_without_ddp = model.module
        iou_types = ["bbox"]
        if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
            iou_types.append("segm")
        if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
            iou_types.append("keypoints")
        return iou_types

    def on_validation_start(self, trainer: "pl.Trainer", pl_module: "Mask_RCNN") -> None:
        self.coco = get_coco_api_from_dataset(self.datamodule.v_dataloader.dataset)
        self.iou_types = self._get_iou_types(pl_module.model)
        self.coco_evaluator = CocoEvaluator(self.coco, self.iou_types)
    
    def on_validation_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.coco_evaluator = CocoEvaluator(self.coco, self.iou_types)

    def on_validation_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs, batch, batch_idx: int) -> None:
        if self.coco_evaluator is not None:
            res = outputs['evaluate']
            evaluator_time = time.time()
            self.coco_evaluator.update(res)
            evaluator_time = time.time() - evaluator_time

            evaluate_params = {
                'batch_size': len(batch),
                'prog_bar': True,
                # 'sync_dist': True,
            }
            pl_module.log('evaluator_time', evaluator_time, **evaluate_params)

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.coco_evaluator is None:
            return

        self.coco_evaluator.synchronize_between_processes()
        self.coco_evaluator.accumulate()
        summaries = self.coco_evaluator.summarize()

        for iou_type in self.iou_types:
            summary = summaries[iou_type]
            map = summary["map"]
            map_50 = summary["map_50"]
            mar = summary["mar"]
            table = summary["table"]

            pl_module.print(table)

            params = {
                'prog_bar': True,
                'logger': True,
                # 'sync_dist': True,
            }
            pl_module.log(f"map_{iou_type}", map, **params)
            pl_module.log(f'map_50_{iou_type}', map_50, **params)
            pl_module.log(f'mar_{iou_type}', mar, **params)

            # skip since too many for logging
            # per_class_AP_dict = summary["per_class_AP_dict"]
            # per_class_AR_dict = summary["per_class_AR_dict"]
            # pl_module.log_dict(per_class_AP_dict)
            # pl_module.log_dict(per_class_AR_dict)
