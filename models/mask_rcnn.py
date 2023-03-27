import datetime
import time
import pytorch_lightning as pl
from pprint import pprint

import torch
import torch.utils.data
import torchvision.models
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection import maskrcnn_resnet50_fpn, maskrcnn_resnet50_fpn_v2
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights, MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.resnet import ResNet50_Weights
from torchmetrics.detection.mean_ap import MeanAveragePrecision

import optimizers

from dataset.coco import get_coco_api_from_dataset
from dataset.coco_eval import CocoEvaluator
from utils.dist import reduce_dict


def get_maskrcnn(version: str = 'v2', pretrained: bool = True, pretrained_backbone: bool = True, num_classes: int = 91) -> MaskRCNN:
    """
    function to return the Mask RCNN model
    :param version: which version we want to use
    :param pretrained: boolean value, if we want to use pretrained weights on COCO dataset
    :param pretrained_backbone: boolean value, if we want to use pretrained backbone weights
    :param num_classes: the number of classes we want to classify (Note: classes + background)
    :return:
    """

    assert version in ['v1', 'v2'], 'You have to choose which version you want to use:\n ' \
                                    'v1 is Mask-RCNN with Resnet50 backbone and FPN network \n' \
                                    'v2 is improved version of Mask-RCNN with vision transformer.'

    if version == 'v1':
        if pretrained:
            model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
        elif pretrained_backbone:
            model = maskrcnn_resnet50_fpn(weights_backbone=ResNet50_Weights.DEFAULT)
        else:
            model = maskrcnn_resnet50_fpn()
        if num_classes != 91:
            # get the number of input features for the classifier
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            # now get the number of input features for the mask classifier
            in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
            hidden_layer = 256
            # replace the pre-trained head with a new one
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
            model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    elif version == 'v2':
        if pretrained:
            model = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
        elif pretrained_backbone:
            model = maskrcnn_resnet50_fpn_v2(weights_backbone=ResNet50_Weights.DEFAULT)
        else:
            model = maskrcnn_resnet50_fpn_v2()

        if num_classes != 91:
            # get the number of input features for the classifier
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            # now get the number of input features for the mask classifier
            in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
            hidden_layer = 256
            # replace the pre-trained head with a new one
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
            model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model



class Mask_RCNN(pl.LightningModule):
    def __init__(self, cfg):
        """
        Constructor for the Mask_RCNN class
        :param cfg: yacs configuration that contains all the necessary information about the available backbones
        :param model: the model
        :param optimizer: the optimizer
        """
        super(Mask_RCNN, self).__init__()

        self.save_hyperparameters()

        self.cfg = cfg

        model_params = {
            'num_classes': cfg.DATASET.NUM_CLASSES,
            'version': cfg.MODEL.VERSION,
            'pretrained': cfg.MODEL.PRETRAINED,
            'pretrained_backbone': cfg.MODEL.PRETRAINED_BACKBONE,
        }
        self.model = get_maskrcnn(**model_params)

        if not self._use_coco_evaluator():
            self.metric_bbox = MeanAveragePrecision(class_metrics=True)


    def forward(self, inputs, targets=None):
        return self.model(inputs, targets)


    def on_train_start(self) -> None:
        self.start_time = time.time()


    def on_train_end(self) -> None:
        total_time = time.time() - self.start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        pprint(f"Training time {total_time_str}")


    def on_train_epoch_start(self) -> None:
        self.training_step_outputs = []
        self.epoch_time = time.time()


    def on_train_epoch_end(self) -> None:
        iter_time = time.time() - self.epoch_time
        iter_time_str = str(datetime.timedelta(seconds=int(iter_time)))
        pprint(f"Total time on {self.current_epoch} epoch: {iter_time_str}")

        epoch_losses = torch.tensor([batch_loss.item() for batch_loss in self.training_step_outputs])
        loss_mean = torch.mean(epoch_losses)

        self.log('training_loss', loss_mean, on_epoch=True)

        self.training_step_outputs.clear()


    def training_step(self, train_batch, batch_idx):
        images, targets = train_batch

        targets = [{k: v for k, v in t.items()} for t in targets]
        loss_dict = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_value = losses_reduced.item()

        self.log('training_step_loss', loss_value)
        self.log_dict(loss_dict_reduced)

        self.training_step_outputs.append(losses)

        return losses


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

    def set_coco_api_from_dataset(self, val_dataloader: torch.utils.data.DataLoader):
        if self._use_coco_evaluator():
            self.coco = get_coco_api_from_dataset(val_dataloader.dataset)
            self.iou_types = self._get_iou_types(self.model)
            self.coco_evaluator = CocoEvaluator(self.coco, self.iou_types)
            self.cpu_device = torch.device("cpu")


    def on_validation_epoch_start(self) -> None:
        if self._use_coco_evaluator():
            self.coco_evaluator = CocoEvaluator(self.coco, self.iou_types)


    def validation_step(self, val_batch, batch_idx) -> float:
        images, targets = val_batch

        model_time = time.time()

        outputs = self.model(images)
        if self._use_coco_evaluator():
            outputs = [{k: v.to(self.cpu_device) for k, v in t.items()} for t in outputs]
        else:
            outputs = [{k: v for k, v in t.items()} for t in outputs]

        model_time = time.time() - model_time
        self.log('model_time', model_time)

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}

        if self._use_coco_evaluator():
            evaluator_time = time.time()
            self.coco_evaluator.update(res)
            evaluator_time = time.time() - evaluator_time
            self.log('evaluator_time', evaluator_time)
        else:
            self.metric_bbox.update(outputs, targets)
        
        return outputs
    
    def _use_coco_evaluator(self):
        return self.cfg.METRICS.COCO_EVALUATOR

    def on_validation_epoch_end(self) -> None:
        if self._use_coco_evaluator():
            self.coco_evaluator.synchronize_between_processes()
            self.coco_evaluator.accumulate()
            self.coco_evaluator.summarize()
        else:
            pprint(self.metric_bbox.compute())

            mAPs = {"val_" + k: v for k, v in self.metric_bbox.compute().items()}
            mAPs_per_class = mAPs.pop("val_map_per_class")
            mARs_per_class = mAPs.pop("val_mar_100_per_class")

            self.log_dict(mAPs, sync_dist=True)
            self.log('val_loss', mAPs['val_map'])
            self.metric_bbox.reset()

            # skip since to many for logging
            # self.log_dict(
            #     {
            #         f"val_map_{label}": value
            #         for label, value in zip(self.categories, mAPs_per_class)
            #     },
            #     sync_dist=True,
            # )
            # self.log_dict(
            #     {
            #         f"val_mar_100_{label}": value
            #         for label, value in zip(self.categories, mARs_per_class)
            #     },
            #     sync_dist=True,
            # )


    def configure_optimizers(self):
        optimizer_params = {
            'parameters': self.parameters(),
            'opt': self.cfg.OPTIMIZER.NAME,
            'lr': self.cfg.OPTIMIZER.LR,
            'weight_decay': self.cfg.OPTIMIZER.WEIGHT_DECAY,
            'nesterov': self.cfg.OPTIMIZER.NESTEROV,
            'momentum': self.cfg.OPTIMIZER.MOMENTUM,
        }
        optimizer = optimizers.get_optimizer(**optimizer_params)
        return optimizer
