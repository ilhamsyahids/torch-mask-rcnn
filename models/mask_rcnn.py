import torch
import pytorch_lightning as pl

from torchmetrics.detection.mean_ap import MeanAveragePrecision

from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection import maskrcnn_resnet50_fpn, maskrcnn_resnet50_fpn_v2
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights, MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.resnet import ResNet50_Weights


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



class Mask_RCNN_Lightning(pl.LightningModule):

    def __init__(self,
                 cfg,
                 model: MaskRCNN,
                 optimizer: torch.optim.Optimizer,
                 ):
        """
        Constructor for the Mask_RCNN class
        :param cfg: yacs configuration that contains all the necessary information about the available backbones
        :param model: the model
        :param optimizer: the optimizer
        """
        super(Mask_RCNN_Lightning, self).__init__()

        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.metric = MeanAveragePrecision(class_metrics=True)

        self.save_hyperparameters()

    def forward(self, inputs, targets=None):
        return self.model(inputs, targets)
    
    def on_train_epoch_start(self):
        self.training_step_outputs = []

    def training_step(self, train_batch, batch_idx) -> float:
        images, targets = train_batch

        targets = [{k: v for k, v in t.items()} for t in targets]
        loss_dict = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        self.training_step_outputs.append(losses)

        return losses

    def on_train_epoch_end(self) -> None:
        epoch_losses = torch.tensor([batch_loss['loss'].item() for batch_loss in self.training_step_outputs])
        loss_mean = torch.mean(epoch_losses)
        self.log('training_loss', loss_mean)

        self.training_step_outputs.clear()

    def on_validation_epoch_start(self) -> None:
        self.validation_step_outputs = []

    def validation_step(self, val_batch, batch_idx) -> float:
        images, targets = val_batch

        targets = [{k: v for k, v in t.items()} for t in targets]
        loss_dict = self.model(images, targets)
        va_losses = sum(loss for loss in loss_dict.values())

        self.validation_step_outputs.append(va_losses)

        return va_losses

    def validation_epoch_end(self, validation_step_outputs) -> dict:
        epoch_losses = torch.tensor([batch_loss.item() for batch_loss in validation_step_outputs])
        loss_mean = torch.mean(epoch_losses)
        self.log('val_loss', loss_mean)

        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        return self.optimizer
