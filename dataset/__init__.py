import torchvision

from . import presets
from .coco import get_coco


def get_transform(train, cfg):
    if train:
        return presets.DetectionPresetTrain(data_augmentation=cfg.DATA_AUGMENTATION)
    elif cfg.WEIGHTS and cfg.test_only:
        weights = torchvision.models.get_weight(cfg.WEIGHTS)
        trans = weights.transforms()
        return lambda img, target: (trans(img), target)
    else:
        return presets.DetectionPresetEval()


def get_datasets(name, cfg):
    if name == 'coco':
        train = get_coco(cfg.DATA_PATH, 'train', get_transform(train=True, cfg=cfg))
        val = get_coco(cfg.DATA_PATH, 'val', get_transform(train=False, cfg=cfg))
    else:
        raise RuntimeError("Only COCO dataset is supported for now.")

    return train, val

