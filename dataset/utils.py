import torchvision

from . import presets
from .coco import get_coco, COCO_CLASS_NAMES
from .transforms import SimpleCopyPaste, InterpolationMode


def get_transform(train, cfg):
    if train:
        return presets.DetectionPresetTrain(data_augmentation=cfg.DATA_AUGMENTATION)
    elif cfg.WEIGHTS and cfg.test_only:
        weights = torchvision.models.get_weight(cfg.WEIGHTS)
        trans = weights.transforms()
        return lambda img, target: (trans(img), target)
    else:
        return presets.DetectionPresetEval()


def get_datasets(name, cfg, mode='train'):
    if name == 'coco':
        if mode == 'train':
            return get_coco(cfg.DATA_PATH, 'train', get_transform(train=True, cfg=cfg))
        elif mode == 'val':
            return get_coco(cfg.DATA_PATH, 'val', get_transform(train=False, cfg=cfg))
        elif mode == 'class_names':
            return COCO_CLASS_NAMES
        raise RuntimeError("Unknown mode: {}".format(mode))
    raise RuntimeError("Only COCO dataset is supported for now.")


def collate_fn(batch):
    return tuple(zip(*batch))


def copypaste_collate_fn(batch):
    copypaste = SimpleCopyPaste(blending=True, resize_interpolation=InterpolationMode.BILINEAR)
    return copypaste(*collate_fn(batch))

