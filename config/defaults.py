from yacs.config import CfgNode as CN


# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()


# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------

_C.OUTPUT_DIR = "checkpoint"

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------

_C.DATASET = CN()
_C.DATASET.NAME = "coco"
_C.DATASET.DATA_PATH = "/data/coco"
_C.DATASET.DATA_AUGMENTATION = "hflip"
_C.DATASET.WEIGHTS = None

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------

_C.MODEL = CN()


cfg = _C
