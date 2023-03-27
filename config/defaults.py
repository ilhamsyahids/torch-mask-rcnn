from yacs.config import CfgNode as CN


# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()


# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------

_C.PROJECT_NAME = "PyTorch Mask R-CNN Training"

_C.CONFIG_NAME = "Default"
_C.CONFIG_FILE = "config/defaults.py"

_C.OUTPUT_DIR = "checkpoint"

_C.EPOCHS = 1

_C.USE_SIMPLE_COPY_PASTE = False

_C.PRINT_FREQ = 20

_C.STRATEGY = "auto"

# -----------------------------------------------------------------------------
# Accelerator
# -----------------------------------------------------------------------------

_C.ACCELERATOR = CN()
_C.ACCELERATOR.NAME = "gpu"
_C.ACCELERATOR.DEVICES = 8
_C.ACCELERATOR.PRECISION = 16

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------

_C.DATASET = CN()
_C.DATASET.NAME = "coco"
_C.DATASET.DATA_PATH = "data/coco"
_C.DATASET.DATA_AUGMENTATION = "hflip"
_C.DATASET.WEIGHTS = None
_C.DATASET.NUM_CLASSES = 91

# -----------------------------------------------------------------------------
# Dataloader
# -----------------------------------------------------------------------------

_C.DATALOADER = CN()
_C.DATALOADER.WORKERS = 4
_C.DATALOADER.TRAIN_BATCH_SIZE = 16
_C.DATALOADER.VAL_BATCH_SIZE = 8

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------

_C.MODEL = CN()
_C.MODEL.VERSION = "v2"
_C.MODEL.PRETRAINED = False
_C.MODEL.PRETRAINED_BACKBONE = True

# -----------------------------------------------------------------------------
# Optimizer
# -----------------------------------------------------------------------------

_C.OPTIMIZER = CN()
_C.OPTIMIZER.NAME = "sgd"
_C.OPTIMIZER.LR = 0.02
_C.OPTIMIZER.WEIGHT_DECAY = 0.0001
_C.OPTIMIZER.MOMENTUM = 0.9
_C.OPTIMIZER.NESTEROV = False


# -----------------------------------------------------------------------------
# Logger
# -----------------------------------------------------------------------------

_C.LOGGER = CN()
_C.LOGGER.OUTPUT_DIR = "logs"
_C.LOGGER.VERSION = 1


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

_C.METRICS = CN()
_C.METRICS.COCO_EVALUATOR = True

cfg = _C
