import torch
import torch.nn
import torch.optim

from typing import Iterator

from optimizers.lars import LARS
from optimizers.lamb import LAMB

def get_optimizer(opt: str, parameters: Iterator[torch.nn.Parameter], lr: float, weight_decay: float, **kwargs):
    parameters = [p for p in parameters if p.requires_grad]

    opt_name = opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
            nesterov=kwargs['nesterov'],
            momentum=kwargs['momentum'],
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(
            parameters,
            lr=lr,
            weight_decay=weight_decay
        )
    elif opt_name == "lars":
        optimizer = LARS(
            parameters,
            lr=lr,
            momentum=kwargs['momentum'],
            dampening=kwargs['dampening'],
            weight_decay=weight_decay,
            nesterov=kwargs['nesterov'],
            log=kwargs['log'],
        )
    elif opt_name == "lamb":
        optimizer = LAMB(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
        )
    else:
        raise RuntimeError(f"Invalid optimizer {opt_name}. Only SGD, AdamW, LARS are supported.")

    return optimizer
