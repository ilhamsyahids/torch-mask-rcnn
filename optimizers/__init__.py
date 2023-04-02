import torch
import torch.optim

def get_optimizer(opt: str, parameters, lr: float, weight_decay: float, **kwargs):
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
    else:
        raise RuntimeError(f"Invalid optimizer {opt_name}. Only SGD and AdamW are supported.")

    return optimizer
