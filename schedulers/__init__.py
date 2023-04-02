
from schedulers.lr_scheduler import CosineAnnealingWarmupRestarts, WarmupMultiStepLR

def get_scheduler(name: str, optimizer, cfg):
    name = name.lower()
    if name == "multisteplr":
        lr_scheduler = WarmupMultiStepLR(
                optimizer,
                cfg.SCHEDULER.STEPS,
                cfg.SCHEDULER.GAMMA,
                warmup_factor=cfg.SCHEDULER.WARMUP_FACTOR,
                warmup_iters=cfg.SCHEDULER.WARMUP_ITERS,
                warmup_method=cfg.SCHEDULER.WARMUP_METHOD,
            )
    elif name == "cosineannealinglr":
        lr_scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=cfg.SCHEDULER.MAX_ITER, # total steps SCHEDULER.max_iter T_0
            max_lr = cfg.SCHEDULER.BASE_LR, # max lr or base lr init_lr max_lr
            gamma=cfg.SCHEDULER.GAMMA, # lr decay gamma
            warmup_steps=cfg.SCHEDULER.WARMUP_ITERS, # warmup steps T_up
        )
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{name}'. Only MultiStepLR and CosineAnnealingLR are supported."
        )
    
    return lr_scheduler
