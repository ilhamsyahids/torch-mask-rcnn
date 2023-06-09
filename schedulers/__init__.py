
import torch
from schedulers.lr_scheduler import CosineAnnealingWarmupRestarts, WarmupCosineLR, WarmupMultiStepLR, WarmupPolynomialLR

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
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.SCHEDULER.MAX_ITER,
        )
    elif name == "cosineannealinglrwarmup":
        lr_scheduler = WarmupCosineLR(
            optimizer,
            max_iters=cfg.SCHEDULER.MAX_ITER,
            warmup_iters=cfg.SCHEDULER.WARMUP_ITERS,
            warmup_factor=cfg.SCHEDULER.WARMUP_FACTOR,
            warmup_method=cfg.SCHEDULER.WARMUP_METHOD,
        )
    elif name == "cosineannealinglrwarmuprestarts":
        lr_scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=cfg.SCHEDULER.FIRST_CYCLE, # first cycle T_0
            max_lr = cfg.SCHEDULER.BASE_LR, # max lr or base lr init_lr max_lr
            min_lr = cfg.SCHEDULER.MIN_LR,
            gamma=cfg.SCHEDULER.GAMMA, # lr decay gamma
            warmup_steps=cfg.SCHEDULER.WARMUP_ITERS, # warmup steps T_up
        )
    elif name == "polynomialwarmup":
        lr_scheduler = WarmupPolynomialLR(
            optimizer,
            total_iters=cfg.SCHEDULER.MAX_ITER,
            power=cfg.SCHEDULER.GAMMA,
            warmup_iters=cfg.SCHEDULER.WARMUP_ITERS,
            warmup_factor=cfg.SCHEDULER.WARMUP_FACTOR,
            warmup_method=cfg.SCHEDULER.WARMUP_METHOD,
        )
    elif name == "polynomial":
        lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(
            optimizer,
            total_iters=cfg.SCHEDULER.MAX_ITER,
            power=cfg.SCHEDULER.GAMMA,
        )
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{name}'. \
            Only MultiStepLR, \
            CosineAnnealingLR, \
            CosineAnnealingLRWarmUp, \
            CosineAnnealingLRWarmUpRestarts, \
            and Polynomial \
            are supported."
        )
    
    return lr_scheduler
