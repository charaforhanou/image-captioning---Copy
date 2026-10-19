"""AdamW optimizer + linear-warmup/cosine-decay learning rate schedule."""

import math

import torch
import torch.nn as nn


def build_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.AdamW:
    """Build AdamW with weight decay disabled for biases and norm/embedding scales.

    Args:
        model: The model whose trainable parameters are optimized.
        lr: Base learning rate.
        weight_decay: Weight decay applied to all parameters with ndim > 1.

    Returns:
        A configured :class:`torch.optim.AdamW`.
    """
    decay, no_decay = [], []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1:
            no_decay.append(param)
        else:
            decay.append(param)

    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    base_lr: float,
    min_lr: float = 1e-6,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup followed by cosine decay to ``min_lr``.

    Args:
        optimizer: Optimizer whose learning rate is scheduled.
        warmup_steps: Number of linear-warmup optimizer steps.
        total_steps: Total number of optimizer steps over training.
        base_lr: The optimizer's base (post-warmup peak) learning rate.
        min_lr: The learning rate at the end of the cosine decay.

    Returns:
        A :class:`torch.optim.lr_scheduler.LambdaLR`.
    """
    min_lr_ratio = min_lr / base_lr

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(progress, 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
