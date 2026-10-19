"""Random seed management for reproducible training and evaluation."""

import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Seed all relevant RNGs and configure cuDNN for determinism.

    Args:
        seed: The random seed applied to ``random``, ``numpy``, and
            ``torch`` (CPU and all CUDA devices).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int, base_seed: Optional[int] = None) -> None:
    """``DataLoader`` worker_init_fn that seeds numpy/random per worker.

    Args:
        worker_id: Index of the DataLoader worker (passed by PyTorch).
        base_seed: Optional base seed; defaults to ``torch.initial_seed()``.
    """
    seed = (base_seed if base_seed is not None else torch.initial_seed()) % 2**32
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)
