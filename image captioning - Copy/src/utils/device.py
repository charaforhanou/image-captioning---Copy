"""Device selection utilities."""

import torch


def get_device(preferred: str = "cuda") -> torch.device:
    """Return the requested device, falling back to CPU if unavailable.

    Args:
        preferred: Either ``"cuda"`` or ``"cpu"``.

    Returns:
        The resolved ``torch.device``.
    """
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
