"""Checkpoint saving and loading with versioning for training resumption."""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch

from src.utils.logger import get_logger

logger = get_logger(__name__)


class Checkpointer:
    """Manages saving/loading of training checkpoints.

    Maintains three kinds of checkpoints inside ``checkpoint_dir``:
      - ``latest.pt``: overwritten after every save call.
      - ``best.pt``: overwritten whenever a new best metric is reached.
      - ``epoch_<N>.pt``: periodic snapshots (every ``save_every`` epochs).

    Args:
        checkpoint_dir: Directory in which checkpoints are stored.
        save_every: Save a periodic ``epoch_<N>.pt`` snapshot every N epochs.
        mode: Either ``"max"`` (higher metric is better, e.g. METEOR) or
            ``"min"`` (lower metric is better, e.g. loss).
    """

    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        save_every: int = 5,
        mode: str = "max",
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_every = save_every
        if mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min', got {mode!r}")
        self.mode = mode
        self.best_metric: Optional[float] = None

    def is_better(self, metric: float) -> bool:
        """Return True if ``metric`` improves on the best seen so far."""
        if self.best_metric is None:
            return True
        if self.mode == "max":
            return metric > self.best_metric
        return metric < self.best_metric

    def save(
        self,
        state: Dict[str, Any],
        epoch: int,
        metric: Optional[float] = None,
    ) -> Dict[str, bool]:
        """Persist a checkpoint dict, updating latest/best/periodic copies.

        Args:
            state: Dictionary containing at least ``model_state_dict``;
                typically also ``optimizer_state_dict``, ``scheduler_state_dict``,
                ``epoch``, ``global_step``, and ``config``.
            epoch: Current epoch number (1-indexed).
            metric: Validation metric used to decide whether this is the
                new best checkpoint (e.g. METEOR score).

        Returns:
            Dict with boolean flags ``{"saved_best": bool, "saved_periodic": bool}``.
        """
        state = dict(state)
        state["epoch"] = epoch
        if metric is not None:
            state["metric"] = metric

        latest_path = self.checkpoint_dir / "latest.pt"
        torch.save(state, latest_path)
        logger.info(f"Saved latest checkpoint to {latest_path} (epoch {epoch})")

        saved_best = False
        if metric is not None and self.is_better(metric):
            self.best_metric = metric
            best_path = self.checkpoint_dir / "best.pt"
            torch.save(state, best_path)
            logger.info(f"New best metric={metric:.4f}; saved {best_path}")
            saved_best = True

        saved_periodic = False
        if epoch % self.save_every == 0:
            periodic_path = self.checkpoint_dir / f"epoch_{epoch}.pt"
            torch.save(state, periodic_path)
            logger.info(f"Saved periodic checkpoint to {periodic_path}")
            saved_periodic = True

        return {"saved_best": saved_best, "saved_periodic": saved_periodic}

    def load(self, name: str = "best.pt", map_location: Optional[str] = None) -> Dict[str, Any]:
        """Load a checkpoint by filename from ``checkpoint_dir``.

        Args:
            name: Filename within ``checkpoint_dir`` (e.g. ``"best.pt"``,
                ``"latest.pt"``, or an absolute/relative path to a checkpoint).
            map_location: Passed through to ``torch.load``.

        Returns:
            The deserialized checkpoint dictionary.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
        """
        path = Path(name)
        if not path.is_file():
            path = self.checkpoint_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        logger.info(f"Loading checkpoint from {path}")
        return torch.load(path, map_location=map_location, weights_only=False)
