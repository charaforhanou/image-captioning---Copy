"""Training callbacks."""

from typing import Optional


class EarlyStopping:
    """Stops training when a monitored metric has not improved for ``patience`` epochs.

    Args:
        patience: Number of epochs without improvement before stopping.
        mode: ``"max"`` if higher metric values are better, ``"min"`` otherwise.
        min_delta: Minimum change to qualify as an improvement.
    """

    def __init__(self, patience: int = 5, mode: str = "max", min_delta: float = 0.0) -> None:
        if mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min', got {mode!r}")
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best: Optional[float] = None
        self.counter = 0
        self.should_stop = False

    def step(self, metric: float) -> bool:
        """Update state with the latest metric value.

        Args:
            metric: The monitored metric value for the current epoch.

        Returns:
            ``True`` if training should stop.
        """
        if self.best is None or self._is_improvement(metric):
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop

    def _is_improvement(self, metric: float) -> bool:
        if self.mode == "max":
            return metric > self.best + self.min_delta
        return metric < self.best - self.min_delta
