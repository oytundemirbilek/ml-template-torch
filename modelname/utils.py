"""Module for other utilities."""

from __future__ import annotations


class EarlyStopping:
    """Early stopping utility class."""

    def __init__(self, patience: int | None):
        self.patience = patience
        self.trigger = 0.0
        self.last_loss = 100.0

    def step(self, current_loss: float) -> None:
        """Inspect the current loss comparing to previous one."""
        if current_loss > self.last_loss:
            self.trigger += 1
        else:
            self.trigger = 0
        self.last_loss = current_loss

    def check_patience(self) -> bool:
        """Determine whether the training should be stopped."""
        if self.patience is None:
            return False
        return self.trigger >= self.patience
