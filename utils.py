from typing import Optional


class EarlyStopping:
    def __init__(self, patience: Optional[int]):
        self.patience = patience
        self.trigger = 0.0
        self.last_loss = 100.0

    def step(self, current_loss: float) -> None:
        if current_loss > self.last_loss:
            self.trigger += 1
        else:
            self.trigger = 0
        self.last_loss = current_loss

    def check_patience(self) -> bool:
        if self.patience is None:
            return False
        if self.trigger >= self.patience:
            return True
        else:
            return False
