"""Module that provides calculations for loss functions and evaluation metrics."""

import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss  # noqa: PLC2701


class MockLoss(_Loss):
    """Loss that combines Loss 1 and Loss 2."""

    def __init__(
        self,
        batch_first: bool = True,
        reduction: str = "mean",
        another_loss_weight: float = 1.0,
    ) -> None:
        super().__init__(reduction=reduction)
        self.another_loss_weight = another_loss_weight
        self.batch_first = batch_first
        self.reduction = reduction

    def forward(self, input_data: Tensor, target_data: Tensor) -> Tensor:
        """Calculate the total loss."""
        if self.batch_first:
            loss = torch.sqrt(torch.square(input_data - target_data).sum(dim=(1, 2)))
        else:
            loss = torch.sqrt(torch.square(input_data - target_data).sum())
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            raise NotImplementedError(
                "Batch reduction options are only 'mean' or 'sum'"
            )
