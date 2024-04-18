"""Module to define neural network, which is our solution."""

from __future__ import annotations

from torch import Tensor
from torch.nn import Linear, Module, ReLU, Sequential


class MockModel(Module):
    """Your model that predicts something."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        batch_size: int,
        layer_sizes: tuple[int, ...],
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.batch_size = batch_size
        self.layer_sizes = layer_sizes
        self.layers = Sequential(
            Linear(self.in_features, self.layer_sizes[0]),
            ReLU(),
            Linear(self.layer_sizes[0], self.layer_sizes[1]),
            ReLU(),
            Linear(self.layer_sizes[1], self.out_features),
        )

    def forward(self, input_data: Tensor) -> Tensor:
        """Forward pass of the network."""
        return self.layers(input_data)
