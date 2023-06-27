# Our solution
from typing import List
from torch import Tensor
from torch.nn import Module, Sequential, Linear, ReLU


class MockModel(Module):
    """ """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        batch_size: int,
        layer_sizes: List[int],
    ) -> None:
        super(MockModel, self).__init__()

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
        return self.layers(input_data)
