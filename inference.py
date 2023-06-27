import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from typing import List

from evaluation import MockLoss
from dataset import (
    MockDataset,
    mock_batch_collate_fn,
)

DATASETS = {
    "mock_dataset": MockDataset,
    "another_mock_dataset": MockDataset,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BaseInferer:
    """Inference loop for a trained model. Run the testing scheme."""

    def __init__(
        self,
        model: Module,
        dataset: str,
        metric_name: str = "mock_loss",
    ) -> None:
        self.model = model
        self.dataset = dataset
        if metric_name == "mock_loss":
            self.metric = MockLoss()
        else:
            raise NotImplementedError()

    @torch.no_grad()
    def run(self) -> List[float]:
        self.model.eval()
        test_losses = []
        test_dataset = DATASETS[self.dataset](
            mode="inference",
            n_folds=1,
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=1,
            collate_fn=mock_batch_collate_fn,
        )
        for input_graph, target_graph in test_dataloader:
            pred_graph = self.model(input_graph)
            # TODO: Find better metric common with the benchmarks
            test_loss = self.metric(pred_graph, target_graph)
            test_losses.append(test_loss.item())
            # TODO: Collect metrics in a table.

        self.model.train()
        return test_losses
