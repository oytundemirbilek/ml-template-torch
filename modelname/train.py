"""Module for training script."""

from __future__ import annotations

import json
import os

import numpy as np
import torch
from torch.backends import cudnn
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader

from modelname.dataset import MockDataset, mock_batch_collate_fn
from modelname.evaluation import MockLoss
from modelname.model import MockModel
from modelname.utils import EarlyStopping

# We used 35813 (part of the Fibonacci Sequence) as the seed when we conducted experiments
np.random.seed(35813)
torch.manual_seed(35813)

# These two options should be seed to ensure reproducible (If you are using cudnn backend)
# https://pytorch.org/docs/stable/notes/randomness.html
cudnn.deterministic = True
cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FILE_PATH = os.path.dirname(__file__)

DATASETS = {
    "mock_dataset": MockDataset,
    "another_mock_dataset": MockDataset,
}


class BaseTrainer:
    """Wrapper around training function to save all the training parameters."""

    def __init__(
        self,
        # Data related:
        dataset: str,
        timepoint: str | None,
        # Training related:
        n_epochs: int,
        learning_rate: float,
        weight_decay: float = 0.001,
        batch_size: int = 1,
        validation_period: int = 5,
        patience: int | None = None,
        # Model related:
        n_folds: int = 5,
        layer_sizes: tuple[int, ...] = (8, 16),
        loss_weight: float = 1.0,
        loss_name: str = "mock_loss",
        model_name: str = "default_model_name",
    ) -> None:
        self.dataset = dataset
        self.timepoint = timepoint
        self.n_epochs = n_epochs
        self.n_folds = n_folds
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.validation_period = validation_period
        self.loss_weight = loss_weight
        self.loss_name = loss_name
        self.layer_sizes = layer_sizes
        self.model_name = model_name
        self.model_save_path = os.path.join(FILE_PATH, "..", "models", model_name)
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

        self.model_params_save_path = os.path.join(
            FILE_PATH, "..", "models", model_name + "_params.json"
        )
        with open(self.model_params_save_path, "w", encoding="utf-8") as f:
            json.dump(self.__dict__, f, indent=4)

        self.loss_fn: _Loss
        if loss_name == "mock_loss":
            self.loss_fn = MockLoss()
        else:
            raise NotImplementedError("Specified loss function is not defined.")

        self.val_loss_per_epoch: list[float] = []

    def __repr__(self) -> str:
        """Return string representation of the Trainer as training parameters."""
        return str(self.__dict__)

    @torch.no_grad()
    def validate(self, model: Module, val_dataloader: DataLoader) -> float:
        """Run validation loop."""
        model.eval()
        val_losses = []
        for input_data, target_label in val_dataloader:
            prediction = model(input_data)
            val_loss = self.loss_fn(prediction, target_label)
            val_losses.append(val_loss)

        model.train()
        return torch.stack(val_losses).mean().item()

    def train(self, current_fold: int = 0) -> Module:
        """Train model."""
        tr_dataset = DATASETS[self.dataset](
            mode="train",
            n_folds=self.n_folds,
            current_fold=current_fold,
        )
        val_dataset = DATASETS[self.dataset](
            mode="validation",
            n_folds=self.n_folds,
            current_fold=current_fold,
        )
        tr_dataloader = DataLoader(
            tr_dataset,
            batch_size=self.batch_size,
            collate_fn=mock_batch_collate_fn,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            collate_fn=mock_batch_collate_fn,
        )
        model = MockModel(
            in_features=3,
            out_features=1,
            batch_size=self.batch_size,
            layer_sizes=self.layer_sizes,
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        early_stopping = EarlyStopping(self.patience)
        best_loss = 99999999999999999999999.0
        for epoch in range(self.n_epochs):
            tr_losses = []
            for input_data, target_data in tr_dataloader:
                pred_data = model(input_data)
                tr_loss = self.loss_fn(pred_data, target_data)
                optimizer.zero_grad()
                tr_loss.backward()
                optimizer.step()
                tr_losses.append(tr_loss.detach())
            avg_tr_loss = torch.stack(tr_losses).mean().item()
            if (epoch + 1) % self.validation_period == 0:
                val_loss = self.validate(model, val_dataloader)
                print(
                    f"Epoch: {epoch + 1}/{self.n_epochs} | Tr.Loss: {avg_tr_loss} | Val.Loss: {val_loss}"
                )
                self.val_loss_per_epoch.append(val_loss)
                early_stopping.step(val_loss)
                if early_stopping.check_patience():
                    break

                if val_loss < best_loss:
                    torch.save(
                        model.state_dict(),
                        os.path.join(self.model_save_path, f"fold{current_fold}.pth"),
                    )
                    best_loss = val_loss

        return model


if __name__ == "__main__":
    trainer = BaseTrainer(
        dataset="mock_dataset",
        timepoint=None,
        n_epochs=100,
        learning_rate=0.01,
    )
    trainer.train()
