"""Test graph dataset classes."""

from __future__ import annotations

import os
from typing import Any

from modelname.dataset import MockDataset
from modelname.inference import BaseInferer
from modelname.model import MockModel
from modelname.train import BaseTrainer

DATA_PATH = os.path.join(os.path.dirname(__file__), "datasets")
GOLD_STANDARD_PATH = os.path.join(os.path.dirname(__file__), "expected")
MODELS_PATH = os.path.join(os.path.dirname(__file__), "..", "models")
DEVICE = "cpu"


def test_simple_iteration() -> None:
    """Test if the model can be iterated - cpu based."""
    model = MockModel(5, 5, 5, (8, 16))
    assert model is not None
    # out = model.forward(data)
    # assert out


def test_dataset() -> None:
    """Test if the model can be iterated - cpu based."""
    dataset = MockDataset()
    assert dataset is not None
    # out = model.forward(data)
    # assert out


def test_reproducibility() -> None:
    """Test if the model can give same results always - compare cpu based results with cuda results."""


def test_trainer() -> None:
    """Test if the experiment module works properly."""
    training_params: dict[str, Any] = {
        "dataset": "mock_dataset",
        "timepoint": None,
        "n_epochs": 5,
        "learning_rate": 0.005,
    }
    trainer = BaseTrainer(**training_params)
    trainer.train()


def test_inferer() -> None:
    """Test if the experiment module works properly."""
    target_model_path = os.path.join(MODELS_PATH, "default_model_name", "fold0")
    inference_params: dict[str, Any] = {
        # "conv_size": 48,
        "model_params": {
            "in_features": 3,
            "out_features": 1,
            "batch_size": 1,
            "layer_sizes": (8, 16),
        },
        "model_path": target_model_path,
        "dataset": "mock_dataset",
    }
    inferer = BaseInferer(**inference_params)
    current_results = inferer.run()
    assert current_results is not None
