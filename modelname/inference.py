"""Module for inference and testing scripts."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader

from modelname.dataset import MockDataset
from modelname.evaluation import MockLoss
from modelname.model import MockModel

DATASETS = {
    "mock_dataset": MockDataset,
    "another_mock_dataset": MockDataset,
}


class BaseInferer:
    """Inference loop for a trained model. Run the testing scheme."""

    def __init__(
        self,
        dataset: str,
        model: Module | None = None,
        model_path: str | None = None,
        model_params: dict[str, Any] | None = None,
        out_path: str | None = None,
        metric_name: str = "mock_loss",
        random_seed: int = 0,
        device: str | None = None,
    ) -> None:
        """
        Initialize the inference (or testing) setup.

        Parameters
        ----------
        dataset: string
            Which dataset should be used for inference.
        model: torch Module, optional
            The model needs to be tested or inferred. If None, model_path
            and model_params should be specified to load a model.
        model_path: string, optional
            Path to the expected model.
        model_params: dictionary, optional
            Parameters that was specified before the training of the model.
        out_path: string, optional
            If you want to save the predictions, specify a path.
        metric_name: string
            Metric to evaluate the test performance of the model.
        """
        self.dataset = dataset
        self.device = device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        self.model_path = model_path
        self.out_path = out_path
        self.random_seed = random_seed
        self.model_params = model_params
        self.model: Module
        if model is None:
            if model_params is None:
                raise ValueError("Specify a model or model params and its path.")
            if model_path is None:
                raise ValueError("Specify a model or model params and its path.")
            self.model = self.load_model_from_file(
                model_path, model_params, self.device
            )
        else:
            self.model = model

        self.dataset = dataset
        self.metric_name = metric_name
        if metric_name == "mock_loss":
            self.metric = MockLoss()
        else:
            raise NotImplementedError()

    @torch.no_grad()
    def run(self, mode: str = "test", save_predictions: bool = False) -> Tensor:
        """
        Run inference loop whether for testing purposes or in-production.

        Parameters
        ----------
        mode: string
            Either 'test' or 'infer'. Whether to use all dataset samples or just the testing split.
            This can be handy when testing a pretrained model on your private dataset. Set 'infer'
            if you want to use your model in production.

        Returns
        -------
        test_losses: list of floats
            Test loss for each sample. Or any metric you will define. Calculates only if test_split_only is True.
        """
        self.model.eval()

        test_dataset = DATASETS[self.dataset](
            mode=mode,
            n_folds=1,
            device=self.device,
        )
        test_dataloader = DataLoader(test_dataset, batch_size=1)
        save_path = os.path.join(
            ".",
            "benchmarks",
            "modelname",
            self.metric_name,
            self.dataset,
        )
        if not os.path.exists(os.path.join(save_path, "predictions")):
            os.makedirs(os.path.join(save_path, "predictions"))
        if not os.path.exists(os.path.join(save_path, "targets")):
            os.makedirs(os.path.join(save_path, "targets"))

        test_losses = []
        for s_idx, (input_data, target_label) in enumerate(test_dataloader):
            prediction = self.model(input_data)
            if mode == "test":
                test_loss = self.metric(prediction, target_label)
                test_losses.append(test_loss.item())
            if save_predictions:
                np.savetxt(
                    os.path.join(save_path, "predictions", f"subject{s_idx}_pred.txt"),
                    prediction.squeeze(0).cpu().numpy(),
                )
            if save_predictions and mode == "test":
                np.savetxt(
                    os.path.join(save_path, "targets", f"subject{s_idx}_tgt.txt"),
                    target_label.squeeze(0).cpu().numpy(),
                )

        self.model.train()
        return torch.tensor(test_losses)

    @staticmethod
    def load_model_from_file(
        model_path: str, model_params: dict[str, Any], device: str | None = None
    ) -> Module:
        """
        Load a pretrained model from file.

        Parameters
        ----------
        model_path: string
            Path to the file which is model saved.
        model_params: dictionary
            Parameters of the model is needed to initialize.

        Returns
        -------
        model: pytorch Module
            Pretrained model ready for inference, or continue training.
        """
        model = MockModel(**model_params).to(device)
        if not model_path.endswith(".pth"):
            model_path += ".pth"
        model.load_state_dict(
            torch.load(
                model_path,
                map_location=torch.device(device) if device is not None else None,
            )
        )
        return model
