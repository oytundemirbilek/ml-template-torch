"""Module to wrap experiments, automatically handle cross validation and collect results."""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
from torch.nn import Module

from modelname.inference import BaseInferer
from modelname.train import BaseTrainer

FILE_PATH = os.path.dirname(__file__)


class Experiment:
    """Make it easy to track experiments, properly name models and results then compare them."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize with kwargs, all parameters for the Trainer."""
        self.kwargs = kwargs
        self.results_save_path = os.path.join(
            FILE_PATH, "..", "results", self.kwargs["model_name"]
        )
        self.model_per_fold: list[Module] = []
        self.val_result_per_fold: list[float] = []
        self.test_result_per_fold: list[float] = []
        self.trainer = BaseTrainer(**self.kwargs)
        self.n_folds = self.kwargs.get("n_folds", 5)

    def train_model(
        self,
        fold: int | None = None,
    ) -> None:
        """Run training loop for each cross validation fold."""
        self.all_results: dict[str, list[float]] = {}
        if fold is None:
            iter_folds = self.n_folds
            start_fold = 0
        else:
            start_fold = fold
            iter_folds = fold + 1
        for fold_id in range(start_fold, iter_folds):
            print(f"--------------------- FOLD {fold_id} ---------------------")
            model = self.trainer.train(current_fold=fold_id)
            self.model_per_fold.append(model)
            # Last epochs validation score:
            self.last_val_result = self.trainer.val_loss_per_epoch[-1]
            self.val_result_per_fold.append(self.last_val_result)

    def run_inference(
        self,
        load: bool = False,
        mode: str = "test",
        save_predictions: bool = False,
        fold: int | None = None,
    ) -> None:
        """Run inference with all models, each trained per fold."""
        if fold is None:
            iter_folds = self.n_folds
            start_fold = 0
        else:
            start_fold = fold
            iter_folds = fold + 1
        for fold_id in range(start_fold, iter_folds):
            print(f"--------------------- FOLD {fold_id} ---------------------")
            if load:
                model_params = {
                    "in_features": 3,
                    "out_features": 1,
                    "batch_size": 1,
                    "layer_sizes": (8, 16),
                }
                model_path = os.path.join(
                    self.trainer.model_save_path,
                    f"fold{fold_id}",
                )
                self.inferer = BaseInferer(
                    dataset=self.kwargs["dataset"],
                    model_path=model_path,
                    model_params=model_params,
                )
            else:
                self.inferer = BaseInferer(
                    dataset=self.kwargs["dataset"],
                    model=self.model_per_fold[fold_id],
                )
            self.test_results = self.inferer.run(mode, save_predictions)
            self.mean_test_score = sum(self.test_results) / len(self.test_results)
            self.test_result_per_fold.append(self.mean_test_score)

    def get_results_table(self) -> None:
        """Save experiment results in a table, this function should be called at the end."""
        results_df = pd.DataFrame(
            {
                "Avg. Test Scores": self.test_result_per_fold,
                "Last Val. Scores": self.val_result_per_fold,
            }
        )
        # Index of the dataframe will indicate the fold id.
        results_df.to_csv(self.results_save_path + "_results.csv", index=True)
