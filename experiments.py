import os
import pandas as pd
from typing import List, Any
from torch.nn import Module

from train import BaseTrainer
from inference import BaseInferer

FILE_PATH = os.path.dirname(__file__)


class Experiment:
    """Make it easy to track experiments, properly name models and results then compare them."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.results_save_path = os.path.join(
            FILE_PATH, "results", self.kwargs["model_name"]
        )
        self.model_per_fold: List[Module] = []
        self.val_result_per_fold: List[float] = []
        self.test_result_per_fold: List[float] = []

    def train_model(self) -> None:
        if "n_folds" in self.kwargs.keys():
            n_folds = self.kwargs["n_folds"]
        else:
            n_folds = 5

        for fold in range(n_folds):
            self.trainer = BaseTrainer(**self.kwargs)
            model = self.trainer.train(current_fold=fold)
            self.model_per_fold.append(model)
            # Last epochs validation score:
            self.last_val_result = self.trainer.val_loss_per_epoch[-1]
            self.val_result_per_fold.append(self.last_val_result)

    def run_inference(self) -> None:
        for model_fold in self.model_per_fold:
            self.inferer = BaseInferer(
                model_fold,
                dataset=self.kwargs["dataset"],
            )
            self.test_results = self.inferer.run()
            self.mean_test_score = sum(self.test_results) / len(self.test_results)
            self.test_result_per_fold.append(self.mean_test_score)

    def get_results_table(self) -> None:
        # TODO: Test score per fold or select a model then measure test score?
        results_df = pd.DataFrame(
            {
                "Avg. Test Scores": self.test_result_per_fold,
                "Last Val. Scores": self.val_result_per_fold,
            }
        )
        # Index of the dataframe will indicate the fold id.
        results_df.to_csv(self.results_save_path + ".csv", index=True)
