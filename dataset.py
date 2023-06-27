import os
from typing import Tuple, List
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from torch import Tensor
from sklearn.model_selection import KFold

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT_PATH = os.path.dirname(__file__)
DATA_PATH = os.path.join(ROOT_PATH, "datasets")

# You can use such a random seed 35813 (part of the Fibonacci Sequence).
np.random.seed(35813)


def mock_batch_collate_fn(batch_data: List[Tensor]) -> Tensor:
    return default_collate(batch_data)


class BaseDataset(Dataset):
    """
    Base class for common functionalities of all datasets.

    Examples
    --------
    Data loading with batches:

    >>> from torch_geometric.loader import DenseDataLoader as PygDataLoader
    >>> tr_dataset = GraphDataset(hemisphere="left", mode="train")
    >>> tr_dataloader = PygDataLoader(tr_dataset, batch_size=5)
    >>> for g in tr_dataloader:
    ...     print(g)

    Data loading without batching (no batch dimension):

    >>> from torch_geometric.loader import DataLoader as PygDataLoader
    >>> tr_dataset = GraphDataset(hemisphere="left", mode="train")
    >>> tr_dataloader = PygDataLoader(tr_dataset, batch_size=1)
    >>> for g in tr_dataloader:
    ...     print(g)

    Data loading with batches but a view selection, useful if the task is graph-to-graph prediction:

    >>> from torch.utils.data import DataLoader
    >>> tr_dataset = GraphDataset(hemisphere="left", mode="train")
    >>> tr_dataloader = DataLoader(tr_dataset, batch_size=5, collate_fn=graph_to_graph_batch_collate_fn)
    >>> for g1, g2 in tr_dataloader:
    ...     print(g1)

    """

    def __init__(
        self,
        path_to_data: str,
        mode: str = "inference",
        n_folds: int = 5,
        current_fold: int = 0,
        n_features: int = 3,
        in_memory: bool = False,
    ):
        super(BaseDataset, self).__init__()
        assert (
            current_fold < n_folds
        ), "selected fold index cannot be more than number of folds."
        self.mode = mode
        self.n_folds = n_folds
        self.in_memory = in_memory
        self.path_to_data = path_to_data
        self.current_fold = current_fold
        self.n_features = n_features
        self.samples_labels = self.get_labels()

        self.n_samples_total = len(self.samples_labels)

        # Keep half of the data as 'unseen' to be used in inference.
        self.seen_data_indices, self.unseen_data_indices = self.get_fold_indices(
            self.n_samples_total, 2, 0
        )

        if self.in_memory:
            self.loaded_samples = self.get_all_samples()

        if mode == "train" or mode == "validation":
            # Here split the 'seen' data to train and validation.
            self.seen_samples_labels = self.samples_labels[self.seen_data_indices]
            if self.in_memory:
                self.seen_samples_data = self.loaded_samples[self.seen_data_indices]

            self.n_samples_seen = len(self.samples_labels)
            self.tr_indices, self.val_indices = self.get_fold_indices(
                self.n_samples_seen,
                self.n_folds,
                self.current_fold,
            )

        if mode == "train":
            self.selected_indices = self.tr_indices
            self.samples_labels = self.seen_samples_labels[self.tr_indices]
            if self.in_memory:
                self.loaded_samples = self.seen_samples_data[self.tr_indices]
        elif mode == "validation":
            self.selected_indices = self.val_indices
            self.samples_labels = self.seen_samples_labels[self.val_indices]
            if self.in_memory:
                self.loaded_samples = self.seen_samples_data[self.val_indices]
        elif mode == "inference":
            self.selected_indices = self.unseen_data_indices
            self.samples_labels = self.samples_labels[self.unseen_data_indices]
            if self.in_memory:
                self.loaded_samples = self.loaded_samples[self.unseen_data_indices]
        else:
            raise ValueError("mode should be 'train', 'validation', or 'inference'")

        self.n_samples_in_split = len(self.samples_labels)

    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        if self.in_memory:
            label = self.samples_labels[index]
            sample = self.loaded_samples[index]
        else:
            label = self.samples_labels[index]
            sample = self.get_sample_data(index)
        return self.preprocess(sample), label

    def __len__(self) -> int:
        return self.n_samples_in_split

    def get_labels(self) -> np.ndarray:
        mock_dataset_row_count = 42
        return np.random.choice([0, 1], size=mock_dataset_row_count)

    def get_sample_data(self, index: int) -> Tensor:
        skip = np.arange(self.n_samples_in_split) - self.selected_indices[index]
        return torch.from_numpy(
            pd.read_csv(self.path_to_data, skiprows=skip).drop("Sample ID").values
        )

    def preprocess(self, data: Tensor) -> Tensor:
        return data

    def get_all_samples(self) -> np.ndarray:
        """
        Convert data from all samples to the Torch Tensor objects to store in a list later.
        This function can be memory-consuming but time-saving, recommended to be used on small datasets.

        Returns
        -------
        all_data: np.ndarray
            A numpy array represents all data.
        """
        return pd.read_csv(self.path_to_data).drop("Sample ID").values

    def get_fold_indices(
        self, all_data_size: int, n_folds: int, fold_id: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create folds and get indices of train and validation datasets.

        Parameters
        --------
        all_data_size: int
            Size of all data.
        fold_id: int
            Which cross validation fold to get the indices for.

        Returns
        --------
        train_indices: numpy ndarray
            Indices to get the training dataset.
        val_indices: numpy ndarray
            Indices to get the validation dataset.
        """
        kf = KFold(n_splits=n_folds, shuffle=True)
        split_indices = kf.split(range(all_data_size))
        train_indices, val_indices = [
            (np.array(train), np.array(val)) for train, val in split_indices
        ][fold_id]
        # Split train and test
        return train_indices, val_indices

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__} dataset ({self.mode}) with"
            f" n_features={self.n_features}, n_samples={self.n_samples_in_split}, "
            f"current fold:{self.current_fold+1}/{self.n_folds}"
        )


class MockDataset(BaseDataset):
    """
    Mock data
    """

    def __init__(
        self,
        mode: str = "inference",
        n_folds: int = 5,
        current_fold: int = 0,
        in_memory: bool = False,
    ):
        mock_data_path = os.path.join(DATA_PATH, "mock_dataset.csv")
        super(MockDataset, self).__init__(
            mock_data_path,
            mode,
            n_folds,
            current_fold,
            in_memory,
        )
