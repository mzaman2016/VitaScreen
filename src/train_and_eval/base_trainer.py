from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from data.utils import split_data


class BaseTrainer(ABC):
    """
    Base trainer class with shared functionality for data setup.
    """

    def __init__(self):
        self.X: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None
        self.k_fold_indices: Optional[list] = None
        self.test_idx: Optional[np.ndarray] = None

    def setup(self, data_cfg: DictConfig):
        """
        Load and prepare the CDC dataset for training, validation, and testing.

        Parameters
        ----------
        data_cfg : DictConfig
            Hydra/OmegaConf configuration containing:
                - file_path: Path to the CDC dataset file.
                - feature_cols: List of feature column names.
                - target_col: Name of the target variable column.
        """
        df = pd.read_csv(data_cfg.file_path)

        if "feature_cols" in data_cfg:
            feature_cols = data_cfg.feature_cols
        else:
            feature_cols = [col for col in df.columns if col != data_cfg.target_col]
        self.X, self.y = df[feature_cols].values, df[data_cfg.target_col].values

        k_fold_indices, test_idx = split_data(
            X=self.X,
            y=self.y,
        )

        self.k_fold_indices = k_fold_indices
        self.test_idx = test_idx

    @abstractmethod
    def train(self):
        """Train the model. To be implemented by subclasses."""
        raise NotImplementedError

    @abstractmethod
    def cross_validate(self):
        """Train the model with cross validation. To be implemented by subclasses."""
        raise NotImplementedError

    @abstractmethod
    def evaluate(self):
        """Evaluate the model. To be implemented by subclasses."""
        raise NotImplementedError

    @abstractmethod
    def export_prob(self):
        """Export predicted probabilities. To be implemented by subclasses."""
        raise NotImplementedError
