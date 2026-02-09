from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, List, Tuple

import lightning as L
import numpy as np
import pandas as pd
import torch
from loguru import logger
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms


class NeuralNetCDCDataset(Dataset):
    """
    PyTorch Dataset for the CDC data transformed using the NCTD algorithm.
    """

    def __init__(
        self, data: np.ndarray, labels: np.ndarray, transform: Callable = None
    ):
        """
        Initialize the dataset.

        Parameters
        ----------
        data : np.ndarray
            Array of shape (n_samples, n_features) containing CDC data.
        labels : np.ndarray
            Array of shape (n_samples,) containing target labels.
        transform : Callable, optional
            A callable that takes a single sample and the number of features,
            and returns a transformed sample. If None, no transformation is applied.
        """

        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        Returns
        -------
        int
            Number of samples.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieve a single transformed data sample and its label.

        The data is transformed (if a transform is provided), converted to a
        torch.FloatTensor, and reshaped to add a channel dimension for
        compatibility with convolutional neural networks.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple (x, y) where:
            - x is the transformed data sample as a float32 torch.Tensor
              with shape (1, n_transformed_features).
            - y is the corresponding label as a float32 torch.Tensor scalar.
        """
        n_features = len(self.data[idx])

        if self.transform:
            x = self.transform(self.data[idx], n_features)

        else:
            x = self.data[idx]

        # logger.debug(f"Transformed data shape: {x.shape}")

        # Add channel dimension for CNNs
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        y = torch.tensor(self.labels[idx], dtype=torch.float32)

        return x, y


class IgtdCDCDataset(Dataset):
    """
    PyTorch Dataset for the CDC data transformed using the IGTD algorithm.
    """

    def __init__(self, img_dir: Path, labels: pd.Series):
        """
        Initialize the CDC Dataset transformed by the IGTD algorithm.

        Parameters
        ----------
        img_dir : Path
            Path to the directory containing IGTD-transformed PNG images.
        labels : pd.Series
            Series containing target labels. The Series index corresponds
            to the integer indices extracted from the filenames.

        Raises
        ------
        AssertionError
            If the labels series does not cover all required indices.
        """
        self.img_dir = img_dir
        # Sort files numerically by index to match CSV row order
        self.img_files = sorted(
            img_dir.glob("*_image.png"), key=lambda f: int(f.name.split("_")[1])
        )

        # Extract indices from filenames like _123_image.png
        self.indices = [int(f.name.split("_")[1]) for f in self.img_files]

        logger.debug(f"Found {len(self.img_files)} images in {self.img_dir}")
        logger.debug(f"Number of labels: {len(labels)}")
        assert (
            len(labels) >= max(self.indices) + 1
        ), "Label series does not cover all image indices"

        self.labels = labels
        self.transform = transforms.Compose(
            [transforms.Grayscale(num_output_channels=1), transforms.ToTensor()]
        )

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        Returns
        -------
        int
            Number of samples.
        """
        return len(self.img_files)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
        """
        Retrieve a single image and its corresponding label.

        Loads the image, converts it to a single-channel grayscale
        tensor, and retrieves the label from the labels Series
        based on the index extracted from the filename.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.

        Returns
        -------
        x : torch.Tensor
            Image tensor of shape (1, H, W) and dtype float32.
        y : torch.Tensor
            Scalar tensor containing the label as float32.
        """
        index = self.indices[idx]
        x = self.transform(Image.open(self.img_files[idx]))
        y = torch.tensor(self.labels.iloc[index], dtype=torch.float32)
        return x, y


class CDCDataModule(L.LightningDataModule, ABC):
    """PyTorch Lightning DataModule for handling CDC datasets."""

    def __init__(
        self,
        data_file: Path,
        target_col: str,
        batch_size: int = 64,
        num_workers: int = 4,
    ):
        super().__init__()
        self.data_file = data_file
        self.target_col = target_col
        self.batch_size = batch_size
        self.num_workers = num_workers

    @abstractmethod
    def setup(self, stage: str = None, **kwargs):
        """
        Abstract method for loading the splitting data.

        Subclasses must implement this method to perform:
          - Reading and processing the data file.
          - Splitting the dataset into training, validation, and test sets.
          - Assigning datasets to:
              self.train_dataset
              self.val_dataset
              self.test_dataset

        Parameters
        ----------
        stage : str, optional
            One of 'fit', 'validate', 'test', or 'predict'. Allows
            different data preparation logic depending on the stage.

        Raises
        ------
        NotImplementedError
            Always raised unless overridden in a subclass.
        """
        raise NotImplementedError(
            "The setup method must be implemented in the subclass."
        )

    def train_dataloader(self) -> DataLoader:
        """
        Return the DataLoader for the training dataset.

        Returns
        -------
        DataLoader
            PyTorch DataLoader configured for training.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Return the DataLoader for the validation dataset.

        Returns
        -------
        DataLoader
            PyTorch DataLoader configured for validation.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        """
        Return the DataLoader for the test dataset.

        Returns
        -------
        DataLoader
            PyTorch DataLoader configured for testing.
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )


class NeuralNetDataModule(CDCDataModule):
    """
    PyTorch Lightning DataModule for CDC data transformed
    using the NCTD algorithm.

    Inherits from CDCDataModule.
    """

    def __init__(
        self,
        data_file: Path,
        target_col: str = "Label",
        feature_cols: List[str] = None,
        batch_size: int = 64,
        num_workers: int = 4,
    ):
        """
        Initialize the NeuralNetDataModule.

        Parameters
        ----------
        data_file : Path
            Path to the file containing the CDC dataset.
        target_col : str
            Name of the column to be used as the target variable.
        batch_size : int, optional
            Number of samples per batch (default is 64).
        num_workers : int, optional
            Number of subprocesses to use for data loading (default is 4).
        """
        super().__init__(
            data_file=data_file,
            target_col=target_col,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        if not self.data_file.exists():
            raise FileNotFoundError(f"Data file {self.data_file} does not exist.")

        df = pd.read_csv(self.data_file)
        if feature_cols is None:
            feature_cols = [col for col in df.columns if col != self.target_col]

        self.X, self.y = df[feature_cols].values, df[self.target_col].values

    def setup(
        self,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        test_idx: np.ndarray = None,
        transform: Callable = None,
        # downsample: bool = False
    ):
        """
        Prepare the data splits and initialize datasets.

        Parameters
        ----------
        transform : Callable, optional
            A function to transform individual data samples. The function
            should accept two arguments:
              - the feature vector for a single sample
              - the number of features
            and return a transformed feature vector.
            If None, no transformation is applied.
        """

        X_train, y_train = self.X[train_idx], self.y[train_idx]
        X_val, y_val = self.X[val_idx], self.y[val_idx]

        # Scale the features
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        # Initialize datasets
        self.train_dataset = NeuralNetCDCDataset(X_train, y_train, transform)
        self.val_dataset = NeuralNetCDCDataset(X_val, y_val, transform)

        if test_idx is not None:
            X_test, y_test = self.X[test_idx], self.y[test_idx]
            X_test = scaler.transform(X_test)
            self.test_dataset = NeuralNetCDCDataset(X_test, y_test, transform)

        else:
            self.test_dataset = None


class IgtdDataModule(CDCDataModule):
    """
    PyTorch Lightning DataModule for CDC data transformed
    using the IGTD algorithm.

    Inherits from CDCDataModule.
    """

    def __init__(
        self,
        data_file: Path,
        img_dir: Path,
        target_col: str = "Label",
        batch_size=64,
        num_workers=4,
    ):
        """
        Initialize the IgtdDataModule.

        Parameters
        ----------
        data_file : Path
            Path to the CSV file containing CDC labels and features.
        img_dir : Path
            Path to the directory containing IGTD-generated PNG images.
        target_col : str, optional
            Name of the column in the CSV file to use as the target variable.
            Default is "Label".
        batch_size : int, optional
            Number of samples per batch (default is 64).
        num_workers : int, optional
            Number of subprocesses to use for data loading (default is 4).
        """
        super().__init__(
            data_file=data_file,
            target_col=target_col,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        self.img_dir = img_dir

    def setup(
        self, train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray = None
    ):
        """
        Load and split the IGTD images for cross-validation.

        Raises
        ------
        FileNotFoundError
            If the specified image directory does not exist.
        """
        if not self.img_dir.exists():
            raise FileNotFoundError(
                f"IGTD data directory {self.img_dir} does not exist."
            )

        df = pd.read_csv(self.data_file)
        full_dataset = IgtdCDCDataset(img_dir=self.img_dir, labels=df[self.target_col])

        self.train_dataset = Subset(full_dataset, train_idx)
        self.val_dataset = Subset(full_dataset, val_idx)

        if test_idx is not None:
            self.test_dataset = Subset(full_dataset, test_idx)

        else:
            self.test_dataset = None
