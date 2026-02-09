from typing import List, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split


def split_data(
    X: np.ndarray, y: np.ndarray, n_splits=5
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """Split data into cross-validation folds and test set.

    Downsampling should be performed inside the cross-validation loop
    to prevent data leakage between folds.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target labels
    n_splits : int, optional
        Number of cross-validation folds, by default 5

    Returns
    -------
    Tuple[List[Tuple[np.ndarray, np.ndarray]], np.ndarray]
        List of (train_indices, val_indices) tuples for each fold, and test indices
    """
    indices = np.arange(len(X))

    train_val_idx, test_idx = train_test_split(
        indices, stratify=y, test_size=0.2, random_state=42
    )

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    folds: List[Tuple[np.ndarray, np.ndarray]] = []

    y_train_val = y[train_val_idx]  # avoid re-indexing y inside the loop
    for tr_local, va_local in cv.split(range(len(train_val_idx)), y_train_val):
        folds.append((train_val_idx[tr_local], train_val_idx[va_local]))

    return folds, test_idx


def nctd_transform(x: np.ndarray, n_features: int) -> np.ndarray:
    """
    Transform the input data for NCTD model.
    This function can be customized based on the specific requirements of the NCTD model.
    """

    x = np.tile(x * 255, (n_features, 1))
    x = np.array([np.roll(row, -i) for i, row in enumerate(x)])

    return np.tile(x, (2, 2))
