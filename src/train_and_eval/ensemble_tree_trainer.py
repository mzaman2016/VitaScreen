from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from imblearn.under_sampling import EditedNearestNeighbours
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from model.ensemble_tree import EnsembleTreeClassifier, Pool
from train_and_eval.base_trainer import BaseTrainer
from train_and_eval.evaluate import compute_metrics


class EnsembleTreeTrainer(BaseTrainer):
    """
    Trainer class for ensemble tree-based models, in particular
    CatBoostClassifier on CDC data.
    """

    def __init__(self, hyperparams: DictConfig):
        super().__init__()
        self.hyperparams = OmegaConf.to_container(hyperparams, resolve=True)
        self.best_model = None

    def train(self, enn: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Train the ensemble tree model.

        Parameters
        ----------
        enn : bool, optional
            Whether to apply Edited Nearest Neighbours resampling to training data, by default False
        """
        # Pick the first fold from k-fold indices
        train_idx, val_idx = self.k_fold_indices[0]
        logger.info(
            f"Size of training set: {len(train_idx)}, validation set: {len(val_idx)}"
        )

        X_train, y_train = self.X[train_idx], self.y[train_idx]

        if enn:
            logger.info(
                "Applying Edited Nearest Neighbours resampling to training data"
            )
            resampler = EditedNearestNeighbours()
            X_train, y_train = resampler.fit_resample(X_train, y_train)
            logger.info(
                f"Resampled training data shape: {X_train.shape}, {y_train.shape}"
            )

        train_pool = Pool(data=X_train, label=y_train)
        val_pool = Pool(data=self.X[val_idx], label=self.y[val_idx])

        model = EnsembleTreeClassifier(**self.hyperparams)
        model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50)

        y_preds = model.predict(self.X[val_idx])

        return self.y[val_idx], y_preds

    def cross_validate(self, enn: bool = False) -> pd.DataFrame:
        """Train the ensemble tree model with cross validation.

        Parameters
        ----------
        enn : bool, optional
            Whether to apply Edited Nearest Neighbours resampling to training data in each fold, by default False
        """
        best_f1_score = 0.0
        best_cv_metrics = None

        cv_results = []
        for i, (train_idx, val_idx) in enumerate(self.k_fold_indices):
            logger.info(f"Training fold {i + 1}/{len(self.k_fold_indices)}")

            X_train, y_train = self.X[train_idx], self.y[train_idx]

            if enn:
                logger.info(
                    "Applying Edited Nearest Neighbours resampling to training data"
                )
                resampler = EditedNearestNeighbours()
                X_train, y_train = resampler.fit_resample(X_train, y_train)
                logger.info(
                    f"Resampled training data shape: {X_train.shape}, {y_train.shape}"
                )

            train_pool = Pool(data=X_train, label=y_train)
            val_pool = Pool(data=self.X[val_idx], label=self.y[val_idx])

            model = EnsembleTreeClassifier(**self.hyperparams)
            model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50)

            y_preds = model.predict(self.X[val_idx])
            val_metrics = compute_metrics(self.y[val_idx], y_preds, avg_option="macro")

            logger.info(f"Validation Accuracy: {val_metrics['accuracy']:.4f}")
            logger.info(f"Validation Precision: {val_metrics['precision']:.4f}")
            logger.info(f"Validation Recall: {val_metrics['recall']:.4f}")
            logger.info(f"Validation F1 Score: {val_metrics['f1_score']:.4f}")

            val_metrics["fold"] = i + 1
            cv_results.append(val_metrics)

            if val_metrics["f1_score"] > best_f1_score:
                best_f1_score = val_metrics["f1_score"]
                self.best_model = model
                best_cv_metrics = val_metrics

        logger.info("Test validation metrics of the best model:")
        for metric, value in best_cv_metrics.items():
            logger.info(f"Best Model {metric}: {value}")

        return pd.DataFrame(cv_results)

    def evaluate(self):
        """
        Evaluate the ensemble tree model on validation and test sets.

        Returns
        -------
        y_test : np.ndarray
            True labels for the test set.
        y_preds : np.ndarray
            Predicted labels for the test set.
        """
        y_pred = self.best_model.predict(self.X[self.test_idx])

        return self.y[self.test_idx], y_pred

    def export_prob(self, output_path: Path):
        if self.best_model is None:
            raise ValueError("No best model found. Please train the model first.")

        prob_dfs = []
        train_idx, val_idx = self.k_fold_indices[0]
        for split, indices in zip(
            ["train", "val", "test"], [train_idx, val_idx, self.test_idx]
        ):
            df = pd.DataFrame(
                {
                    "id": indices,
                    "split": split,
                    "y_true": self.y[indices],
                    "y_prob": self.best_model.predict_proba(self.X[indices])[:, 1],
                }
            )
            prob_dfs.append(df)

        all_probs = pd.concat(prob_dfs).sort_values(by="id")
        all_probs.to_csv(output_path, index=False)
        logger.info(f"Predicted probabilities exported to {output_path}")
