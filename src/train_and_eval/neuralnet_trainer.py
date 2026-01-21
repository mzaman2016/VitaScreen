import time
from pathlib import Path
from typing import List, Tuple

import lightning as L
import pandas as pd
import torch
from imblearn.under_sampling import EditedNearestNeighbours
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from loguru import logger
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchinfo import summary

from data.cdc import IgtdDataModule, NeuralNetDataModule
from model.classifier import DiabetesRiskClassifier
from model.cnn import ConvNet
from model.mlp import MLP
from model.model_type import ModelType
from train_and_eval.base_trainer import BaseTrainer
from train_and_eval.evaluate import compute_metrics
from train_and_eval.metrics_logger import MetricsLogger


class NeuralNetTrainer(BaseTrainer):
    def __init__(
        self,
        model_cfg: DictConfig,
        train_cfg: DictConfig,
    ):
        super().__init__()
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.best_model_path = None

    def init_trainer(self, model: ConvNet | MLP = ConvNet()):
        """Initialize the PyTorch Lightning trainer."""

        self.model = model
        input_size = (
            (1, self.model_cfg.input_dim)
            if self.model_cfg.name == ModelType.MLP.value
            else (1, 1, self.model_cfg.input_dim.nrows, self.model_cfg.input_dim.ncols)
        )
        summary(model=self.model, input_size=input_size)

        metrics_logger = MetricsLogger()
        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            min_delta=0.001,
            patience=5,
            verbose=True,
            mode="min",
        )
        self.checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",  # same metric as EarlyStopping
            mode="min",
            save_top_k=1,  # keep only the best model
            filename=f"best-checkpoint-{self.model_cfg.name}",
            verbose=False,
            save_weights_only=False,  # save full model (or True for just weights)
        )

        self.trainer = L.Trainer(
            accelerator="auto",
            max_epochs=self.train_cfg.max_epochs,
            devices=self.train_cfg.devices,
            deterministic=True,  # Enable reproducibility
            enable_progress_bar=True,
            log_every_n_steps=self.train_cfg.log_every_n_steps,
            enable_model_summary=False,  # optional
            callbacks=[metrics_logger, early_stop_callback, self.checkpoint_callback],
        )

    def cross_validate(
        self,
        data_file: Path,
        img_dir: Path = None,
        transform=None,
        feature_cols: List[str] = None,
    ) -> pd.DataFrame:
        """Train the model with cross validation.

        Parameters
        ----------
        data_file : Path
            Path to the data file
        img_dir : Path, optional
            Path to image directory for IGTD model
        transform : Callable, optional
            Transform function for data preprocessing
        feature_cols : List[str], optional
            List of feature column names
        """
        match self.model_cfg.name:
            case ModelType.MLP.value:
                dm = NeuralNetDataModule(data_file=data_file, feature_cols=feature_cols)
                model = MLP(input_dim=self.model_cfg.input_dim)

            case ModelType.NCTD.value:
                dm = NeuralNetDataModule(data_file=data_file, feature_cols=feature_cols)
                model = ConvNet()

            case ModelType.IGTD.value:
                dm = IgtdDataModule(data_file=data_file, img_dir=img_dir)
                model = ConvNet()

            case _:
                raise ValueError(f"Unsupported model type: {self.model_cfg.name}")

        best_f1_score = 0.0
        best_cv_metrics = None

        cv_results = []
        for i, (train_idx, val_idx) in enumerate(self.k_fold_indices):
            logger.info(f"Training fold {i + 1}/{len(self.k_fold_indices)}")

            # Apply ENN resampling if enabled
            if self.train_cfg.enn:
                logger.info(
                    "Applying Edited Nearest Neighbours resampling to training data"
                )
                resampler = EditedNearestNeighbours()
                X_train_resampled, y_train_resampled = resampler.fit_resample(
                    self.X[train_idx], self.y[train_idx]
                )
                logger.info(
                    f"Resampled training data shape: {X_train_resampled.shape}, {y_train_resampled.shape}"
                )
                # Get the resampled indices by finding which samples were kept
                train_idx = train_idx[resampler.sample_indices_]

            if self.model_cfg.name == ModelType.IGTD.value:
                dm.setup(train_idx=train_idx, val_idx=val_idx)

            else:
                dm.setup(train_idx=train_idx, val_idx=val_idx, transform=transform)

            self.init_trainer(model=model)
            self.train(
                train_loader=dm.train_dataloader(),
                val_loader=dm.val_dataloader(),
            )

            # Evaluate on validation set
            y_val, y_pred = self.evaluate(dm.val_dataloader())
            val_metrics = compute_metrics(y_val, y_pred, avg_option="macro")

            logger.info(f"Validation Accuracy: {val_metrics['accuracy']:.4f}")
            logger.info(f"Validation Precision: {val_metrics['precision']:.4f}")
            logger.info(f"Validation Recall: {val_metrics['recall']:.4f}")
            logger.info(f"Validation F1 Score: {val_metrics['f1_score']:.4f}")

            val_metrics["fold"] = i + 1
            cv_results.append(val_metrics)

            # Track best model
            if val_metrics["f1_score"] > best_f1_score:
                best_f1_score = val_metrics["f1_score"]
                self.best_model_path = self.checkpoint_callback.best_model_path
                best_cv_metrics = val_metrics

        logger.info("Test validation metrics of the best model:")
        for metric, value in best_cv_metrics.items():
            logger.info(f"Best Model {metric}: {value}")

        return pd.DataFrame(cv_results)

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Train the model.

        Parameters
        ----------
        train_loader : DataLoader
            Training data loader
        val_loader : DataLoader
            Validation data loader
        """
        logger.info("Training the model...")

        classifier = DiabetesRiskClassifier(
            model=self.model,
            batch_size=self.train_cfg.batch_size,
            pos_weight=self.model_cfg.pos_weight,
        )

        start_time = time.time()
        self.trainer.fit(
            classifier,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )
        end_time = time.time()

        logger.info(f"Training time: {(end_time - start_time):.2f} seconds")

    def evaluate(self, data_loader: DataLoader) -> Tuple[List[float], List[float]]:
        """
        Evaluate the model on the validation / test set.
        """
        if self.best_model_path is not None:
            # For evaluating on the test set
            classifier = DiabetesRiskClassifier.load_from_checkpoint(
                self.best_model_path
            )

        else:
            classifier = self.model

        classifier.eval()
        y_pred = []
        y_true = []

        with torch.no_grad():
            for x, y in data_loader:

                logits = self.model(x)  # [B, 1]
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).long().squeeze()

                y_pred.extend(preds.cpu().numpy())
                y_true.extend(y.cpu().numpy())

        return y_true, y_pred

    def export_prob(
        self,
        data_file: Path,
        output_path: Path,
        img_dir: Path = None,
        transform=None,
        feature_cols: List[str] = None,
    ):
        if self.best_model_path is not None:
            # For evaluating on the test set
            classifier = DiabetesRiskClassifier.load_from_checkpoint(
                self.best_model_path
            )

        else:
            classifier = self.model

        match self.model_cfg.name:
            case ModelType.MLP.value:
                dm = NeuralNetDataModule(data_file=data_file, feature_cols=feature_cols)

            case ModelType.NCTD.value:
                dm = NeuralNetDataModule(data_file=data_file, feature_cols=feature_cols)

            case ModelType.IGTD.value:
                dm = IgtdDataModule(data_file=data_file, img_dir=img_dir)

            case _:
                raise ValueError(f"Unsupported model type: {self.model_cfg.name}")

        train_idx, val_idx = self.k_fold_indices[0]
        if self.model_cfg.name == ModelType.IGTD.value:
            dm.setup(train_idx=train_idx, val_idx=val_idx, test_idx=self.test_idx)

        else:
            dm.setup(
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=self.test_idx,
                transform=transform,
            )

        prob_dfs = []
        classifier.eval()

        for split, indices, dataset in zip(
            ["train", "val", "test"],
            [train_idx, val_idx, self.test_idx],
            [dm.train_dataset, dm.val_dataset, dm.test_dataset],
        ):
            # Create dataloader without shuffling to preserve index order
            data_loader = DataLoader(
                dataset,
                batch_size=self.train_cfg.batch_size,
                shuffle=False,  # Important: no shuffling for export
                num_workers=0,  # Use single worker to ensure consistent order
            )

            y_prob = []
            y_true = []

            with torch.no_grad():
                for x, y in data_loader:

                    logits = self.model(x)  # [B, 1]
                    probs = torch.sigmoid(logits).squeeze().reshape(-1)

                    y_prob.extend(probs.cpu().numpy())
                    y_true.extend(y.cpu().numpy())

            df = pd.DataFrame(
                {
                    "id": indices,
                    "split": split,
                    "y_true": y_true,
                    "y_prob": y_prob,
                }
            )
            prob_dfs.append(df)

        all_probs = pd.concat(prob_dfs).sort_values(by="id")
        all_probs.to_csv(output_path, index=False)
        logger.info(f"Predicted probabilities exported to {output_path}")
