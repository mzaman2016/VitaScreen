import random
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from data.cdc import IgtdDataModule, NeuralNetDataModule
from data.utils import nctd_transform
from model.model_type import ModelType
from train_and_eval.ensemble_tree_trainer import EnsembleTreeTrainer
from train_and_eval.evaluate import compute_metrics
from train_and_eval.neuralnet_trainer import NeuralNetTrainer


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))

    # Enable reproducibility
    torch.manual_seed(seed=42)
    np.random.seed(42)
    random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Initialize results directory
    (Path("results") / cfg.results.model_name).mkdir(parents=True, exist_ok=True)

    # Initialize trainer based on model type - all use cross-validation
    transform = nctd_transform if cfg.model.name == ModelType.NCTD.value else None
    match cfg.model.name:
        case ModelType.MLP.value | ModelType.NCTD.value:
            trainer = NeuralNetTrainer(
                model_cfg=cfg.model,
                train_cfg=cfg.train,
            )
            trainer.setup(data_cfg=cfg.data)
            cv_results = trainer.cross_validate(
                data_file=Path(cfg.data.file_path),
                transform=transform,
                feature_cols=cfg.data.feature_cols
                if "feature_cols" in cfg.data
                else None,
            )

            logger.info("Evaluating the model on the test set ...")
            dm = NeuralNetDataModule(
                data_file=Path(cfg.data.file_path),
                feature_cols=cfg.data.feature_cols
                if "feature_cols" in cfg.data
                else None,
            )
            train_idx, val_idx = trainer.k_fold_indices[0]
            dm.setup(
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=trainer.test_idx,
                transform=transform,
            )
            y_test, y_pred = trainer.evaluate(dm.test_dataloader())

            # Export predicted probabilities
            trainer.export_prob(
                data_file=Path(cfg.data.file_path),
                output_path=Path(cfg.results.prob_path),
                transform=transform,
                feature_cols=cfg.data.feature_cols
                if "feature_cols" in cfg.data
                else None,
            )

        case ModelType.IGTD.value:
            trainer = NeuralNetTrainer(
                model_cfg=cfg.model,
                train_cfg=cfg.train,
            )
            trainer.setup(data_cfg=cfg.data)
            cv_results = trainer.cross_validate(
                data_file=Path(cfg.data.file_path), img_dir=Path(cfg.igtd.img_dir)
            )

            logger.info("Evaluating the model on the test set ...")
            dm = IgtdDataModule(
                data_file=Path(cfg.data.file_path), img_dir=Path(cfg.igtd.img_dir)
            )
            train_idx, val_idx = trainer.k_fold_indices[0]
            dm.setup(train_idx=train_idx, val_idx=val_idx, test_idx=trainer.test_idx)
            y_test, y_pred = trainer.evaluate(dm.test_dataloader())

            # Export predicted probabilities
            trainer.export_prob(
                data_file=Path(cfg.data.file_path),
                output_path=Path(cfg.results.prob_path),
                img_dir=Path(cfg.igtd.img_dir),
                transform=transform,
                feature_cols=cfg.data.feature_cols
                if "feature_cols" in cfg.data
                else None,
            )

        case ModelType.CatBoost.value:
            trainer = EnsembleTreeTrainer(hyperparams=cfg.model.hyperparams)
            trainer.setup(data_cfg=cfg.data)
            cv_results = trainer.cross_validate(enn=cfg.train.enn)

            logger.info("Evaluating the model on the test set ...")
            y_test, y_pred = trainer.evaluate()

            # Export predicted probabilities
            trainer.export_prob(output_path=Path(cfg.results.prob_path))

        case _:
            raise ValueError(f"Unsupported model type: {cfg.model.name}")

    # Compute metrics
    test_results = [
        compute_metrics(y_test, y_pred, avg) for avg in cfg.results.avg_options
    ]

    # Export results
    cv_results.to_csv(cfg.results.cv_path, index=False)
    logger.info(f"Cross-validation results saved to {cfg.results.cv_path}")

    pd.DataFrame(test_results).to_csv(cfg.results.test_path, index=False)
    logger.info(f"Predictions for the test set saved to {cfg.results.test_path}")

    # Clean up GPU memory if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
