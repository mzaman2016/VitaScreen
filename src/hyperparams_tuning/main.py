from functools import partial

import hydra
import optuna
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import f1_score

from model.model_type import ModelType
from train_and_eval.ensemble_tree_trainer import EnsembleTreeTrainer


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))

    if cfg.model.name != ModelType.CatBoost.value:
        raise ValueError(
            f"Hyperparameter tuning is only supported for {ModelType.CatBoost.value} model type."
        )

    study = optuna.create_study(direction="maximize")
    study.optimize(partial(objective, cfg=cfg), n_trials=cfg.n_trials, n_jobs=4)

    logger.info(f"Number of finished trials: {len(study.trials)}")

    logger.info("Best trial:")
    trial = study.best_trial

    logger.info(f"  Value: {trial.value}")
    logger.info("  Params: ")
    for key, value in trial.params.items():
        logger.info(f"    {key}: {value}")


def objective(trial, cfg: DictConfig):
    search_space = cfg.model.hyperparams_search_space
    params = {}

    for param_name, param_cfg in search_space.items():
        if hasattr(param_cfg, "choices"):
            params[param_name] = trial.suggest_categorical(
                param_name, param_cfg.choices
            )
        elif hasattr(param_cfg, "low") and hasattr(param_cfg, "high"):
            # Determine integer vs float
            if isinstance(param_cfg.low, int) and isinstance(param_cfg.high, int):
                params[param_name] = trial.suggest_int(
                    param_name, param_cfg.low, param_cfg.high
                )
            else:
                params[param_name] = trial.suggest_float(
                    param_name,
                    param_cfg.low,
                    param_cfg.high,
                    log=param_cfg.get("log", False),
                )
        else:
            # skip unexpected keys like nested groups
            continue

    params_cfg = OmegaConf.create(params)
    trainer = EnsembleTreeTrainer(
        hyperparams=OmegaConf.merge({}, cfg.model.hyperparams, params_cfg)
    )
    trainer.setup(data_cfg=cfg.data)

    # Evaluate model performance
    y_true, y_pred = trainer.train(enn=cfg.train.enn)

    return f1_score(y_true=y_true, y_pred=y_pred, average="macro")


if __name__ == "__main__":
    main()
