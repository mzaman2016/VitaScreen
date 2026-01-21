from typing import List

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from train_and_eval.evaluate import compute_metrics


def majority_vote(
    prob_df: pd.DataFrame,
    models: List[str],
    cut: float = 0.5,
    tie_break_up: bool = True,  # True => ties go to 1 (>= ceil(n/2)); False => strict majority (> n/2)
) -> None:
    test_prob_df = prob_df[prob_df["split"] == "test"].copy()

    votes = (test_prob_df[models].to_numpy() >= cut).sum(axis=1)
    n = len(models)
    thr = np.ceil(n / 2) if tie_break_up else (n / 2)

    if len(models) % 2 == 0:
        test_prob_df["y_pred"] = (votes > thr).astype(int)
    else:
        test_prob_df["y_pred"] = (votes >= thr).astype(int)

    res = compute_metrics(
        test_prob_df["y_true"], test_prob_df["y_pred"], avg_option="macro"
    )
    res["n_obs"] = len(test_prob_df)
    logger.info(f"Majority Vote Ensemble Results:\n{res}")
    del test_prob_df


def avg_prob(
    prob_df: pd.DataFrame,
    models: List[str],
    cut: float = 0.5,
) -> None:
    test_prob_df = prob_df[prob_df["split"] == "test"].copy()

    test_prob_df["y_prob"] = test_prob_df[models].mean(axis=1)
    test_prob_df["y_pred"] = (test_prob_df["y_prob"] >= cut).astype(int)

    res = compute_metrics(
        test_prob_df["y_true"], test_prob_df["y_pred"], avg_option="macro"
    )
    res["n_obs"] = len(test_prob_df)
    logger.info(f"Average Probability Ensemble Results:\n{res}")
    del test_prob_df


def logistic_regression(
    prob_df: pd.DataFrame,
    models: List[str],
) -> None:
    # Use only validation split to train meta-learner (out-of-fold predictions)
    # This avoids overfitting issues from using training predictions
    train_prob_df = prob_df[prob_df["split"] == "val"].copy()
    test_prob_df = prob_df[prob_df["split"] == "test"].copy()
    X_train = train_prob_df[models].values
    y_train = train_prob_df["y_true"].values
    X_test = test_prob_df[models].values
    y_test = test_prob_df["y_true"].values

    logger.info(f"Training meta-learner on {len(X_train)} validation samples")

    # Use stronger regularization and enable more solver options
    lr_model = LogisticRegression(
        class_weight="balanced",
        C=0.001,  # Can tune this
        max_iter=1000,
        solver="lbfgs",
        random_state=42,
    )
    lr_model.fit(X_train, y_train)

    # Log learned weights
    logger.info(f"Learned weights: {dict(zip(models, lr_model.coef_[0]))}")
    logger.info(f"Intercept: {lr_model.intercept_[0]}")

    y_pred = lr_model.predict(X_test)
    res = compute_metrics(y_test, y_pred, avg_option="macro")
    res["n_obs"] = len(test_prob_df)
    logger.info(f"Logistic Regression Ensemble Results:\n{res}")


def random_forest(
    prob_df: pd.DataFrame,
    models: List[str],
) -> None:
    # Use only validation split to train meta-learner (out-of-fold predictions)
    train_prob_df = prob_df[prob_df["split"] == "val"].copy()
    # train_prob_df = prob_df[prob_df["split"] != "test"].copy()
    test_prob_df = prob_df[prob_df["split"] == "test"].copy()
    X_train = train_prob_df[models].values
    y_train = train_prob_df["y_true"].values
    X_test = test_prob_df[models].values
    y_test = test_prob_df["y_true"].values

    logger.info(
        f"Training Random Forest meta-learner on {len(X_train)} validation samples"
    )
    logger.info(f"Class distribution in validation: {np.bincount(y_train.astype(int))}")

    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=3,
        min_samples_split=20,
        min_samples_leaf=20,
        class_weight={0: 1, 1: 2},
        random_state=42,
    )
    rf_model.fit(X_train, y_train)

    # Log feature importances
    importances = dict(zip(models, rf_model.feature_importances_))
    logger.info(f"Feature importances: {importances}")

    y_pred = rf_model.predict(X_test)
    res = compute_metrics(y_test, y_pred, avg_option="macro")
    res["n_obs"] = len(test_prob_df)
    logger.info(f"Random Forest Ensemble Results:\n{res}")
