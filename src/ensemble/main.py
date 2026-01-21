import hydra
import pandas as pd
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from ensemble.combine_predictions import avg_prob, majority_vote, random_forest
from train_and_eval.evaluate import compute_metrics


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))

    # Load predicted probabilities from ensemble members
    models = cfg.ensemble.models
    prob_path = cfg.ensemble.prob_path.format(model_name=models[0])
    agg_prob_df = pd.read_csv(prob_path)
    agg_prob_df.columns = ["id", "split", "y_true", f"{models[0]}"]
    logger.debug(f"\n{agg_prob_df.head(10)}")
    n_test_obs = agg_prob_df[agg_prob_df["split"] == "test"].shape[0]
    logger.info(f"Number of test data points: {n_test_obs}")

    for model in models[1:]:
        prob_path = cfg.ensemble.prob_path.format(model_name=model)
        prob_df = pd.read_csv(prob_path)

        # Sanity check individual model
        logger.info(f"Processing model: {model}")
        test_prob_df = prob_df[prob_df["split"] == "test"].copy()
        logger.info(f"Number of test samples: {len(test_prob_df)}")

        # Number of test samples should be the same
        assert n_test_obs == len(test_prob_df)

        y_pred = (test_prob_df["y_prob"] >= 0.5).astype(int)
        res = compute_metrics(test_prob_df["y_true"], y_pred, avg_option="macro")
        logger.info(f"Metrics: {res}")
        del test_prob_df

        # Verify that the ids match
        assert all(agg_prob_df["id"] == prob_df["id"])
        assert all(agg_prob_df["split"] == prob_df["split"])
        assert all(agg_prob_df["y_true"] == prob_df["y_true"])

        prob_df.columns = ["id", "split", "y_true", model]
        agg_prob_df = pd.merge(
            agg_prob_df, prob_df, on=["id", "split", "y_true"], how="inner"
        )

    logger.info(f"Aggregated probabilities shape: {agg_prob_df.shape}")
    logger.debug(f"\n{agg_prob_df.head(10)}")

    majority_vote(agg_prob_df, models)
    avg_prob(agg_prob_df, models)
    random_forest(agg_prob_df, models)


if __name__ == "__main__":
    main()
