from pathlib import Path

import pandas as pd
import scikit_posthocs as sp
from loguru import logger
from scipy.stats import friedmanchisquare


def main():
    results_dir = Path("results")
    results = {}
    for model_name in results_dir.iterdir():
        if model_name.is_dir() and model_name.name in (
            "cb_f21_enn",
            "mlp_f15_enn",
            "nctd_f21_enn",
            "igtd_f15_enn",
        ):
            logger.info(f"Processing model: {model_name.name}")
            df = pd.read_csv(model_name / "cv_results.csv", usecols=["f1_score"])
            results[model_name.name] = df["f1_score"].values

    stat, p = friedmanchisquare(*results.values())
    logger.info(f"Friedman chi2={stat:.4f}, p={p:.6g}")

    results_df = pd.DataFrame(results)
    pvals_nemenyi = sp.posthoc_nemenyi_friedman(results_df.values)  # returns k×k matrix
    pvals_nemenyi.index = results_df.columns
    pvals_nemenyi.columns = results_df.columns
    logger.info("\nNemenyi p-values (pairwise):")
    logger.info(f"\n{pvals_nemenyi}")


if __name__ == "__main__":
    main()
