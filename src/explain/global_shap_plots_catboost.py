import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv", default="data/cdcNormalDiabetic.csv")
    parser.add_argument("--label_col", default="Label")
    parser.add_argument("--prob_csv", default="results/cb_f21_enn/prob.csv")
    parser.add_argument("--model_path", default="results/cb_f21_enn/catboost_best.cbm")
    parser.add_argument("--out_dir", default="results/cb_f21_enn/explain_global")
    parser.add_argument("--max_test_rows", type=int, default=5000)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_csv(args.data_csv)
    feature_cols = [c for c in df.columns if c != args.label_col]

    # Load test indices from prob.csv (created by export_prob)
    probs = pd.read_csv(args.prob_csv)
    test_ids = probs.loc[probs["split"] == "test", "id"].astype(int).to_numpy()

    # Sample for speed
    if args.max_test_rows and len(test_ids) > args.max_test_rows:
        rng = np.random.default_rng(42)
        test_ids = rng.choice(test_ids, size=args.max_test_rows, replace=False)

    X_test = df.loc[test_ids, feature_cols].to_numpy()
    y_test = df.loc[test_ids, args.label_col].to_numpy()

    # Load model
    model = CatBoostClassifier()
    model.load_model(args.model_path)

    # Compute SHAP values (CatBoost built-in)
    pool = Pool(X_test, label=y_test, feature_names=feature_cols)
    shap_full = model.get_feature_importance(pool, type="ShapValues")
    shap_values = shap_full[:, :-1]  # (n, n_features)
    expected_value = shap_full[0, -1]  # baseline (approx.)

    # Global importance table
    mean_abs = np.abs(shap_values).mean(axis=0)
    imp_df = pd.DataFrame({"feature": feature_cols, "mean_abs_shap": mean_abs})
    imp_df = imp_df.sort_values("mean_abs_shap", ascending=False)
    imp_df.to_csv(out_dir / "global_importance_mean_abs_shap.csv", index=False)

    # Plot 1: Top-20 bar chart
    top_k = min(20, len(imp_df))
    top_df = imp_df.head(top_k).iloc[::-1]
    plt.figure(figsize=(8, 6))
    plt.barh(top_df["feature"], top_df["mean_abs_shap"])
    plt.xlabel("Mean |SHAP value| (global impact)")
    plt.title(f"Global Feature Importance (Top {top_k})")
    plt.tight_layout()
    plt.savefig(out_dir / "shap_global_bar_top20.png", dpi=300)
    plt.close()

    # Plot 2: SHAP summary (beeswarm) using shap if available
    # If shap is not installed, we skip this plot rather than failing.
    try:
        import shap

        shap.summary_plot(shap_values, X_test, feature_names=feature_cols, show=False)
        plt.tight_layout()
        plt.savefig(out_dir / "shap_summary_beeswarm.png", dpi=300)
        plt.close()
    except Exception as e:
        print("Could not generate beeswarm summary plot (shap not available or error).")
        print("Error:", str(e))

    # Save arrays (optional but useful)
    np.save(out_dir / "shap_values_test.npy", shap_values)
    np.save(out_dir / "X_test_sample.npy", X_test)

    print(f"\nSaved outputs to: {out_dir}")
    print("Saved: shap_global_bar_top20.png")
    print("Saved: shap_summary_beeswarm.png (if shap worked)")
    print(f"Baseline (expected value approx): {expected_value}")

    print("\nTop 10 features:")
    print(imp_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
