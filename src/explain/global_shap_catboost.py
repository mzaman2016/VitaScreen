import argparse
from pathlib import Path

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
    parser.add_argument("--max_test_rows", type=int, default=5000)  # start small
    args = parser.parse_args()

    data_csv = Path(args.data_csv)
    prob_csv = Path(args.prob_csv)
    model_path = Path(args.model_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset (features = all except Label, same as BaseTrainer) :contentReference[oaicite:2]{index=2}
    df = pd.read_csv(data_csv)
    feature_cols = [c for c in df.columns if c != args.label_col]
    if args.label_col not in df.columns:
        raise ValueError(f"Label column '{args.label_col}' not found in {data_csv}")

    # Load test indices from prob.csv created by trainer.export_prob() :contentReference[oaicite:3]{index=3}
    probs = pd.read_csv(prob_csv)
    test_ids = probs.loc[probs["split"] == "test", "id"].astype(int).to_numpy()
    if len(test_ids) == 0:
        raise ValueError("No rows with split=='test' found in prob.csv")

    # Sample for speed (optional)
    if args.max_test_rows and len(test_ids) > args.max_test_rows:
        rng = np.random.default_rng(42)
        test_ids = rng.choice(test_ids, size=args.max_test_rows, replace=False)

    X_test = df.loc[test_ids, feature_cols].to_numpy()
    y_test = df.loc[test_ids, args.label_col].to_numpy()

    # Load CatBoost model
    model = CatBoostClassifier()
    model.load_model(str(model_path))

    # Compute SHAP values using CatBoost built-in SHAP (TreeSHAP under the hood)
    pool = Pool(X_test, label=y_test, feature_names=feature_cols)
    shap = model.get_feature_importance(pool, type="ShapValues")
    # shap shape: (n_rows, n_features + 1), last column is expected value
    shap_values = shap[:, :-1]

    mean_abs = np.abs(shap_values).mean(axis=0)
    imp_df = pd.DataFrame({"feature": feature_cols, "mean_abs_shap": mean_abs})
    imp_df = imp_df.sort_values("mean_abs_shap", ascending=False)

    out_csv = out_dir / "global_importance_mean_abs_shap.csv"
    imp_df.to_csv(out_csv, index=False)

    print(f"\nSaved: {out_csv}")
    print(f"Explained rows: {len(test_ids)} (sampled from test split)")
    print("\nTop 10 global drivers (mean |SHAP|):")
    print(imp_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
