from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import shap
from catboost import CatBoostClassifier, Pool

OUT_DIR = Path("results/cb_f21_enn/explain_local")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_CSV = "data/cdcNormalDiabetic.csv"
PROB_CSV = "results/cb_f21_enn/prob.csv"
MODEL_PATH = "results/cb_f21_enn/catboost_best.cbm"

LABEL_COL = "Label"
THRESH = 0.5


def main():
    # Load data
    df = pd.read_csv(DATA_CSV)
    feature_cols = [c for c in df.columns if c != LABEL_COL]

    # Load prob.csv (gives exact split ids + y_prob)
    prob = pd.read_csv(PROB_CSV)
    test = prob[prob["split"] == "test"].copy()

    # Pick cases
    high_ids = (
        test.sort_values("y_prob", ascending=False).head(3)["id"].astype(int).tolist()
    )
    low_ids = (
        test.sort_values("y_prob", ascending=True).head(3)["id"].astype(int).tolist()
    )

    test["dist_to_thresh"] = (test["y_prob"] - THRESH).abs()
    border_ids = test.sort_values("dist_to_thresh").head(2)["id"].astype(int).tolist()

    selected = (
        [(i, "high") for i in high_ids]
        + [(i, "low") for i in low_ids]
        + [(i, "borderline") for i in border_ids]
    )

    # Load model
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)

    # Build a single batch for CatBoost SHAP computation
    ids = [i for i, _ in selected]
    X_sel = df.loc[ids, feature_cols]
    y_sel = df.loc[ids, LABEL_COL].astype(int).to_numpy()

    # CatBoost built-in SHAP values
    pool = Pool(X_sel, label=y_sel, feature_names=feature_cols)
    shap_full = model.get_feature_importance(pool, type="ShapValues")
    # shap_full shape: (n_cases, n_features + 1). last column = expected value (baseline)
    shap_vals = shap_full[:, :-1]
    base_val = float(shap_full[0, -1])

    # predicted probability for each selected id (for filename / reporting)
    pred_prob = model.predict_proba(X_sel)[:, 1]

    summary_rows = []

    for row_idx, (idx, tag) in enumerate(selected):
        y_true = int(df.loc[idx, LABEL_COL])
        y_prob = float(prob.loc[prob["id"] == idx, "y_prob"].values[0])  # from pipeline

        values = shap_vals[row_idx]
        data_row = X_sel.iloc[row_idx].to_numpy()

        # Build SHAP Explanation object manually (no TreeExplainer needed)
        exp = shap.Explanation(
            values=values,
            base_values=base_val,
            data=data_row,
            feature_names=feature_cols,
        )

        # Waterfall plot
        shap.plots.waterfall(exp, show=False)

        # --- Overlay Ground Truth + Prediction on the image ---
        gt_text = "No diabetes" if y_true == 0 else "Prediabetes/Diabetes"
        pred_label = 1 if y_prob >= THRESH else 0
        pred_text = "High risk" if pred_label == 1 else "Low risk"
        match_text = "MATCH" if pred_label == y_true else "MISMATCH"

        plt.gcf().text(
            0.01,
            0.99,
            f"ID: {idx} | Ground Truth: {y_true} ({gt_text}) | Model: {y_prob:.3f} ({pred_text}) | {match_text}",
            ha="left",
            va="top",
            fontsize=10,
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                alpha=0.85,
                edgecolor="gray",
            ),
        )
        # ------------------------------------------------------

        out_path = OUT_DIR / f"waterfall_{tag}_id{idx}_prob{y_prob:.3f}_y{y_true}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()

        # Save top contributors table
        contrib = pd.DataFrame(
            {"feature": feature_cols, "feature_value": data_row, "shap_value": values}
        ).sort_values("shap_value", key=lambda s: s.abs(), ascending=False)

        contrib_path = OUT_DIR / f"top_contrib_{tag}_id{idx}.csv"
        contrib.to_csv(contrib_path, index=False)

        summary_rows.append(
            {
                "id": idx,
                "tag": tag,
                "y_true": y_true,
                "y_prob_pipeline": y_prob,
                "y_prob_model": float(pred_prob[row_idx]),
                "top_feature_1": contrib.iloc[0]["feature"],
                "top_feature_2": contrib.iloc[1]["feature"],
                "top_feature_3": contrib.iloc[2]["feature"],
            }
        )

    pd.DataFrame(summary_rows).to_csv(
        OUT_DIR / "selected_cases_summary.csv", index=False
    )

    print("Saved local explanation outputs to:", OUT_DIR)
    print("Baseline (expected value):", base_val)
    print("Selected cases:", selected)


if __name__ == "__main__":
    main()
