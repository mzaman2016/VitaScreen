from pathlib import Path

import pandas as pd

BASE = Path("results/cb_f21_enn/explain_local")
SUMMARY = BASE / "selected_cases_summary.csv"
OUT_TXT = BASE / "case_narratives.txt"

THRESH = 0.5


def label_text(y):
    return "No diabetes" if int(y) == 0 else "Prediabetes/Diabetes"


def pred_text(p):
    return "High risk" if float(p) >= THRESH else "Low risk"


def fmt_feature(r):
    # Prefer decoded meaning, fallback to raw value
    val = r.get("value_meaning", "")
    if pd.isna(val) or str(val).strip() == "":
        val = r.get("feature_value", "")
    return f"{r['feature']} = {val}"


def main():
    summary = pd.read_csv(SUMMARY)

    lines = []
    for _, row in summary.iterrows():
        idx = int(row["id"])
        tag = str(row["tag"])
        y_true = int(row["y_true"])
        y_prob = float(row["y_prob_pipeline"])

        pred = 1 if y_prob >= THRESH else 0
        match = "MATCH" if pred == y_true else "MISMATCH"

        # Load decoded contributions
        contrib_path = BASE / f"top_contrib_{tag}_id{idx}_with_desc_decoded.csv"
        df = pd.read_csv(contrib_path)

        # Split into positive and negative SHAP drivers
        pos = (
            df[df["shap_value"] > 0].sort_values("shap_value", ascending=False).head(3)
        )
        neg = df[df["shap_value"] < 0].sort_values("shap_value", ascending=True).head(2)

        pos_list = (
            "; ".join(fmt_feature(r) for _, r in pos.iterrows()) if len(pos) else "None"
        )
        neg_list = (
            "; ".join(fmt_feature(r) for _, r in neg.iterrows()) if len(neg) else "None"
        )

        lines.append(f"Case ID {idx} ({tag.upper()}):")
        lines.append(
            f"  Ground truth: {y_true} ({label_text(y_true)}) | Model: {y_prob:.3f} ({pred_text(y_prob)}) | {match}"
        )
        lines.append(f"  Main factors increasing risk: {pos_list}")
        lines.append(f"  Main factors decreasing risk: {neg_list}")
        lines.append("")

    OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
    print("Saved:", OUT_TXT)


if __name__ == "__main__":
    main()
