# XAI-RF: Random Forest explainability

Explainable AI using a **Random Forest** classifier for diabetes risk prediction. Uses Marzia’s dataset (`train_data.csv` + `test_data.csv` in the parent `Diabetes_Risk_XAI/` folder).

## What it does

- Loads the full dataset (train + test merged when both exist)
- Trains a Random Forest (`n_estimators=100`, `max_depth=15`, `random_state=42`)
- Saves **4 CSV files** (TP, TN, FP, FN) with **all samples** per category under `csv/`
- Generates **4 heatmap images** (mean normalized feature values per category) under `output/images/`
- **On each run:** deletes previous CSVs and images, then writes fresh outputs

## How to run

```bash
cd Diabetes_Risk_XAI/XAI-RF
python rf_xai_method.py
```

On Windows you can also use `run_xai.bat`.

## Output layout

```
XAI-RF/
├── csv/
│   ├── TP.csv
│   ├── TN.csv
│   ├── FP.csv
│   └── FN.csv
├── output/
│   └── images/
│       ├── rf_TP.png
│       ├── rf_TN.png
│       ├── rf_FP.png
│       └── rf_FN.png
├── rf_xai_method.py
├── requirements.txt
└── run_xai.bat
```

## Results (latest run on committed data)

These metrics match the current `train_data.csv` / `test_data.csv` in `Diabetes_Risk_XAI/` (80/20 stratified split, `random_state=42`):

| Metric    | Value   |
|----------|---------|
| Accuracy | **0.95** |
| Precision | **0.50** |
| Recall   | **0.20** |
| F1-Score | **0.29** |

The diabetic class is small in the test set, so precision/recall/F1 for the positive class are lower than overall accuracy.

## Dependencies

```bash
pip install -r requirements.txt
```

## Related folders

See [`../README.md`](../README.md) for **GRAD-CAM** (IGTD + CNN + Grad-CAM) and **IGTD** (IGTD + CNN only).
