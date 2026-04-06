# GRAD-CAM Pipeline

IGTD + CNN + Grad-CAM for diabetes risk prediction. Uses Marzia’s dataset (`train_data.csv` + `test_data.csv` in the parent `Diabetes_Risk_XAI/` folder).

## What it does

- Loads the full dataset (train + test merged when both exist)
- Runs IGTD → CNN → Grad-CAM
- Saves **4 CSV files** (TP, TN, FP, FN) with **all samples** per category under `csv/`
- Generates **4 images** (mean IGTD + mean Grad-CAM per category) under `output/images/`
- **On each run:** clears old CSVs and images, then regenerates outputs

## How to run

From the repo root:

```bash
cd Diabetes_Risk_XAI/GRAD-CAM
python gradcam_pipeline.py
```

## Output layout

```
GRAD-CAM/
├── csv/
│   ├── TP.csv
│   ├── TN.csv
│   ├── FP.csv
│   └── FN.csv
├── output/
│   └── images/
│       ├── gradcam_TP.png
│       ├── gradcam_TN.png
│       ├── gradcam_FP.png
│       └── gradcam_FN.png
├── gradcam_pipeline.py
├── IGTD_Functions.py
├── IGTD_Results/      # created on run
├── IGTD_Images/       # created on run
├── GradCAM_Output/    # legacy / optional outputs from older runs
├── results_summary.csv
└── cnn_diabetes_model.pt
```

## Results (latest `results_summary.csv`)

| Metric    | Value   |
|----------|---------|
| Accuracy | **0.83** |
| Precision | **0.37** |
| Recall   | **0.58** |
| F1-Score | **0.45** |

Values are from the same run as the committed `results_summary.csv` (single train/test split, stratified).

## Dependencies

```bash
pip install torch pandas numpy scikit-learn scipy matplotlib ucimlrepo seaborn
```

## Related folders in this repo

See the parent [`../README.md`](../README.md) for **IGTD** (IGTD + CNN only) and **XAI-RF** (Random Forest + heatmap XAI).
