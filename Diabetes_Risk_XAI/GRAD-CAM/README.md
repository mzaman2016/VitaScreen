# GRAD-CAM Pipeline – New Dataset with Explainable AI

This folder contains the **Grad-CAM** pipeline for diabetes risk prediction using the **new CDC dataset** (cdcNormalDiabeticFE1_20RFFSQ.csv or UCI fallback). It adds Grad-CAM explainability to show which parts of each image the model used to make its prediction.

## Contents

- `run_pipeline.py` – Main script (IGTD → CNN → Grad-CAM)
- `IGTD_Functions.py` – IGTD algorithm functions
- `IGTD_Results/` – Created on run; IGTD optimization outputs
- `IGTD_Images/` – Created on run; diabetic_X.png, non_diabetic_X.png
- `GradCAM_Output/` – Created on run; gradcam_sample_1.png … gradcam_sample_8.png
- `results_summary.csv` – Created on run; Accuracy, Precision, Recall, F1-Score

## How to Run

```bash
cd GRAD-CAM
python run_pipeline.py
```

## Behavior

- **Old images and data are removed** before each run: `IGTD_Results`, `IGTD_Images`, and `GradCAM_Output` are cleared.
- **New images and data are generated** each time you run the script.
- **Dataset:** New CDC (cdcNormalDiabeticFE1_20RFFSQ.csv) or UCI fallback, 500 samples, 15 features, 80:20 split.
- **Pipeline:** IGTD → ENN (k=3) → CNN → Grad-CAM with feature names and color scales.

## Expected Results

- Accuracy: ~0.84
- Precision: ~0.38
- Recall: ~0.50
- F1-Score: ~0.43

## Dependencies

```bash
pip install torch pandas numpy scikit-learn scipy matplotlib ucimlrepo
```

## GitHub

This folder is one of two folders in the repo:

1. **IGTD/** – Old dataset, IGTD + CNN only
2. **GRAD-CAM/** – New dataset, IGTD + CNN + Grad-CAM (this folder)
