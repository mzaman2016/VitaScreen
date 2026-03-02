# IGTD Pipeline – Old CDC Dataset (No Grad-CAM)

This folder contains the **IGTD-only** pipeline for diabetes risk prediction using the **old CDC Diabetes Health Indicators dataset** (500 samples). No Grad-CAM or explainability is used here.

## Contents

- `run_pipeline.py` – Main script to run the full pipeline
- `IGTD_Functions.py` – IGTD algorithm functions (Zhu et al.)
- `IGTD_Results/` – Created on run; contains IGTD optimization outputs
- `IGTD_Images/` – Created on run; contains diabetic_X.png and non_diabetic_X.png
- `results_summary.csv` – Created on run; contains Accuracy, Precision, Recall, F1-Score

## How to Run

```bash
cd IGTD
python run_pipeline.py
```

## Behavior

- **Old images are removed** before each run: `IGTD_Results` and `IGTD_Images` are cleared.
- **New images are generated** each time you run the script.
- **Dataset:** CDC Diabetes Health Indicators from UCI (500 samples, 15 features, 80:20 split).
- **Pipeline:** IGTD → ENN (k=3) → CNN (cnn_igtd_f15_enn).

## Expected Results

- Accuracy: ~0.89  
- Precision: ~0.54  
- Recall: ~0.58  
- F1-Score: ~0.56  

## Dependencies

```bash
pip install torch pandas numpy scikit-learn scipy matplotlib ucimlrepo
```

## GitHub

This folder is intended to be used as one of two folders in the repo:

1. **IGTD/** – Old dataset, IGTD + CNN only (this folder)
2. **Root or XAI folder** – New dataset, IGTD + CNN + Grad-CAM
