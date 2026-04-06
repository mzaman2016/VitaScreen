# Diabetes Risk XAI (VitaScreen)

Self-contained experiments for **diabetes risk prediction** with **explainable AI**: tabular data is used either as-is (Random Forest + heatmaps) or transformed to images (IGTD + CNN, optionally with Grad-CAM).

All three pipelines below read **`train_data.csv`** and **`test_data.csv`** from this directory (Marzia’s merged CDC-style features, 15 columns + label). Place those files here before running, or rely on the copies committed in this folder.

---

## Folder layout

| Folder | Method | Role |
|--------|--------|------|
| **[IGTD/](IGTD/)** | IGTD → CNN | Baseline image pipeline on the classic CDC-style sampling (see IGTD README). |
| **[GRAD-CAM/](GRAD-CAM/)** | IGTD → CNN → Grad-CAM | Same image route as IGTD, plus Grad-CAM saliency and TP/TN/FP/FN CSVs + aggregate figures. |
| **[XAI-RF/](XAI-RF/)** | Random Forest + heatmaps | Tabular model with per-category mean feature heatmaps (`rf_TP.png` … `rf_FN.png`) and matching CSVs. |

---

## Results snapshot (aligned with committed outputs)

### GRAD-CAM (`GRAD-CAM/results_summary.csv`)

| Metric | Value |
|--------|--------|
| Accuracy | 0.83 |
| Precision | 0.37 |
| Recall | 0.58 |
| F1-Score | 0.45 |

Grad-CAM also writes category CSVs and **`output/images/gradcam_*.png`** (see [GRAD-CAM/README.md](GRAD-CAM/README.md)).

### XAI-RF (latest run on committed `train_data.csv` / `test_data.csv`)

| Metric | Value |
|--------|--------|
| Accuracy | 0.95 |
| Precision | 0.50 |
| Recall | 0.20 |
| F1-Score | 0.29 |

Outputs: **`XAI-RF/csv/*.csv`** and **`XAI-RF/output/images/rf_*.png`**. The positive class is rare in the test split, so class-wise precision/recall/F1 are lower than overall accuracy.

### IGTD-only (reference)

Expected ballpark metrics are documented in [IGTD/README.md](IGTD/README.md) (~0.89 accuracy); that subfolder uses its own run script and dataset wiring.

---

## Quick start

```bash
# From repository root
cd Diabetes_Risk_XAI

# Random Forest XAI
cd XAI-RF
python rf_xai_method.py

# Grad-CAM pipeline (IGTD + CNN + Grad-CAM)
cd ../GRAD-CAM
python gradcam_pipeline.py

# IGTD + CNN only
cd ../IGTD
python run_pipeline.py
```

Install dependencies per subfolder (`requirements.txt` where present, or see each README).

---

## Data files

- `train_data.csv`, `test_data.csv` — used by **GRAD-CAM** and **XAI-RF** (parent directory is the project root for those scripts).

---

## References

- Zhu, Y., Brettin, T., Xia, F. et al. *Converting tabular data into images for deep learning with convolutional neural networks.* Sci Rep 11, 11325 (2021). [DOI](https://doi.org/10.1038/s41598-021-90923-y)
- IGTD code: [github.com/zhuyitan/IGTD](https://github.com/zhuyitan/IGTD)
