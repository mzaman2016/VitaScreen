# Diabetes Risk Prediction Pipeline

End-to-end pipeline for diabetes risk prediction using IGTD (Image Generator for Tabular Data) and CNN, based on **Shenghao Wang et al.** - "Diabetes Risk Modeling through Tabular-to-Image Transformations and Ensemble Learning".

## Model: cnn_igtd_f15_enn

Uses the best CNN configuration from the paper's Results section:
- **IGTD** for tabular-to-image transformation (15 features → 5×3 grayscale images)
- **ENN** (Edited Nearest Neighbors, k=3) for class imbalance
- **CNN** architecture: 4 conv blocks (64 filters, 3×3), BatchNorm, ReLU, Global Average Pooling, Dropout 50%

## Tasks Completed

1. ✅ **Create images from dataset using IGTD** - Converts tabular features to 2D images
2. ✅ **Create CSV file with labels** - `labels.csv` with sample_id and label
3. ✅ **Shuffle and split 80:20** - `train_data.csv` and `test_data.csv`
4. ✅ **Train CNN model** - Using cnn_igtd_f15_enn architecture
5. ✅ **Save CNN model** - `cnn_diabetes_model.pt` (PyTorch) or `.keras` (TensorFlow)
6. ✅ **Test with test dataset** - Evaluation on held-out 20%
7. ✅ **Results** - Accuracy, F1-Score, Precision, Recall in `results_summary.csv`

## FDI File - Not Required

**No separate FDI file is needed.** "FDI" in FDIToImage.py refers to the **output folder name** for IGTD results, not an input file. The pipeline creates images directly from the tabular CSV dataset using IGTD.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python diabetes_prediction_pipeline.py
```

Or with custom CSV path:
```python
from diabetes_prediction_pipeline import run_pipeline
run_pipeline(csv_path="path/to/your/data.csv")
```

## Data

- **Local CSV**: Place `cdcNormalDiabeticFE1_20RFFSQ.csv` in Downloads (or specify path)
- **Fallback**: If local file is invalid/corrupted, automatically loads CDC Diabetes dataset from UCI (id=891)

## Output Files

| File | Description |
|------|-------------|
| `labels.csv` | Image filenames (image_file, label) |
| `train_data.csv` | Training split (80%) - after ENN |
| `test_data.csv` | Test split (20%) |
| `IGTD_Images/` | Images named diabetic_X.png, non_diabetic_X.png |
| `IGTD_Results/train/` | IGTD optimization results |
| `cnn_diabetes_model.pt` | Saved CNN weights |
| `results_summary.csv` | Accuracy, Precision, Recall, F1-Score |

## Expected Results (from paper)

- **Accuracy**: ~0.82
- **F1-Score**: ~0.68
- **Precision**: ~0.67
- **Recall**: ~0.70
