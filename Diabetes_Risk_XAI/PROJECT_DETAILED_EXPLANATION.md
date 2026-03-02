# Diabetes Prediction Pipeline - Detailed Explanation

## 1. DATA USED

### 1.1 Dataset Source
- **Primary**: CDC Diabetes Health Indicators Dataset (UCI Repository, ID: 891)
- **Original source**: 2015 Behavioral Risk Factor Surveillance System (BRFSS) survey by U.S. CDC
- **Total size**: 253,680 rows (we use 500 samples for faster execution)
- **Local fallback**: `cdcNormalDiabeticFE1_20RFFSQ.csv` (if valid CSV; currently loads from UCI as local file contains HTML)

### 1.2 Data Type & Structure
- **Format**: Tabular (rows = patients, columns = features)
- **Target variable**: `Diabetes_binary` (0 = No diabetes, 1 = Diabetes or prediabetes)
- **Original target**: `Diabetes_012` (0=No, 1=Prediabetic, 2=Diabetic) → converted to binary (0 vs 1)
- **Class imbalance**: ~84% non-diabetic, ~16% diabetic

### 1.3 Features Used (15 features - from Shenghao's paper)
| # | Feature | Type | Description |
|---|---------|------|--------------|
| 1 | HighBP | Binary | High blood pressure |
| 2 | HighChol | Binary | High cholesterol |
| 3 | CholCheck | Binary | Cholesterol check in 5 years |
| 4 | BMI | Numerical | Body Mass Index |
| 5 | Smoker | Binary | Ever smoked |
| 6 | Stroke | Binary | History of stroke |
| 7 | HeartDiseaseorAttack | Binary | Heart disease or attack |
| 8 | PhysActivity | Binary | Physical activity |
| 9 | Fruits | Binary | Fruit consumption |
| 10 | Veggies | Binary | Vegetable consumption |
| 11 | HvyAlcoholConsump | Binary | Heavy alcohol use |
| 12 | AnyHealthcare | Binary | Has healthcare |
| 13 | NoDocbcCost | Binary | No doctor due to cost |
| 14 | GenHlth | Categorical | General health (1-5) |
| 15 | MentHlth | Numerical | Mental health days |

---

## 2. PIPELINE STEPS (What Happens When You Run)

### Step 0: Clean Old Outputs
- Deletes `IGTD_Results/` folder
- Deletes `IGTD_Images/` folder
- **Purpose**: Every run starts fresh; all outputs are updated

### Step 1: Load Data
- Fetches CDC dataset from UCI (or local CSV if valid)
- Takes 500 random samples (`max_samples=500`)
- Converts `Diabetes_012` → `Diabetes_binary` (0 or 1)

### Step 2: Shuffle & Split 80:20
- Shuffles data (random_state=42 for reproducibility)
- **Train**: 80% = 400 samples
- **Test**: 20% = 100 samples
- Uses **stratified split** (keeps same % of diabetic in train & test)

### Step 3: ENN (Edited Nearest Neighbors)
- **What**: Removes majority-class (non-diabetic) samples that KNN misclassifies
- **k=3**: Uses 3 nearest neighbors
- **Result**: Train reduces from 400 → ~360-380 (some noisy non-diabetic samples removed)
- **Purpose**: Reduces class imbalance, improves decision boundary (per Shenghao's paper)

### Step 4: IGTD - Create Images
**4a. IGTD Optimization (on train subset)**
- Converts 15 features → 5×3 grayscale image (15 pixels)
- Computes feature similarity (Euclidean distance)
- Swaps feature positions to match image grid layout (minimizes error)
- Runs ~800 steps; saves optimal feature arrangement (index)

**4b. Generate Images**
- Applies same arrangement to ALL train + test samples
- Each row of data → one 5×3 grayscale PNG
- Saves to `IGTD_Images/` as `diabetic_0.png`, `diabetic_1.png`, `non_diabetic_0.png`, etc.

### Step 5: Create Labels CSV
- Columns: `image_file`, `label`
- Maps each PNG filename to 0 (non-diabetic) or 1 (diabetic)

### Step 6: Save Train/Test CSVs
- `train_data.csv`: Training data (after ENN) - 15 features + Diabetes_binary
- `test_data.csv`: Test data - 15 features + Diabetes_binary

### Step 7: Prepare for CNN
- Normalize images: 0-255 → 0-1
- Reshape: (5, 3, N) → (N, 1, 5, 3) for PyTorch (batch, channel, height, width)

### Step 8: Train CNN
- **Architecture**: 4 conv blocks (64 filters, 3×3), BatchNorm, ReLU, Global Avg Pool, Dropout 50%
- **Loss**: BCEWithLogitsLoss with pos_weight=3 (penalize missing diabetic more)
- **Optimizer**: Adam, lr=8e-4
- **Epochs**: 30
- **Batch size**: 64
- Saves model to `cnn_diabetes_model.pt`

### Step 9: Evaluate & Report
- Predicts on test set (threshold 0.5)
- Computes Accuracy, Precision, Recall, F1-Score
- Saves to `results_summary.csv`

---

## 3. CODE CHANGES FROM ORIGINAL

### 3.1 Original Files (Before Our Work)
- `IGTD_Functions.py.removeme` - IGTD algorithm
- `FDIToImage.py.removeme` - Used X_combined20.csv, 5×6 grid, saved to FDI folder

### 3.2 What We Created/Changed

| File | Change |
|------|--------|
| **IGTD_Functions.py** | New file (from .removeme). Added: `save_images_diabetic_naming()`, `labels` param in `generate_image_data`, `matplotlib.use('Agg')`, `os.path.join` for Windows, pickle instead of _pickle |
| **diabetes_prediction_pipeline.py** | **New file** - Full pipeline (didn't exist before) |
| **requirements.txt** | New - dependencies |
| **README.md** | New - documentation |

### 3.3 Key Additions in Pipeline (Not in Original)
1. **Data loading**: UCI fallback when local CSV invalid
2. **Sample limit**: `max_samples=500` (configurable)
3. **ENN**: `apply_enn()` - not in FDIToImage
4. **Clean on run**: Delete old folders at start
5. **Image naming**: `diabetic_X.png`, `non_diabetic_X.png` (original used `_0_image.png`)
6. **Labels CSV**: `image_file` + `label` columns
7. **Train/test split**: 80:20, stratified
8. **CNN training**: Full PyTorch/TensorFlow training loop
9. **Results**: Accuracy, Precision, Recall, F1-Score

### 3.4 IGTD Changes from Original
- **Original**: Used `X_combined20.csv`, 5×6 grid (30 features), saved to FDI folder
- **Ours**: Uses CDC data, 5×3 grid (15 features), saves to IGTD_Images with diabetic/non_diabetic names
- **Original**: No ENN, no train/test split, no CNN training
- **Ours**: Full pipeline with ENN, split, CNN, evaluation

---

## 4. RESULTS EXPLAINED (From Terminal / results_summary.csv)

### 4.1 Example Output (500 samples run)
```
Using 500 samples
Train: 400, Test: 100
After ENN (k=3): Train 361 samples
Saved 461 images to IGTD_Images/
```

**Number breakdown:**
- **500** samples loaded
- **400** train, **100** test (80:20 split)
- **361** train after ENN (39 noisy non-diabetic samples removed)
- **461** total images = 361 train + 100 test

### 4.2 Metrics Explained

| Metric | Value | Meaning |
|--------|-------|---------|
| **Accuracy** | 0.88 | 88% of all predictions are correct (88 out of 100 test samples) |
| **Precision** | 0.50 | When model says "diabetic", it's right 50% of the time |
| **Recall** | 0.50 | Model catches 50% of actual diabetic cases |
| **F1-Score** | 0.50 | Balance of Precision and Recall |

### 4.3 Classification Report Explained
```
              precision    recall  f1-score   support
Non-Diabetic       0.92      0.94      0.93        84
    Diabetic       0.50      0.50      0.50        16
    accuracy                           0.88       100
   macro avg       0.71      0.72      0.71       100
weighted avg       0.86      0.88      0.87       100
```

- **support**: Actual count in test set (84 non-diabetic, 16 diabetic)
- **Non-Diabetic row**: Model is good at identifying non-diabetic (92% precision, 94% recall)
- **Diabetic row**: Model struggles with diabetic (50% precision, 50% recall) - class imbalance
- **macro avg**: Average of both classes (equal weight)
- **weighted avg**: Average weighted by support (84% non-diabetic, 16% diabetic)

### 4.4 Why Diabetic Metrics Are Lower
- Only 16 diabetic samples in test (16%)
- Model trained on ~58 diabetic samples (361 train × ~16%)
- Class imbalance: model tends to predict "non-diabetic" more often
- With 500 samples, diabetic metrics improved from 0 to 0.50 vs 200 samples

---

## 5. FILE STRUCTURE AFTER RUN

```
New folder (4)/
├── diabetes_prediction_pipeline.py   # Main script
├── IGTD_Functions.py                 # IGTD algorithm
├── requirements.txt
├── README.md
├── labels.csv                        # image_file, label
├── train_data.csv                    # 361 rows, 16 cols
├── test_data.csv                     # 100 rows, 16 cols
├── results_summary.csv               # Accuracy, Precision, Recall, F1
├── cnn_diabetes_model.pt             # Saved CNN weights
├── IGTD_Images/                      # diabetic_0.png, non_diabetic_0.png, ...
└── IGTD_Results/
    └── train/                        # Optimization results, pkl files
```

---

## 6. MODEL REFERENCE

**Model**: cnn_igtd_f15_enn (Shenghao Wang et al., Table 7.5, Page 15-16)
- **Paper**: "Diabetes Risk Modeling through Tabular-to-Image Transformations and Ensemble Learning"
- **Best CNN config**: IGTD + 15 features + ENN (k=3) + 4 conv blocks
