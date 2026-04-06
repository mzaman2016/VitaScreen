"""
XAI RF Method: CSV files (TP, TN, FP, FN) + 4 heatmap images
- Loads data from CSV
- Saves 4 CSV files (TP.csv, TN.csv, FP.csv, FN.csv) - ALL samples per category
- Generates 4 images - one heatmap per category (mean of samples)
- On each run: deletes old outputs and creates new data
"""

import os
import sys
import shutil
import subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

CSV_DIR = os.path.join(SCRIPT_DIR, 'csv')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output')
IMAGES_DIR = os.path.join(OUTPUT_DIR, 'images')

# CDC Diabetes Health Indicators - 21 features
CDC_FEATURES = [
    'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
    'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
    'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
    'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income'
]


def _is_html_or_corrupted(filepath):
    """Check if file is HTML or corrupted (e.g. SharePoint download page)."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            peek = f.read(500)
        s = peek.strip().lower()
        return s.startswith('<!') or '<html' in s or '<doctype' in s
    except Exception:
        return True


def _try_load_csv(path):
    """Try loading CSV with multiple strategies."""
    try:
        df = pd.read_csv(path, low_memory=False)
        if len(df.columns) > 1 and len(df) > 10:
            return df
    except Exception:
        pass
    try:
        df = pd.read_csv(path, low_memory=False, on_bad_lines='skip')
        if len(df.columns) > 1 and len(df) > 10:
            return df
    except (TypeError, Exception):
        try:
            df = pd.read_csv(path, low_memory=False, error_bad_lines=False)
            if len(df.columns) > 1 and len(df) > 10:
                return df
        except Exception:
            pass
    try:
        df = pd.read_csv(path, sep=None, engine='python', low_memory=False, on_bad_lines='skip')
        if len(df.columns) > 1 and len(df) > 10:
            return df
    except Exception:
        pass
    return None


def load_dataset(csv_path=None):
    """Load dataset from CSV. Tries multiple paths and strategies."""
    default_paths = [
        os.path.join(PROJECT_ROOT, 'cdcNormalDiabeticFE1_20RFFSQ.csv'),
        os.path.join(SCRIPT_DIR, 'cdcNormalDiabeticFE1_20RFFSQ.csv'),
        r'c:\Users\parva\Downloads\cdcNormalDiabeticFE1_20RFFSQ.csv',
        os.path.join(PROJECT_ROOT, 'train_data.csv'),
    ]
    if csv_path and os.path.exists(csv_path):
        default_paths = [csv_path] + default_paths

    # Try merging train_data + test_data
    train_p = os.path.join(PROJECT_ROOT, 'train_data.csv')
    test_p = os.path.join(PROJECT_ROOT, 'test_data.csv')
    if os.path.exists(train_p) and os.path.exists(test_p) and not _is_html_or_corrupted(train_p) and not _is_html_or_corrupted(test_p):
        t1 = _try_load_csv(train_p)
        t2 = _try_load_csv(test_p)
        if t1 is not None and t2 is not None and list(t1.columns) == list(t2.columns):
            df = pd.concat([t1, t2], ignore_index=True)
            print(f"Loaded {len(df)} rows from train_data.csv + test_data.csv")
            return df

    html_skipped = []
    for path in default_paths:
        if not os.path.exists(path):
            continue
        if _is_html_or_corrupted(path):
            html_skipped.append(os.path.basename(path))
            continue
        df = _try_load_csv(path)
        if df is not None:
            print(f"Loaded {len(df)} rows from {path}")
            return df

    if html_skipped:
        print(f"Note: Skipped {len(html_skipped)} file(s) - not valid CSV: {', '.join(html_skipped)}")
    print("Loading from UCI repository (fallback)...")
    try:
        from ucimlrepo import fetch_ucirepo
        cdc = fetch_ucirepo(id=891)
        X = cdc.data.features
        y = cdc.data.targets
        df = pd.concat([X, y], axis=1)
        if 'Diabetes_012' in df.columns:
            df['Diabetes_binary'] = (df['Diabetes_012'] > 0).astype(int)
            df = df.drop('Diabetes_012', axis=1)
        return df
    except ImportError:
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'ucimlrepo', '-q'], check=False)
        return load_dataset(csv_path)


def get_feature_columns(df):
    """Get feature columns."""
    available = [c for c in CDC_FEATURES if c in df.columns]
    if len(available) >= 15:
        return available[:21]
    label_cols = ['Diabetes_binary', 'Diabetes_012', 'Diabetes', 'target', 'label', 'class']
    numeric = [c for c in df.columns if c not in label_cols and df[c].dtype in ['int64', 'float64']]
    return numeric[:21]


def get_label_column(df):
    """Get the label column name."""
    for c in ['Diabetes_binary', 'Diabetes_012', 'Diabetes', 'target', 'label', 'class']:
        if c in df.columns:
            if c == 'Diabetes_012':
                df['Diabetes_binary'] = (df['Diabetes_012'] > 0).astype(int)
                return 'Diabetes_binary'
            return c
    return df.columns[-1]


def get_tp_tn_fp_fn_indices(y_true, y_pred):
    """Get indices for TP, TN, FP, FN."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    tp = np.where((y_true == 1) & (y_pred == 1))[0]
    tn = np.where((y_true == 0) & (y_pred == 0))[0]
    fp = np.where((y_true == 0) & (y_pred == 1))[0]
    fn = np.where((y_true == 1) & (y_pred == 0))[0]
    return tp, tn, fp, fn


def run_pipeline(csv_path=None, max_samples=None):
    """
    XAI RF pipeline:
    1. Load data, train RF
    2. Save 4 CSV files (TP, TN, FP, FN) - ALL samples per category
    3. Generate 4 images (mean per category)
    4. On each run: delete old data and create new
    """
    os.system('cls' if os.name == 'nt' else 'clear')

    print("=" * 70)
    print("XAI RF Method: 4 CSV files + 4 Images (TP, TN, FP, FN)")
    print("=" * 70)

    # 1. Delete old outputs
    for d in [CSV_DIR, OUTPUT_DIR, IMAGES_DIR]:
        if os.path.exists(d):
            try:
                shutil.rmtree(d)
            except Exception:
                pass
    os.makedirs(CSV_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)

    # 2. Load data (use full dataset from Marzia - no sampling unless max_samples set)
    df = load_dataset(csv_path)
    if max_samples is not None and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
        print(f"Using {max_samples} samples")
    else:
        print(f"Using all {len(df)} samples from dataset")

    feature_cols = get_feature_columns(df)
    label_col = get_label_column(df)
    if label_col not in df.columns:
        df['Diabetes_binary'] = (df[df.columns[-1]] > 0).astype(int)
        label_col = 'Diabetes_binary'

    print(f"Using {len(feature_cols)} features: {feature_cols[:5]}...")

    # 3. Prepare data
    X = df[feature_cols].fillna(0).values
    y = df[label_col].values
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    train_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=0.2, stratify=y, random_state=42
    )
    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    test_df = df.iloc[test_idx].reset_index(drop=True)

    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # 4. Train Random Forest
    print("\nTraining Random Forest...")
    model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    pred_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("\n" + "=" * 70)
    print("Model Performance")
    print("=" * 70)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Non-Diabetic', 'Diabetic']))

    # 5. Get ALL samples per category (TP, TN, FP, FN)
    tp_idx, tn_idx, fp_idx, fn_idx = get_tp_tn_fp_fn_indices(y_test, y_pred)
    categories = [
        ('TP', tp_idx, 'True Positive', 'Diabetic', 'Diabetic'),
        ('TN', tn_idx, 'True Negative', 'Non-Diabetic', 'Non-Diabetic'),
        ('FP', fp_idx, 'False Positive', 'Non-Diabetic', 'Diabetic'),
        ('FN', fn_idx, 'False Negative', 'Diabetic', 'Non-Diabetic'),
    ]

    # 6. Save 4 CSV files - ALL samples per category
    print("\n" + "=" * 70)
    print("Saving CSV files (TP, TN, FP, FN) - all samples per category:")
    print("=" * 70)
    category_data = {}
    for cat, indices, _, true_str, pred_str in categories:
        if len(indices) == 0:
            continue
        rows = []
        X_cat = []
        for idx in indices:
            row = {
                'test_index': int(idx),
                'true_label': int(y_test[idx]),
                'pred_label': int(y_pred[idx]),
                'pred_prob': float(pred_proba[idx]),
                'true_str': true_str,
                'pred_str': pred_str,
            }
            for col in feature_cols:
                row[col] = float(test_df.iloc[idx][col])
            rows.append(row)
            X_cat.append(X_test[idx])
        cat_df = pd.DataFrame(rows)
        csv_path_out = os.path.join(CSV_DIR, f'{cat}.csv')
        cat_df.to_csv(csv_path_out, index=False)
        print(f"  Saved {csv_path_out} ({len(rows)} samples)")
        category_data[cat] = (np.array(X_cat), true_str, pred_str)

    # 7. Generate 4 images - one heatmap per category (mean of samples)
    print("\n" + "=" * 70)
    print("Generating 4 heatmaps (mean per category):")
    print("=" * 70)
    for cat in ['TP', 'TN', 'FP', 'FN']:
        if cat not in category_data:
            continue
        X_cat, true_str, pred_str = category_data[cat]
        agg_vals = np.mean(X_cat, axis=0).reshape(1, -1)
        fig, ax = plt.subplots(figsize=(max(12, len(feature_cols) * 0.4), 3))
        sns.heatmap(agg_vals, xticklabels=feature_cols, yticklabels=['Mean Feature Value'],
                    cmap='YlOrRd', annot=False, fmt='.2f', vmin=0, vmax=1,
                    cbar_kws={'label': 'Normalized Value'})
        plt.xticks(rotation=45, ha='right')
        plt.title(f'{cat} (n={len(X_cat)}): True={true_str} | Pred={pred_str}')
        plt.tight_layout()
        out_path = os.path.join(IMAGES_DIR, f'rf_{cat}.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved {out_path}")

    n_images = len(category_data)
    print("\n" + "=" * 70)
    print(f"Output: 4 CSV files (all samples) in {CSV_DIR}")
    print(f"        4 images in {IMAGES_DIR}")
    print("=" * 70)

    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}


if __name__ == '__main__':
    # Use dataset from Marzia (train_data + test_data); use CDC_CSV env or path if set
    default_csv = os.path.join(PROJECT_ROOT, 'train_data.csv')
    csv_path = os.environ.get('CDC_CSV') or (default_csv if os.path.exists(default_csv) else None)
    run_pipeline(csv_path=csv_path, max_samples=None)  # None = use all samples
