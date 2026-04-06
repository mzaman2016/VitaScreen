"""
GRAD-CAM Pipeline: IGTD + CNN + Grad-CAM with TP, TN, FP, FN
- Uses dataset from Marzia (train_data + test_data)
- Saves 4 CSV files (TP.csv, TN.csv, FP.csv, FN.csv) - ALL samples per category
- Produces 4 images (mean IGTD + Grad-CAM per category)
- On each run: deletes old outputs and creates new images
"""

from datetime import datetime
import os
import sys
import shutil
import numpy as np
import pandas as pd
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Project root for IGTD_Functions
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    raise ImportError("Grad-CAM requires PyTorch: pip install torch")

# Output directories (inside GRAD-CAM folder)
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output')
CSV_DIR = os.path.join(SCRIPT_DIR, 'csv')
IMAGES_DIR = os.path.join(OUTPUT_DIR, 'images')


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
    # Strategy 1: Standard read
    try:
        df = pd.read_csv(path, low_memory=False)
        if len(df.columns) > 1 and len(df) > 10:
            return df
    except Exception:
        pass

    # Strategy 2: Skip bad lines (pandas >= 1.3)
    try:
        df = pd.read_csv(path, low_memory=False, on_bad_lines='skip')
        if len(df.columns) > 1 and len(df) > 10:
            return df
    except TypeError:
        try:
            df = pd.read_csv(path, low_memory=False, error_bad_lines=False)
            if len(df.columns) > 1 and len(df) > 10:
                return df
        except Exception:
            pass
    except Exception:
        pass

    # Strategy 3: Auto-detect delimiter
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

    # Try merging train_data + test_data for larger dataset
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
        # Skip HTML/corrupted files (e.g. SharePoint download page instead of CSV)
        if _is_html_or_corrupted(path):
            html_skipped.append(os.path.basename(path))
            continue
        df = _try_load_csv(path)
        if df is not None:
            print(f"Loaded {len(df)} rows from {path}")
            return df

    if html_skipped:
        print(f"Note: Skipped {len(html_skipped)} file(s) - not valid CSV (e.g. HTML or corrupted): {', '.join(html_skipped)}")
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
        os.system(f"{sys.executable} -m pip install ucimlrepo -q")
        return load_dataset(csv_path)


def get_feature_columns(df):
    """Get feature columns - CDC or generic."""
    cdc_features = [
        'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
        'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
        'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
        'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income'
    ]
    label_cols = ['Diabetes_binary', 'Diabetes_012', 'Diabetes', 'target', 'label', 'class']
    available = [c for c in cdc_features if c in df.columns]
    if len(available) >= 15:
        return available[:21]
    other = [c for c in df.columns if c not in label_cols and df[c].dtype in ['int64', 'float64']]
    return other[:21] if other else [c for c in df.columns if c not in label_cols][:21]


def create_igtd_images(data_df, feature_cols, output_dir, num_row=5, num_col=3, max_step=1500):
    from IGTD_Functions import table_to_image, min_max_transform
    n_features = len(feature_cols)
    if num_row * num_col < n_features:
        num_col = (n_features + num_row - 1) // num_row
    X = data_df[feature_cols].values
    X_norm = min_max_transform(X)
    norm_data = pd.DataFrame(X_norm, columns=feature_cols, index=data_df.index)
    scale = [num_row, num_col]
    fea_dist_method = 'Euclidean'
    image_dist_method = 'Euclidean'
    save_image_size = 3
    val_step = 150
    os.makedirs(output_dir, exist_ok=True)
    table_to_image(norm_data, scale, fea_dist_method, image_dist_method,
                   save_image_size, max_step, val_step, output_dir, 'abs')
    with open(os.path.join(output_dir, 'Results.pkl'), 'rb') as f:
        _ = pickle.load(f)
        image_data = pickle.load(f)
        samples = pickle.load(f)
    with open(os.path.join(output_dir, 'Results_Auxiliary.pkl'), 'rb') as f:
        for _ in range(5):
            pickle.load(f)
        index = pickle.load(f)
    return image_data, samples, index


def create_igtd_images_with_index(data_df, feature_cols, index, num_row, num_col, output_dir=None):
    from IGTD_Functions import generate_images_with_index
    return generate_images_with_index(data_df, feature_cols, index, num_row, num_col, output_dir)


def apply_enn(X, y, n_neighbors=3):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X, y)
    pred = knn.predict(X)
    return X[pred == y], y[pred == y]


def build_cnn_pytorch(input_shape, num_classes=1):
    class DiabetesCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.dropout = nn.Dropout(0.5)
            self.fc = nn.Linear(64, num_classes)

        def forward(self, x):
            x = self.features(x)
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.dropout(x)
            x = self.fc(x)
            return x

    return DiabetesCNN()


class GradCAM:
    """Grad-CAM for PyTorch CNN."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None

    def _save_activation(self, module, input, output):
        self.activations = output

    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        self.activations = None
        h_act = self.target_layer.register_forward_hook(self._save_activation)
        input_tensor = input_tensor.clone().detach().requires_grad_(True)
        output = self.model(input_tensor)

        if output.dim() == 1 or output.shape[1] == 1:
            target = output.sum()
        else:
            target_class = target_class if target_class is not None else output.argmax(dim=1).item()
            target = output[0, target_class]

        self.activations.retain_grad()
        self.model.zero_grad()
        target.backward()
        h_act.remove()

        if self.activations is None or not self.activations.requires_grad:
            return None
        gradients = self.activations.grad
        if gradients is None:
            return None

        weights = gradients.mean(dim=(2, 3))
        cam = (weights.unsqueeze(-1).unsqueeze(-1) * self.activations.detach()).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam.squeeze().cpu().numpy()


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
    Full pipeline: Load CSV -> IGTD -> CNN -> Grad-CAM.
    Outputs: 4 CSV files (ALL samples per category) + 4 images.
    Deletes old outputs on each run.
    """
    os.system('cls' if os.name == 'nt' else 'clear')

    print("=" * 70)
    print("GRAD-CAM Pipeline: IGTD + CNN + Grad-CAM (TP, TN, FP, FN)")
    print("=" * 70)

    # 1. Delete old outputs
    for d in [OUTPUT_DIR, CSV_DIR, IMAGES_DIR]:
        if os.path.exists(d):
            try:
                shutil.rmtree(d)
            except Exception:
                pass
    os.makedirs(CSV_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)

    igtd_dir = os.path.join(SCRIPT_DIR, 'IGTD_Results')
    if os.path.exists(igtd_dir):
        try:
            shutil.rmtree(igtd_dir)
        except Exception:
            pass

    # 2. Load data (use full dataset from Marzia - no sampling unless max_samples set)
    df = load_dataset(csv_path)
    if max_samples is not None and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
        print(f"Using {max_samples} samples")
    else:
        print(f"Using all {len(df)} samples from dataset")

    feature_cols = get_feature_columns(df)
    label_col = 'Diabetes_binary' if 'Diabetes_binary' in df.columns else df.columns[-1]
    if label_col not in df.columns:
        df['Diabetes_binary'] = (df[df.columns[-1]] > 0).astype(int)
        label_col = 'Diabetes_binary'

    n_features = min(15, len(feature_cols))
    feature_cols = feature_cols[:n_features]
    print(f"Using {n_features} features: {feature_cols[:5]}...")

    # 3. Split
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    train_df, test_df = train_test_split(df_shuffled, test_size=0.2, stratify=df_shuffled[label_col], random_state=42)
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")

    # 4. ENN
    X_train = train_df[feature_cols].values
    y_train = train_df[label_col].values
    X_train_enn, y_train_enn = apply_enn(X_train, y_train, n_neighbors=3)
    train_df_enn = pd.DataFrame(X_train_enn, columns=feature_cols)
    train_df_enn[label_col] = y_train_enn
    print(f"After ENN: {len(train_df_enn)} train samples")

    # 5. IGTD
    num_row, num_col = (5, 3) if n_features <= 15 else (5, 4)
    igtd_subset_size = min(3000, len(train_df_enn))
    if igtd_subset_size < 100:
        igtd_subset_size = len(train_df_enn)
    train_for_igtd = train_df_enn.sample(n=igtd_subset_size, random_state=42)

    train_igtd_dir = os.path.join(igtd_dir, 'train')
    print("\nCreating IGTD images...")
    train_images_sub, _, igtd_index = create_igtd_images(
        train_for_igtd, feature_cols, train_igtd_dir, num_row, num_col, max_step=800
    )
    train_images, train_samples = create_igtd_images_with_index(
        train_df_enn, feature_cols, igtd_index, num_row, num_col, output_dir=None
    )
    test_images, test_samples = create_igtd_images_with_index(
        test_df, feature_cols, igtd_index, num_row, num_col, output_dir=None
    )

    # 6. Prepare CNN inputs
    train_images = np.nan_to_num(train_images, nan=0.0)
    test_images = np.nan_to_num(test_images, nan=0.0)
    train_images = np.clip(train_images, 0, 255).astype(np.float32) / 255.0
    test_images = np.clip(test_images, 0, 255).astype(np.float32) / 255.0

    X_train_img = np.expand_dims(np.transpose(train_images, (2, 0, 1)), axis=1)
    X_test_img = np.expand_dims(np.transpose(test_images, (2, 0, 1)), axis=1)
    y_train_cnn = y_train_enn.astype(np.float32)
    y_test_cnn = test_df[label_col].values.astype(np.float32)

    # 7. Train CNN
    print("\nTraining CNN...")
    input_shape = (1, train_images.shape[0], train_images.shape[1])
    model = build_cnn_pytorch(input_shape)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0]))
    optimizer = optim.Adam(model.parameters(), lr=8e-4)

    X_t = torch.FloatTensor(X_train_img)
    y_t = torch.FloatTensor(y_train_cnn).unsqueeze(1)
    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)

    model.train()
    for epoch in range(30):
        total_loss = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/30, Loss: {total_loss/len(loader):.4f}")

    model.eval()
    with torch.no_grad():
        preds = torch.sigmoid(model(torch.FloatTensor(X_test_img))).numpy().flatten()
    y_pred = (preds >= 0.5).astype(int)

    accuracy = accuracy_score(y_test_cnn, y_pred)
    precision = precision_score(y_test_cnn, y_pred, zero_division=0)
    recall = recall_score(y_test_cnn, y_pred, zero_division=0)
    f1 = f1_score(y_test_cnn, y_pred, zero_division=0)

    print("\n" + "=" * 70)
    print("Model Performance")
    print("=" * 70)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test_cnn.astype(int), y_pred, target_names=['Non-Diabetic', 'Diabetic']))

    # 8. Get ALL samples per category (TP, TN, FP, FN)
    tp_idx, tn_idx, fp_idx, fn_idx = get_tp_tn_fp_fn_indices(y_test_cnn, y_pred)
    categories = [
        ('TP', tp_idx, 'True Positive', 'Diabetic', 'Diabetic'),
        ('TN', tn_idx, 'True Negative', 'Non-Diabetic', 'Non-Diabetic'),
        ('FP', fp_idx, 'False Positive', 'Non-Diabetic', 'Diabetic'),
        ('FN', fn_idx, 'False Negative', 'Diabetic', 'Non-Diabetic'),
    ]

    # 9. Save 4 CSV files - ALL samples per category
    print("\n" + "=" * 70)
    print("Saving CSV files (TP, TN, FP, FN) - all samples per category:")
    print("=" * 70)
    category_indices = {}
    for cat, indices, _, true_str, pred_str in categories:
        if len(indices) == 0:
            continue
        rows = []
        for idx in indices:
            row = {
                'test_index': int(idx),
                'true_label': int(y_test_cnn[idx]),
                'pred_label': int(y_pred[idx]),
                'pred_prob': float(preds[idx]),
                'true_str': true_str,
                'pred_str': pred_str,
            }
            for col in feature_cols:
                row[col] = float(test_df.iloc[idx][col])
            rows.append(row)
        cat_df = pd.DataFrame(rows)
        csv_path_out = os.path.join(CSV_DIR, f'{cat}.csv')
        cat_df.to_csv(csv_path_out, index=False)
        print(f"  Saved {csv_path_out} ({len(rows)} samples)")
        category_indices[cat] = indices

    # 10. Generate 4 images (mean IGTD + mean Grad-CAM per category)
    print("\n" + "=" * 70)
    print("Generating 4 images (IGTD + Grad-CAM per category):")
    print("=" * 70)

    target_layer = model.features[9]
    gradcam = GradCAM(model, target_layer)

    from IGTD_Functions import generate_matrix_distance_ranking
    (coord_rows, coord_cols), _ = generate_matrix_distance_ranking(num_row, num_col)
    feature_at_cell = {}
    for pos in range(len(igtd_index)):
        r, c = int(coord_rows[pos]), int(coord_cols[pos])
        feature_at_cell[(r, c)] = feature_cols[igtd_index[pos]]

    try:
        import cv2
        has_cv2 = True
    except ImportError:
        from scipy.ndimage import zoom
        has_cv2 = False

    for cat in ['TP', 'TN', 'FP', 'FN']:
        if cat not in category_indices:
            continue
        indices = category_indices[cat]
        igtd_list = []
        cam_list = []
        for idx in indices:
            img = X_test_img[idx:idx+1]
            img_t = torch.FloatTensor(img).requires_grad_(True)
            cam = gradcam.generate(img_t, target_class=None)
            if cam is None:
                continue
            orig = img[0, 0]
            if has_cv2:
                cam_resized = cv2.resize(cam.astype(np.float32), (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_LINEAR)
            else:
                zoom_factors = (orig.shape[0] / cam.shape[0], orig.shape[1] / cam.shape[1])
                cam_resized = zoom(cam.astype(np.float32), zoom_factors, order=1)
            igtd_list.append(orig)
            cam_list.append(cam_resized)

        if not igtd_list:
            continue

        agg_igtd = np.mean(igtd_list, axis=0)
        agg_cam = np.mean(cam_list, axis=0)

        true_str = 'Diabetic' if y_test_cnn[indices[0]] == 1 else 'Non-Diabetic'
        pred_str = 'Diabetic' if y_pred[indices[0]] == 1 else 'Non-Diabetic'

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        extent = [0, num_col, num_row, 0]

        im0 = axes[0].imshow(agg_igtd, cmap='gray', extent=extent, aspect='equal', vmin=0, vmax=1)
        axes[0].set_title('IGTD Image')
        cbar0 = plt.colorbar(im0, ax=axes[0], shrink=0.8)
        cbar0.set_label('Feature value (White=Low, Black=High)')
        for (r, c), fname in feature_at_cell.items():
            axes[0].text(c + 0.5, r + 0.5, fname, ha='center', va='center', fontsize=5,
                         color='yellow', weight='bold')
        axes[0].set_xlim(0, num_col)
        axes[0].set_ylim(num_row, 0)
        axes[0].axis('off')

        im1 = axes[1].imshow(agg_cam, cmap='jet', extent=extent, aspect='equal', vmin=0, vmax=1)
        axes[1].set_title('Grad-CAM Heatmap')
        cbar1 = plt.colorbar(im1, ax=axes[1], shrink=0.8)
        cbar1.set_label('Importance (Red=High, Blue=Low)')
        for (r, c), fname in feature_at_cell.items():
            axes[1].text(c + 0.5, r + 0.5, fname, ha='center', va='center', fontsize=5,
                         color='white', weight='bold')
        axes[1].set_xlim(0, num_col)
        axes[1].set_ylim(num_row, 0)
        axes[1].axis('off')

        fig.suptitle(f'{cat} (n={len(indices)}): True={true_str} | Pred={pred_str}', fontsize=10)
        plt.tight_layout()
        out_path = os.path.join(IMAGES_DIR, f'gradcam_{cat}.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved {out_path}")

    print("\n" + "=" * 70)
    print(f"Output: 4 CSV files (all samples) in {CSV_DIR}")
    print(f"        4 images in {IMAGES_DIR}")
    print("=" * 70)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'images_dir': IMAGES_DIR,
        'csv_dir': CSV_DIR,
    }


if __name__ == '__main__':
    # Use dataset from Marzia (train_data + test_data); use CDC_CSV env or path if set
    default_csv = os.path.join(PROJECT_ROOT, 'train_data.csv')
    csv_path = os.environ.get('CDC_CSV') or (default_csv if os.path.exists(default_csv) else None)
    run_pipeline(csv_path=csv_path, max_samples=None)  # None = use all samples
