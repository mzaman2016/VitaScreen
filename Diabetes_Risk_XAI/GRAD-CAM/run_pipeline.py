"""
GRAD-CAM Pipeline - New Dataset with Explainable AI
Uses new CDC dataset (cdcNormalDiabeticFE1_20RFFSQ.csv or UCI fallback).
Pipeline: IGTD -> ENN -> CNN -> Grad-CAM
On each run: old images and data are removed and new ones are generated.
"""

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

# GRAD-CAM folder is the script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    raise ImportError("Grad-CAM requires PyTorch: pip install torch")


def load_dataset_new(csv_path=None):
    """
    Load new dataset. Tries cdcNormalDiabeticFE1_20RFFSQ.csv first, then UCI fallback.
    """
    if csv_path and os.path.exists(csv_path):
        try:
            with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
                first_line = f.readline()
            if first_line.strip().lower().startswith('<!'):
                print(f"File appears to be HTML (SharePoint/corrupted). Using UCI fallback.")
            else:
                df = pd.read_csv(csv_path, nrows=5)
                if len(df.columns) > 1 and not any('html' in str(v).lower() for v in df.iloc[0].astype(str)):
                    df = pd.read_csv(csv_path, low_memory=False)
                    print(f"Loaded {len(df)} rows from {csv_path}")
                    return df
        except Exception as e:
            print(f"Could not load {csv_path}: {e}")

    print("Loading CDC Diabetes dataset from UCI repository...")
    try:
        from ucimlrepo import fetch_ucirepo
        cdc_diabetes = fetch_ucirepo(id=891)
        X = cdc_diabetes.data.features
        y = cdc_diabetes.data.targets
        df = pd.concat([X, y], axis=1)
        if 'Diabetes_012' in df.columns:
            df['Diabetes_binary'] = (df['Diabetes_012'] > 0).astype(int)
            df = df.drop('Diabetes_012', axis=1)
        print(f"Loaded {len(df)} rows from UCI")
        return df
    except ImportError:
        os.system(f"{sys.executable} -m pip install ucimlrepo -q")
        return load_dataset_new(csv_path)


def get_feature_columns(df):
    """Get feature columns - supports CDC datasets."""
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
    """CNN architecture - cnn_igtd_f15_enn (Shenghao Wang et al.)."""
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
    """Grad-CAM for PyTorch CNN - explains which image regions influenced the prediction."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None

    def _save_activation(self, module, input, output):
        self.activations = output

    def generate(self, input_tensor, target_class=None):
        """Generate Grad-CAM heatmap for input image."""
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


def safe_rmtree(directory):
    """Safely remove a directory (clears old outputs before each run)."""
    if not os.path.exists(directory):
        return
    for attempt in range(3):
        try:
            shutil.rmtree(directory)
            return
        except (OSError, PermissionError):
            if attempt < 2:
                import time
                time.sleep(0.5)
            else:
                for root, dirs, files in os.walk(directory, topdown=False):
                    for name in files:
                        try:
                            os.remove(os.path.join(root, name))
                        except Exception:
                            pass
                return


def run_pipeline(csv_path=None, max_samples=500, num_gradcam_samples=8):
    """
    Full pipeline: Load new data -> IGTD -> CNN -> Grad-CAM.
    On each run: old images and data are removed, new ones are generated.
    """
    os.system('cls' if os.name == 'nt' else 'clear')

    print("=" * 60)
    print("GRAD-CAM Pipeline - New Dataset with Explainable AI")
    print("Dataset: cdcNormalDiabeticFE1_20RFFSQ.csv (or UCI fallback)")
    print("=" * 60)

    igtd_dir = os.path.join(SCRIPT_DIR, 'IGTD_Results')
    images_dir = os.path.join(SCRIPT_DIR, 'IGTD_Images')
    gradcam_dir = os.path.join(SCRIPT_DIR, 'GradCAM_Output')

    # Remove old outputs so each run produces fresh images and data
    for d in [igtd_dir, images_dir, gradcam_dir]:
        if os.path.exists(d):
            safe_rmtree(d)
            print(f"Removed old {os.path.basename(d)}/")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(gradcam_dir, exist_ok=True)

    # 1. Load data
    default_csv = os.path.join(os.path.expanduser('~'), 'Downloads', 'cdcNormalDiabeticFE1_20RFFSQ.csv')
    csv_path = csv_path or default_csv
    df = load_dataset_new(csv_path)
    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
        print(f"Using {max_samples} samples")

    feature_cols = get_feature_columns(df)
    label_col = 'Diabetes_binary' if 'Diabetes_binary' in df.columns else df.columns[-1]
    if label_col not in df.columns:
        df['Diabetes_binary'] = (df[df.columns[-1]] > 0).astype(int)
        label_col = 'Diabetes_binary'

    n_features = min(15, len(feature_cols))
    feature_cols = feature_cols[:n_features]
    print(f"Using {n_features} features: {feature_cols[:5]}...")

    # 2. Split
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    train_df, test_df = train_test_split(df_shuffled, test_size=0.2, stratify=df_shuffled[label_col], random_state=42)
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")

    # 3. ENN
    X_train = train_df[feature_cols].values
    y_train = train_df[label_col].values
    X_train_enn, y_train_enn = apply_enn(X_train, y_train, n_neighbors=3)
    train_df_enn = pd.DataFrame(X_train_enn, columns=feature_cols)
    train_df_enn[label_col] = y_train_enn
    print(f"After ENN: {len(train_df_enn)} train samples")

    # 4. IGTD
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

    from IGTD_Functions import save_images_diabetic_naming
    all_images = np.concatenate([train_images, test_images], axis=2)
    all_labels = np.concatenate([y_train_enn, test_df[label_col].values])
    save_images_diabetic_naming(all_images, all_labels, images_dir)
    print(f"Saved images to IGTD_Images/")

    # 5. Prepare CNN inputs
    train_images = np.nan_to_num(train_images, nan=0.0)
    test_images = np.nan_to_num(test_images, nan=0.0)
    train_images = np.clip(train_images, 0, 255).astype(np.float32) / 255.0
    test_images = np.clip(test_images, 0, 255).astype(np.float32) / 255.0

    X_train_img = np.expand_dims(np.transpose(train_images, (2, 0, 1)), axis=1)
    X_test_img = np.expand_dims(np.transpose(test_images, (2, 0, 1)), axis=1)
    y_train_cnn = y_train_enn.astype(np.float32)
    y_test_cnn = test_df[label_col].values.astype(np.float32)

    # 6. Train CNN
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

    print("\n" + "=" * 60)
    print("RESULTS - Test Dataset Performance")
    print("=" * 60)
    print(f"\nAccuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test_cnn.astype(int), y_pred, target_names=['Non-Diabetic', 'Diabetic']))

    results_path = os.path.join(SCRIPT_DIR, 'results_summary.csv')
    pd.DataFrame([{
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    }]).to_csv(results_path, index=False)
    print(f"Results saved to results_summary.csv")

    model_path = os.path.join(SCRIPT_DIR, 'cnn_diabetes_model.pt')
    torch.save(model.state_dict(), model_path)

    # 7. Grad-CAM
    print("\n" + "=" * 60)
    print("Grad-CAM Explainability")
    print("=" * 60)

    target_layer = model.features[9]
    gradcam = GradCAM(model, target_layer)

    n_show = min(num_gradcam_samples, len(X_test_img))
    indices = np.arange(len(X_test_img))
    np.random.seed(42)
    np.random.shuffle(indices)
    indices = indices[:n_show]

    from IGTD_Functions import generate_matrix_distance_ranking
    (coord_rows, coord_cols), _ = generate_matrix_distance_ranking(num_row, num_col)
    feature_at_cell = {}
    for pos in range(len(igtd_index)):
        r, c = int(coord_rows[pos]), int(coord_cols[pos])
        feature_at_cell[(r, c)] = feature_cols[igtd_index[pos]]

    for idx, i in enumerate(indices):
        img = X_test_img[i:i+1]
        img_t = torch.FloatTensor(img).requires_grad_(True)
        cam = gradcam.generate(img_t, target_class=None)

        if cam is None:
            continue

        orig = img[0, 0]
        try:
            import cv2
            cam_resized = cv2.resize(cam.astype(np.float32), (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_LINEAR)
        except ImportError:
            from scipy.ndimage import zoom
            zoom_factors = (orig.shape[0] / cam.shape[0], orig.shape[1] / cam.shape[1])
            cam_resized = zoom(cam.astype(np.float32), zoom_factors, order=1)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        extent = [0, num_col, num_row, 0]

        im0 = axes[0].imshow(orig, cmap='gray', extent=extent, aspect='equal', vmin=0, vmax=1)
        axes[0].set_title('IGTD Image (with feature names)')
        cbar0 = plt.colorbar(im0, ax=axes[0], shrink=0.8)
        cbar0.set_label('Feature value (White=Low, Black=High)')
        for (r, c), fname in feature_at_cell.items():
            axes[0].text(c + 0.5, r + 0.5, fname, ha='center', va='center', fontsize=5,
                         color='yellow', weight='bold')
        axes[0].set_xlim(0, num_col)
        axes[0].set_ylim(num_row, 0)
        axes[0].axis('off')

        im1 = axes[1].imshow(cam_resized, cmap='jet', extent=extent, aspect='equal', vmin=0, vmax=1)
        axes[1].set_title('Grad-CAM Heatmap')
        cbar1 = plt.colorbar(im1, ax=axes[1], shrink=0.8)
        cbar1.set_label('Importance (Red=High, Blue=Low)')
        for (r, c), fname in feature_at_cell.items():
            axes[1].text(c + 0.5, r + 0.5, fname, ha='center', va='center', fontsize=5,
                         color='white', weight='bold')
        axes[1].set_xlim(0, num_col)
        axes[1].set_ylim(num_row, 0)
        axes[1].axis('off')

        true_label = int(y_test_cnn[i])
        pred_label = int(y_pred[i])
        pred_prob = float(preds[i])
        fig.suptitle(f'Sample {idx+1}: True={true_label} ({"Diabetic" if true_label else "Non-Diabetic"}) | '
                     f'Pred={pred_label} ({pred_prob:.2f})', fontsize=10)
        plt.tight_layout()
        out_path = os.path.join(gradcam_dir, f'gradcam_sample_{idx+1}.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved {out_path}")

    print(f"\nGrad-CAM outputs saved to GradCAM_Output/")
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'gradcam_dir': gradcam_dir,
        'results_path': results_path
    }


if __name__ == '__main__':
    run_pipeline(csv_path=None, max_samples=500, num_gradcam_samples=8)
