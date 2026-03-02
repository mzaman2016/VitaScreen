"""
IGTD Pipeline - Old CDC Dataset (No Grad-CAM)
Uses CDC Diabetes Health Indicators dataset with 500 samples.
Pipeline: IGTD -> ENN -> CNN (cnn_igtd_f15_enn)
On each run: old images are removed and new ones are generated.
"""

import os
import sys
import time
import shutil
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# IGTD folder is the script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    BACKEND = 'pytorch'
except ImportError:
    raise ImportError("Please install PyTorch: pip install torch")


def load_dataset_old():
    """
    Load OLD CDC Diabetes dataset from UCI (500 samples).
    CDC Diabetes Health Indicators dataset.
    """
    print("Loading CDC Diabetes dataset from UCI repository (old dataset)...")
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
        print("Installing ucimlrepo...")
        os.system(f"{sys.executable} -m pip install ucimlrepo -q")
        return load_dataset_old()


def get_feature_columns(df):
    """Get CDC feature columns (15 features)."""
    cdc_features = [
        'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
        'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
        'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
        'MentHlth'
    ]
    available = [c for c in cdc_features if c in df.columns]
    return available[:15]


def create_igtd_images(data_df, feature_cols, output_dir, num_row=5, num_col=3, max_step=1500):
    """Create IGTD images from tabular data."""
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
    """Generate IGTD images using pre-computed index."""
    from IGTD_Functions import generate_images_with_index
    return generate_images_with_index(data_df, feature_cols, index, num_row, num_col, output_dir)


def apply_enn(X, y, n_neighbors=3):
    """Edited Nearest Neighbors - remove noisy samples."""
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


def safe_rmtree(directory):
    """Safely remove a directory (clears old images before each run)."""
    if not os.path.exists(directory):
        return
    for attempt in range(3):
        try:
            shutil.rmtree(directory)
            return
        except (OSError, PermissionError):
            if attempt < 2:
                time.sleep(0.5)
            else:
                for root, dirs, files in os.walk(directory, topdown=False):
                    for name in files:
                        try:
                            os.remove(os.path.join(root, name))
                        except Exception:
                            pass
                return


def run_pipeline(max_samples=500):
    """
    Execute full pipeline: Load old dataset -> IGTD -> ENN -> CNN.
    On each run: clears old IGTD_Results and IGTD_Images, then generates new ones.
    """
    # Clear terminal for fresh output
    os.system('cls' if os.name == 'nt' else 'clear')

    print("=" * 60)
    print("IGTD Pipeline - Old CDC Dataset (No Grad-CAM)")
    print("Dataset: CDC Diabetes Health Indicators, 500 samples")
    print("=" * 60)

    igtd_dir = os.path.join(SCRIPT_DIR, 'IGTD_Results')
    images_dir = os.path.join(SCRIPT_DIR, 'IGTD_Images')

    # Remove old outputs so each run produces fresh images
    for d in [igtd_dir, images_dir]:
        if os.path.exists(d):
            safe_rmtree(d)
            print(f"Removed old {os.path.basename(d)}/")
    os.makedirs(images_dir, exist_ok=True)

    # 1. Load old dataset (UCI CDC)
    df = load_dataset_old()
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

    # 2. Split 80:20
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    train_df, test_df = train_test_split(
        df_shuffled, test_size=0.2, stratify=df_shuffled[label_col], random_state=42
    )
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")

    # 3. ENN (k=3)
    X_train = train_df[feature_cols].values
    y_train = train_df[label_col].values
    X_train_enn, y_train_enn = apply_enn(X_train, y_train, n_neighbors=3)
    train_df_enn = pd.DataFrame(X_train_enn, columns=feature_cols)
    train_df_enn[label_col] = y_train_enn
    print(f"After ENN (k=3): {len(train_df_enn)} train samples")

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
    print(f"Saved images to IGTD_Images/ (diabetic_X.png, non_diabetic_X.png)")

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
    print(f"\nResults saved to results_summary.csv")

    model_path = os.path.join(SCRIPT_DIR, 'cnn_diabetes_model.pt')
    torch.save(model.state_dict(), model_path)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'igtd_dir': igtd_dir,
        'images_dir': images_dir,
        'results_path': results_path
    }


if __name__ == '__main__':
    run_pipeline(max_samples=500)
