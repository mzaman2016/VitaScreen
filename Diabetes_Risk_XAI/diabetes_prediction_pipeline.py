"""
Diabetes Risk Prediction Pipeline
Based on Shenghao Wang et al. - Diabetes Risk Modeling through Tabular-to-Image Transformations
Uses best model: cnn_igtd_f15_enn (CNN with IGTD, 15 features, ENN)
"""

from datetime import datetime
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
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# Try PyTorch first, fallback to TensorFlow
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    BACKEND = 'pytorch'
except ImportError:
    try:
        import tensorflow as tf
        from tensorflow import keras # type: ignore
        BACKEND = 'tensorflow'
    except ImportError:
        raise ImportError("Please install PyTorch or TensorFlow: pip install torch OR pip install tensorflow")


def load_dataset(csv_path=None):
    """
    Load CDC Diabetes dataset. Tries local CSV first, then UCI repository.
    """
    # Try local file first
    if csv_path and os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path, nrows=5)
            # Check if it's valid CSV (not HTML)
            if len(df.columns) > 1 and not any('html' in str(v).lower() for v in df.iloc[0]):
                df = pd.read_csv(csv_path, low_memory=False)
                print(f"Loaded {len(df)} rows from {csv_path}")
                return df
        except Exception as e:
            print(f"Could not load {csv_path}: {e}")

    # Fallback: Load from UCI repository
    print("Loading CDC Diabetes dataset from UCI repository...")
    try:
        from ucimlrepo import fetch_ucirepo
        cdc_diabetes = fetch_ucirepo(id=891)
        X = cdc_diabetes.data.features
        y = cdc_diabetes.data.targets
        df = pd.concat([X, y], axis=1)
        # Convert to binary: 0=no diabetes, 1=diabetes or prediabetes
        if 'Diabetes_012' in df.columns:
            df['Diabetes_binary'] = (df['Diabetes_012'] > 0).astype(int)
            df = df.drop('Diabetes_012', axis=1)
        print(f"Loaded {len(df)} rows from UCI")
        return df
    except ImportError:
        print("Installing ucimlrepo...")
        os.system(f"{sys.executable} -m pip install ucimlrepo -q")
        return load_dataset(csv_path)

    # Create sample data if all else fails
    print("Creating sample data for demonstration...")
    np.random.seed(42)
    n_samples = 5000
    n_features = 21
    X = np.random.rand(n_samples, n_features)
    y = (np.random.rand(n_samples) > 0.84).astype(int)  # ~16% positive
    feature_names = [f'Feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['Diabetes_binary'] = y
    return df


def get_cdc_feature_columns(df):
    """Get standard CDC diabetes feature names."""
    cdc_features = [
        'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
        'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
        'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
        'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income'
    ]
    # Find matching columns
    available = [c for c in cdc_features if c in df.columns]
    if len(available) >= 15:
        return available[:21]  # Use up to 21
    return [c for c in df.columns if c != 'Diabetes_binary' and c != 'Diabetes_012'][:21]


def create_igtd_images(data_df, feature_cols, output_dir, num_row=5, num_col=3, max_step=1500):
    """
    Create IGTD images from tabular data. Runs full IGTD optimization.
    Returns (image_data, samples, index) - index can be reused for test data.
    """
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

    # Load results and index
    with open(os.path.join(output_dir, 'Results.pkl'), 'rb') as f:
        _ = pickle.load(f)
        image_data = pickle.load(f)
        samples = pickle.load(f)

    with open(os.path.join(output_dir, 'Results_Auxiliary.pkl'), 'rb') as f:
        for _ in range(5):  # Skip to index (6th item)
            pickle.load(f)
        index = pickle.load(f)

    return image_data, samples, index


def create_igtd_images_with_index(data_df, feature_cols, index, num_row, num_col, output_dir=None):
    """Generate IGTD images using pre-computed index (for test data)."""
    from IGTD_Functions import generate_images_with_index
    return generate_images_with_index(data_df, feature_cols, index, num_row, num_col, output_dir)


def apply_enn(X, y, n_neighbors=3):
    """Edited Nearest Neighbors - remove majority samples misclassified by KNN."""
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X, y)
    pred = knn.predict(X)
    mask = (pred == y)
    return X[mask], y[mask]


def build_cnn_pytorch(input_shape, num_classes=1):
    """
    CNN architecture from Shenghao's paper (Figure 6.2):
    4 conv blocks (64 filters, 3x3), BatchNorm, ReLU
    Global Average Pooling, Dropout 50%, FC output
    """
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


def build_cnn_tensorflow(input_shape, num_classes=1):
    """CNN architecture for TensorFlow."""
    model = keras.Sequential([
        keras.layers.Conv2D(64, 3, padding='same', input_shape=input_shape),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.Conv2D(64, 3, padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.Conv2D(64, 3, padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.Conv2D(64, 3, padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(num_classes, activation='sigmoid')
    ])
    return model


def safe_rmtree(directory):
    """Safely remove a directory, handling Windows file locks with retry."""
    if not os.path.exists(directory):
        return
    for attempt in range(3):
        try:
            shutil.rmtree(directory)
            return
        except (OSError, PermissionError) as e:
            if attempt < 2:
                time.sleep(0.5)
            else:
                # Clear files individually if directory removal fails
                for root, dirs, files in os.walk(directory, topdown=False):
                    for name in files:
                        try:
                            os.remove(os.path.join(root, name))
                        except:
                            pass
                return


def run_pipeline(csv_path=None, max_samples=500):
    """Execute full pipeline: IGTD -> Labels -> Split -> Train -> Test -> Report."""
    print("=" * 60)
    print("Diabetes Risk Prediction Pipeline")
    print("Model: cnn_igtd_f15_enn (Shenghao Wang et al.)")
    print("=" * 60)

    # 0. Clean old outputs (delete old images - everything updated on each run)
    igtd_dir = os.path.join(SCRIPT_DIR, 'IGTD_Results')
    images_dir = os.path.join(SCRIPT_DIR, 'IGTD_Images')
    if os.path.exists(igtd_dir):
        safe_rmtree(igtd_dir)
        print("Cleaned old IGTD_Results folder")
    if os.path.exists(images_dir):
        safe_rmtree(images_dir)
        print("Cleaned old IGTD_Images folder")

    # 1. Load data
    df = load_dataset(csv_path)
    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
        print(f"Using {max_samples} samples")
    feature_cols = get_cdc_feature_columns(df)
    label_col = 'Diabetes_binary' if 'Diabetes_binary' in df.columns else df.columns[-1]

    if label_col not in df.columns:
        df['Diabetes_binary'] = (df[df.columns[-1]] > 0).astype(int)
        label_col = 'Diabetes_binary'

    # Use top 15 features (as per paper's best model)
    n_features = min(15, len(feature_cols))
    feature_cols = feature_cols[:n_features]
    print(f"Using {n_features} features: {feature_cols[:5]}...")

    # 2. Shuffle and split 80:20 (before IGTD to avoid data leakage)
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    train_df, test_df = train_test_split(df_shuffled, test_size=0.2, stratify=df_shuffled[label_col], random_state=42)
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")

    # 3. Apply ENN on training set (k=3 as per Shenghao's paper - cnn_igtd_f15_enn)
    X_train = train_df[feature_cols].values
    y_train = train_df[label_col].values
    X_train_enn, y_train_enn = apply_enn(X_train, y_train, n_neighbors=3)
    train_df_enn = pd.DataFrame(X_train_enn, columns=feature_cols)
    train_df_enn[label_col] = y_train_enn
    print(f"After ENN (k=3): Train {len(train_df_enn)} samples")

    # Use subset for IGTD optimization (faster - feature arrangement is stable)
    igtd_subset_size = min(3000, len(train_df_enn))
    if igtd_subset_size < 100:
        igtd_subset_size = len(train_df_enn)
    train_for_igtd = train_df_enn.sample(n=igtd_subset_size, random_state=42)

    # 4. Create IGTD images (run IGTD once on train subset, reuse index for all)
    num_row, num_col = (5, 3) if n_features <= 15 else (5, 4)
    igtd_dir = os.path.join(SCRIPT_DIR, 'IGTD_Results')
    train_igtd_dir = os.path.join(igtd_dir, 'train')
    print("\nCreating IGTD images (this may take several minutes)...")

    # Run IGTD on training subset (max_step=800 for faster execution)
    train_images_sub, _, igtd_index = create_igtd_images(
        train_for_igtd, feature_cols, train_igtd_dir, num_row, num_col, max_step=800
    )
    # Generate full train images with same index
    train_images, train_samples = create_igtd_images_with_index(
        train_df_enn, feature_cols, igtd_index, num_row, num_col, output_dir=None
    )
    # Generate test images using same feature arrangement (no IGTD optimization)
    test_images, test_samples = create_igtd_images_with_index(
        test_df, feature_cols, igtd_index, num_row, num_col, output_dir=None
    )

    # 4b. Save all images with diabetic/non_diabetic naming (200 total)
    from IGTD_Functions import save_images_diabetic_naming
    all_images = np.concatenate([train_images, test_images], axis=2)
    all_labels = np.concatenate([y_train_enn, test_df[label_col].values])
    image_labels = save_images_diabetic_naming(all_images, all_labels, images_dir)
    print(f"Saved {len(all_labels)} images to IGTD_Images/ (diabetic_X.png, non_diabetic_X.png)")

    # Remove old-named images from IGTD_Results (we use IGTD_Images with diabetic/non_diabetic names)
    igtd_data_dir = os.path.join(train_igtd_dir, 'data')
    if os.path.exists(igtd_data_dir):
        shutil.rmtree(igtd_data_dir)

    # 5. Create labels CSV (with image filenames - updated each run)
    labels_df = pd.DataFrame(image_labels)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    labels_path = os.path.join(SCRIPT_DIR, f'labels_{timestamp}.csv')
    labels_df.to_csv(labels_path, index=False)
    print(f"Saved labels to {labels_path}")

    # 6. Save train/test splits
    train_df_enn.to_csv(os.path.join(SCRIPT_DIR, f'train_data_{timestamp}.csv'), index=False)
    test_df[feature_cols + [label_col]].to_csv(os.path.join(SCRIPT_DIR, f'test_data_{timestamp}.csv'), index=False)
    print(f"Saved train_data_{timestamp}.csv and test_data_{timestamp}.csv")

    # 7. Prepare image arrays for CNN
    # Handle NaN in images
    train_images = np.nan_to_num(train_images, nan=0.0)
    test_images = np.nan_to_num(test_images, nan=0.0)
    train_images = np.clip(train_images, 0, 255).astype(np.float32) / 255.0
    test_images = np.clip(test_images, 0, 255).astype(np.float32) / 255.0

    # Add channel dimension: (H, W, N) -> (N, 1, H, W) for PyTorch or (N, H, W, 1) for TF
    if BACKEND == 'pytorch':
        X_train_img = np.expand_dims(np.transpose(train_images, (2, 0, 1)), axis=1)
        X_test_img = np.expand_dims(np.transpose(test_images, (2, 0, 1)), axis=1)
    else:
        X_train_img = np.transpose(train_images, (2, 0, 1))
        X_train_img = np.expand_dims(X_train_img, axis=-1)
        X_test_img = np.transpose(test_images, (2, 0, 1))
        X_test_img = np.expand_dims(X_test_img, axis=-1)

    y_train_cnn = y_train_enn.astype(np.float32)
    y_test_cnn = test_df[label_col].values.astype(np.float32)

    # 8. Train CNN
    print("\nTraining CNN model...")
    input_shape = (1, train_images.shape[0], train_images.shape[1]) if BACKEND == 'pytorch' else (train_images.shape[0], train_images.shape[1], 1)

    if BACKEND == 'pytorch':
        model = build_cnn_pytorch(input_shape)
        criterion = nn.BCEWithLogitsLoss()
        # Class weight for imbalance
        pos_weight = torch.tensor([3.0])  # 1:3 as per paper
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
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
            X_test_t = torch.FloatTensor(X_test_img)
            preds = torch.sigmoid(model(X_test_t)).numpy().flatten()
        y_pred = (preds >= 0.5).astype(int)

        # Save model
        model_path = os.path.join(SCRIPT_DIR, 'cnn_diabetes_model.pt')
        torch.save(model.state_dict(), model_path)
    else:
        model = build_cnn_tensorflow(input_shape)
        model.compile(optimizer=keras.optimizers.Adam(8e-4), loss='binary_crossentropy', metrics=['accuracy'])
        class_weight = {0: 1, 1: 3}
        model.fit(X_train_img, y_train_cnn, epochs=30, batch_size=64, class_weight=class_weight, verbose=1)
        y_pred = (model.predict(X_test_img).flatten() >= 0.5).astype(int)
        model.save(os.path.join(SCRIPT_DIR, 'cnn_diabetes_model.keras'))

    # 9. Evaluate and report
    print("\n" + "=" * 60)
    print("RESULTS - Test Dataset Performance")
    print("=" * 60)

    accuracy = accuracy_score(y_test_cnn, y_pred)
    precision = precision_score(y_test_cnn, y_pred, zero_division=0)
    recall = recall_score(y_test_cnn, y_pred, zero_division=0)
    f1 = f1_score(y_test_cnn, y_pred, zero_division=0)

    print(f"\nAccuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test_cnn.astype(int), y_pred, target_names=['Non-Diabetic', 'Diabetic']))

    # Save results
    results = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    }
    pd.DataFrame([results]).to_csv(os.path.join(SCRIPT_DIR, 'results_summary.csv'), index=False)
    print(f"\nResults saved to results_summary.csv")

    return results


if __name__ == '__main__':
    csv_path = r"c:\Users\parva\Downloads\cdcNormalDiabeticFE1_20RFFSQ.csv"
    # 500 samples; set max_samples=None for full dataset
    run_pipeline(csv_path, max_samples=500)
