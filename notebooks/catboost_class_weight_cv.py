# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold, train_test_split
from utils import compute_metrics

# %%
# Load the dataset
df = pd.read_csv("../data/cdcNormalDiabetic.csv")
print(df.shape)
print(df.columns)

# %%
df["Label"].value_counts()

# %%
feature_cols = [col for col in df.columns if col != "Label"]
print(f"Number of features: {len(feature_cols)}")

# %%
X = df[feature_cols]
y = df["Label"]

X.shape, y.shape

# %%
# Split the data into training and test sets
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train_val.shape, y_train_val.shape, X_test.shape, y_test.shape

# %% Training loop
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
random_state = 42
early_stopping_rounds = 50
best_score = 0
best_cv_metrics = None
for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_val, y_train_val), 1):
    print(f"\nFold {fold}")
    X_train, X_val = X_train_val.iloc[train_idx], X_train_val.iloc[val_idx]
    y_train, y_val = y_train_val.iloc[train_idx], y_train_val.iloc[val_idx]

    train_pool = Pool(data=X_train, label=y_train)
    val_pool = Pool(data=X_val, label=y_val)

    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.01,
        max_depth=10,
        verbose=0,
        random_seed=random_state,
        # auto_class_weights='SqrtBalanced',  # better than 'Balanced'
        class_weights=[1, 3],  # Adjust class weights for imbalance
        eval_metric="AUC",
    )

    model.fit(
        train_pool, eval_set=val_pool, early_stopping_rounds=early_stopping_rounds
    )

    y_pred = model.predict(X_val)
    val_metrics = compute_metrics(y_val, y_pred, avg_option="macro")

    print(f"Validation Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"Validation Precision: {val_metrics['precision']:.4f}")
    print(f"Validation Recall: {val_metrics['recall']:.4f}")
    print(f"Validation F1 Score: {val_metrics['f1_score']:.4f}")

    if val_metrics["f1_score"] > best_score:
        best_score = val_metrics["f1_score"]
        model.save_model("best_model.cbm")
        best_cv_metrics = val_metrics

print("Test validation metrics of the best model:")
for metric, value in best_cv_metrics.items():
    print(f"Best Model {metric}: {value}")

# %%
# Examine feature importance
model.load_model("best_model.cbm")
importances = model.get_feature_importance(type="PredictionValuesChange")
for name, score in zip(model.feature_names_, importances):
    print(f"{name}: {score:.4f}")

indices = np.argsort(importances)[::-1]

plt.figure(figsize=(8, 6))
plt.title("CDC Dataset Feature Importance")
plt.bar(range(len(importances)), importances[indices], align="center")
plt.xticks(
    range(len(importances)), np.array(model.feature_names_)[indices], rotation=90
)
plt.tight_layout()
plt.show()


# %%
# Evaluate on the test set
model.load_model("best_model.cbm")
y_pred = model.predict(X_test)
avg_options = ["micro", "macro", "weighted", "binary"]

results = [compute_metrics(y_test, y_pred, avg) for avg in avg_options]
results_df = pd.DataFrame(results)
# results_df.to_csv("results.csv", index=False)

print(results_df)
