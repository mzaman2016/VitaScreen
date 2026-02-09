# %%
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split

from notebooks.utils import compute_metrics

# %%
# Load the dataset
df = pd.read_csv("data/cdcNormalDiabetic.csv")
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

# %%
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.2, random_state=42
)
X_train.shape, y_train.shape, X_val.shape, y_val.shape

# %%
train_pool = Pool(data=X_train, label=y_train)
val_pool = Pool(data=X_val, label=y_val)

random_state = 42
early_stopping_rounds = 50

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

model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=early_stopping_rounds)

y_pred = model.predict(X_val)
val_metrics = compute_metrics(y_val, y_pred, avg_option="binary")

print(f"Validation Accuracy: {val_metrics['accuracy']:.4f}")
print(f"Validation Precision: {val_metrics['precision']:.4f}")
print(f"Validation Recall: {val_metrics['recall']:.4f}")
print(f"Validation F1 Score: {val_metrics['f1_score']:.4f}")

## auto_class_weights='SqrtBalanced'
# Validation Accuracy: 0.8259
# Validation Precision: 0.4522
# Validation Recall: 0.5050
# Validation F1 Score: 0.4771

## class_weights=[1, 3]
# Validation Accuracy: 0.8025
# Validation Precision: 0.4122
# Validation Recall: 0.5997
# Validation F1 Score: 0.4886

# %%
# Evaluate on the test set
y_pred = model.predict(X_test)
avg_options = ["micro", "macro", "weighted", "binary"]

results = [compute_metrics(y_test, y_pred, avg) for avg in avg_options]
results_df = pd.DataFrame(results)
results_df.to_csv("results.csv", index=False)

print(results_df)

# binary  0.803473   0.413954  0.614910  0.494807
