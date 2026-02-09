# %%
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from utils import compute_metrics

# %% Load predicted probabilities
# model_names = ["cb_f21_cw", "mlp_f15_enn", "mlp_f21_cw"]
model_names = ["cb_f21_cw", "mlp_f15_enn", "igtd_f15_cw"]
# model_names = ["cb_f21_cw", "mlp_f15_enn", "nctd_f15_enn", "igtd_f15_cw"]
# model_names = ["cb_f21_cw", "mlp_f15_enn", "mlp_f21_cw", "igtd_f15_cw", "nctd_f15_enn"]
agg_prob_df = pd.read_csv(f"../prob/{model_names[0]}_prob.csv")
agg_prob_df.columns = ["id", "split", "y_true", f"{model_names[0]}"]

agg_prob_df.head(10)
n_test_obs = agg_prob_df[agg_prob_df["split"] == "test"].shape[0]
print(f"Number of test data points: {n_test_obs}")

# %%
for model_name in model_names[1:]:
    prob_df = pd.read_csv(f"../prob/{model_name}_prob.csv")

    # Sanity check individual model
    print(model_name)
    test_prob_df = prob_df[prob_df["split"] == "test"].copy()
    print(len(test_prob_df))

    # Number of test samples should be the same
    assert n_test_obs == len(test_prob_df)

    y_pred = (test_prob_df["y_prob"] >= 0.5).astype(int)
    res = compute_metrics(test_prob_df["y_true"], y_pred, avg_option="macro")
    print(res)
    del test_prob_df

    # Verify that the ids match
    # assert all(agg_prob_df["id"] == prob_df["id"])
    # assert all(agg_prob_df["split"] == prob_df["split"])
    # assert all(agg_prob_df["y_true"] == prob_df["y_true"])

    prob_df.columns = ["id", "split", "y_true", model_name]
    agg_prob_df = pd.merge(
        agg_prob_df, prob_df, on=["id", "split", "y_true"], how="inner"
    )

print(agg_prob_df.shape)
agg_prob_df.head(10)

# %%
# Ensemble by majority voting

# params
cut = 0.5  # probability threshold per model
tie_break_up = (
    True  # True => ties go to 1 (>= ceil(n/2)); False => strict majority (> n/2)
)

test_prob_df = agg_prob_df[agg_prob_df["split"] == "test"].copy()

votes = (test_prob_df[model_names].to_numpy() >= cut).sum(axis=1)
n = len(model_names)
thr = np.ceil(n / 2) if tie_break_up else (n / 2)

if len(model_name) % 2 == 0:
    test_prob_df["y_pred"] = (votes > thr).astype(int)
else:
    test_prob_df["y_pred"] = (votes >= thr).astype(int)

res = compute_metrics(
    test_prob_df["y_true"], test_prob_df["y_pred"], avg_option="macro"
)
res["n_obs"] = len(test_prob_df)
print(res)
del test_prob_df

# %%
# Ensemble by averaging probabilities
test_prob_df = agg_prob_df[agg_prob_df["split"] == "test"].copy()

# Simple (equal-weight) average over all models listed in model_names
test_prob_df["y_prob"] = test_prob_df[model_names].mean(axis=1)

# Classify with your chosen threshold
threshold = 0.5
test_prob_df["y_pred"] = (test_prob_df["y_prob"] >= threshold).astype(int)

res = compute_metrics(
    test_prob_df["y_true"], test_prob_df["y_pred"], avg_option="macro"
)
res["n_obs"] = len(test_prob_df)
print(res)
del test_prob_df

# %%
# Fit a logistic regression model

train_prob_df = agg_prob_df[agg_prob_df["split"] != "test"].copy()
test_prob_df = agg_prob_df[agg_prob_df["split"] == "test"].copy()
X_train = train_prob_df[model_names].values
y_train = train_prob_df["y_true"].values
X_test = test_prob_df[model_names].values
y_test = test_prob_df["y_true"].values

lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
y_pred = lr_model.predict(X_test)
res = compute_metrics(y_test, y_pred, avg_option="macro")
res["n_obs"] = len(test_prob_df)
print(res)

# %%
