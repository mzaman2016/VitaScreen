# %%
import os
import pickle
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_classif

from IGTD_Functions import (
    generate_feature_distance_ranking,
    generate_matrix_distance_ranking,
    min_max_transform,
    select_features_by_variation,
    table_to_image,
    IGTD
)

# %%
# Load the dataset
df = pd.read_csv("../data/cdcNormalDiabetic.csv")
print(df.shape)

target_col = "Label"
feature_cols = [col for col in df.columns if col != target_col]
print(f"Number of features: {len(feature_cols)}")

# %%
X, y = df[feature_cols], df[target_col]

selector = SelectKBest(score_func=f_classif, k=12)
X_selected = selector.fit_transform(X, y)

# Get names of selected features
selected_mask = selector.get_support()
selected_features = X.columns[selected_mask]

print("Selected features:", selected_features.tolist())

# %%
# Scale the features
X = df[selected_features]
id = select_features_by_variation(
    X, variation_measure="var", num=len(selected_features)
)
X = X.iloc[:, id]

X_norm = min_max_transform(X.values)
X_norm = pd.DataFrame(X_norm, columns=X.columns, index=X.index)

# %%
# Get source matrix
ranking_feature, corr = generate_feature_distance_ranking(data=X_norm)
print(ranking_feature.shape)
print(corr.shape)

ranking_feature, corr

# %%
# Plot the source matrix
def plot_dissimilarity_matrices(corr, ranking_feature, feature_names=None, figsize=(15, 6)):
    """
    Visualize the dissimilarity matrix and dissimilarity ranking matrix with values annotated.
    
    Parameters:
        corr (ndarray): Dissimilarity matrix (1 - correlation).
        ranking_feature (ndarray): Dissimilarity ranking matrix.
        feature_names (list): Optional list of feature names.
        figsize (tuple): Size of the figure.
    """
    if feature_names is None:
        feature_names = [f"F{i}" for i in range(corr.shape[0])]

    corr_df = pd.DataFrame(corr, index=feature_names, columns=feature_names)
    rank_df = pd.DataFrame(ranking_feature, index=feature_names, columns=feature_names)

    fig, axs = plt.subplots(1, 2, figsize=figsize)

    sns.heatmap(
        corr_df,
        ax=axs[0],
        # cmap="coolwarm",
        cmap="Purples",
        square=True,
        cbar_kws={"label": "1 - Correlation"},
        annot=True,
        fmt=".2f"
    )
    axs[0].set_title("Dissimilarity Matrix (1 - Correlation)")

    sns.heatmap(
        rank_df,
        ax=axs[1],
        # cmap="viridis",
        cmap="Oranges",
        square=True,
        cbar_kws={"label": "Dissimilarity Rank"},
        annot=True,
        fmt=".0f"
    )
    axs[1].set_title("Dissimilarity Ranking")

    plt.tight_layout()
    plt.show()

plot_dissimilarity_matrices(
    corr=corr,
    ranking_feature=ranking_feature,
    feature_names=list(X.columns)  # or None if not available
)

# %%
# Get Target Matrix
scale = [3, 4]
image_dist_method = "Euclidean"
coordinate, ranking_image = generate_matrix_distance_ranking(
    num_r=scale[0], num_c=scale[1], method=image_dist_method, num=X_norm.shape[1]
)
print(ranking_image.shape)

coordinate, ranking_image

# %%
# Plot the target matrix
def plot_ranking_matrix(ranking: np.ndarray, feature_names: list, title="Ranking Matrix", cmap="viridis"):
    """
    Plots a ranking matrix with feature names annotated along the axes.

    Parameters:
    - ranking: A symmetric 2D NumPy array of shape (n_features, n_features)
    - feature_names: List of feature names of length n_features
    - title: Title of the plot
    - cmap: Colormap for the heatmap
    """

    assert ranking.shape[0] == ranking.shape[1], "Matrix must be square."
    assert ranking.shape[0] == len(feature_names), "Feature name count must match matrix size."

    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(
        ranking,
        xticklabels=feature_names,
        yticklabels=feature_names,
        cmap=cmap,
        annot=True,
        fmt=".0f",
        cbar_kws={"label": "Ranking"},
        square=True
    )

    ax.set_title(title, fontsize=14)
    ax.tick_params(axis='x', rotation=45, labelsize=9)
    ax.tick_params(axis='y', rotation=0, labelsize=9)
    plt.tight_layout()
    plt.show()

plot_ranking_matrix(ranking_image, selected_features, "Ranking Image", "Blues")

# %%
# Run the IGTD algorithm
nrows, ncols = 3, 4
save_image_size = 3
max_step, val_step = 30000, 300
fea_dist_method = "Euclidean"
image_dist_method = "Euclidean"
error = "abs"
result_dir = "IGTD/Test_2"
os.makedirs(name=result_dir, exist_ok=True)

start_time = time.time()
index_coord = table_to_image(
    X_norm,
    [nrows, ncols],
    fea_dist_method,
    image_dist_method,
    save_image_size,
    max_step,
    val_step,
    result_dir,
    error,
)
end_time = time.time()

print(f"Training time: {(end_time - start_time):.2f} seconds")

# %%
type(index_coord)

# %%
# Load results
result_dir = "IGTD/Test_2"
with open(Path(result_dir) / "Results_Auxiliary.pkl", "rb") as f:
    ranking_feature = pickle.load(f)
    ranking_image = pickle.load(f)
    coordinate = pickle.load(f)
    err = pickle.load(f)
    time = pickle.load(f)

print(f"Ranking feature: {ranking_feature}")
print(f"Ranking image: {ranking_image}")
print(f"Coordinate: {coordinate}")
print(f"Error: {err}")
print(f"Time: {time}")

# %%
# Re-run IGTD to recover index_record
index_record, _, _ = IGTD(
    source=ranking_feature,
    target=ranking_image,
    err_measure="abs",         # use the same error used previously
    max_step=30000,            # use the same as before
    switch_t=0,
    val_step=300,
    min_gain=0.000001,
    random_state=1,
    save_folder=result_dir,         # don't need to save plots again
    file_name=""
)

# %%
type(index_coord)

# %%
plot_ranking_matrix(ranking_image, selected_features, "Ranking Image", "Blues")
# %%
