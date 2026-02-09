# %%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# %%
df_orig = pd.read_csv("../data/cdcNormalDiabetic.csv")
df = pd.read_csv("../data/cdcNormalDiabeticFE1.csv")

df_orig.shape, df.shape

# %%
df_orig.info()

# %%
df.info()

# %%
for col in df.columns:
    if not (df[col] == df_orig[col]).all():
        print(f"Column '{col}' has changed:")
        # print(df[col].value_counts(), df_orig[col].value_counts())

# %%
df_orig["Income"].value_counts()

# %%
# Plot the distribution of Age, Education, Income, and GenHlth

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
sns.histplot(df_orig["Age"], bins=30, ax=axes[0, 0])
axes[0, 0].set_title("Age Distribution")
sns.histplot(df_orig["Education"], bins=30, ax=axes[0, 1])
axes[0, 1].set_title("Education Distribution")
sns.histplot(df_orig["Income"], bins=30, ax=axes[1, 0])
axes[1, 0].set_title("Income Distribution")
sns.histplot(df_orig["GenHlth"], bins=30, ax=axes[1, 1])
axes[1, 1].set_title("GenHlth Distribution")
plt.tight_layout()
plt.show()
# %%
