import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid") # Note: set_theme is the modern seaborn method

# 1. LOAD DATA
# df = pd.read_csv("./titanic/train_and_test2.csv")
# age = df["Age"].dropna()
# fare = df["Fare"].dropna()
# sibsp = df["sibsp"].dropna()

# Mocking Kaggle data for demonstration
age = np.random.normal(loc=30, scale=10, size=700)
fare = np.random.lognormal(mean=3, sigma=1, size=700)
sibsp = np.random.poisson(lam=0.5, size=700)

# 2. SIMULATE DATA
np.random.seed(42) # Set seed for reproducibility
data = {
    "Normal (Age - Mocked)": age,
    "Lognormal (Fare - Mocked)": fare,
    "Exponential": np.random.exponential(scale=1, size=1000),
    "Gamma": np.random.gamma(shape=2, scale=1, size=1000),
    "Poisson (Simulated)": np.random.poisson(lam=2, size=1000),
    "Poisson (SibSp - Mocked)": sibsp,
    "Beta": np.random.beta(a=2, b=5, size=1000),
    "Binomial": np.random.binomial(n=10, p=0.5, size=1000),
    "Weibull": np.random.weibull(a=1.5, size=1000),
    "Triangular": np.random.triangular(left=0, mode=5, right=10, size=1000),
    "Uniform": np.random.uniform(low=0, high=1, size=1000),
}

# 3. PLOT
# Use an object-oriented approach for better control
fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(18, 16))
axes = axes.flatten() # Flatten the 4x3 grid into a 1D array for easy iteration

for ax, (name, values) in zip(axes, data.items()):
    sns.histplot(values, kde=True, ax=ax, color="steelblue", bins=30)
    ax.set_title(name, fontsize=12, fontweight='bold')
    ax.set_ylabel('') # Remove repetitive y-labels for cleaner look

# Hide any unused subplots (since we have 11 items and 12 slots)
for i in range(len(data), len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()