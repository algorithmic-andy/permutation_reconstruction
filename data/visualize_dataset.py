# data/visualize_dataset.py

from pathlib import Path
import pickle

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Load dataset
# ============================================================

DATASET_PATH = Path("datasets/dataset.pkl")

with DATASET_PATH.open("rb") as f:
    dataset = pickle.load(f)


# ============================================================
# Extract quantities
# ============================================================

branches = np.array([
    d["solver_stats"].branches
    for d in dataset
])

conflicts = np.array([
    d["solver_stats"].conflicts
    for d in dataset
])

runtime = np.array([
    d["solver_stats"].runtime
    for d in dataset
])

num_solutions = np.array([
    d["solver_stats"].num_solutions
    for d in dataset
])

status = [
    d["solver_stats"].status
    for d in dataset
]

obs_size = np.array([
    len(d["raw_observations"])
    for d in dataset
])

closure_size = np.array([
    len(d["closed_observations"])
    for d in dataset
])


# ============================================================
# Summary
# ============================================================

print("=" * 60)
print("DATASET SUMMARY")
print("=" * 60)

print(f"Instances          : {len(dataset)}")
print(f"Solved             : {sum(s == 'OPTIMAL' for s in status)}")

print()

print(f"Mean branches      : {branches.mean():.2f}")
print(f"Median branches    : {np.median(branches):.2f}")

print(f"Mean conflicts     : {conflicts.mean():.2f}")
print(f"Median conflicts   : {np.median(conflicts):.2f}")

print(f"Mean runtime (s)   : {runtime.mean():.5f}")

print(f"Mean observations  : {obs_size.mean():.2f}")
print(f"Mean closure size  : {closure_size.mean():.2f}")


# ============================================================
# Histogram helper
# ============================================================

def histogram(values, title, xlabel):

    plt.figure(figsize=(6,4))

    plt.hist(values, bins=30)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")

    plt.tight_layout()


# ============================================================
# Histograms
# ============================================================

histogram(branches,
          "Branches",
          "Branches")

histogram(np.log1p(branches),
          "log(1 + branches)",
          "log(1 + branches)")

histogram(conflicts,
          "Conflicts",
          "Conflicts")

histogram(np.log1p(conflicts),
          "log(1 + conflicts)",
          "log(1 + conflicts)")

histogram(runtime,
          "Runtime",
          "Seconds")

histogram(obs_size,
          "Observation Count",
          "Number of observations")

histogram(closure_size,
          "Closure Size",
          "Number of observations")


# ============================================================
# Scatter helper
# ============================================================

def scatter(x, y, title, xlabel, ylabel):

    plt.figure(figsize=(6,4))

    plt.scatter(x, y, s=8)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.tight_layout()


# ============================================================
# Scatter plots
# ============================================================

scatter(
    obs_size,
    np.log1p(branches),
    "Observation Count vs log(Branches)",
    "Observation Count",
    "log(1 + branches)",
)

scatter(
    closure_size,
    np.log1p(branches),
    "Closure Size vs log(Branches)",
    "Closure Size",
    "log(1 + branches)",
)

scatter(
    conflicts,
    branches,
    "Conflicts vs Branches",
    "Conflicts",
    "Branches",
)

plt.show()