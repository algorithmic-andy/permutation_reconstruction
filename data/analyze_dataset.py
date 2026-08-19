# data/analyze_dataset.py

from pathlib import Path
import pickle

import numpy as np


# ============================================================
# Load dataset
# ============================================================

DATASET_PATH = Path("datasets/dataset.pkl")

with DATASET_PATH.open("rb") as f:
    dataset = pickle.load(f)


# ============================================================
# Extract arrays
# ============================================================

branches = np.array([
    d["solver_stats"].branches
    for d in dataset
], dtype=float)

conflicts = np.array([
    d["solver_stats"].conflicts
    for d in dataset
], dtype=float)

runtime = np.array([
    d["solver_stats"].runtime
    for d in dataset
], dtype=float)

obs = np.array([
    len(d["raw_observations"])
    for d in dataset
], dtype=float)

closure = np.array([
    len(d["closed_observations"])
    for d in dataset
], dtype=float)

gain = closure - obs

target = np.log1p(branches)


# ============================================================
# Helpers
# ============================================================

def corr(x, y):
    return np.corrcoef(x, y)[0, 1]


def describe(name, x):

    print(f"\n{name}")
    print("-" * len(name))

    print(f"Mean      : {x.mean():.3f}")
    print(f"Std       : {x.std():.3f}")
    print(f"Min       : {x.min():.3f}")
    print(f"Median    : {np.median(x):.3f}")
    print(f"95%       : {np.quantile(x,0.95):.3f}")
    print(f"99%       : {np.quantile(x,0.99):.3f}")
    print(f"Max       : {x.max():.3f}")


# ============================================================
# Report
# ============================================================

print("=" * 70)
print("DATASET ANALYSIS")
print("=" * 70)

print(f"Instances : {len(dataset)}")

describe("Branches", branches)
describe("Conflicts", conflicts)
describe("Runtime", runtime)
describe("Observations", obs)
describe("Closure size", closure)
describe("Closure gain", gain)

print("\n")
print("=" * 70)
print("CORRELATIONS WITH log(1 + branches)")
print("=" * 70)

print(f"Observations : {corr(obs,target): .4f}")
print(f"Closure size : {corr(closure,target): .4f}")
print(f"Closure gain : {corr(gain,target): .4f}")
print(f"Conflicts    : {corr(conflicts,target): .4f}")
print(f"Runtime      : {corr(runtime,target): .4f}")

print("\n")
print("=" * 70)
print("INTER-METRIC CORRELATIONS")
print("=" * 70)

print(f"Branches vs Conflicts : {corr(branches,conflicts): .4f}")
print(f"Branches vs Runtime   : {corr(branches,runtime): .4f}")
print(f"Conflicts vs Runtime  : {corr(conflicts,runtime): .4f}")