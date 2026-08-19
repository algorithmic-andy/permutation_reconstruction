"""
Analyze branch statistics in the dataset.
"""

import pickle
from pathlib import Path

import numpy as np

DATASET_PATH = Path("datasets/dataset.pkl")


def main():

    with DATASET_PATH.open("rb") as f:
        dataset = pickle.load(f)

    branches = np.array(
        [
            d["solver_stats"].branches
            for d in dataset
        ]
    )

    zero = np.sum(branches == 0)
    positive = np.sum(branches > 0)

    print("=" * 60)
    print("BRANCH STATISTICS")
    print("=" * 60)

    print(f"Dataset size            : {len(branches):,}")
    print(f"Zero-branch instances   : {zero:,}")
    print(f"Positive-branch         : {positive:,}")

    print()
    print(f"Proportion zero         : {zero / len(branches):.4f}")
    print(f"Proportion positive     : {positive / len(branches):.4f}")

    print()
    print("Positive instances only")
    print("-" * 60)

    positive_values = branches[branches > 0]

    if len(positive_values) > 0:
        print(f"Mean branches           : {positive_values.mean():.2f}")
        print(f"Median branches         : {np.median(positive_values):.2f}")
        print(f"Std branches            : {positive_values.std():.2f}")
        print(f"95th percentile         : {np.quantile(positive_values, 0.95):.2f}")
        print(f"Maximum                 : {positive_values.max():.2f}")


if __name__ == "__main__":
    main()