# models/pipeline.py

"""
Utilities for converting the permutation reasoning dataset
into machine learning feature matrices.
"""

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils.features import build_feature_vector


TARGET_NAMES = [
    "Branches",
]


# ============================================================
# Train/test split
# ============================================================

def split_dataset(
    X,
    y,
    test_size=0.2,
    seed=42,
):
    """
    Reproducible train/test split.
    """

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
    )


# ============================================================
# Feature extraction
# ============================================================

def extract_features(dataset):
    """
    Construct feature matrix X.
    """

    X = np.array(
        [
            build_feature_vector(d["closed_observations"])
            for d in dataset
        ],
        dtype=float,
    )

    return X


# ============================================================
# Target extraction
# ============================================================

def extract_target(dataset):
    """
    Regression targets.

    Columns
    -------

    0 : log(1 + branches)

    """

    y = np.array(
        [
            [
                np.log1p(d["solver_stats"].branches),

            ]
            for d in dataset
        ],
        dtype=float,
    )

    return y


# ============================================================
# Build full dataset 
# ============================================================

def build_xy(dataset, return_scaler=False):
    """
    Construct ML dataset with standardization.

    IMPORTANT:
    - scaler is fit ONLY on training data outside this function.
    - this function only constructs raw X, y.
    """

    X = extract_features(dataset)
    y = extract_target(dataset)

    return X, y