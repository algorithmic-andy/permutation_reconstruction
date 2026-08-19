"""
Feature extraction from observation sets.
"""

import numpy as np

from .config import N
from core.types import Observation


def build_feature_vector(
    observations: set[Observation],
):
    """
    Convert an observation set into a fixed-length feature vector.

    Feature ordering:

        row sums
        row products
        column sums
        column products
        diagonal sum
        diagonal product
        antidiagonal sum
        antidiagonal product

    Missing observations are encoded as 0.
    """

    lookup = {
        (obs.family, obs.statistic, obs.index): obs.value
        for obs in observations
    }

    features = []

    # --------------------------------------------------------
    # Row observations
    # --------------------------------------------------------

    for statistic in ("sum", "product"):
        for i in range(N):
            value = lookup.get(("row", statistic, i), 0)
            

            if statistic == "product" and value != 0:

                value = np.log(value)

            features.append(value)
        

    # --------------------------------------------------------
    # Column observations
    # --------------------------------------------------------

    for statistic in ("sum", "product"):
        for i in range(N):
            value = lookup.get(("column", statistic, i), 0)

            if statistic == "product" and value != 0:

                value = np.log(value)

            features.append(value)

    # --------------------------------------------------------
    # Diagonal observations
    # --------------------------------------------------------

    for statistic in ("sum", "product"):

        value = lookup.get(
                ("diag", statistic, None),
                0,
            )

        if statistic == "product" and value != 0:

            value = np.log(value)

        features.append(value)

    # --------------------------------------------------------
    # Anti-diagonal observations
    # --------------------------------------------------------

    for statistic in ("sum", "product"):

        value = lookup.get(
                ("antidiag", statistic, None),
                0,
            )

        if statistic == "product" and value != 0:

            value = np.log(value)

        features.append(value)



    return np.asarray(
        features,
        dtype=float,
    )