# utils/observations.py

"""
Compute all observable invariants (constraints) of a permutation matrix.
"""

import numpy as np

from core.types import Observation


def compute_observations(matrix: np.ndarray) -> set[Observation]:
    """
    Compute the complete observation set for a permutation matrix.

    Parameters
    ----------
    matrix : np.ndarray
        An N x N permutation matrix containing the integers
        1, ..., N^2.

    Returns
    -------
    set[Observation]
        The complete set of observable invariants.
    """

    n = matrix.shape[0]

    observations = set()

    # ==========================================================
    # Row observations
    # ==========================================================

    for i in range(n):
        row = matrix[i]

        observations.add(
            Observation(
                family="row",
                statistic="sum",
                index=i,
                value=int(row.sum()),
            )
        )

        observations.add(
            Observation(
                family="row",
                statistic="product",
                index=i,
                value=int(np.prod(row)),
            )
        )

    # ==========================================================
    # Column observations
    # ==========================================================

    for j in range(n):
        col = matrix[:, j]

        observations.add(
            Observation(
                family="column",
                statistic="sum",
                index=j,
                value=int(col.sum()),
            )
        )

        observations.add(
            Observation(
                family="column",
                statistic="product",
                index=j,
                value=int(np.prod(col)),
            )
        )

    # ==========================================================
    # Main diagonal observations
    # ==========================================================

    diag = np.diag(matrix)

    observations.add(
        Observation(
            family="diag",
            statistic="sum",
            index=None,
            value=int(diag.sum()),
        )
    )

    observations.add(
        Observation(
            family="diag",
            statistic="product",
            index=None,
            value=int(np.prod(diag)),
        )
    )

    # ==========================================================
    # Anti-diagonal observations
    # ==========================================================

    anti = np.fliplr(matrix).diagonal()

    observations.add(
        Observation(
            family="antidiag",
            statistic="sum",
            index=None,
            value=int(anti.sum()),
        )
    )

    observations.add(
        Observation(
            family="antidiag",
            statistic="product",
            index=None,
            value=int(np.prod(anti)),
        )
    )

    return observations