# utils/canonical.py

import numpy as np


def canonicalize(matrix: np.ndarray) -> np.ndarray:
    """
    Return the canonical representative of a permutation matrix
    under transpose equivalence.

    The first integer among {1, ..., N+1} that is not on the main
    diagonal determines the orientation:

        - Above the diagonal  -> matrix is already canonical.
        - Below the diagonal  -> transpose is canonical.

    By the pigeonhole principle, at least one of {1, ..., N+1}
    must lie off the diagonal, since there are only N diagonal
    positions.
    """

    n = matrix.shape[0]

    for k in range(1, n + 2):
        i, j = np.argwhere(matrix == k)[0]

        if i < j:
            return matrix.copy()

        if i > j:
            return matrix.T.copy()

    # This should never be reached for a valid permutation matrix.
    raise RuntimeError("Canonicalization failed: no off-diagonal pivot found.")