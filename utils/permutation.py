# utils/permutation.py

import numpy as np

def sample_permutation_matrix(n, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    perm = rng.permutation(np.arange(1, n*n + 1))
    return perm.reshape(n, n)