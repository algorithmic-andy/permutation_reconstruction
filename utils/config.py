"""
Global configuration for the Permutation Reconstruction benchmark.
"""

import math

# ============================================================================
# Problem Definition
# ============================================================================

# Matrix size (matrix is N x N containing 1,...,N^2)
N = 5

TOTAL_SUM = sum(range(1, N * N + 1))
TOTAL_PRODUCT = math.factorial(N * N)

MAX_PRODUCT = math.prod([(N**2)-i for i in range(N)])

# ============================================================================
# Dataset Generation
# ============================================================================

# Number of benchmark instances
NUM_INSTANCES = 100000

# Maximum nmber of solutions to track
MAX_SOLUTIONS = 50

# Independent observation probability
OBSERVATION_PROB = 0.5

# Random seed for reproducibility
RANDOM_SEED = 42

# ============================================================================
# Solver
# ============================================================================

# Maximum solver time (seconds)
SOLVER_TIME_LIMIT = 120

# ============================================================================
# Dataset Split
# ============================================================================

TRAIN_FRACTION = 0.80

# ============================================================================
# Output Locations
# ============================================================================

DATASET_DIR = "datasets"

MODEL_DIR = "trained_models"

RESULTS_DIR = "results"