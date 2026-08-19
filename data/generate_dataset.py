# data/generate_dataset.py

from pathlib import Path
import pickle

import numpy as np

from utils.config import (
    N,
    NUM_INSTANCES,
    OBSERVATION_PROB,
    RANDOM_SEED,
)

from utils.permutation import sample_permutation_matrix
from utils.canonical import canonicalize
from utils.observations import compute_observations
from utils.closure import closure

from solver.reconstruct import ReconstructionSolver


rng = np.random.default_rng(RANDOM_SEED)


def sample_observation_set(observations: set):
    """
    Randomly subsample observations.
    """
    return {
        obs
        for obs in observations
        if rng.random() < OBSERVATION_PROB
    }


def generate_instance():
    # --------------------------------------------------------
    # 1. Ground truth
    # --------------------------------------------------------
    A = sample_permutation_matrix(N, rng)
    A = canonicalize(A)

    # --------------------------------------------------------
    # 2. Observations
    # --------------------------------------------------------
    full_observations = compute_observations(A)
    raw_observations = sample_observation_set(full_observations)
    closed_observations = closure(raw_observations)

    # --------------------------------------------------------
    # 3. Solve
    # --------------------------------------------------------
    solver = ReconstructionSolver()
    stats, solution = solver.solve(closed_observations)

    # --------------------------------------------------------
    # 4. Target
    # --------------------------------------------------------
    targets = {
                "branches" : stats.branches,
                "conflicts" : stats.conflicts,
                "propagations" : stats.propagations,
                "solutions" : stats.num_solutions,
            }  

    return {
        "ground_truth": A,
        "raw_observations": raw_observations,
        "closed_observations": closed_observations,
        "solution": solution,
        "solver_stats": stats,
        "targets": targets,
    }


def generate_dataset():
    return [
        generate_instance()
        for _ in range(NUM_INSTANCES)
    ]


def save_dataset(dataset, filename="dataset.pkl"):
    """
    Save dataset to the datasets/ directory.
    """
    output_dir = Path("datasets")
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / filename

    with output_path.open("wb") as f:
        pickle.dump(dataset, f)

    return output_path


if __name__ == "__main__":
    dataset = generate_dataset()
    path = save_dataset(dataset)

    print(f"Generated {len(dataset)} instances.")
    print(f"Saved dataset to: {path}")