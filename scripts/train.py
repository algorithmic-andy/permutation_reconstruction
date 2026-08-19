"""
Main training script for permutation reasoning models.
"""

from models.pipeline import build_xy
from evaluation.experiments import run_experiments

from models.linear import train_linear
from models.ridge import train_ridge
from models.tree import train_tree

from utils.dataset import load_dataset


def main():

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    dataset = load_dataset("datasets/dataset.pkl")

    # --------------------------------------------------------
    # Build ML data
    # --------------------------------------------------------

    X, y = build_xy(dataset)

    # --------------------------------------------------------
    # Define models
    # --------------------------------------------------------

    model_dict = {

        "Linear": train_linear,
        "Ridge": train_ridge,
        "Tree": train_tree,
    }

    # --------------------------------------------------------
    # Run experiments
    # --------------------------------------------------------

    results_tables = run_experiments(
        X,
        y,
        model_dict,
    )

    return results_tables


if __name__ == "__main__":
    main()