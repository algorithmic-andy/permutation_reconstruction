"""
Full experiment pipeline runner.

Used for:
- batch experiments
- report generation
- dataset-scale evaluation
"""

from models.pipeline import build_xy

from evaluation.experiments import run_experiments
from evaluation.plots import plot_all_metrics
from evaluation.report_tables import save_table_csv

from models.linear import train_linear
from models.ridge import train_ridge
from models.tree import train_tree

from utils.dataset import load_dataset


def run():

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    dataset = load_dataset("datasets/dataset.pkl")

    # --------------------------------------------------------
    # Build features + target
    # --------------------------------------------------------

    X, y = build_xy(dataset)

    # --------------------------------------------------------
    # Define models
    # --------------------------------------------------------

    models = {

        "Linear": train_linear,
        "Ridge": train_ridge,
        "Tree": train_tree,
    }

    # --------------------------------------------------------
    # Run experiments
    # --------------------------------------------------------

    tables = run_experiments(X, y, models)

    for target_name, table in tables.items():

        save_table_csv(
            table,
            f"results/{target_name.lower()}_comparison.csv",
        )

        plot_all_metrics(table)

    return tables


if __name__ == "__main__":
    run()