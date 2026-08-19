# models/compare.py

"""
Utilities for comparing regression models.
"""

import pandas as pd
import matplotlib.pyplot as plt


def compare_models(results):
    """
    Build a model comparison table.

    Parameters
    ----------
    results : dict
        Dictionary mapping model names to metric dictionaries.

        Example:
        {
            "Linear": {
                "mae": 10.2,
                "rmse": 15.4,
                "r2": 0.72,
            },
            "Ridge": {
                "mae": 9.8,
                "rmse": 14.7,
                "r2": 0.75,
            },
        }

    Returns
    -------
    pandas.DataFrame
        Comparison table sorted by R^2 in descending order.
    """

    rows = [
        {
            "Model": model_name,
            "MAE": metrics["overall"]["mae"],
            "RMSE": metrics["overall"]["rmse"],
            "R^2": metrics["overall"]["r2"],
        }
        for model_name, metrics in results.items()
    ]

    return (
        pd.DataFrame(rows)
        .sort_values("R^2", ascending=False)
        .reset_index(drop=True)
    )


def print_comparison(table):
    """Print a model comparison table."""
    print("=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    print(
        table.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )


def plot_r2(table):
    """Plot R^2 for each model."""
    _plot_metric(
        table,
        metric="R^2",
        ylabel="R^2",
        title="Model Comparison: R^2",
    )


def plot_rmse(table):
    """Plot RMSE for each model."""
    _plot_metric(
        table,
        metric="RMSE",
        ylabel="RMSE",
        title="Model Comparison: RMSE",
    )


def plot_mae(table):
    """Plot MAE for each model."""
    _plot_metric(
        table,
        metric="MAE",
        ylabel="MAE",
        title="Model Comparison: MAE",
    )


def _plot_metric(table, metric, ylabel, title):
    """Plot one metric for each model."""
    plt.figure(figsize=(8, 4))

    plt.bar(
        table["Model"],
        table[metric],
    )

    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.show()

