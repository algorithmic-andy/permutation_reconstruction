"""
Common plotting utilities.
"""

import matplotlib.pyplot as plt


def plot_metric(
    table,
    metric,
):
    """
    Bar plot for one metric.
    """

    plt.figure(figsize=(7, 4))

    plt.bar(
        table["Model"],
        table[metric],
    )

    plt.ylabel(metric)

    plt.title(f"Model comparison ({metric})")

    plt.tight_layout()

    plt.show()


def plot_all_metrics(table):
    """
    Plot RMSE, MAE and R².
    """

    for metric in [
        "RMSE",
        "MAE",
        "R²",
    ]:

        plot_metric(
            table,
            metric,
        )