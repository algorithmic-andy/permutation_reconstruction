# models/metrics.py

"""
Common regression evaluation metrics.
Supports both single-output and multi-output regression.
"""

import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from models.pipeline import TARGET_NAMES


# ============================================================
# Metric computation
# ============================================================

def evaluate_predictions(
    y_true,
    y_pred,
):
    """
    Compute regression metrics.

    Supports:

        (N,)
        (N, K)

    targets.

    Returns
    -------
    dict
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # --------------------------------------------------------
    # Single-output regression
    # --------------------------------------------------------

    if y_true.ndim == 1:

        return {
            "overall": {
                "mae": mean_absolute_error(
                    y_true,
                    y_pred,
                ),

                "rmse": np.sqrt(
                    mean_squared_error(
                        y_true,
                        y_pred,
                    )
                ),

                "r2": r2_score(
                    y_true,
                    y_pred,
                ),
            }
        }

    # --------------------------------------------------------
    # Multi-output regression
    # --------------------------------------------------------

    metrics = {}

    maes = []
    rmses = []
    r2s = []

    for i, name in enumerate(TARGET_NAMES):

        mae = mean_absolute_error(
            y_true[:, i],
            y_pred[:, i],
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_true[:, i],
                y_pred[:, i],
            )
        )

        r2 = r2_score(
            y_true[:, i],
            y_pred[:, i],
        )

        metrics[name] = {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
        }

        maes.append(mae)
        rmses.append(rmse)
        r2s.append(r2)

    # --------------------------------------------------------
    # Overall summary
    # --------------------------------------------------------

    metrics["overall"] = {

        "mae": np.mean(maes),

        "rmse": np.mean(rmses),

        "r2": np.mean(r2s),
    }

    return metrics


# ============================================================
# Reporting
# ============================================================

def print_metrics(metrics):
    """
    Pretty-print metrics.
    """

    print("=" * 60)
    print("MODEL PERFORMANCE")
    print("=" * 60)

    # --------------------------------------------------------
    # Single-output
    # --------------------------------------------------------

    if (
        len(metrics) == 1
        and "overall" in metrics
    ):

        overall = metrics["overall"]

        print(
            f"MAE  : {overall['mae']:.4f}"
        )

        print(
            f"RMSE : {overall['rmse']:.4f}"
        )

        print(
            f"R^2   : {overall['r2']:.4f}"
        )

        print("=" * 60)

        return

    # --------------------------------------------------------
    # Multi-output
    # --------------------------------------------------------

    for name in TARGET_NAMES:

        m = metrics[name]

        print(f"\n{name}")
        print("-" * len(name))

        print(
            f"MAE  : {m['mae']:.4f}"
        )

        print(
            f"RMSE : {m['rmse']:.4f}"
        )

        print(
            f"R^2   : {m['r2']:.4f}"
        )

    overall = metrics["overall"]

    print("\n" + "=" * 60)
    print("AVERAGE")
    print("=" * 60)

    print(
        f"MAE  : {overall['mae']:.4f}"
    )

    print(
        f"RMSE : {overall['rmse']:.4f}"
    )

    print(
        f"R^2   : {overall['r2']:.4f}"
    )

    print("=" * 60)