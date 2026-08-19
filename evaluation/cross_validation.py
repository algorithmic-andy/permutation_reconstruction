"""
Cross-validation utilities.
"""

import numpy as np

from sklearn.model_selection import KFold

from models.metrics import evaluate_predictions


# ============================================================
# Cross Validation
# ============================================================

def cross_validate(
    trainer,
    X,
    y,
    k=5,
    random_state=42,
):
    """
    Perform k-fold cross-validation for a single target.

    Parameters
    ----------
    trainer
        Callable with signature

            trainer(
                X_train,
                y_train,
                X_test,
                y_test,
                target_name,
            )

    Returns
    -------
    dict

        {
            "overall": {
                "mae": ...,
                "rmse": ...,
                "r2": ...,
            }
        }
    """

    cv = KFold(
        n_splits=k,
        shuffle=True,
        random_state=random_state,
    )

    fold_metrics = []

    # --------------------------------------------------------
    # Run folds
    # --------------------------------------------------------

    for train_idx, test_idx in cv.split(X):

        X_train = X[train_idx]
        X_test = X[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        _, predictions, _ = trainer(
            X_train,
            y_train,
            X_test,
            y_test,
        )

        metrics = evaluate_predictions(
            y_test,
            predictions,
        )

        fold_metrics.append(metrics)

    # --------------------------------------------------------
    # Aggregate metrics
    # --------------------------------------------------------

    summary = {
        "overall": {

            "mae": np.mean(
                [
                    fold["overall"]["mae"]
                    for fold in fold_metrics
                ]
            ),

            "rmse": np.mean(
                [
                    fold["overall"]["rmse"]
                    for fold in fold_metrics
                ]
            ),

            "r2": np.mean(
                [
                    fold["overall"]["r2"]
                    for fold in fold_metrics
                ]
            ),
        }
    }

    return summary


# ============================================================
# Reporting
# ============================================================

def print_cv_results(
    results,
    target_name,
):
    """
    Pretty-print cross-validation results.
    """
    
    metrics = results["overall"]

    print("=" * 70)
    print(f"CROSS VALIDATION RESULTS ({target_name})")
    print("=" * 70)

    print(
        f"MAE  : {metrics['mae']:.4f}"
    )

    print(
        f"RMSE : {metrics['rmse']:.4f}"
    )

    print(
        f"R²   : {metrics['r2']:.4f}"
    )