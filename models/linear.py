# models/linear.py

"""
Baseline linear regression model.
"""

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from models.metrics import (
    evaluate_predictions,
    print_metrics,
)

from models.pipeline import TARGET_NAMES


# ============================================================
# Training
# ============================================================

def train_linear(
    X_train,
    y_train,
    X_test,
    y_test,
):
    """
    Train a linear regression model.

    Targets
    -------
    0 : log(1 + branches)
    """

    model = LinearRegression()

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model.fit(
        X_train,
        y_train,
    )

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    predictions = model.predict(
        X_test,
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    metrics = evaluate_predictions(
        y_test,
        predictions,
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print("=" * 60)
    print("LINEAR REGRESSION")
    print("=" * 60)

    print_metrics(metrics)

    return (
        model,
        predictions,
        metrics,
    )


# ============================================================
# Prediction
# ============================================================

def predict_linear(
    model,
    X,
):
    """
    Predict all four solver metrics.
    """

    return model.predict(X)


# ============================================================
# Coefficients
# ============================================================

def print_coefficients(model):
    """
    Print learned coefficients for each task.
    """

    print("=" * 60)
    print("LINEAR REGRESSION COEFFICIENTS")
    print("=" * 60)

    for task, name in enumerate(TARGET_NAMES):

        print(f"\n{name}")
        print("-" * len(name))

        for feature, coef in enumerate(model.coef_[task]):
            print(
                f"Feature {feature:02d}: {coef:10.4f}"
            )

        print(
            f"Intercept: {model.intercept_[task]:10.4f}"
        )


# ============================================================
# Diagnostic plots
# ============================================================

def plot_predictions(
    y_true,
    predictions,
    task=0,
):
    """
    Predicted vs true for one task.

    Parameters
    ----------
    task
        0 Branches
    """

    names = [
        "Branches",
    ]

    plt.figure(figsize=(6, 6))

    plt.scatter(
        y_true[:, task],
        predictions[:, task],
        alpha=0.4,
        s=12,
    )

    lower = min(
        y_true[:, task].min(),
        predictions[:, task].min(),
    )

    upper = max(
        y_true[:, task].max(),
        predictions[:, task].max(),
    )

    plt.plot(
        [lower, upper],
        [lower, upper],
        "--",
        linewidth=2,
    )

    plt.xlabel("True")

    plt.ylabel("Predicted")

    plt.title(names[task])

    plt.tight_layout()

    plt.show()


def plot_residuals(
    y_true,
    predictions,
    task=0,
):
    """
    Residual plot for one task.
    """

    import matplotlib.pyplot as plt

    residuals = (
        y_true[:, task]
        - predictions[:, task]
    )

    plt.figure(figsize=(7, 5))

    plt.scatter(
        predictions[:, task],
        residuals,
        alpha=0.4,
        s=12,
    )

    plt.axhline(
        0,
        linestyle="--",
        linewidth=2,
    )

    plt.xlabel("Predicted")

    plt.ylabel("Residual")

    plt.title(f"Residuals ({task})")

    plt.tight_layout()

    plt.show()