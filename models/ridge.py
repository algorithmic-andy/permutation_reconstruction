"""
Ridge regression model with automatic cross-validation.
"""

from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from models.metrics import (
    evaluate_predictions,
    print_metrics,
)

from models.pipeline import TARGET_NAMES


def train_ridge(
    X_train,
    y_train,
    X_test,
    y_test,
):
    """
    Train a Ridge regression model using
    cross-validation to choose alpha.
    """

    candidate_alphas = [
        1e-3,
        1e-1,
        1,
        10,
        100,
        1000,
        10000,
    ]

    model = Pipeline([
        ("scaler", StandardScaler()),
        (
            "ridge",
            RidgeCV(
                alphas=candidate_alphas,
                cv=5,
            ),
        ),
    ])

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model.fit(X_train, y_train)

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    predictions = model.predict(X_test)

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    metrics = evaluate_predictions(
        y_test,
        predictions,
    )

    ridge = model.named_steps["ridge"]

    print("=" * 60)
    print("RIDGE REGRESSION")
    print("=" * 60)

    print(f"Selected alpha : {ridge.alpha_:.4f}")

    print_metrics(metrics)

    return (
        model,
        predictions,
        metrics,
    )


def predict_ridge(model, X):
    """
    Predict using a trained Ridge model.
    """

    return model.predict(X)


def print_coefficients(model):
    """
    Print Ridge coefficients for each prediction task.
    """

    ridge = model.named_steps["ridge"]

    print("=" * 60)
    print("RIDGE COEFFICIENTS")
    print("=" * 60)

    for task, name in enumerate(TARGET_NAMES):

        print(f"\n{name}")
        print("-" * len(name))

        for feature, coef in enumerate(ridge.coef_[task]):

            print(
                f"Feature {feature:02d}: {coef:10.4f}"
            )

        print(
            f"Intercept: {ridge.intercept_[task]:10.4f}"
        )