"""
Decision Tree regression model.
"""

import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree as sklearn_plot_tree

from models.metrics import (
    evaluate_predictions,
    print_metrics,
)


def train_tree(
    X_train,
    y_train,
    X_test,
    y_test,
):
    """
    Train a Decision Tree regressor using
    cross-validation.
    """

    parameter_grid = {

        "max_depth": [
            2,
            3,
            5,
            7,
            10,
            50,
            None
        ],

        "min_samples_leaf": [
            5,
            10,
            50,
            100,
            500,
        ],
    }

    search = GridSearchCV(

        DecisionTreeRegressor(
            random_state=42,
        ),

        parameter_grid,

        cv=5,

        scoring="neg_root_mean_squared_error",

        n_jobs=-1,
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    search.fit(
        X_train,
        y_train,
    )

    model = search.best_estimator_

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

    print("=" * 60)
    print("DECISION TREE")
    print("=" * 60)

    print("Best parameters:")

    print(search.best_params_)

    print_metrics(metrics)

    return (
        model,
        predictions,
        metrics,
    )


def predict_tree(
    model,
    X,
):
    """
    Predict using a trained tree.
    """

    return model.predict(X)


def print_feature_importance(model):
    """
    Display feature importance.
    """

    print("=" * 60)
    print("FEATURE IMPORTANCE")
    print("=" * 60)

    importances = model.feature_importances_

    ranking = sorted(
        enumerate(importances),
        key=lambda x: x[1],
        reverse=True,
    )

    for feature, importance in ranking:

        print(
            f"Feature {feature:02d}: {importance:.4f}"
        )


def plot_tree(model):
    """
    Visualize the fitted decision tree.
    """

    plt.figure(figsize=(18, 10))

    sklearn_plot_tree(
        model,
        filled=True,
        rounded=True,
        fontsize=8,
    )

    plt.tight_layout()
    plt.show()