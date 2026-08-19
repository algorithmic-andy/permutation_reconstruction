"""
Experiment runner (multi-target, single-output per target).
"""

from models.pipeline import TARGET_NAMES

from models.compare import (
    compare_models,
    print_comparison,
)

from evaluation.cross_validation import cross_validate


# ============================================================
# Experiments
# ============================================================

def run_experiments(
    X,
    y,
    model_dict,
):
    """
    Evaluate multiple models across multiple regression targets.

    Returns
    -------
    dict[str, pd.DataFrame]

        {
            "Branches": DataFrame,
            "Conflicts": DataFrame,
            ...
        }
    """

    all_tables = {}

    # --------------------------------------------------------
    # Loop over targets
    # --------------------------------------------------------

    for target_idx, target_name in enumerate(TARGET_NAMES):

        print()
        print("=" * 80)
        print(f"TARGET: {target_name}")
        print("=" * 80)

        y_target = y[:, target_idx]

        results_for_target = {}

        # ----------------------------------------------------
        # Loop over models
        # ----------------------------------------------------

        for model_name, trainer in model_dict.items():

            print()
            print("-" * 70)
            print(model_name)
            print("-" * 70)

            metrics = cross_validate(
                trainer,
                X,
                y_target,
            )

            results_for_target[model_name] = metrics

        # ----------------------------------------------------
        # Build comparison table for this target
        # ----------------------------------------------------

        table = compare_models(results_for_target)

        print()
        print_comparison(table)

        all_tables[target_name] = table

    return all_tables