"""
Utilities for exporting report tables.
"""

from pathlib import Path


def save_table_csv(
    table,
    filename,
):
    """
    Save comparison table as CSV.
    """

    path = Path(filename)

    table.to_csv(
        path,
        index=False,
    )

    print(f"Saved {path}")


def save_table_latex(
    table,
    filename,
):
    """
    Save comparison table as LaTeX.
    """

    path = Path(filename)

    with open(
        path,
        "w",
        encoding="utf8",
    ) as f:

        f.write(
            table.to_latex(
                index=False,
                float_format="%.4f",
            )
        )

    print(f"Saved {path}")