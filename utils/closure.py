# utils/closure.py

"""
Closure operator for observation sets.
"""

from utils.config import N, TOTAL_SUM, TOTAL_PRODUCT
from core.types import Observation
from math import prod
from utils.query import observation_dict


def _infer_missing(
    observations: set[Observation],
    lookup: dict[tuple[str, str, int], Observation],
    family: str,
    statistic: str,
    total: int,
) -> bool:
    """
    Infer the final missing observation for a given
    family/statistic.

    Returns
    -------
    bool
        True iff a new observation was added.
    """

    present = [
        i
        for i in range(N)
        if (family, statistic, i) in lookup
    ]

    if len(present) != N - 1:
        return False

    missing = next(
        i
        for i in range(N)
        if i not in present
    )

    if statistic == "sum":
        known_total = sum(
        lookup[(family, statistic, i)].value
        for i in present
        )
        inferred_value = TOTAL_SUM - known_total
    else:
        known_total = prod(
        lookup[(family, statistic, i)].value
        for i in present
        )
        inferred_value = TOTAL_PRODUCT // known_total

    observations.add(
        Observation(
            family=family,
            statistic=statistic,
            index=missing,
            value=inferred_value,
        )
    )

    return True


def closure(
    observations: set[Observation],
) -> set[Observation]:
    """
    Compute the closure cl(O).

    The returned observation set contains every observation
    deterministically implied by the input.
    """

    closed = set(observations)

    while True:

        changed = False

        lookup = observation_dict(closed)

        changed |= _infer_missing(
            closed,
            lookup,
            "row",
            "sum",
            TOTAL_SUM,
        )

        lookup = observation_dict(closed)

        changed |= _infer_missing(
            closed,
            lookup,
            "row",
            "product",
            TOTAL_PRODUCT,
        )

        lookup = observation_dict(closed)

        changed |= _infer_missing(
            closed,
            lookup,
            "column",
            "sum",
            TOTAL_SUM,
        )

        lookup = observation_dict(closed)

        changed |= _infer_missing(
            closed,
            lookup,
            "column",
            "product",
            TOTAL_PRODUCT,
        )

        if not changed:
            break

    return closed