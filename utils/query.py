# utils/query.py

"""
Convenience functions for querying observation sets.
"""

from collections import defaultdict

from core.types import Observation


def get_observations(
    observations: set[Observation],
    family: str,
    statistic: str,
) -> list[Observation]:
    """
    Return all observations satisfying the requested filters.

    Parameters
    ----------
    observations
        Observation set.

    family
        Optional family filter.

    statistic
        Optional statistic filter.

    Returns
    -------
    list[Observation]
        Matching observations sorted by index.
    """

    result = []

    for obs in observations:

        if family is not None and obs.family != family:
            continue

        if statistic is not None and obs.statistic != statistic:
            continue

        result.append(obs)

    return sorted(
        result,
        key=lambda obs: (
            obs.index is None,
            obs.index if obs.index is not None else -1,
        ),
    )


def find_observation(
    observations: set[Observation],
    family: str,
    statistic: str,
    index: int,
) -> Observation:
    """
    Find a single observation.
    """

    for obs in observations:

        if (
            obs.family == family
            and obs.statistic == statistic
            and obs.index == index
        ):
            return obs

    return None


def has_observation(
    observations: set[Observation],
    family: str,
    statistic: str,
    index: int,
) -> bool:
    """
    Determine whether an observation exists.
    """

    return (
        find_observation(
            observations,
            family,
            statistic,
            index,
        )
        is not None
    )


def group_by_family_and_statistic(
    observations: set[Observation],
) -> dict[tuple[str, str], list[Observation]]:
    """
    Group observations by (family, statistic).
    """

    grouped = defaultdict(list)

    for obs in observations:
        grouped[(obs.family, obs.statistic)].append(obs)

    return dict(grouped)


def observation_dict(
    observations: set[Observation],
) -> dict[tuple[str, str, int], Observation]:
    """
    Build a dictionary for constant-time lookup.
    """

    return {
        (obs.family, obs.statistic, obs.index): obs
        for obs in observations
    }