"""
Fundamental data structures for the Permutation Reconstruction benchmark.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Observation:
    """
    A single observed invariant.

    Attributes
    ----------
    family
        One of {"row", "column", "diag", "antidiag"}.

    statistic
        One of {"sum", "product"}.

    index
        Row/column index (0,...,N-1), or None for diagonals.

    value
        Exact integer value of the invariant.
    """

    family: str
    statistic: str
    index: int
    value: int


@dataclass
class SolverStatistics:
    """
    Statistics returned by the reconstruction solver.
    """

    solved: bool
    status: str

    runtime: float

    branches: int
    conflicts: int
    propagations: int

    num_solutions: int
