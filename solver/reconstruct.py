# solver/reconstruct.py

"""
CP-SAT solver for permutation reconstruction from observations.
"""

import time
import numpy as np

from ortools.sat.python import cp_model

from utils.config import N, MAX_PRODUCT, MAX_SOLUTIONS
from core.types import Observation
from utils.query import observation_dict
from solver.status import decode_status
from core.types import SolverStatistics


# ============================================================
# Solver
# ============================================================

class ReconstructionSolver:
    def __init__(self):
        self.model = cp_model.CpModel()
        self.vars = {}

    # --------------------------------------------------------
    # Variable creation
    # --------------------------------------------------------

    def _create_variables(self):
        """
        Create N^2 integer variables in [1, N^2].
        """
        for i in range(N):
            for j in range(N):
                self.vars[(i, j)] = self.model.NewIntVar(
                    1,
                    N * N,
                    f"x_{i}_{j}",
                )

        # AllDifferent constraint
        self.model.AddAllDifferent(list(self.vars.values()))

    # --------------------------------------------------------
    # Constraint helpers
    # --------------------------------------------------------

    def _get_row(self, i):
        return [self.vars[(i, j)] for j in range(N)]

    def _get_col(self, j):
        return [self.vars[(i, j)] for i in range(N)]

    def _get_diag(self):
        return [self.vars[(i, i)] for i in range(N)]

    def _get_antidiag(self):
        return [self.vars[(i, N - i - 1)] for i in range(N)]

    # --------------------------------------------------------
    # Constraint application
    # --------------------------------------------------------

    def _apply_observation(self, obs: Observation):
        """
        Convert a single Observation into a CP-SAT constraint.
        """

        if obs.family == "row":
            vars_ = self._get_row(obs.index)

        elif obs.family == "column":
            vars_ = self._get_col(obs.index)

        elif obs.family == "diag":
            vars_ = self._get_diag()

        elif obs.family == "antidiag":
            vars_ = self._get_antidiag()

        else:
            raise ValueError(f"Unknown family: {obs.family}")

        if obs.statistic == "sum":
            self.model.Add(sum(vars_) == obs.value)

        elif obs.statistic == "product":
            # CP-SAT multiplication constraint
            prod_var = self.model.NewIntVar(
                1,
                MAX_PRODUCT,
                "prod_tmp",
            )

            self.model.AddMultiplicationEquality(
                prod_var,
                vars_,
            )

            self.model.Add(prod_var == obs.value)

        else:
            raise ValueError(
                f"Unknown statistic: {obs.statistic}"
            )

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    def _build_model(
        self,
        observations: set[Observation],
    ):
        self._create_variables()

        for obs in observations:
            self._apply_observation(obs)

    # --------------------------------------------------------
    # Solve
    # --------------------------------------------------------

    def solve(
        self,
        observations: set[Observation],
    ):
        """
        Solve reconstruction problem.

        Returns
        -------
        SolverStatistics
        solution : np.ndarray
        """

        self.model = cp_model.CpModel()
        self.vars = {}

        self._build_model(observations)

        solver = cp_model.CpSolver()

        # ----------------------------------------------------
        # Solution callback
        # ----------------------------------------------------

        class SolutionCounter(
            cp_model.CpSolverSolutionCallback
        ):
            def __init__(self, vars_):
                super().__init__()

                self.vars = vars_

                self.num_solutions = 0
                self.first_solution = None

            def OnSolutionCallback(self):

                self.num_solutions += 1

                # Save only the first solution
                if self.first_solution is None:

                    solution = np.zeros(
                        (N, N),
                        dtype=int,
                    )

                    for i in range(N):
                        for j in range(N):
                            solution[i, j] = self.Value(
                                self.vars[(i, j)]
                            )

                    self.first_solution = solution

                # Stop if we hit the limit
                if self.num_solutions >= MAX_SOLUTIONS:
                    self.StopSearch()

        callback = SolutionCounter(self.vars)

        # ----------------------------------------------------
        # Enumerate solutions
        # ----------------------------------------------------

        start = time.time()

        status = solver.SearchForAllSolutions(
            self.model,
            callback,
        )

        runtime = time.time() - start

        # ----------------------------------------------------
        # Extract solution
        # ----------------------------------------------------

        if callback.first_solution is not None:
            solution = callback.first_solution
        else:
            solution = np.zeros((N, N), dtype=int)

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        stats = SolverStatistics(
            solved=callback.num_solutions > 0,
            status=decode_status(status),

            runtime=runtime,

            branches=solver.NumBranches(),
            conflicts=solver.NumConflicts(),
            propagations=solver.NumBooleans(),

            num_solutions=callback.num_solutions,
        )

        return stats, solution