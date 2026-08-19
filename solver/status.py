# solver/status.py

"""
Centralized mapping of CP-SAT solver statuses.
"""

from ortools.sat.python import cp_model


STATUS_MAP = {
    cp_model.OPTIMAL: "OPTIMAL",
    cp_model.FEASIBLE: "FEASIBLE",
    cp_model.INFEASIBLE: "INFEASIBLE",
    cp_model.MODEL_INVALID: "MODEL_INVALID",
    cp_model.UNKNOWN: "UNKNOWN",
}


def decode_status(status_code: int) -> str:
    """
    Convert OR-Tools status code to human-readable string.
    """

    return STATUS_MAP.get(status_code, f"UNRECOGNIZED_{status_code}")