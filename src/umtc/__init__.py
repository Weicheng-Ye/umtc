"""UMTC data, callable symbols and numerical consistency checks."""

from .data import DataError, Label, Symmetry, UMTC, load_json
from .gap import GAPError, GAPUnavailableError, GAPGroup, GAPZ2Homomorphism
from .checks import (
    CheckReport, EquationFailure, check_all, check_eta_consistency,
    check_fusion, check_hexagon, check_modularity, check_pentagon,
    check_ribbon, check_symmetry, check_symmetry_action, check_symmetry_f,
    check_symmetry_r, check_u_consistency, check_unitarity,
    coherence_report, s_matrix,
)

__version__ = "0.4.0"
__all__ = [
    "UMTC", "Symmetry", "Label", "DataError", "load_json", "CheckReport", "EquationFailure",
    "GAPGroup", "GAPZ2Homomorphism", "GAPError", "GAPUnavailableError",
    "check_all", "check_eta_consistency", "check_fusion", "check_hexagon",
    "check_modularity", "check_pentagon", "check_ribbon", "check_symmetry",
    "check_symmetry_action", "check_symmetry_f", "check_symmetry_r",
    "check_u_consistency", "check_unitarity", "coherence_report", "s_matrix",
]
