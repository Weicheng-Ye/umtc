"""All eight Ising UMTCs, using arXiv:2309.15118 Sec. VII, Eqs. (46)–(48).

nu is an odd integer modulo 16. The F gauge contains the Frobenius–Schur
indicator kappa=(-1)^((nu^2-1)/8); retaining it is essential for nu=3,5,11,13.
"""

from __future__ import annotations

import cmath
import itertools
import math

from scripts.generate_examples import json_number


def ising(nu: int = 1) -> dict:
    """Return complete tables for Ising^(nu), with trivial intrinsic symmetry."""
    if type(nu) is not int or nu % 2 != 1:
        raise ValueError("Ising nu must be an odd integer (defined modulo 16)")
    nu %= 16
    anyons = ["1", "sigma", "psi"]
    kappa = (-1) ** ((nu * nu - 1) // 8)

    def outcomes(a, b):
        if a == "1":
            return [b]
        if b == "1":
            return [a]
        if a == b == "sigma":
            return ["1", "psi"]
        return ["1"] if a == b == "psi" else ["sigma"]

    f_symbols = []
    for a, b, c, d in itertools.product(anyons, repeat=4):
        left = [e for e in anyons if e in outcomes(a, b) and d in outcomes(e, c)]
        if not left:
            continue
        if (a, b, c, d) == ("sigma",) * 4:
            scale = kappa / math.sqrt(2)
            matrix = [[scale, scale], [scale, -scale]]
        elif (a, b, c, d) in (("psi", "sigma", "psi", "sigma"),
                               ("sigma", "psi", "sigma", "psi")):
            matrix = [[-1]]
        else:
            matrix = [[1]]
        f_symbols.append({"a": a, "b": b, "c": c, "d": d, "matrix": matrix})

    r_symbols = []
    for a, b in itertools.product(anyons, repeat=2):
        for c in outcomes(a, b):
            if a == b == "psi":
                value = -1
            elif {a, b} == {"psi", "sigma"}:
                value = (-1j) ** nu
            elif a == b == "sigma":
                value = kappa * cmath.exp(((-1 if c == "1" else 3) * nu * math.pi * 1j) / 8)
            else:
                value = 1
            r_symbols.append({"a": a, "b": b, "c": c, "matrix": [[json_number(value)]]})

    return {
        "schema_version": 1,
        "name": f"Ising^(nu={nu}) UMTC",
        "symbol_format": "tables",
        "anyons": anyons,
        "vacuum": "1",
        "fusion_rules": [{"a": a, "b": b, "c": c, "multiplicity": 1}
                         for a, b in itertools.product(anyons, repeat=2) for c in outcomes(a, b)],
        "quantum_dimensions": [{"anyon": a, "value": math.sqrt(2) if a == "sigma" else 1}
                               for a in anyons],
        "topological_spins": [
            {"anyon": "1", "value": 1},
            {"anyon": "sigma", "value": json_number(cmath.exp(nu * math.pi * 1j / 8))},
            {"anyon": "psi", "value": -1},
        ],
        "F_symbols": f_symbols,
        "R_symbols": r_symbols,
        "symmetry": None,
    }


def build_ising_examples() -> dict[str, dict]:
    return {f"ising_nu{nu}.json": ising(nu) for nu in range(1, 16, 2)}
