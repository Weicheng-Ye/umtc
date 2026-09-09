"""Regenerate the original examples and the category catalog from both papers."""

from __future__ import annotations

import cmath
import itertools
import json
import math
from pathlib import Path
import sys

# Support both ``python scripts/generate_examples.py`` and ``python -m ...``.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def json_number(value: complex | float | int) -> dict[str, float] | float | int:
    """Encode a numerical entry in an explicit symbol table."""
    if not isinstance(value, complex):
        return value
    if abs(value.imag) < 1e-15:
        return value.real
    return {"re": value.real, "im": value.imag}


def toric_code(*, symbol_format: str = "functions") -> dict:
    """The toric code with commuting unitary X and antiunitary T, both e↔m.

    The public example uses compact functions for all anyon-dependent data.
    Explicit tables remain available for tests of vertex gauge transformations.
    """
    if symbol_format not in ("tables", "functions"):
        raise ValueError("symbol_format must be 'tables' or 'functions'")
    anyons = list(itertools.product(range(2), repeat=2))
    groups = ["1", "X", "T", "X*T"]
    group_bits = [(0, 0), (1, 0), (0, 1), (1, 1)]

    def fusion(a, b):
        return tuple(x ^ y for x, y in zip(a, b))

    def exchange(a, b):
        return (-1) ** (a[0] * b[1])

    def spin(a):
        return (-1) ** (a[0] * a[1])

    def parity(g):
        x, t = group_bits[groups.index(g)]
        return x ^ t

    data = {
        "schema_version": 1,
        "name": "Toric code with Z2 x Z2^T intrinsic symmetry",
        "symbol_format": symbol_format,
        "anyons": {"type": "tuples", "moduli": [2, 2]},
        "vacuum": (0, 0),
        "fusion_rules": [
            {"a": a, "b": b, "c": fusion(a, b), "multiplicity": 1}
            for a, b in itertools.product(anyons, repeat=2)
        ],
        "quantum_dimensions": [{"anyon": a, "value": 1} for a in anyons],
        "topological_spins": [{"anyon": a, "value": spin(a)} for a in anyons],
        "F_symbols": {"expression": "1"},
        "R_symbols": {"expression": "(-1) ** (a[0] * b[1])"},
    }
    if symbol_format == "tables":
        data["F_symbols"] = [
            {"a": a, "b": b, "c": c, "d": fusion(fusion(a, b), c), "matrix": [[1]]}
            for a, b, c in itertools.product(anyons, repeat=3)
        ]
        data["R_symbols"] = [
            {"a": a, "b": b, "c": fusion(a, b), "matrix": [[exchange(a, b)]]}
            for a, b in itertools.product(anyons, repeat=2)
        ]
    data["symmetry"] = {
        "elements": groups,
        "identity": "1",
        "multiplication_table": [
            [groups[group_bits.index((x ^ u, t ^ v))] for u, v in group_bits]
            for x, t in group_bits
        ],
        "antiunitary": ["T", "X*T"],
        "action": [
            {"g": g, "images": [a[::-1] if parity(g) else a for a in anyons]}
            for g in groups
        ],
        "U_symbols": [
            {"g": g, "a": a, "b": b, "c": fusion(a, b),
             "matrix": [[exchange(a, b) ** parity(g)]]}
            for g, a, b in itertools.product(groups, anyons, anyons)
        ],
        "eta_symbols": [
            {"anyon": a, "g": g, "h": h,
             "value": spin(a) ** (parity(g) * parity(h))}
            for a, g, h in itertools.product(anyons, groups, groups)
        ],
    }
    if symbol_format == "functions":
        data["fusion_rules"] = {
            "expression": "1 if c == ((a[0] + b[0]) % 2, (a[1] + b[1]) % 2) else 0"
        }
        data["quantum_dimensions"] = {"expression": "1"}
        data["topological_spins"] = {"expression": "(-1) ** (a[0] * a[1])"}
        data["symmetry"] = {
            "group": {"gap": "AbelianGroup([2,2])", "generator_names": ["X", "T"]},
            "rho": {"generator_images": [0, 1]},
            "action": {"expression": '(a[1], a[0]) if g == "X" or g == "T" else a'},
            "U_symbols": {
                "expression": '(-1) ** (a[0] * b[1]) if g == "X" or g == "T" else 1'
            },
            "eta_symbols": {
                "expression": '(-1) ** (a[0] * a[1]) if (g == "X" or g == "T") and (h == "X" or h == "T") else 1'
            },
        }
    return data


def fibonacci() -> dict:
    """Fibonacci theory in the usual real F-symbol gauge, with no symmetry data."""
    anyons = ["1", "tau"]
    phi = (1 + math.sqrt(5)) / 2

    def outcomes(a, b):
        if a == "1":
            return [b]
        if b == "1":
            return [a]
        return anyons

    f_symbols = []
    for a, b, c, d in itertools.product(anyons, repeat=4):
        left = [e for e in outcomes(a, b) if d in outcomes(e, c)]
        right = [f for f in outcomes(b, c) if d in outcomes(a, f)]
        if not left:
            continue
        assert len(left) == len(right)
        matrix = (
            [[1 / phi, 1 / math.sqrt(phi)], [1 / math.sqrt(phi), -1 / phi]]
            if (a, b, c, d) == ("tau",) * 4
            else [[1]]
        )
        f_symbols.append({"a": a, "b": b, "c": c, "d": d, "matrix": matrix})

    r_symbols = []
    for a, b in itertools.product(anyons, repeat=2):
        for c in outcomes(a, b):
            value = 1
            if a == b == "tau":
                value = cmath.exp((-4 if c == "1" else 3) * math.pi * 1j / 5)
            r_symbols.append({"a": a, "b": b, "c": c, "matrix": [[json_number(value)]]})

    return {
        "schema_version": 1,
        "name": "Fibonacci UMTC",
        "symbol_format": "tables",
        "anyons": anyons,
        "vacuum": "1",
        "fusion_rules": [
            {"a": a, "b": b, "c": c, "multiplicity": 1}
            for a, b in itertools.product(anyons, repeat=2) for c in outcomes(a, b)
        ],
        "quantum_dimensions": [{"anyon": "1", "value": 1}, {"anyon": "tau", "value": phi}],
        "topological_spins": [
            {"anyon": "1", "value": 1},
            {"anyon": "tau", "value": json_number(cmath.exp(4 * math.pi * 1j / 5))},
        ],
        "F_symbols": f_symbols,
        "R_symbols": r_symbols,
    }


def all_examples() -> dict[str, dict]:
    """Return every shipped category; family modules keep their own formulas."""
    from scripts.categories_abelian import build_abelian_examples
    from scripts.categories_ising import build_ising_examples
    from scripts.categories_su2 import build_su2_examples

    return {
        "toric_code_z2_z2t.json": toric_code(),
        "fibonacci.json": fibonacci(),
        **build_abelian_examples(),
        **build_ising_examples(),
        **build_su2_examples(),
    }


def main() -> None:
    directory = Path(__file__).resolve().parents[1] / "examples"
    directory.mkdir(exist_ok=True)
    for filename, data in all_examples().items():
        (directory / filename).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
