"""Complete SU(2)_k example tables in the Racah--Wigner gauge.

Source: arXiv:2608.14180v1, Appendix A, equations (79)--(99).
https://arxiv.org/html/2608.14180v1#A1

Integer JSON labels are twice the spin label in the paper: a = 2j. Negative
levels reverse chirality by conjugating R and twists; the real F and intrinsic
symmetry symbols are unchanged. These are examples of the UMTC itself and its
intrinsic symmetry, not the paper's microscopic p4 x SO(3) SET classifications.
"""

from __future__ import annotations

import cmath
from functools import cache
import itertools
import math


def _number(value: complex | float | int) -> dict[str, float] | float | int:
    """JSON numerical encoding with exact zero and unit cleanup."""
    def clean(x: float) -> float | int:
        for special in (0, 1, -1):
            if abs(x - special) < 1e-14:
                return special
        return x

    value = complex(value)
    real, imaginary = clean(value.real), clean(value.imag)
    return real if imaginary == 0 else {"re": real, "im": imaginary}


def su2(k: int) -> dict:
    """Return all admissible F/R blocks and intrinsic symmetry of SU(2)_k.

    ``k`` must be a nonzero integer. Anyons are the listed integer labels
    ``0,...,abs(k)``, with physical spin label ``j=a/2``. Enumeration of all
    blocks and subsequent pentagon checks become expensive at large levels.
    """
    if type(k) is not int or k == 0:
        raise ValueError("SU(2) level k must be a nonzero integer")
    level = abs(k)
    chirality = 1 if k > 0 else -1
    anyons = list(range(level + 1))
    angle = math.pi / (level + 2)

    @cache
    def outcomes(a: int, b: int) -> tuple[int, ...]:
        return tuple(range(abs(a - b), min(a + b, 2 * level - a - b) + 1, 2))

    @cache
    def q_integer(n: int) -> float:
        if n == 0 or n == level + 2:
            return 0.0
        return math.sin(n * angle) / math.sin(angle)

    @cache
    def q_factorial(n: int) -> float:
        if n < 0:
            raise ValueError("A quantum factorial argument must be nonnegative")
        if n >= level + 2:
            return 0.0  # [k+2] is exactly zero at this root of unity.
        return math.prod(q_integer(i) for i in range(1, n + 1))

    @cache
    def triangle(a: int, b: int, c: int) -> float:
        if c not in outcomes(a, b):
            return 0.0
        numerator = (
            q_factorial((a + b - c) // 2)
            * q_factorial((a - b + c) // 2)
            * q_factorial((-a + b + c) // 2)
        )
        return math.sqrt(numerator / q_factorial((a + b + c) // 2 + 1))

    def f_symbol(a: int, b: int, c: int, d: int, e: int, f: int) -> float:
        # In the paper's 6j array, top row is (a,b,e)/2 and bottom
        # row is (c,d,f)/2. This gives the package's left-e/right-f basis.
        triangles = ((a, b, e), (a, d, f), (c, b, f), (c, d, e))
        lower = tuple(sum(t) // 2 for t in triangles)
        upper = ((a + b + c + d) // 2, (a + c + e + f) // 2, (b + d + e + f) // 2)
        terms = []
        # Terms with z >= k+1 have numerator [z+1]!=0. Denominator
        # factorial arguments remain below k+2 for admissible tetrahedra.
        for z in range(max(lower), min(min(upper), level) + 1):
            denominator = math.prod(q_factorial(z - s) for s in lower)
            denominator *= math.prod(q_factorial(s - z) for s in upper)
            terms.append((-1) ** z * q_factorial(z + 1) / denominator)
        six_j = math.prod(triangle(*t) for t in triangles) * math.fsum(terms)
        return (-1) ** ((a + b + c + d) // 2) * math.sqrt(q_integer(e + 1) * q_integer(f + 1)) * six_j

    def r_symbol(a: int, b: int, c: int) -> complex:
        casimir_difference = c * (c + 2) - a * (a + 2) - b * (b + 2)
        return (-1) ** ((c - a - b) // 2) * cmath.exp(
            chirality * 1j * math.pi * casimir_difference / (4 * (level + 2))
        )

    def spin(a: int) -> complex:
        return cmath.exp(chirality * 1j * math.pi * a * (a + 2) / (2 * (level + 2)))

    f_symbols = []
    for a, b, c, d in itertools.product(anyons, repeat=4):
        left = [e for e in outcomes(a, b) if d in outcomes(e, c)]
        if not left:
            continue
        right = [f for f in outcomes(b, c) if d in outcomes(a, f)]
        f_symbols.append({
            "a": a, "b": b, "c": c, "d": d,
            "matrix": [[_number(f_symbol(a, b, c, d, e, f)) for f in right] for e in left],
        })
    channels = [(a, b, c) for a, b in itertools.product(anyons, repeat=2) for c in outcomes(a, b)]
    nontrivial_symmetry = level >= 6 and level % 4 == 2
    elements = ["1", "X"] if nontrivial_symmetry else ["1"]

    def u_symbol(g: str, a: int, b: int, c: int) -> int:
        if g == "1":
            return 1
        floor_sum = a // 2 + b // 2 + c // 2
        return (-1) ** (floor_sum + (a % 2) * floor_sum + (b % 2) * (a // 2))

    return {
        "schema_version": 1,
        "name": f"SU(2)_{k} UMTC with {'Z2' if nontrivial_symmetry else 'trivial'} intrinsic symmetry",
        "symbol_format": "tables",
        "anyons": anyons,
        "vacuum": 0,
        "fusion_rules": [{"a": a, "b": b, "c": c, "multiplicity": 1} for a, b, c in channels],
        "quantum_dimensions": [{"anyon": a, "value": _number(q_integer(a + 1))} for a in anyons],
        "topological_spins": [{"anyon": a, "value": _number(spin(a))} for a in anyons],
        "F_symbols": f_symbols,
        "R_symbols": [
            {"a": a, "b": b, "c": c, "matrix": [[_number(r_symbol(a, b, c))]]}
            for a, b, c in channels
        ],
        "symmetry": {
            "group": {
                "gap": "CyclicGroup(2)" if nontrivial_symmetry else "CyclicGroup(1)",
                "generator_names": ["X"] if nontrivial_symmetry else [],
            },
            "rho": {"generator_images": [0] if nontrivial_symmetry else []},
            "action": {"expression": f'{level} - a if g == "X" and a % 2 == 1 else a'},
            "U_symbols": [
                {"g": g, "a": a, "b": b, "c": c, "matrix": [[u_symbol(g, a, b, c)]]}
                for g in elements for a, b, c in channels
            ],
            "eta_symbols": [
                {"anyon": a, "g": g, "h": h, "value": 1}
                for a, g, h in itertools.product(anyons, elements, elements)
            ],
        } if nontrivial_symmetry else None,
    }


def build_su2_examples() -> dict[str, dict]:
    """Representative levels, including the first nontrivial intrinsic action."""
    return {
        (f"su2_k{k}.json" if k > 0 else f"su2_k_minus{-k}.json"): su2(k)
        for k in (1, 2, 3, 4, 6, -3)
    }
