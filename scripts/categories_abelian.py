"""Abelian categories and full intrinsic symmetries from arXiv:2309.15118.

Parameters are restricted to the ranges whose full intrinsic groups are given
in Sections VI, VIII, and IX. The doubled theory uses the tensor product of
Eq. (10) with its conjugate: the prefactors printed in Eqs. (61)-(62) do not
produce the stated double-semion theory at N=1.

Source: https://arxiv.org/html/2309.15118v3
"""

from __future__ import annotations


_D8_GAP = "Group((1,2,3,4),(2,4))"


def _expression(value: str) -> dict:
    return {"expression": value}


def _pointed(name: str, moduli: list[int], f: str, r: str, spin: str) -> dict:
    coordinates = ", ".join(f"(a[{i}] + b[{i}]) % {n}" for i, n in enumerate(moduli))
    if len(moduli) == 1:
        coordinates += ","
    return {
        "schema_version": 1,
        "name": name,
        "symbol_format": "functions",
        "anyons": {"type": "tuples", "moduli": moduli},
        "vacuum": [0] * len(moduli),
        "fusion_rules": _expression(f"1 if c == ({coordinates}) else 0"),
        "quantum_dimensions": _expression("1"),
        "topological_spins": _expression(spin),
        "F_symbols": _expression(f),
        "R_symbols": _expression(r),
    }


def u1(level: int) -> dict:
    """U(1)_level for level=2,4,6,8,10 with its full intrinsic symmetry.

    Larger even levels can have additional automorphisms and are rejected
    rather than silently returning only the charge-conjugation subgroup.
    """
    if type(level) is not int or level not in (2, 4, 6, 8, 10):
        raise ValueError("u1 supports levels 2, 4, 6, 8, 10; full intrinsic symmetry outside this range is not implemented")
    data = _pointed(
        f"U(1)_{level}", [level],
        f"(-1) ** (a[0] * ((b[0] + c[0]) // {level}))",
        f"exp(pi * 1j * a[0] * b[0] / {level})",
        f"exp(pi * 1j * a[0] * a[0] / {level})",
    )
    data["symmetry"] = None if level == 2 else {
        "group": {"gap": "CyclicGroup(2)", "generator_names": ["C"]},
        "rho": {"generator_images": [0]},
        "action": _expression(f"((-a[0]) % {level},) if g == 'C' else a"),
        "U_symbols": _expression("(-1) ** a[0] if g == 'C' and b[0] > 0 else 1"),
        "eta_symbols": _expression("1"),
    }
    return data


def _d8_action(modulus: int) -> str:
    """Action of T^i S^j, with T(x,y)=(y,-x), S(x,y)=(y,x).

    GAP's stable breadth-first names are 1,T,S,T*T,T*S,S*T,T*T*T,T*T*S.
    In normal form S*T=T^3*S, so the word spelling must not be read as a
    count of T's. Both the action and U cases below use this normal form.
    """
    x = "a[1] if g == 'T' or g == 'S' else -a[0] if g == 'T*T' or g == 'S*T' else -a[1] if g == 'T*T*T' or g == 'T*T*S' else a[0]"
    y = "a[0] if g == 'S' or g == 'T*T*T' else -a[0] if g == 'T' or g == 'T*T*S' else -a[1] if g == 'T*T' or g == 'T*S' else a[1]"
    return f"(({x}) % {modulus}, ({y}) % {modulus})"


def zn_gauge(n: int) -> dict:
    """Untwisted Z_n gauge theory, n=2,3,4, in the paper's R(a,b) gauge."""
    if type(n) is not int or n not in (2, 3, 4):
        raise ValueError("zn_gauge supports n=2,3,4; full intrinsic symmetry outside this range is not implemented")
    data = _pointed(
        f"Z{n} gauge theory", [n, n], "1",
        f"exp(2 * pi * 1j * a[1] * b[0] / {n})",
        f"exp(2 * pi * 1j * a[0] * a[1] / {n})",
    )
    if n == 2:
        permutation = "g == 'S' or g == 'T'"
        other_permutation = "h == 'S' or h == 'T'"
        data["symmetry"] = {
            "group": {"gap": "AbelianGroup([2,2])", "generator_names": ["S", "T"]},
            "rho": {"generator_images": [0, 1]},
            "action": _expression(f"(a[1], a[0]) if {permutation} else a"),
            "U_symbols": _expression(f"(-1) ** (a[1] * b[0]) if {permutation} else 1"),
            "eta_symbols": _expression(f"(-1) ** (a[0] * a[1]) if ({permutation}) and ({other_permutation}) else 1"),
        }
    else:
        odd = "g == 'T' or g == 'S' or g == 'T*T*T' or g == 'T*T*S'"
        odd_h = odd.replace("g ==", "h ==")
        data["symmetry"] = {
            "group": {"gap": _D8_GAP, "generator_names": ["T", "S"]},
            "rho": {"generator_images": [1, 0]},
            "action": _expression(_d8_action(n)),
            "U_symbols": _expression(f"exp(2 * pi * 1j * a[1] * b[0] / {n}) if {odd} else 1"),
            "eta_symbols": _expression(f"exp(2 * pi * 1j * a[0] * a[1] / {n}) if ({odd}) and ({odd_h}) else 1"),
        }
    return data


def doubled_u1(level: int) -> dict:
    """U(1)_level x U(1)_-level, level=2 or 4, with full intrinsic symmetry."""
    if type(level) is not int or level not in (2, 4):
        raise ValueError("doubled_u1 supports levels 2 and 4; full intrinsic symmetry outside this range is not implemented")
    data = _pointed(
        "Double semion" if level == 2 else f"U(1)_{level} x U(1)_{-level}", [level, level],
        f"(-1) ** (a[0] * ((b[0] + c[0]) // {level}) - a[1] * ((b[1] + c[1]) // {level}))",
        f"exp(pi * 1j * (a[0] * b[0] - a[1] * b[1]) / {level})",
        f"exp(pi * 1j * (a[0] * a[0] - a[1] * a[1]) / {level})",
    )
    if level == 2:
        data["symmetry"] = {
            "group": {"gap": "CyclicGroup(2)", "generator_names": ["S"]},
            "rho": {"generator_images": [1]},
            "action": _expression("(a[1], a[0]) if g == 'S' else a"),
            "U_symbols": _expression("1"),
            "eta_symbols": _expression("1"),
        }
    else:
        bar_u = "(-1) ** a[1] if b[1] != 0 else 1"
        s_u = "(-1) ** a[0] if b[0] != 0 else 1"
        data["symmetry"] = {
            "group": {"gap": _D8_GAP, "generator_names": ["T", "S"]},
            "rho": {"generator_images": [1, 1]},
            "action": _expression(_d8_action(level)),
            "U_symbols": _expression(
                f"({bar_u}) if g == 'T' or g == 'T*S' else "
                f"({s_u}) * ({bar_u}) if g == 'T*T' or g == 'T*T*S' else "
                f"({s_u}) if g == 'T*T*T' or g == 'S*T' else 1"
            ),
            "eta_symbols": _expression("1"),
        }
    return data


def build_abelian_examples() -> dict[str, dict]:
    """All paper cases whose full intrinsic groups are implemented here."""
    result = {f"u1_{level}.json": u1(level) for level in (2, 4, 6, 8, 10)}
    result.update({f"z{n}_gauge.json": zn_gauge(n) for n in (2, 3, 4)})
    result["double_semion.json"] = doubled_u1(2)
    result["u1_4_x_u1_minus4.json"] = doubled_u1(4)
    return result
