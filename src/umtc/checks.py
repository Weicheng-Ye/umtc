"""Numerical coherence tests in the conventions of arXiv:2210.02444, §II.

F rows label ((ab)e c)d and columns label (a (bc)f)d. R^{ab}_c
has rows in V^{ba}_c and columns in V^{ab}_c. U uses destination
anyon labels. All fusion-space indices are zero based.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from itertools import product
from typing import Any, Iterator, Mapping, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from .data import UMTC

Matrix = tuple[tuple[complex, ...], ...]
Equation = tuple[str, dict[str, Any], complex, complex]


def _adjoint(a: Matrix) -> Matrix:
    return tuple(tuple(z.conjugate() for z in col) for col in zip(*a))


def _conjugate(a: Matrix) -> Matrix:
    return tuple(tuple(z.conjugate() for z in row) for row in a)


def _mm(a: Matrix, b: Matrix) -> Matrix:
    if not a:
        return ()
    if len(a[0]) != len(b):
        raise ValueError("Incompatible matrix dimensions")
    columns = tuple(zip(*b))
    return tuple(tuple(sum(x * y for x, y in zip(row, col)) for col in columns)
                 for row in a)


def _chain(*matrices: Matrix) -> Matrix:
    result = matrices[0]
    for matrix in matrices[1:]:
        result = _mm(result, matrix)
    return result


def _identity(n: int, scale: complex = 1) -> Matrix:
    return tuple(tuple(complex(scale if i == j else 0) for j in range(n))
                 for i in range(n))


def _condition(name: str, labels: dict[str, Any], condition: bool) -> Equation:
    # Discrete prerequisites must remain exact even with a large tolerance.
    return name, labels, bool(condition), True


def _matrix_equations(name: str, labels: dict[str, Any],
                      lhs: Matrix, rhs: Matrix) -> Iterator[Equation]:
    shape_l = len(lhs), len(lhs[0]) if lhs else 0
    shape_r = len(rhs), len(rhs[0]) if rhs else 0
    if shape_l != shape_r:
        yield _condition(name + ".shape", {**labels, "lhs_shape": shape_l,
                                          "rhs_shape": shape_r}, False)
        return
    for i, (left, right) in enumerate(zip(lhs, rhs)):
        for j, (x, y) in enumerate(zip(left, right)):
            yield name, {**labels, "row": i, "column": j}, x, y


def _unitary_equations(name: str, labels: dict[str, Any],
                       matrix: Matrix) -> Iterator[Equation]:
    n = len(matrix)
    m = len(matrix[0]) if matrix else 0
    if n != m:
        yield _condition(name + ".square", {**labels, "shape": (n, m)}, False)
        return
    yield from _matrix_equations(name, labels, _mm(matrix, _adjoint(matrix)),
                                _identity(n))


def _fusion_equations(cat: UMTC) -> Iterator[Equation]:
    anyons, one = cat.anyons, cat.vacuum
    for a in anyons:
        if a == one:
            yield "dimension.vacuum", {"a": a}, cat.quantum_dimension(a), 1.0
        yield _condition("dimension.positive", {"a": a}, cat.quantum_dimension(a) > 0)
        for b in anyons:
            yield "fusion.left_unit", {"a": a, "b": b}, cat.N(one, a, b), int(a == b)
            yield "fusion.right_unit", {"a": a, "b": b}, cat.N(a, one, b), int(a == b)
            yield "dimension.fusion", {"a": a, "b": b}, (
                cat.quantum_dimension(a) * cat.quantum_dimension(b)), sum(
                    cat.N(a, b, c) * cat.quantum_dimension(c) for c in anyons)
            for c in anyons:
                yield "fusion.commutativity", {"a": a, "b": b, "c": c}, cat.N(a, b, c), cat.N(b, a, c)
        duals = [b for b in anyons if cat.N(a, b, one)]
        yield _condition("fusion.dual", {"a": a},
                         len(duals) == 1 and cat.N(a, duals[0], one) == 1)
    for a, b, c, d in product(anyons, repeat=4):
        yield "fusion.associativity", {"a": a, "b": b, "c": c, "d": d}, sum(
            cat.N(a, b, e) * cat.N(e, c, d) for e in anyons), sum(
            cat.N(b, c, f) * cat.N(a, f, d) for f in anyons)


def _unitarity_equations(cat: UMTC) -> Iterator[Equation]:
    for a, b, c, d in product(cat.anyons, repeat=4):
        left = cat.F_left_basis(a, b, c, d)
        right = cat.F_right_basis(a, b, c, d)
        if left or right:
            # A zero-row, positive-column matrix cannot encode its second
            # dimension in nested tuples; compare the fusion bases explicitly.
            yield _condition("F.square", {"a": a, "b": b, "c": c, "d": d},
                             len(left) == len(right))
            yield from _unitary_equations("F.unitarity", {"a": a, "b": b, "c": c, "d": d}, cat.F_matrix(a, b, c, d))
    for a, b, c in product(cat.anyons, repeat=3):
        if cat.N(a, b, c) or cat.N(b, a, c):
            yield _condition("R.square", {"a": a, "b": b, "c": c},
                             cat.N(a, b, c) == cat.N(b, a, c))
            yield from _unitary_equations("R.unitarity", {"a": a, "b": b, "c": c}, cat.R_matrix(a, b, c))


def _left_four(cat: UMTC, a: Any, b: Any, c: Any, d: Any, t: Any):
    for e, n_ab in cat.fusion(a, b).items():
        for f, n_ec in cat.fusion(e, c).items():
            for alpha, beta, gamma in product(range(n_ab), range(n_ec), range(cat.N(f, d, t))):
                yield e, f, alpha, beta, gamma


def _right_four(cat: UMTC, a: Any, b: Any, c: Any, d: Any, t: Any):
    for h, n_cd in cat.fusion(c, d).items():
        for i, n_bh in cat.fusion(b, h).items():
            for delta, epsilon, zeta in product(range(n_cd), range(n_bh), range(cat.N(a, i, t))):
                yield h, i, delta, epsilon, zeta


def _pentagon_equations(cat: UMTC) -> Iterator[Equation]:
    F, N = cat.F, cat.N
    for a, b, c, d, t in product(cat.anyons, repeat=5):
        left = tuple(_left_four(cat, a, b, c, d, t))
        right = tuple(_right_four(cat, a, b, c, d, t))
        if len(left) != len(right):
            yield _condition("pentagon.fusion_space", {"external": (a, b, c, d), "total": t}, False)
        for source, target in product(left, right):
            e, f, alpha, beta, gamma = source
            h, i, delta, epsilon, zeta = target
            short = sum(
                F(e, c, d, t, f, h, beta, gamma, delta, lam)
                * F(a, b, h, t, e, i, alpha, lam, epsilon, zeta)
                for lam in range(N(e, h, t)))
            long = sum(
                F(a, b, c, f, e, j, alpha, beta, eta, theta)
                * F(a, j, d, t, f, i, theta, gamma, kappa, zeta)
                * F(b, c, d, i, j, h, eta, kappa, delta, epsilon)
                for j, n_bc in cat.fusion(b, c).items()
                for eta, theta, kappa in product(range(n_bc), range(N(a, j, f)), range(N(j, d, i))))
            yield "pentagon", {"external": (a, b, c, d), "total": t,
                               "source": source, "target": target}, short, long


def _braid(cat: UMTC, a: Any, b: Any, c: Any, d: Any, kind: str) -> Matrix:
    """Row-source tree maps; the paper's R requires reversed arguments."""
    if kind == "inner_left":
        source, target = cat.F_left_basis(a, b, c, d), cat.F_left_basis(b, a, c, d)
        def entry(s, t):
            e, alpha, beta = s
            f, mu, nu = t
            return cat.R(b, a, e, alpha, mu) if e == f and beta == nu else 0j
    elif kind == "inner_right":
        source, target = cat.F_right_basis(a, b, c, d), cat.F_right_basis(a, c, b, d)
        def entry(s, t):
            e, alpha, beta = s
            f, mu, nu = t
            return cat.R(c, b, e, alpha, mu) if e == f and beta == nu else 0j
    elif kind == "outer_left":
        source, target = cat.F_left_basis(a, b, c, d), cat.F_right_basis(c, a, b, d)
        def entry(s, t):
            e, alpha, beta = s
            f, mu, nu = t
            return cat.R(c, e, d, beta, nu) if e == f and alpha == mu else 0j
    else:  # a(bc) -> (bc)a
        source, target = cat.F_right_basis(a, b, c, d), cat.F_left_basis(b, c, a, d)
        def entry(s, t):
            e, alpha, beta = s
            f, mu, nu = t
            return cat.R(e, a, d, beta, nu) if e == f and alpha == mu else 0j
    return tuple(tuple(entry(s, t) for t in target) for s in source)


def _hexagon_equations(cat: UMTC) -> Iterator[Equation]:
    # In a UMTC F^{-1}=F†. Include unitarity so a nonunitary input cannot
    # pass an equation in which an adjoint has been used as an inverse.
    yield from _unitarity_equations(cat)
    for a, b, c, d in product(cat.anyons, repeat=4):
        permutations = ((a, b, c), (b, a, c), (b, c, a), (a, c, b), (c, a, b))
        sizes = [(len(cat.F_left_basis(x, y, z, d)), len(cat.F_right_basis(x, y, z, d)))
                 for x, y, z in permutations]
        if any(x != sizes[0][0] or y != sizes[0][0] for x, y in sizes):
            yield _condition("hexagon.fusion_space", {"a": a, "b": b, "c": c, "d": d}, False)
            continue
        if not sizes[0][0]:
            continue
        labels = {"a": a, "b": b, "c": c, "d": d}
        path1 = _chain(_adjoint(cat.F_matrix(a, b, c, d)),
                       _braid(cat, a, b, c, d, "inner_left"),
                       cat.F_matrix(b, a, c, d),
                       _braid(cat, b, a, c, d, "inner_right"),
                       _adjoint(cat.F_matrix(b, c, a, d)))
        yield from _matrix_equations("hexagon_1", labels,
                                    _braid(cat, a, b, c, d, "outer_right"), path1)
        path2 = _chain(cat.F_matrix(a, b, c, d),
                       _braid(cat, a, b, c, d, "inner_right"),
                       _adjoint(cat.F_matrix(a, c, b, d)),
                       _braid(cat, a, c, b, d, "inner_left"),
                       cat.F_matrix(c, a, b, d))
        yield from _matrix_equations("hexagon_2", labels,
                                    _braid(cat, a, b, c, d, "outer_left"), path2)


def _ribbon_equations(cat: UMTC) -> Iterator[Equation]:
    yield "spin.vacuum", {}, cat.spin(cat.vacuum), 1 + 0j
    for a in cat.anyons:
        yield "spin.phase", {"a": a}, complex(abs(cat.spin(a))), 1 + 0j
        yield "spin.trace", {"a": a}, cat.quantum_dimension(a) * cat.spin(a), sum(
            cat.quantum_dimension(c) * sum(cat.R(a, a, c, mu, mu) for mu in range(cat.N(a, a, c)))
            for c in cat.anyons)
        for b in cat.anyons:
            if cat.N(a, b, cat.vacuum):
                yield "spin.dual", {"a": a, "b": b}, cat.spin(a), cat.spin(b)
            for c in cat.anyons:
                n = cat.N(a, b, c)
                if not n:
                    continue
                if n != cat.N(b, a, c):
                    yield _condition("ribbon.fusion_space", {"a": a, "b": b, "c": c}, False)
                    continue
                double = _mm(cat.R_matrix(b, a, c), cat.R_matrix(a, b, c))
                lhs = tuple(tuple(x * cat.spin(a) * cat.spin(b) for x in row) for row in double)
                yield from _matrix_equations("ribbon.balancing", {"a": a, "b": b, "c": c},
                                            lhs, _identity(n, cat.spin(c)))


def s_matrix(cat: UMTC) -> Matrix:
    """Normalized Hopf-link S matrix, Eq. (12), in ``cat.anyons`` order."""
    D = cat.total_quantum_dimension
    if not D or any(cat.spin(a) == 0 for a in cat.anyons):
        raise ValueError("S requires a nonzero total dimension and nonzero spins")
    rows = []
    for a in cat.anyons:
        row = []
        for b in cat.anyons:
            denominator = D * cat.spin(a) * cat.spin(b)
            if not denominator or not cmath.isfinite(denominator):
                raise ValueError("S denominator underflowed or is not finite")
            value = sum(cat.N(a, b, c) * cat.spin(c) * cat.quantum_dimension(c)
                        for c in cat.anyons) / denominator
            if not cmath.isfinite(value):
                raise ValueError("S matrix contains a nonfinite value")
            row.append(value)
        rows.append(tuple(row))
    return tuple(rows)


def _modularity_equations(cat: UMTC) -> Iterator[Equation]:
    try:
        matrix = s_matrix(cat)
    except (ValueError, ArithmeticError):
        yield _condition("modularity.defined", {}, False)
        return
    yield from _unitary_equations("modularity.S_unitarity", {}, matrix)


def _symmetry_action_equations(cat: UMTC) -> Iterator[Equation]:
    sym = cat.symmetry
    if sym is None:
        return
    elements, one = sym.elements, sym.identity
    yield _condition("group.identity_unitary", {}, not sym.is_antiunitary(one))
    for g in elements:
        yield _condition("group.left_unit", {"g": g}, sym.mul(one, g) == g)
        yield _condition("group.right_unit", {"g": g}, sym.mul(g, one) == g)
        inverses = [h for h in elements if sym.mul(g, h) == one and sym.mul(h, g) == one]
        yield _condition("group.inverse", {"g": g}, len(inverses) == 1)
        yield _condition("action.permutation", {"g": g},
                         {sym.act(g, a) for a in cat.anyons} == set(cat.anyons))
        yield _condition("action.vacuum", {"g": g}, sym.act(g, cat.vacuum) == cat.vacuum)
        for h in elements:
            gh = sym.mul(g, h)
            yield _condition("group.antiunitary_grading", {"g": g, "h": h},
                             sym.is_antiunitary(gh) == (sym.is_antiunitary(g) ^ sym.is_antiunitary(h)))
            for k in elements:
                yield _condition("group.associativity", {"g": g, "h": h, "k": k},
                                 sym.mul(gh, k) == sym.mul(g, sym.mul(h, k)))
            for a in cat.anyons:
                yield _condition("action.composition", {"g": g, "h": h, "a": a},
                                 sym.act(g, sym.act(h, a)) == sym.act(gh, a))
        for a in cat.anyons:
            ga = sym.act(g, a)
            spin = cat.spin(a).conjugate() if sym.is_antiunitary(g) else cat.spin(a)
            yield "action.spin", {"g": g, "a": a}, cat.spin(ga), spin
            yield "action.dimension", {"g": g, "a": a}, cat.quantum_dimension(ga), cat.quantum_dimension(a)
            for b, c in product(cat.anyons, repeat=2):
                yield "action.fusion", {"g": g, "a": a, "b": b, "c": c}, (
                    cat.N(ga, sym.act(g, b), sym.act(g, c))), cat.N(a, b, c)
    for a in cat.anyons:
        yield _condition("action.identity", {"a": a}, sym.act(one, a) == a)


def _symmetry_normalization_equations(cat: UMTC) -> Iterator[Equation]:
    sym = cat.symmetry
    if sym is None:
        return
    for g, a, b, c in product(sym.elements, cat.anyons, cat.anyons, cat.anyons):
        n = cat.N(a, b, c)
        if n:
            labels = {"g": g, "a": a, "b": b, "c": c}
            U = sym.U_matrix(g, a, b, c)
            yield from _unitary_equations("U.unitarity", labels, U)
            if g == sym.identity or a == cat.vacuum or b == cat.vacuum:
                yield from _matrix_equations("U.normalization", labels, U, _identity(n))
    for a, g, h in product(cat.anyons, sym.elements, sym.elements):
        value = sym.eta(a, g, h)
        labels = {"a": a, "g": g, "h": h}
        yield "eta.phase", labels, complex(abs(value)), 1 + 0j
        if a == cat.vacuum or g == sym.identity or h == sym.identity:
            yield "eta.normalization", labels, value, 1 + 0j


def _u_tree(cat: UMTC, g: Any, a: Any, b: Any, c: Any, d: Any, side: str) -> Matrix:
    """U blocks indexed in the original basis, mapping channels through g."""
    sym = cat.symmetry
    assert sym is not None
    ga, gb, gc, gd = (sym.act(g, x) for x in (a, b, c, d))
    basis = (cat.F_left_basis if side == "left" else cat.F_right_basis)(a, b, c, d)
    def entry(source, target):
        e, alpha, beta = source
        f, mu, nu = target
        if e != f:
            return 0j
        ge = sym.act(g, e)
        if side == "left":
            return sym.U(g, ga, gb, ge, alpha, mu) * sym.U(g, ge, gc, gd, beta, nu)
        return sym.U(g, gb, gc, ge, alpha, mu) * sym.U(g, ga, ge, gd, beta, nu)
    return tuple(tuple(entry(s, t) for t in basis) for s in basis)


def _symmetry_f_equations(cat: UMTC) -> Iterator[Equation]:
    sym = cat.symmetry
    if sym is None:
        return
    for g, a, b, c, d in product(sym.elements, cat.anyons, cat.anyons, cat.anyons, cat.anyons):
        left, right = cat.F_left_basis(a, b, c, d), cat.F_right_basis(a, b, c, d)
        if not left and not right:
            continue
        ga, gb, gc, gd = (sym.act(g, x) for x in (a, b, c, d))
        mapped = tuple(tuple(cat.F(ga, gb, gc, gd, sym.act(g, e), sym.act(g, f), alpha, beta, mu, nu)
                             for f, mu, nu in right) for e, alpha, beta in left)
        original = cat.F_matrix(a, b, c, d)
        if sym.is_antiunitary(g):
            original = _conjugate(original)
        # Cross-multiply Eq. (19); U unitarity is checked separately. This
        # avoids numerical matrix inversion without assuming scalar U.
        yield from _matrix_equations("symmetry.F", {"g": g, "a": a, "b": b, "c": c, "d": d},
                                    _mm(_u_tree(cat, g, a, b, c, d, "left"), mapped),
                                    _mm(original, _u_tree(cat, g, a, b, c, d, "right")))


def _symmetry_r_equations(cat: UMTC) -> Iterator[Equation]:
    sym = cat.symmetry
    if sym is None:
        return
    for g, a, b, c in product(sym.elements, cat.anyons, cat.anyons, cat.anyons):
        if not cat.N(a, b, c):
            continue
        ga, gb, gc = (sym.act(g, x) for x in (a, b, c))
        original = cat.R_matrix(a, b, c)
        if sym.is_antiunitary(g):
            original = _conjugate(original)
        yield from _matrix_equations("symmetry.R", {"g": g, "a": a, "b": b, "c": c},
                                    _mm(sym.U_matrix(g, gb, ga, gc), cat.R_matrix(ga, gb, gc)),
                                    _mm(original, sym.U_matrix(g, ga, gb, gc)))


def _u_consistency_equations(cat: UMTC) -> Iterator[Equation]:
    sym = cat.symmetry
    if sym is None:
        return
    for g, h in product(sym.elements, repeat=2):
        inverse = sym.inverse(g)
        for a, b, c in product(cat.anyons, repeat=3):
            if not cat.N(a, b, c):
                continue
            ia, ib, ic = (sym.act(inverse, x) for x in (a, b, c))
            uh = sym.U_matrix(h, ia, ib, ic)
            if sym.is_antiunitary(g):
                uh = _conjugate(uh)
            # Eq. (23): U_gh = κ (U_h pulled back and conjugated) U_g.
            # The order matters for fusion multiplicities greater than one.
            composed = _mm(uh, sym.U_matrix(g, a, b, c))
            phase_ab, phase_c = sym.eta(a, g, h) * sym.eta(b, g, h), sym.eta(c, g, h)
            lhs = tuple(tuple(phase_c * x for x in row) for row in sym.U_matrix(sym.mul(g, h), a, b, c))
            rhs = tuple(tuple(phase_ab * x for x in row) for row in composed)
            yield from _matrix_equations("symmetry.U_composition", {"g": g, "h": h, "a": a, "b": b, "c": c}, lhs, rhs)


def _eta_consistency_equations(cat: UMTC) -> Iterator[Equation]:
    sym = cat.symmetry
    if sym is None:
        return
    for g, h, k in product(sym.elements, repeat=3):
        inverse = sym.inverse(g)
        for a in cat.anyons:
            pulled = sym.eta(sym.act(inverse, a), h, k)
            if sym.is_antiunitary(g):
                pulled = pulled.conjugate()
            yield "symmetry.eta_cocycle", {"g": g, "h": h, "k": k, "a": a}, (
                sym.eta(a, g, h) * sym.eta(a, sym.mul(g, h), k)), (
                sym.eta(a, g, sym.mul(h, k)) * pulled)


def _symmetry_equations(cat: UMTC, part: str, atol: float, rtol: float) -> Iterator[Equation]:
    if cat.symmetry is None:
        return
    valid_action = True
    for equation in _symmetry_action_equations(cat):
        yield equation
        valid_action &= _close(equation[2], equation[3], atol, rtol)
    # Subsequent matrix maps require a fusion-preserving group action.
    if not valid_action or part == "symmetry_action":
        return
    valid_symbols = True
    for equation in _symmetry_normalization_equations(cat):
        yield equation
        valid_symbols &= _close(equation[2], equation[3], atol, rtol)
    if not valid_symbols:
        return
    for name, generator in (("symmetry_f", _symmetry_f_equations),
                            ("symmetry_r", _symmetry_r_equations),
                            ("u_consistency", _u_consistency_equations),
                            ("eta_consistency", _eta_consistency_equations)):
        if part == "symmetry" or part == name:
            yield from generator(cat)


def _close(lhs: complex, rhs: complex, atol: float, rtol: float) -> bool:
    if isinstance(lhs, bool) or isinstance(rhs, bool) or (type(lhs) is int and type(rhs) is int):
        return lhs == rhs
    return (cmath.isfinite(lhs) and cmath.isfinite(rhs)
            and cmath.isclose(lhs, rhs, abs_tol=atol, rel_tol=rtol))


@dataclass(frozen=True)
class EquationFailure:
    """One failed scalar coefficient, including the fusion-space indices."""
    equation: str
    labels: Mapping[str, Any]
    lhs: complex
    rhs: complex
    error: float


@dataclass(frozen=True)
class CheckReport:
    checked: int
    failure_count: int
    max_error: float
    failures: tuple[EquationFailure, ...]

    @property
    def ok(self) -> bool:
        return self.failure_count == 0

    def __bool__(self) -> bool:
        return self.ok

    def to_dict(self) -> dict[str, Any]:
        def finite(x):
            return x if math.isfinite(x) else None
        def number(z):
            z = complex(z)
            return {"re": finite(z.real), "im": finite(z.imag)}
        return {"ok": self.ok, "checked": self.checked,
                "failure_count": self.failure_count, "max_error": finite(self.max_error),
                "failures": [{"equation": f.equation, "labels": dict(f.labels),
                              "lhs": number(f.lhs), "rhs": number(f.rhs), "error": finite(f.error)}
                             for f in self.failures]}


_CHECKS = {"fusion": _fusion_equations, "unitarity": _unitarity_equations,
           "pentagon": _pentagon_equations, "hexagon": _hexagon_equations,
           "ribbon": _ribbon_equations, "modularity": _modularity_equations}
_SYMMETRY_CHECKS = {"symmetry", "symmetry_action", "symmetry_f", "symmetry_r",
                    "u_consistency", "eta_consistency"}
DEFAULT_CHECKS = (*_CHECKS, "symmetry")


def _equations(cat: UMTC, checks: Sequence[str] | None, atol: float, rtol: float) -> Iterator[Equation]:
    for name, value in (("atol", atol), ("rtol", rtol)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a finite nonnegative number")
    selected = DEFAULT_CHECKS if checks is None else ((checks,) if isinstance(checks, str) else tuple(checks))
    unknown = set(selected) - _CHECKS.keys() - _SYMMETRY_CHECKS
    if unknown:
        raise ValueError(f"Unknown checks: {sorted(unknown)}")
    for name in dict.fromkeys(selected):
        if name in _SYMMETRY_CHECKS:
            yield from _symmetry_equations(cat, name, atol, rtol)
        else:
            yield from _CHECKS[name](cat)


def coherence_report(cat: UMTC, checks: Sequence[str] | None = None, *,
                     atol: float = 1e-9, rtol: float = 1e-9,
                     max_failures: int | None = 20) -> CheckReport:
    """Evaluate every selected equation, retaining at most ``max_failures``.

    ``None`` retains every failure. A missing symmetry section represents the
    trivial action and has no symmetry equations to check. The default checks
    also test fusion, dimensions, unitarity, ribbon identities and modularity.
    Equality is |lhs-rhs| <= max(atol, rtol*max(|lhs|, |rhs|)).
    """
    if max_failures is not None and (type(max_failures) is not int or max_failures < 0):
        raise ValueError("max_failures must be a nonnegative integer or None")
    checked = failed = 0
    max_error = 0.0
    failures = []
    for equation, labels, lhs, rhs in _equations(cat, checks, atol, rtol):
        checked += 1
        error = abs(lhs - rhs)
        max_error = max(max_error, error if math.isfinite(error) else math.inf)
        if not _close(lhs, rhs, atol, rtol):
            failed += 1
            if max_failures is None or len(failures) < max_failures:
                failures.append(EquationFailure(equation, labels, complex(lhs), complex(rhs), error))
    return CheckReport(checked, failed, max_error, tuple(failures))


def _check(cat: UMTC, checks: Sequence[str] | None = None, *, atol: float = 1e-9, rtol: float = 1e-9) -> bool:
    return all(_close(lhs, rhs, atol, rtol) for _, _, lhs, rhs in _equations(cat, checks, atol, rtol))


def check_pentagon(cat: UMTC, *, atol: float = 1e-9, rtol: float = 1e-9) -> bool:
    """True iff all pentagon coefficients agree within tolerance."""
    return _check(cat, ("pentagon",), atol=atol, rtol=rtol)


def check_hexagon(cat: UMTC, *, atol: float = 1e-9, rtol: float = 1e-9) -> bool:
    """Check both hexagons and the unitarity used for F inverse = F adjoint."""
    return _check(cat, ("hexagon",), atol=atol, rtol=rtol)


def check_symmetry(cat: UMTC, *, atol: float = 1e-9, rtol: float = 1e-9) -> bool:
    """Check group action, U/eta normalization and Eqs. (19), (23), (24)."""
    return _check(cat, ("symmetry",), atol=atol, rtol=rtol)


def check_all(cat: UMTC, *, atol: float = 1e-9, rtol: float = 1e-9) -> bool:
    """Check every category and symmetry condition implemented here."""
    return _check(cat, atol=atol, rtol=rtol)


def check_fusion(cat: UMTC, **kwargs: Any) -> bool:
    """Check the fusion ring, duals and dimension identities."""
    return _check(cat, ("fusion",), **kwargs)


def check_unitarity(cat: UMTC, **kwargs: Any) -> bool:
    """Check every admissible F and R matrix is unitary."""
    return _check(cat, ("unitarity",), **kwargs)


def check_ribbon(cat: UMTC, **kwargs: Any) -> bool:
    """Check twist phases, duals, the R trace and balancing identities."""
    return _check(cat, ("ribbon",), **kwargs)


def check_modularity(cat: UMTC, **kwargs: Any) -> bool:
    """Check unitarity of the normalized S matrix."""
    return _check(cat, ("modularity",), **kwargs)


def check_symmetry_action(cat: UMTC, **kwargs: Any) -> bool:
    """Check the finite group, antiunitary grading and anyon action."""
    return _check(cat, ("symmetry_action",), **kwargs)


def check_symmetry_f(cat: UMTC, **kwargs: Any) -> bool:
    """Check F covariance with group-action and unitary U prerequisites."""
    return _check(cat, ("symmetry_f",), **kwargs)


def check_symmetry_r(cat: UMTC, **kwargs: Any) -> bool:
    """Check R covariance with group-action and unitary U prerequisites."""
    return _check(cat, ("symmetry_r",), **kwargs)


def check_u_consistency(cat: UMTC, **kwargs: Any) -> bool:
    """Check U composition against eta, Eq. (23), and prerequisites."""
    return _check(cat, ("u_consistency",), **kwargs)


def check_eta_consistency(cat: UMTC, **kwargs: Any) -> bool:
    """Check the twisted eta cocycle, Eq. (24), and prerequisites."""
    return _check(cat, ("eta_consistency",), **kwargs)
