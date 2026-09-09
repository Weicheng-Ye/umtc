"""Mathematical regressions using complete data, corruptions, and gauge changes."""

from __future__ import annotations

import cmath
import copy
import itertools
import json
import math
from pathlib import Path
import unittest

from umtc import (
    UMTC,
    check_all,
    check_fusion,
    check_hexagon,
    check_modularity,
    check_pentagon,
    check_ribbon,
    check_symmetry,
    check_unitarity,
    coherence_report,
)


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def example_data(name):
    return json.loads((EXAMPLES / name).read_text(encoding="utf-8"))


def number(value):
    return complex(value["re"], value["im"]) if isinstance(value, dict) else complex(value)


def encoded(value):
    value = complex(value)
    return {"re": value.real, "im": value.imag}


def materialize_symbols(data):
    """Build a table fixture when a regression needs to edit individual entries."""
    data = copy.deepcopy(data)
    category = UMTC.from_dict(data)
    data["symbol_format"] = "tables"
    channels = [
        (a, b, c) for a, b, c in itertools.product(category.anyons, repeat=3)
        if category.N(a, b, c)
    ]
    data["fusion_rules"] = [
        {"a": a, "b": b, "c": c, "multiplicity": category.N(a, b, c)}
        for a, b, c in channels
    ]
    data["quantum_dimensions"] = [
        {"anyon": a, "value": category.quantum_dimension(a)} for a in category.anyons
    ]
    data["topological_spins"] = [
        {"anyon": a, "value": encoded(category.spin(a))} for a in category.anyons
    ]
    data["F_symbols"] = [
        {"a": a, "b": b, "c": c, "d": d,
         "matrix": [[encoded(value) for value in row]
                    for row in category.F_matrix(a, b, c, d)]}
        for a, b, c, d in itertools.product(category.anyons, repeat=4)
        if category.F_left_basis(a, b, c, d)
    ]
    data["R_symbols"] = [
        {"a": a, "b": b, "c": c,
         "matrix": [[encoded(value) for value in row]
                    for row in category.R_matrix(a, b, c)]}
        for a, b, c in channels
    ]
    if category.symmetry is not None:
        symmetry = category.symmetry
        data["symmetry"].pop("group", None)
        data["symmetry"].pop("rho", None)
        data["symmetry"].update({
            "elements": list(symmetry.elements),
            "identity": symmetry.identity,
            "multiplication_table": [list(row) for row in symmetry.multiplication_table],
            "antiunitary": [g for g in symmetry.elements if symmetry.is_antiunitary(g)],
            "action": [
                {"g": g, "images": [symmetry.action(g, a) for a in category.anyons]}
                for g in symmetry.elements
            ],
        })
        data["symmetry"]["U_symbols"] = [
            {"g": g, "a": a, "b": b, "c": c,
             "matrix": [[encoded(value) for value in row]
                        for row in symmetry.U_matrix(g, a, b, c)]}
            for g in symmetry.elements for a, b, c in channels
        ]
        data["symmetry"]["eta_symbols"] = [
            {"anyon": a, "g": g, "h": h, "value": encoded(symmetry.eta(a, g, h))}
            for a, g, h in itertools.product(category.anyons, symmetry.elements, symmetry.elements)
        ]
    # These fixtures model JSON inputs, where tuple labels are encoded as lists.
    return json.loads(json.dumps(data))


def gauge_toric(data, *, conjugate_antiunitary=True, inverse_action=True):
    """Apply Eqs. (10),(18), with U indexed by destination anyons.

    This helper also accepts the three-fermion fixture with the same anyon
    labels. Keeping eta unchanged under a vertex basis change follows from
    its action on the anyon line.
    """
    data = materialize_symbols(data)
    anyons = list(itertools.product(range(2), repeat=2))
    vacuum = (0, 0)
    action = {row["g"]: [tuple(a) for a in row["images"]] for row in data["symmetry"]["action"]}

    def gamma(a, b):
        a, b = tuple(a), tuple(b)
        if a == vacuum or b == vacuum:
            return 1
        i, j = anyons.index(a), anyons.index(b)
        return cmath.exp(1j * (0.17 * i * j + 0.11 * i * i * j))

    def fuse(a, b):
        return tuple(x ^ y for x, y in zip(a, b))

    for row in data["F_symbols"]:
        a, b, c = (row[key] for key in ("a", "b", "c"))
        factor = gamma(a, b) * gamma(fuse(a, b), c) / (gamma(b, c) * gamma(a, fuse(b, c)))
        row["matrix"][0][0] = encoded(factor * number(row["matrix"][0][0]))
    for row in data["R_symbols"]:
        a, b = row["a"], row["b"]
        row["matrix"][0][0] = encoded(gamma(b, a) / gamma(a, b) * number(row["matrix"][0][0]))
    for row in data["symmetry"]["U_symbols"]:
        g, a, b = row["g"], row["a"], row["b"]
        if inverse_action:
            inverse_a = anyons[action[g].index(tuple(a))]
            inverse_b = anyons[action[g].index(tuple(b))]
        else:
            inverse_a = action[g][anyons.index(tuple(a))]
            inverse_b = action[g][anyons.index(tuple(b))]
        factor = gamma(inverse_a, inverse_b)
        if conjugate_antiunitary and g in data["symmetry"]["antiunitary"]:
            factor = factor.conjugate()
        row["matrix"][0][0] = encoded(factor / gamma(a, b) * number(row["matrix"][0][0]))
    return data


def three_fermion_z3_data():
    """A four-anyon UMTC with an order-three permutation of its fermions."""
    data = materialize_symbols(example_data("toric_code_z2_z2t.json"))
    data["name"] = "Three-fermion UMTC with Z3 permutation symmetry"
    anyons = list(itertools.product(range(2), repeat=2))
    groups = ["1", "C", "C2"]

    def act(g, a):
        x, y = a
        return [(x, y), (y, x ^ y), (x ^ y, x)][groups.index(g)]

    for row in data["R_symbols"]:
        a, b = row["a"], row["b"]
        row["matrix"] = [[(-1) ** (a[0] * b[0] + a[1] * b[1] + a[0] * b[1])]]
    for row in data["topological_spins"]:
        row["value"] = 1 if row["anyon"] == [0, 0] else -1
    data["symmetry"] = {
        "elements": groups,
        "identity": "1",
        "multiplication_table": [[groups[(i + j) % 3] for j in range(3)] for i in range(3)],
        "antiunitary": [],
        "action": [{"g": g, "images": [act(g, a) for a in anyons]} for g in groups],
        "U_symbols": [
            {"g": g, "a": row["a"], "b": row["b"], "c": row["c"], "matrix": [[1]]}
            for g in groups for row in data["fusion_rules"]
        ],
        "eta_symbols": [{"anyon": a, "g": g, "h": h, "value": 1} for a, g, h in itertools.product(anyons, groups, groups)],
    }
    return data


def multiplicity_two_data():
    """Synthetic rank-two fusion data to exercise every multiplicity index.

    The fusion ring x*x=1+2*x is associative. Identity associators are unitary
    matrices but do not satisfy its pentagon; this is deliberately not a UMTC.
    """
    anyons = ["1", "x"]

    def n(a, b, c):
        if a == "1":
            return int(b == c)
        if b == "1":
            return int(a == c)
        return 1 if c == "1" else 2

    def identity(size):
        return [[int(i == j) for j in range(size)] for i in range(size)]

    f_symbols = []
    for a, b, c, d in itertools.product(anyons, repeat=4):
        size = sum(n(a, b, e) * n(e, c, d) for e in anyons)
        if size:
            f_symbols.append({"a": a, "b": b, "c": c, "d": d, "matrix": identity(size)})
    return {
        "schema_version": 1,
        "name": "Associative fusion ring with inconsistent identity associators",
        "anyons": anyons,
        "vacuum": "1",
        "fusion_rules": [
            {"a": a, "b": b, "c": c, "multiplicity": n(a, b, c)}
            for a, b, c in itertools.product(anyons, repeat=3) if n(a, b, c)
        ],
        "quantum_dimensions": [{"anyon": "1", "value": 1}, {"anyon": "x", "value": 1 + math.sqrt(2)}],
        "topological_spins": [{"anyon": a, "value": 1} for a in anyons],
        "F_symbols": f_symbols,
        "R_symbols": [
            {"a": a, "b": b, "c": c, "matrix": identity(n(a, b, c))}
            for a, b, c in itertools.product(anyons, repeat=3) if n(a, b, c)
        ],
    }


class ConsistencyTests(unittest.TestCase):
    def setUp(self):
        self.toric_data = materialize_symbols(example_data("toric_code_z2_z2t.json"))
        self.fibonacci_data = example_data("fibonacci.json")

    def test_complete_examples_satisfy_every_check(self):
        for filename in ("toric_code_z2_z2t.json", "fibonacci.json"):
            category = UMTC.from_json(EXAMPLES / filename)
            for checker in (
                check_fusion, check_unitarity, check_pentagon, check_hexagon,
                check_ribbon, check_modularity, check_symmetry, check_all,
            ):
                with self.subTest(filename=filename, check=checker.__name__):
                    self.assertIs(checker(category), True)

    def test_toric_code_functions_and_symmetry(self):
        category = UMTC.from_dict(self.toric_data)
        vacuum, e, m, psi = (0, 0), (1, 0), (0, 1), (1, 1)
        self.assertEqual(category.N(e, m, psi), 1)
        self.assertEqual(category.N(e, e, vacuum), 1)
        self.assertEqual(category.N(e, m, vacuum), 0)
        self.assertEqual(category.quantum_dimension(e), 1)
        self.assertEqual(category.spin(psi), -1)
        self.assertEqual(category.R(e, m, psi), -1)
        self.assertEqual(category.R(m, e, psi), 1)
        symmetry = category.symmetry
        for g in ("X", "T"):
            self.assertEqual(symmetry.act(g, e), m)
            self.assertEqual(symmetry.act(g, m), e)
            self.assertEqual(symmetry.act(g, psi), psi)
        self.assertFalse(symmetry.is_antiunitary("X"))
        self.assertTrue(symmetry.is_antiunitary("T"))
        self.assertTrue(symmetry.is_antiunitary("X*T"))
        self.assertEqual(symmetry.act("X*T", e), e)
        self.assertEqual(symmetry.U("X", e, m, psi), -1)
        self.assertEqual(symmetry.eta(psi, "X", "T"), -1)

    def test_fibonacci_functions(self):
        category = UMTC.from_dict(self.fibonacci_data)
        phi = (1 + math.sqrt(5)) / 2
        self.assertIsNone(category.symmetry)
        self.assertAlmostEqual(category.quantum_dimension("tau"), phi)
        self.assertAlmostEqual(category.F("tau", "tau", "tau", "tau", "1", "tau"), 1 / math.sqrt(phi))
        self.assertAlmostEqual(category.F("tau", "tau", "tau", "tau", "tau", "tau"), -1 / phi)
        self.assertAlmostEqual(category.R("tau", "tau", "1"), cmath.exp(-4j * math.pi / 5))

    def test_corrupt_f_symbol_fails_pentagon_and_unitarity(self):
        row = next(row for row in self.fibonacci_data["F_symbols"] if all(row[k] == "tau" for k in "abcd"))
        row["matrix"][0][0] = 0
        category = UMTC.from_dict(self.fibonacci_data)
        self.assertIs(check_pentagon(category), False)
        self.assertIs(check_unitarity(category), False)
        self.assertIs(check_all(category), False)

    def test_corrupt_r_symbol_fails_hexagon(self):
        row = next(row for row in self.fibonacci_data["R_symbols"] if row["a"] == row["b"] == "tau" and row["c"] == "1")
        row["matrix"][0][0] = 1
        self.assertIs(check_hexagon(UMTC.from_dict(self.fibonacci_data)), False)

    def test_corrupt_u_symbol_fails_symmetry(self):
        row = next(row for row in self.toric_data["symmetry"]["U_symbols"] if row["g"] == "X" and row["a"] == row["b"] == [1, 0])
        row["matrix"][0][0] = {"re": 0, "im": 1}
        self.assertIs(check_symmetry(UMTC.from_dict(self.toric_data)), False)

    def test_corrupt_eta_symbol_fails_symmetry(self):
        row = next(row for row in self.toric_data["symmetry"]["eta_symbols"] if row["anyon"] == [1, 1] and row["g"] == row["h"] == "X")
        row["value"] = 1
        self.assertIs(check_symmetry(UMTC.from_dict(self.toric_data)), False)

    def test_corrupt_spin_fails_ribbon(self):
        self.fibonacci_data["topological_spins"][1]["value"] = 1
        self.assertIs(check_ribbon(UMTC.from_dict(self.fibonacci_data)), False)

    def test_corrupt_dimension_fails_fusion(self):
        self.fibonacci_data["quantum_dimensions"][1]["value"] = 2
        self.assertIs(check_fusion(UMTC.from_dict(self.fibonacci_data)), False)

    def test_nonreal_vertex_gauge_preserves_every_equation(self):
        data = gauge_toric(self.toric_data)
        self.assertTrue(any(abs(number(row["matrix"][0][0]).imag) > 0.1 for row in data["F_symbols"]))
        self.assertTrue(any(abs(number(row["matrix"][0][0]).imag) > 0.1 for row in data["symmetry"]["U_symbols"]))
        self.assertIs(check_all(UMTC.from_dict(data)), True)

    def test_missing_antiunitary_conjugation_is_detected(self):
        data = gauge_toric(self.toric_data, conjugate_antiunitary=False)
        category = UMTC.from_dict(data)
        self.assertIs(check_pentagon(category), True)
        self.assertIs(check_hexagon(category), True)
        self.assertIs(check_symmetry(category), False)

    def test_non_involutive_symmetry_uses_inverse_destination_action(self):
        data = three_fermion_z3_data()
        self.assertIs(check_all(UMTC.from_dict(data)), True)
        self.assertIs(check_all(UMTC.from_dict(gauge_toric(data))), True)
        self.assertIs(check_symmetry(UMTC.from_dict(gauge_toric(data, inverse_action=False))), False)

    def test_multiplicity_indices_are_not_ignored(self):
        category = UMTC.from_dict(multiplicity_two_data())
        self.assertEqual(category.N("x", "x", "x"), 2)
        self.assertIs(check_fusion(category), True)
        self.assertIs(check_unitarity(category), True)
        self.assertEqual(category.R("x", "x", "x", mu=1, nu=1), 1)
        self.assertEqual(category.R("x", "x", "x", mu=0, nu=1), 0)
        self.assertEqual(category.F("x", "x", "x", "x", "x", "x", alpha=1, beta=1, mu=1, nu=1), 1)
        self.assertIs(check_pentagon(category), False)

    def test_diagnostics_identify_failed_equations(self):
        row = next(row for row in self.fibonacci_data["F_symbols"] if all(row[k] == "tau" for k in "abcd"))
        row["matrix"][0][0] = 0
        report = coherence_report(UMTC.from_dict(self.fibonacci_data), checks=["pentagon"])
        self.assertFalse(report.ok)
        self.assertGreater(report.checked, 0)
        self.assertGreater(report.failure_count, 0)
        self.assertGreater(report.max_error, 0)
        self.assertTrue(report.failures)
        self.assertTrue(report.failures[0].labels)


if __name__ == "__main__":
    unittest.main()
