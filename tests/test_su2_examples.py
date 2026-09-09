"""SU(2) example coverage and independent closed-form consistency checks."""

import cmath
import copy
import itertools
import json
import math
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch

from scripts.categories_su2 import build_su2_examples, su2
from umtc import UMTC, check_pentagon, check_symmetry, coherence_report, s_matrix


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


class SU2ExampleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = build_su2_examples()

    def test_all_committed_examples_are_regenerable_complete_tables(self):
        self.assertEqual(set(self.data), {
            "su2_k1.json", "su2_k2.json", "su2_k3.json", "su2_k4.json",
            "su2_k6.json", "su2_k_minus3.json",
        })
        for filename, expected in self.data.items():
            with self.subTest(filename=filename):
                self.assertEqual(json.loads((EXAMPLES / filename).read_text()), expected)
                self.assertEqual(expected["symbol_format"], "tables")
                for key in ("anyons", "fusion_rules", "quantum_dimensions", "topological_spins", "F_symbols", "R_symbols"):
                    self.assertIsInstance(expected[key], list)

    def test_all_examples_with_trivial_intrinsic_symmetry_pass_every_equation_without_gap(self):
        with patch("umtc.gap.subprocess.run", side_effect=AssertionError("GAP is unnecessary here")):
            for k in (1, 2, 3, 4, -3):
                with self.subTest(k=k):
                    category = UMTC.from_dict(su2(k))
                    self.assertIsNone(category.symmetry)
                    report = coherence_report(category)
                    self.assertTrue(report.ok, report.to_dict())
                    self.assertGreater(report.checked, 100)
                    self.assertLess(report.max_error, 1e-12)

    def test_modular_matrix_matches_sine_formula(self):
        # This computes S from fusion/twists, independently of the q6j sum.
        for k in (1, 2, 3, 4, -3):
            category = UMTC.from_dict(su2(k))
            level = abs(k)
            matrix = s_matrix(category)
            expected_total = math.sqrt((level + 2) / 2) / math.sin(math.pi / (level + 2))
            self.assertAlmostEqual(category.total_quantum_dimension, expected_total)
            for a, b in itertools.product(category.anyons, repeat=2):
                expected = math.sqrt(2 / (level + 2)) * math.sin(math.pi * (a + 1) * (b + 1) / (level + 2))
                with self.subTest(k=k, a=a, b=b):
                    self.assertAlmostEqual(matrix[a][b], expected, places=12)

    def test_frobenius_schur_signs_and_small_level_boundary_values(self):
        for k in (1, 2, 3, 4):
            category = UMTC.from_dict(su2(k))
            for a in category.anyons:
                # Appendix A: every object is self-dual, with indicator (-1)^a.
                self.assertAlmostEqual(category.quantum_dimension(a) * category.F(a, a, a, a, 0, 0), (-1) ** a)
                self.assertEqual(category.fusion(k, a), {k - a: 1})
        semion = UMTC.from_dict(su2(1))
        self.assertEqual(semion.F(1, 1, 1, 1, 0, 0), -1)
        self.assertEqual(semion.R(1, 1, 0), 1j)
        self.assertEqual(semion.spin(1), 1j)
        level2 = UMTC.from_dict(su2(2))
        self.assertAlmostEqual(level2.spin(1), cmath.exp(3j * math.pi / 8))
        self.assertAlmostEqual(level2.F(1, 1, 1, 1, 0, 0), -1 / math.sqrt(2))

    def test_negative_level_reverses_only_chirality(self):
        positive, negative = UMTC.from_dict(su2(3)), UMTC.from_dict(su2(-3))
        for a in positive.anyons:
            self.assertAlmostEqual(negative.spin(a), positive.spin(a).conjugate())
            self.assertEqual(negative.quantum_dimension(a), positive.quantum_dimension(a))
        for a, b, c in itertools.product(positive.anyons, repeat=3):
            if positive.N(a, b, c):
                self.assertAlmostEqual(negative.R(a, b, c), positive.R(a, b, c).conjugate())
        self.assertEqual(su2(3)["F_symbols"], su2(-3)["F_symbols"])

    def test_corrupt_associator_is_detected_by_pentagon(self):
        data = copy.deepcopy(self.data["su2_k3.json"])
        block = next(row for row in data["F_symbols"] if tuple(row[x] for x in "abcd") == (1, 1, 1, 1))
        block["matrix"][0][0] *= -1
        self.assertFalse(check_pentagon(UMTC.from_dict(data)))

    def test_noninteger_and_zero_levels_are_rejected(self):
        for k in (0, True, 1.5, "3", None):
            with self.subTest(k=k), self.assertRaisesRegex(ValueError, "nonzero integer"):
                su2(k)


@unittest.skipUnless(shutil.which("gap"), "GAP is needed for the nontrivial intrinsic group")
class SU2LevelSixSymmetryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = su2(6)
        cls.category = UMTC.from_dict(cls.data)

    def test_all_pentagons_both_hexagons_and_full_symmetry_pass(self):
        report = coherence_report(self.category)
        self.assertTrue(report.ok, report.to_dict())
        self.assertGreater(report.checked, 50000)
        self.assertLess(report.max_error, 1e-12)

    def test_full_intrinsic_group_and_paper_u_eta(self):
        symmetry = self.category.symmetry
        self.assertEqual(symmetry.elements, ("1", "X"))
        self.assertEqual(symmetry.G.expression, "CyclicGroup(2)")
        self.assertEqual([symmetry.rho(g) for g in symmetry.elements], [0, 0])
        self.assertEqual([symmetry.action("X", a) for a in self.category.anyons], [0, 5, 2, 3, 4, 1, 6])
        self.assertEqual(symmetry.U("X", 2, 2, 2), -1)
        self.assertEqual(symmetry.U("X", 3, 1, 2), -1)
        self.assertEqual(symmetry.U("X", 1, 3, 2), 1)
        self.assertIsInstance(self.data["symmetry"]["U_symbols"], list)
        self.assertIsInstance(self.data["symmetry"]["eta_symbols"], list)
        for a, g, h in itertools.product(self.category.anyons, symmetry.elements, symmetry.elements):
            self.assertEqual(symmetry.eta(a, g, h), 1)

    def test_corrupt_u_and_eta_are_detected(self):
        data = copy.deepcopy(self.data)
        row = next(row for row in data["symmetry"]["U_symbols"] if
                   row["g"] == "X" and (row["a"], row["b"], row["c"]) == (2, 2, 2))
        row["matrix"] = [[1]]
        self.assertFalse(check_symmetry(UMTC.from_dict(data)))
        data = copy.deepcopy(self.data)
        row = next(row for row in data["symmetry"]["eta_symbols"] if
                   row["anyon"] == 2 and row["g"] == row["h"] == "X")
        row["value"] = -1
        self.assertFalse(check_symmetry(UMTC.from_dict(data)))


if __name__ == "__main__":
    unittest.main()
