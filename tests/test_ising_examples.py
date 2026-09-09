"""Checks on the eight Ising theories and their distinct chiral data."""

import cmath
import math
import unittest

from umtc import UMTC, coherence_report, s_matrix
from scripts.categories_ising import build_ising_examples, ising


class IsingExamplesTests(unittest.TestCase):
    def test_all_eight_ising_variants_satisfy_every_equation(self):
        self.assertEqual(len(build_ising_examples()), 8)
        for nu in range(1, 16, 2):
            with self.subTest(nu=nu):
                category = UMTC.from_dict(ising(nu))
                report = coherence_report(category)
                self.assertTrue(report.ok, report.failures)
                self.assertEqual(category.anyons, ("1", "sigma", "psi"))
                self.assertIsNone(category.symmetry)
                self.assertAlmostEqual(category.total_quantum_dimension, 2)
                self.assertAlmostEqual(category.spin("sigma"), cmath.exp(nu * math.pi * 1j / 8))
                gauss_sum = sum(category.quantum_dimension(a)**2 * category.spin(a)
                                for a in category.anyons) / category.total_quantum_dimension
                self.assertAlmostEqual(gauss_sum, cmath.exp(nu * math.pi * 1j / 8))
                self.assertAlmostEqual(s_matrix(category)[1][1], 0)

    def test_indicator_and_braiding_sign_cannot_be_dropped(self):
        data = ising(3)
        category = UMTC.from_dict(data)
        self.assertAlmostEqual(category.F("sigma", "sigma", "sigma", "sigma", "1", "1"),
                               -1 / math.sqrt(2))
        # Change only the FS sign of the sigma^3 associator. F still unitary,
        # but the unchanged braiding no longer satisfies the hexagons.
        for record in data["F_symbols"]:
            if all(record[key] == "sigma" for key in "abcd"):
                record["matrix"] = [[-value for value in row] for row in record["matrix"]]
        corrupted = UMTC.from_dict(data)
        self.assertTrue(corrupted.check_unitarity())
        self.assertFalse(corrupted.check_hexagon())

    def test_reverse_chiral_partner_and_periodicity(self):
        for nu in (1, 3, 5, 7):
            positive, negative = UMTC.from_dict(ising(nu)), UMTC.from_dict(ising(-nu))
            self.assertEqual(ising(nu), ising(nu + 16))
            for a in positive.anyons:
                self.assertAlmostEqual(negative.spin(a), positive.spin(a).conjugate())
                for b in positive.anyons:
                    for c in positive.fusion(a, b):
                        self.assertAlmostEqual(negative.R(a, b, c), positive.R(a, b, c).conjugate())

    def test_non_odd_integer_parameters_are_rejected(self):
        for nu in (0, 2, True, 1.0, "1", None):
            with self.subTest(nu=nu), self.assertRaises(ValueError):
                ising(nu)


if __name__ == "__main__":
    unittest.main()
