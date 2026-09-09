"""Paper formulas, full intrinsic groups, and coherence of Abelian examples."""

import cmath
import itertools
import json
import math
from pathlib import Path
import unittest

from umtc import UMTC, check_all
from scripts.categories_abelian import build_abelian_examples, doubled_u1, u1, zn_gauge


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def linear_quadratic_symmetries(modulus, *, doubled):
    """Independently enumerate GL(2,Z_n) preserving or conjugating the twist.

    Every automorphism of the additive fusion group Z_n^2 is such a matrix.
    Integer congruences avoid numerical comparisons of root-of-unity spins.
    """
    anyons = tuple(itertools.product(range(modulus), repeat=2))
    denominator = 2 * modulus if doubled else modulus

    def q(a):
        return a[0] ** 2 - a[1] ** 2 if doubled else a[0] * a[1]

    result = set()
    for p, r, s, t in itertools.product(range(modulus), repeat=4):
        if math.gcd(p * t - r * s, modulus) != 1:
            continue
        images = tuple(((p * a[0] + r * a[1]) % modulus,
                        (s * a[0] + t * a[1]) % modulus) for a in anyons)
        for parity in (0, 1):
            if all((q(image) - (-1) ** parity * q(a)) % denominator == 0
                   for a, image in zip(anyons, images)):
                result.add((parity, images))
    return result


class AbelianExampleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.generated = build_abelian_examples()
        cls.categories = {name: UMTC.from_json(EXAMPLES / name) for name in cls.generated}

    def test_committed_examples_match_generator_and_use_compact_functions(self):
        self.assertEqual(len(self.generated), 10)
        for filename, data in self.generated.items():
            with self.subTest(filename=filename):
                self.assertEqual(json.loads((EXAMPLES / filename).read_text()), data)
                self.assertEqual(data["symbol_format"], "functions")
                self.assertEqual(data["anyons"]["type"], "tuples")
                for field in ("fusion_rules", "quantum_dimensions", "topological_spins", "F_symbols", "R_symbols"):
                    self.assertEqual(set(data[field]), {"expression"})
                if data["symmetry"] is not None:
                    for field in ("action", "U_symbols", "eta_symbols"):
                        self.assertEqual(set(data["symmetry"][field]), {"expression"})

    def test_every_abelian_example_satisfies_all_equations(self):
        for filename, category in self.categories.items():
            with self.subTest(filename=filename):
                self.assertIs(check_all(category), True)

    def test_u1_gauge_and_charge_conjugation_match_section_vi(self):
        for level in (2, 4, 6, 8, 10):
            category = self.categories[f"u1_{level}.json"]
            with self.subTest(level=level):
                self.assertAlmostEqual(category.spin((1,)), cmath.exp(math.pi * 1j / level))
                self.assertEqual(category.F((1,), (level - 1,), (1,), (1,), (0,), (0,)), -1)
                expected = {
                    (0, tuple(((k * a[0]) % level,) for a in category.anyons))
                    for k in range(level)
                    if math.gcd(k, level) == 1 and (k * k - 1) % (2 * level) == 0
                }
                if level == 2:
                    self.assertIsNone(category.symmetry)
                    actual = {(0, category.anyons)}
                else:
                    symmetry = category.symmetry
                    actual = {(symmetry.rho(g), tuple(symmetry.action(g, a) for a in category.anyons))
                              for g in symmetry.elements}
                    self.assertEqual(symmetry.action("C", (1,)), (level - 1,))
                    self.assertEqual(symmetry.U("C", (1,), (1,), (2 % level,)), -1)
                    self.assertEqual(symmetry.eta((1,), "C", "C"), 1)
                self.assertEqual(actual, expected)

    def test_gauge_examples_use_the_papers_braiding_orientation(self):
        e, m, em = (1, 0), (0, 1), (1, 1)
        for n in (2, 3, 4):
            category = self.categories[f"z{n}_gauge.json"]
            phase = cmath.exp(2 * math.pi * 1j / n)
            with self.subTest(N=n):
                self.assertAlmostEqual(category.R(e, m, em), 1)
                self.assertAlmostEqual(category.R(m, e, em), phase)
                self.assertAlmostEqual(category.spin(em), phase)
                self.assertAlmostEqual(category.symmetry.U("S", m, e, em), phase)
                self.assertAlmostEqual(category.symmetry.eta(em, "S", "S"), phase)

    def test_intrinsic_groups_exhaust_all_linear_twist_symmetries(self):
        for filename, modulus, doubled in (
            ("z2_gauge.json", 2, False), ("z3_gauge.json", 3, False),
            ("z4_gauge.json", 4, False), ("double_semion.json", 2, True),
            ("u1_4_x_u1_minus4.json", 4, True),
        ):
            category = self.categories[filename]
            symmetry = category.symmetry
            actual = {(symmetry.rho(g), tuple(symmetry.action(g, a) for a in category.anyons))
                      for g in symmetry.elements}
            with self.subTest(filename=filename):
                self.assertEqual(actual, linear_quadratic_symmetries(modulus, doubled=doubled))
                self.assertEqual(len(actual), len(symmetry.elements))

    def test_dihedral_generators_and_noncommuting_words_have_correct_actions(self):
        for filename, modulus, s_parity in (("z3_gauge.json", 3, 0), ("z4_gauge.json", 4, 0),
                                           ("u1_4_x_u1_minus4.json", 4, 1)):
            symmetry = self.categories[filename].symmetry
            e = (1, 0)
            with self.subTest(filename=filename):
                self.assertEqual(symmetry.G.pow("T", 4), "1")
                self.assertEqual(symmetry.G.pow("S", 2), "1")
                self.assertEqual(symmetry.mul(symmetry.mul("S", "T"), "S"), symmetry.inverse("T"))
                self.assertEqual(symmetry.rho("T"), 1)
                self.assertEqual(symmetry.rho("S"), s_parity)
                self.assertEqual(symmetry.action("T", e), (0, modulus - 1))
                self.assertEqual(symmetry.action("S", e), (0, 1))
                self.assertEqual(symmetry.action("T*S", e), e)
                self.assertEqual(symmetry.action("S*T", e), (modulus - 1, 0))

    def test_doubled_theory_uses_conjugate_tensor_product_normalization(self):
        semion = self.categories["double_semion.json"]
        self.assertAlmostEqual(semion.spin((1, 0)), 1j)
        self.assertAlmostEqual(semion.spin((0, 1)), -1j)
        self.assertAlmostEqual(semion.spin((1, 1)), 1)
        self.assertEqual(semion.F((1, 0), (1, 0), (1, 0), (1, 0), (0, 0), (0, 0)), -1)
        for level, filename in ((2, "double_semion.json"), (4, "u1_4_x_u1_minus4.json")):
            doubled = self.categories[filename]
            factor = self.categories[f"u1_{level}.json"]
            for a, b in itertools.product(doubled.anyons, repeat=2):
                total = tuple((x + y) % level for x, y in zip(a, b))
                expected = (factor.R((a[0],), (b[0],), (total[0],))
                            * factor.R((a[1],), (b[1],), (total[1],)).conjugate())
                self.assertAlmostEqual(doubled.R(a, b, total), expected)

    def test_unsupported_parameters_do_not_silently_omit_intrinsic_symmetries(self):
        for factory, values in ((u1, (0, 1, 3, 12, True, 2.0)),
                                (zn_gauge, (0, 1, 5, True, 2.0)),
                                (doubled_u1, (0, 1, 6, True, 2.0))):
            for value in values:
                with self.subTest(factory=factory.__name__, value=value), self.assertRaises(ValueError):
                    factory(value)


if __name__ == "__main__":
    unittest.main()
