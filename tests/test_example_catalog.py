"""The requested paper families are actually present as loadable JSON files."""

import json
from pathlib import Path
import unittest

from scripts.generate_examples import all_examples
from umtc import load_json


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


class ExampleCatalogTests(unittest.TestCase):
    def test_all_requested_families_are_shipped_and_match_the_generators(self):
        expected = all_examples()
        filenames = {path.name for path in EXAMPLES.glob("*.json")}
        self.assertEqual(filenames, set(expected))
        self.assertEqual(len(filenames), 26)
        for filename, data in expected.items():
            with self.subTest(filename=filename):
                # Normalize tuple labels in the legacy generator to JSON arrays.
                self.assertEqual(json.loads((EXAMPLES / filename).read_text()),
                                 json.loads(json.dumps(data)))
                category = load_json(EXAMPLES / filename)
                self.assertTrue(callable(category.F))
                self.assertTrue(callable(category.R))
                if category.symmetry is not None:
                    self.assertTrue(callable(category.symmetry.action))
                    self.assertTrue(callable(category.symmetry.rho))
        families = {
            "U(1)_2N": sum(name.startswith("u1_") and "_x_" not in name for name in filenames),
            "Ising": sum(name.startswith("ising_") for name in filenames),
            "Z_N gauge theory": sum(name.startswith("z") and name.endswith("_gauge.json") for name in filenames),
            "U(1)_2N x U(1)_-2N": sum(name == "double_semion.json" or "_x_u1_" in name for name in filenames),
            "SU(2)_k": sum(name.startswith("su2_k") for name in filenames),
        }
        self.assertEqual(families, {
            "U(1)_2N": 5, "Ising": 8, "Z_N gauge theory": 3,
            "U(1)_2N x U(1)_-2N": 2, "SU(2)_k": 6,
        })

    def test_known_overlaps_have_the_same_fusion_dimensions_and_twists(self):
        for first, second, relabel in (
            ("su2_k1.json", "u1_2.json", {0: (0,), 1: (1,)}),
            ("su2_k2.json", "ising_nu3.json", {0: "1", 1: "sigma", 2: "psi"}),
        ):
            left, right = load_json(EXAMPLES / first), load_json(EXAMPLES / second)
            for a in left.anyons:
                with self.subTest(first=first, a=a):
                    self.assertAlmostEqual(left.quantum_dimension(a), right.quantum_dimension(relabel[a]))
                    self.assertAlmostEqual(left.spin(a), right.spin(relabel[a]))
                    for b in left.anyons:
                        self.assertEqual({relabel[c]: n for c, n in left.fusion(a, b).items()},
                                         right.fusion(relabel[a], relabel[b]))


if __name__ == "__main__":
    unittest.main()
