"""GAP groups, their Z2 gradings, and callable anyon actions compose correctly."""

import itertools
import unittest

from umtc import DataError, GAPError, GAPGroup, UMTC, check_all, check_symmetry
from scripts.generate_examples import fibonacci, toric_code
from test_additional_functions import native_toric_functions


class GAPSymmetryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.group = GAPGroup.from_gap("AbelianGroup([2,2])", generator_names=["X", "T"])
        cls.rho = cls.group.homomorphism_to_z2([0, 1])

    def test_public_toric_definition_uses_gap_group_grading_and_action_formula(self):
        data = toric_code()
        symmetry = data["symmetry"]
        self.assertEqual(symmetry["group"], {
            "gap": "AbelianGroup([2,2])", "generator_names": ["X", "T"]
        })
        self.assertEqual(symmetry["rho"], {"generator_images": [0, 1]})
        self.assertEqual(set(symmetry["action"]), {"expression"})
        for field in ("elements", "identity", "multiplication_table", "antiunitary"):
            self.assertNotIn(field, symmetry)
        category = UMTC.from_dict(data)
        self.assertIsInstance(category.symmetry.G, GAPGroup)
        self.assertEqual(category.symmetry.elements, ("1", "X", "T", "X*T"))
        self.assertIs(check_all(category), True)

    def test_gap_native_and_legacy_symmetry_definitions_agree(self):
        legacy = UMTC.from_dict(toric_code(symbol_format="tables"))
        formula = UMTC.from_dict(toric_code())
        data = toric_code()
        data["symmetry"]["group"] = self.group
        data["symmetry"]["rho"] = self.rho
        del data["symmetry"]["action"]
        native = UMTC.from_functions(data, **native_toric_functions())
        self.assertNotIn("action", data["symmetry"])
        for category in (legacy, formula, native):
            symmetry = category.symmetry
            self.assertTrue(callable(symmetry.action))
            self.assertTrue(callable(symmetry.rho))
            for g, a in itertools.product(legacy.symmetry.elements, legacy.anyons):
                self.assertEqual(symmetry.action(g, a), legacy.symmetry.act(g, a))
                self.assertEqual(symmetry.act(g, a), symmetry.action(g, a))
                self.assertEqual(symmetry.rho(g), int(legacy.symmetry.is_antiunitary(g)))
            self.assertIs(check_all(category), True)

    def test_rho_is_the_declared_homomorphism_and_controls_antiunitarity(self):
        symmetry = UMTC.from_dict(toric_code()).symmetry
        self.assertEqual([symmetry.rho(g) for g in symmetry.elements], [0, 0, 1, 1])
        for g, h in itertools.product(symmetry.elements, repeat=2):
            self.assertEqual(symmetry.rho(symmetry.mul(g, h)), (symmetry.rho(g) + symmetry.rho(h)) % 2)
            self.assertIs(symmetry.is_antiunitary(g), bool(symmetry.rho(g)))
        e, m = (1, 0), (0, 1)
        self.assertEqual(symmetry.action("X", e), m)
        self.assertEqual(symmetry.action("T", e), m)
        self.assertEqual(symmetry.action("X*T", e), e)

    def test_invalid_generator_gradings_are_rejected_as_data_errors(self):
        for images in ([0], [0, 2], [False, 1], [0, 1, 0]):
            data = toric_code()
            data["symmetry"]["rho"] = {"generator_images": images}
            with self.subTest(images=images), self.assertRaises(DataError):
                UMTC.from_dict(data)
        # A generator of odd order cannot map to the nontrivial Z2 element.
        data = toric_code()
        data["symmetry"]["group"] = {"gap": "CyclicGroup(3)", "generator_names": ["C"]}
        data["symmetry"]["rho"] = {"generator_images": [1]}
        with self.assertRaises(DataError):
            UMTC.from_dict(data)

    def test_gap_group_metadata_rejects_conflicts_and_a_different_rho_source(self):
        for field, value in (("elements", ["1"]), ("identity", "1"),
                             ("multiplication_table", [["1"]]), ("antiunitary", [])):
            data = toric_code()
            data["symmetry"][field] = value
            with self.subTest(field=field), self.assertRaises(DataError):
                UMTC.from_dict(data)
        data = toric_code()
        data["symmetry"]["group"] = GAPGroup.from_gap("CyclicGroup(2)", generator_names=["S"])
        data["symmetry"]["rho"] = self.rho
        with self.assertRaises(DataError):
            UMTC.from_dict(data)

    def test_native_action_is_lazy_and_normalizes_tuple_labels(self):
        calls = []

        def action(g, a):
            self.assertIsInstance(a, tuple)
            calls.append((g, a))
            return [a[1], a[0]] if g in ("X", "T") else a

        category = UMTC.from_functions(toric_code(), action=action)
        self.assertEqual(calls, [])
        self.assertEqual(category.symmetry.action("X", [1, 0]), (0, 1))
        self.assertEqual(calls, [("X", (1, 0))])

    def test_invalid_action_results_are_rejected(self):
        for result in ((2, 0), True, 1j, (0.0, 1), "unknown", {"anyon": (0, 0)}):
            with self.subTest(result=result), self.assertRaises(DataError):
                category = UMTC.from_functions(toric_code(), action=lambda g, a: result)
                category.symmetry.action("X", (1, 0))
        for expression in ("(2, 0)", "True", "1j", "(0.0, 1)", "'unknown'"):
            data = toric_code()
            data["symmetry"]["action"] = {"expression": expression}
            with self.subTest(expression=expression), self.assertRaises(DataError):
                category = UMTC.from_dict(data)
                category.symmetry.action("X", (1, 0))

    def test_unknown_action_arguments_do_not_reach_the_callback(self):
        def forbidden(g, a):
            self.fail("Unknown labels must be rejected before invoking the action")

        symmetry = UMTC.from_functions(toric_code(), action=forbidden).symmetry
        for g, a in (("missing", (1, 0)), ("X", (2, 0)), (True, (1, 0))):
            with self.subTest(g=g, a=a), self.assertRaises(DataError):
                symmetry.action(g, a)
        with self.assertRaises(GAPError):
            symmetry.rho("missing")
        with self.assertRaises(DataError):
            symmetry.is_antiunitary("missing")

    def test_inconsistent_action_on_valid_labels_fails_the_symmetry_equations(self):
        category = UMTC.from_functions(
            toric_code(),
            action=lambda g, a: (a[1], a[0] ^ a[1]) if g == "X" else a,
        )
        self.assertIs(check_symmetry(category), False)

    def test_action_formulas_work_with_listed_anyons_and_table_symbols(self):
        data = fibonacci()
        elements = ["1", "S"]
        data["symmetry"] = {
            "group": {"gap": "CyclicGroup(2)", "generator_names": ["S"]},
            "rho": {"generator_images": [0]},
            "action": {"expression": "a"},
            "U_symbols": [
                {"g": g, "a": row["a"], "b": row["b"], "c": row["c"], "matrix": [[1]]}
                for g in elements for row in data["fusion_rules"]
            ],
            "eta_symbols": [
                {"anyon": a, "g": g, "h": h, "value": 1}
                for a, g, h in itertools.product(data["anyons"], elements, elements)
            ],
        }
        formula = UMTC.from_dict(data)
        native = UMTC.from_functions(data, action=lambda g, a: a)
        self.assertEqual(formula.symbol_format, "tables")
        for category in (formula, native):
            for g, a in itertools.product(elements, data["anyons"]):
                self.assertEqual(category.symmetry.action(g, a), a)
            self.assertIs(check_all(category), True)

    def test_action_definition_and_callback_signature_are_validated(self):
        for definition in ({}, {"expression": 0}, {"expression": "a", "default": "a"},
                           {"expression": "a.__class__"}):
            data = toric_code()
            data["symmetry"]["action"] = definition
            with self.subTest(definition=definition), self.assertRaises(DataError):
                UMTC.from_dict(data)
        for callback in (1, lambda g: g):
            with self.subTest(callback=callback), self.assertRaises(DataError):
                UMTC.from_functions(toric_code(), action=callback)


if __name__ == "__main__":
    unittest.main()
