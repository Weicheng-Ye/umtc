"""Uniform callable fusion, dimensions, spins, and symmetry symbols."""

import copy
import itertools
import math
import unittest

from umtc import DataError, UMTC, check_all, check_fusion, check_ribbon, check_symmetry
from scripts.generate_examples import fibonacci, toric_code
from test_multiplicity import _a4_data, _a4_symmetry_data


def native_toric_functions():
    return {
        "N": lambda a, b, c: int(c == tuple((x + y) % 2 for x, y in zip(a, b))),
        "quantum_dimension": lambda a: 1,
        "spin": lambda a: (-1) ** (a[0] * a[1]),
        "F": lambda a, b, c, d, e, f: 1,
        "R": lambda a, b, c: (-1) ** (a[0] * b[1]),
        "U": lambda g, a, b, c: (-1) ** (a[0] * b[1]) if g in ("X", "T") else 1,
        "eta": lambda a, g, h: (-1) ** (a[0] * a[1]) if g in ("X", "T") and h in ("X", "T") else 1,
        "action": lambda g, a: a[::-1] if g in ("X", "T") else a,
    }


class AdditionalFunctionTests(unittest.TestCase):
    def assert_same_data(self, expected, actual):
        for name in ("fusion_rules", "quantum_dimensions", "topological_spins"):
            self.assertTrue(callable(getattr(actual, name)))
        self.assertEqual(actual.total_quantum_dimension, expected.total_quantum_dimension)
        for a in expected.anyons:
            self.assertEqual(actual.quantum_dimension(a), expected.quantum_dimension(a))
            self.assertEqual(actual.quantum_dimensions(a), expected.quantum_dimension(a))
            self.assertEqual(actual.spin(a), expected.spin(a))
            self.assertEqual(actual.topological_spins(a), expected.spin(a))
        for a, b, c in itertools.product(expected.anyons, repeat=3):
            self.assertEqual(actual.N(a, b, c), expected.N(a, b, c))
            self.assertEqual(actual.fusion_rules(a, b, c), expected.N(a, b, c))
            self.assertEqual(actual.fusion(a, b), expected.fusion(a, b))
        if expected.symmetry is None:
            self.assertIsNone(actual.symmetry)
            return
        target, result = expected.symmetry, actual.symmetry
        self.assertTrue(callable(result.U_symbols))
        self.assertTrue(callable(result.eta_symbols))
        self.assertTrue(callable(result.action))
        self.assertTrue(callable(result.rho))
        for g, a in itertools.product(target.elements, expected.anyons):
            self.assertEqual(result.action(g, a), target.action(g, a))
            self.assertEqual(result.act(g, a), target.action(g, a))
            self.assertEqual(result.rho(g), target.rho(g))
        for g, a, b, c in itertools.product(target.elements, expected.anyons, expected.anyons, expected.anyons):
            self.assertEqual(result.U_matrix(g, a, b, c), target.U_matrix(g, a, b, c))
            for mu, nu in itertools.product(range(expected.N(a, b, c)), repeat=2):
                self.assertEqual(result.U(g, a, b, c, mu, nu), target.U(g, a, b, c, mu, nu))
                self.assertEqual(result.U_symbols(g, a, b, c, mu, nu), target.U(g, a, b, c, mu, nu))
        for a, g, h in itertools.product(expected.anyons, target.elements, target.elements):
            self.assertEqual(result.eta(a, g, h), target.eta(a, g, h))
            self.assertEqual(result.eta_symbols(a, g, h), target.eta(a, g, h))

    def test_table_formula_and_native_toric_data_agree_and_pass_all_checks(self):
        table = UMTC.from_dict(toric_code(symbol_format="tables"))
        formula = UMTC.from_dict(toric_code())
        native = UMTC.from_functions(toric_code(), **native_toric_functions())
        for category in (table, formula, native):
            self.assert_same_data(table, category)
            self.assertIs(check_all(category), True)

    def test_listed_fibonacci_data_have_the_same_callable_interface(self):
        category = UMTC.from_dict(fibonacci())
        self.assert_same_data(category, category)
        self.assertEqual(category.fusion_rules("tau", "tau", "tau"), 1)
        self.assertAlmostEqual(category.quantum_dimensions("tau"), (1 + math.sqrt(5)) / 2)

    def test_native_definitions_replace_omitted_data_without_mutating_input(self):
        data = toric_code()
        for key in ("F_symbols", "R_symbols", "fusion_rules", "quantum_dimensions", "topological_spins"):
            data.pop(key)
        for key in ("U_symbols", "eta_symbols", "action"):
            data["symmetry"].pop(key)
        original = copy.deepcopy(data)
        category = UMTC.from_functions(data, **native_toric_functions())
        self.assertEqual(data, original)
        self.assert_same_data(UMTC.from_dict(toric_code(symbol_format="tables")), category)
        self.assertIs(check_all(category), True)

    def test_individual_native_overrides_preserve_existing_definitions(self):
        reference = UMTC.from_dict(toric_code(symbol_format="tables"))
        for name, function in native_toric_functions().items():
            with self.subTest(symbol=name):
                category = UMTC.from_functions(toric_code(), **{name: function})
                self.assert_same_data(reference, category)
        self.assert_same_data(reference, UMTC.from_functions(toric_code()))

    def test_legacy_function_mode_accepts_explicit_tables_for_added_fields(self):
        data = toric_code(symbol_format="tables")
        formulas = toric_code()
        data.update({name: formulas[name] for name in ("symbol_format", "F_symbols", "R_symbols")})
        self.assert_same_data(UMTC.from_dict(formulas), UMTC.from_dict(data))
        self.assertIs(check_all(UMTC.from_dict(data)), True)

    def test_table_mode_requires_added_fields_to_be_tables(self):
        for name in ("fusion_rules", "quantum_dimensions", "topological_spins", "U_symbols", "eta_symbols"):
            data = toric_code(symbol_format="tables")
            owner = data["symmetry"] if name in ("U_symbols", "eta_symbols") else data
            owner[name] = {"expression": "1"}
            with self.subTest(symbol=name), self.assertRaises(DataError):
                UMTC.from_dict(data)

    def test_fusion_is_evaluated_once_per_channel_and_uses_canonical_tuples(self):
        calls = []

        def fusion(a, b, c):
            self.assertTrue(all(isinstance(x, tuple) for x in (a, b, c)))
            calls.append((a, b, c))
            return float(c == tuple((x + y) % 2 for x, y in zip(a, b)))

        category = UMTC.from_functions(toric_code(), N=fusion)
        self.assertEqual(set(calls), set(itertools.product(category.anyons, repeat=3)))
        self.assertEqual(len(calls), len(category.anyons) ** 3)
        count = len(calls)
        self.assertEqual(category.N([1, 0], [0, 1], [1, 1]), 1)
        self.assertEqual(category.fusion([1, 0], [0, 1]), {(1, 1): 1})
        self.assertIs(check_fusion(category), True)
        self.assertEqual(len(calls), count)

    def test_other_native_functions_are_lazy_and_receive_canonical_labels(self):
        calls = []

        def dimension(a):
            calls.append(("dimension", a))
            self.assertIsInstance(a, tuple)
            return 1

        def spin(a):
            calls.append(("spin", a))
            self.assertIsInstance(a, tuple)
            return -1

        def u_symbol(g, a, b, c):
            calls.append(("U", g, a, b, c))
            self.assertTrue(all(isinstance(x, tuple) for x in (a, b, c)))
            return -1

        def eta(a, g, h):
            calls.append(("eta", a, g, h))
            self.assertIsInstance(a, tuple)
            return -1

        category = UMTC.from_functions(toric_code(), quantum_dimension=dimension, spin=spin, U=u_symbol, eta=eta)
        self.assertEqual(calls, [])
        category.quantum_dimension([1, 0])
        category.spin([1, 1])
        category.symmetry.U("X", [1, 0], [0, 1], [1, 1])
        category.symmetry.eta([1, 1], "X", "T")
        self.assertEqual([call[0] for call in calls], ["dimension", "spin", "U", "eta"])

    def test_formula_evaluation_errors_are_lazy_except_fusion(self):
        for name in ("quantum_dimensions", "topological_spins", "U_symbols", "eta_symbols"):
            data = toric_code()
            owner = data["symmetry"] if name in ("U_symbols", "eta_symbols") else data
            owner[name] = {"expression": "1 / 0"}
            category = UMTC.from_dict(data)
            vacuum = (0, 0)
            callbacks = {
                "quantum_dimensions": lambda: category.quantum_dimension(vacuum),
                "topological_spins": lambda: category.spin(vacuum),
                "U_symbols": lambda: category.symmetry.U("1", vacuum, vacuum, vacuum),
                "eta_symbols": lambda: category.symmetry.eta(vacuum, "1", "1"),
            }
            with self.subTest(symbol=name), self.assertRaisesRegex(DataError, name):
                callbacks[name]()
        data = toric_code()
        data["fusion_rules"] = {"expression": "1 / 0"}
        with self.assertRaisesRegex(DataError, "fusion_rules"):
            UMTC.from_dict(data)

    def test_u_selection_rules_do_not_invoke_functions_and_unknown_labels_raise(self):
        def forbidden(*args):
            self.fail("An inadmissible entry reached the native function")

        category = UMTC.from_functions(toric_code(), quantum_dimension=forbidden, spin=forbidden, U=forbidden, eta=forbidden)
        vacuum, e, m, psi = (0, 0), (1, 0), (0, 1), (1, 1)
        self.assertEqual(category.symmetry.U("X", e, m, vacuum), 0)
        self.assertEqual(category.symmetry.U_matrix("X", e, m, vacuum), ())
        self.assertEqual(category.symmetry.U("X", e, m, psi, mu=1), 0)
        self.assertEqual(category.symmetry.U("X", e, m, psi, nu=1), 0)
        for index in (-1, True, 0.5):
            with self.assertRaises(DataError):
                category.symmetry.U("X", e, m, psi, mu=index)
        calls = (
            lambda: category.N((2, 0), e, m),
            lambda: category.quantum_dimension((2, 0)),
            lambda: category.spin((2, 0)),
            lambda: category.symmetry.U("missing", e, m, psi),
            lambda: category.symmetry.U("X", (2, 0), m, psi),
            lambda: category.symmetry.eta((2, 0), "X", "T"),
            lambda: category.symmetry.eta(psi, "X", "missing"),
        )
        for call in calls:
            with self.assertRaises(DataError):
                call()

    def test_fusion_functions_reject_nonintegral_negative_and_nonfinite_values(self):
        for value in (0.5, -1, 1j, True, math.inf, math.nan, "1"):
            with self.subTest(value=value), self.assertRaisesRegex(DataError, "fusion_rules"):
                UMTC.from_functions(toric_code(), N=lambda a, b, c: value)
        for expression in ("0.5", "-1", "1j", "True", "exp(10000)"):
            data = toric_code()
            data["fusion_rules"] = {"expression": expression}
            with self.subTest(expression=expression), self.assertRaises(DataError):
                UMTC.from_dict(data)

    def test_dimension_functions_reject_nonreal_or_nonfinite_values(self):
        for value in (1j, math.inf, math.nan, True, "1"):
            category = UMTC.from_functions(toric_code(), quantum_dimension=lambda a: value)
            with self.subTest(value=value), self.assertRaisesRegex(DataError, "quantum_dimensions"):
                category.quantum_dimension((0, 0))

    def test_other_functions_reject_nonfinite_or_nonnumeric_values(self):
        for value in (math.inf, math.nan, True, "1"):
            category = UMTC.from_functions(toric_code(), spin=lambda a: value, U=lambda g, a, b, c: value, eta=lambda a, g, h: value)
            for function, args in (
                (category.spin, ((0, 0),)),
                (category.symmetry.U, ("1", (0, 0), (0, 0), (0, 0))),
                (category.symmetry.eta, ((0, 0), "1", "1")),
            ):
                with self.subTest(value=value, function=function.__name__), self.assertRaises(DataError):
                    function(*args)

    def test_incorrect_finite_functions_fail_consistency(self):
        for field, callback, checker in (
            ("quantum_dimension", lambda a: 2, check_fusion),
            ("spin", lambda a: 1, check_ribbon),
            ("U", lambda g, a, b, c: 1, check_symmetry),
            ("eta", lambda a, g, h: 1, check_symmetry),
        ):
            with self.subTest(symbol=field):
                self.assertIs(checker(UMTC.from_functions(toric_code(), **{field: callback})), False)
        for field, expression, checker in (
            ("quantum_dimensions", "2", check_fusion),
            ("topological_spins", "1", check_ribbon),
            ("U_symbols", "1", check_symmetry),
            ("eta_symbols", "1", check_symmetry),
        ):
            data = toric_code()
            owner = data["symmetry"] if field in ("U_symbols", "eta_symbols") else data
            owner[field] = {"expression": expression}
            with self.subTest(formula=field):
                self.assertIs(checker(UMTC.from_dict(data)), False)

    def test_function_definitions_and_arities_are_validated_on_load(self):
        for name in ("fusion_rules", "quantum_dimensions", "topological_spins", "U_symbols", "eta_symbols"):
            for definition in ({}, {"expression": 1}, {"expression": "1", "default": 1}, {"expression": "a.__class__"}):
                data = toric_code()
                owner = data["symmetry"] if name in ("U_symbols", "eta_symbols") else data
                owner[name] = definition
                with self.subTest(symbol=name, definition=definition), self.assertRaises(DataError):
                    UMTC.from_dict(data)
        for name in ("N", "quantum_dimension", "spin", "U", "eta"):
            for callback in (7, lambda: 1):
                with self.subTest(symbol=name, callback=callback), self.assertRaises(DataError):
                    UMTC.from_functions(toric_code(), **{name: callback})

    def test_symmetry_callbacks_require_explicit_group_metadata(self):
        for value in (None, {}):
            data = toric_code()
            data["symmetry"] = value
            for name in ("U", "eta", "action"):
                with self.subTest(symmetry=value, symbol=name), self.assertRaises(DataError):
                    UMTC.from_functions(data, **{name: native_toric_functions()[name]})
        data = toric_code()
        del data["symmetry"]
        with self.assertRaises(DataError):
            UMTC.from_functions(data, U=native_toric_functions()["U"])


class MultiplicityUFunctionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = _a4_symmetry_data(_a4_data())
        cls.table = UMTC.from_dict(cls.data)

    def test_native_u_forwards_positional_keyword_and_vararg_indices(self):
        seen = set()

        def positional(g, a, b, c, row=0, column=0, **options):
            self.assertEqual(options, {})
            seen.add((row, column))
            return self.table.symmetry.U(g, a, b, c, row, column)

        def keyword(g, a, b, c, *, mu=0, nu=0):
            seen.add((mu, nu))
            return self.table.symmetry.U(g, a, b, c, mu, nu)

        def varargs(g, a, b, c, *extra, mu=0, nu=0):
            self.assertEqual(extra, ())
            seen.add((mu, nu))
            return self.table.symmetry.U(g, a, b, c, mu, nu)

        for callback in (positional, keyword, varargs):
            seen.clear()
            category = UMTC.from_functions(self.data, F=self.table.F, R=self.table.R, U=callback)
            for g in self.table.symmetry.elements:
                self.assertEqual(category.symmetry.U_matrix(g, "v", "v", "v"), self.table.symmetry.U_matrix(g, "v", "v", "v"))
            self.assertEqual(seen, set(itertools.product(range(2), repeat=2)))
            self.assertIs(check_symmetry(category), True)

    def test_formula_u_uses_named_matrix_indices(self):
        category = UMTC.from_functions(self.data, F=self.table.F, R=self.table.R)
        data = copy.deepcopy(self.data)
        data["symbol_format"] = "functions"
        data["F_symbols"], data["R_symbols"] = category.F, category.R
        data["symmetry"]["U_symbols"] = {"expression": "10 * mu + nu"}
        symmetry = UMTC.from_dict(data).symmetry
        self.assertEqual(symmetry.U_matrix("X", "v", "v", "v"), ((0, 1), (10, 11)))

    def test_multiplicity_u_requires_index_arguments(self):
        with self.assertRaisesRegex(DataError, "multiplicit"):
            UMTC.from_functions(self.data, F=self.table.F, R=self.table.R, U=lambda g, a, b, c: 1)


if __name__ == "__main__":
    unittest.main()
