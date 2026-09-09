"""Tables, JSON formulas, and Python callables share one symbol interface."""

import copy
import itertools
import math
import unittest

from umtc import DataError, UMTC, check_all, check_hexagon, check_pentagon
from scripts.generate_examples import fibonacci, toric_code
from test_multiplicity import _a4_data


def fibonacci_functions():
    """A formula fixture independent of the public, explicitly listed example."""
    data = fibonacci()
    data["symbol_format"] = "functions"
    data["F_symbols"] = {
        "expression": (
            "(1 / ((1 + sqrt(5)) / 2) if e == f == '1' "
            "else -1 / ((1 + sqrt(5)) / 2) if e == f == 'tau' "
            "else 1 / sqrt((1 + sqrt(5)) / 2)) "
            "if a == b == c == d == 'tau' else 1"
        )
    }
    data["R_symbols"] = {
        "expression": "exp((-4 if c == '1' else 3) * pi * 1j / 5) if a == b == 'tau' else 1"
    }
    return data


class SymbolModeTests(unittest.TestCase):
    def assert_same_symbols(self, expected, actual):
        self.assertTrue(callable(actual.F_symbols))
        self.assertTrue(callable(actual.R_symbols))
        for a, b, c, d in itertools.product(expected.anyons, repeat=4):
            expected_matrix = expected.F_matrix(a, b, c, d)
            actual_matrix = actual.F_matrix(a, b, c, d)
            self.assertEqual(len(expected_matrix), len(actual_matrix))
            for expected_row, actual_row in zip(expected_matrix, actual_matrix):
                self.assertEqual(len(expected_row), len(actual_row))
                for expected_value, actual_value in zip(expected_row, actual_row):
                    self.assertAlmostEqual(expected_value, actual_value)
            for e, alpha, beta in expected.F_left_basis(a, b, c, d):
                for f, mu, nu in expected.F_right_basis(a, b, c, d):
                    args = (a, b, c, d, e, f, alpha, beta, mu, nu)
                    self.assertAlmostEqual(expected.F(*args), actual.F(*args))
                    self.assertAlmostEqual(expected.F(*args), actual.F_symbols(*args))
        for a, b, c in itertools.product(expected.anyons, repeat=3):
            expected_matrix = expected.R_matrix(a, b, c)
            actual_matrix = actual.R_matrix(a, b, c)
            self.assertEqual(len(expected_matrix), len(actual_matrix))
            for mu, (expected_row, actual_row) in enumerate(zip(expected_matrix, actual_matrix)):
                self.assertEqual(len(expected_row), len(actual_row))
                for nu, (expected_value, actual_value) in enumerate(zip(expected_row, actual_row)):
                    self.assertAlmostEqual(expected_value, actual_value)
                    self.assertAlmostEqual(expected_value, actual.R(a, b, c, mu, nu))
                    self.assertAlmostEqual(expected_value, actual.R_symbols(a, b, c, mu, nu))

    def test_toric_tables_formulas_and_python_functions_are_equivalent(self):
        table = UMTC.from_dict(toric_code(symbol_format="tables"))
        formula = UMTC.from_dict(toric_code())
        native = UMTC.from_functions(
            toric_code(),
            F=lambda a, b, c, d, e, f: 1,
            R=lambda a, b, c: (-1) ** (a[0] * b[1]),
        )
        self.assertEqual(table.symbol_format, "tables")
        for category in (table, formula, native):
            self.assert_same_symbols(table, category)
            self.assertIs(check_all(category), True)
        self.assertEqual(formula.symbol_format, "functions")
        self.assertEqual(native.symbol_format, "functions")

    def test_fibonacci_tables_and_formulas_have_same_callable_semantics(self):
        table = UMTC.from_dict(fibonacci())
        formula = UMTC.from_dict(fibonacci_functions())
        # Native callables can replace the table fields without modifying input.
        data = fibonacci()
        original = copy.deepcopy(data)
        native = UMTC.from_functions(data, F=table.F, R=table.R)
        self.assertEqual(data, original)
        for category in (table, formula, native):
            self.assert_same_symbols(table, category)
            self.assertIs(check_all(category), True)

    def test_symbol_format_is_inferred_for_legacy_tables_and_formulas(self):
        for data, expected in ((fibonacci(), "tables"), (toric_code(), "functions")):
            data.pop("symbol_format")
            category = UMTC.from_dict(data)
            self.assertEqual(category.symbol_format, expected)
            self.assertTrue(callable(category.F_symbols))
            self.assertTrue(callable(category.R_symbols))

    def test_explicit_modes_reject_mixed_or_mismatched_symbols(self):
        invalid = []
        data = toric_code()
        data["symbol_format"] = "tables"
        invalid.append(data)
        data = fibonacci()
        data["symbol_format"] = "functions"
        invalid.append(data)
        data = toric_code()
        data["R_symbols"] = toric_code(symbol_format="tables")["R_symbols"]
        invalid.append(data)
        data = toric_code()
        data["symbol_format"] = "unknown"
        invalid.append(data)
        for data in invalid:
            with self.subTest(mode=data["symbol_format"]):
                with self.assertRaises(DataError):
                    UMTC.from_dict(data)

    def test_native_functions_are_lazy_and_receive_canonical_tuples(self):
        calls = []

        def f_symbol(a, b, c, d, e, f):
            calls.append(("F", a, b, c, d, e, f))
            self.assertTrue(all(isinstance(x, tuple) for x in (a, b, c, d, e, f)))
            return 1

        def r_symbol(a, b, c):
            calls.append(("R", a, b, c))
            self.assertTrue(all(isinstance(x, tuple) for x in (a, b, c)))
            return (-1) ** (a[0] * b[1])

        data = toric_code()
        data.pop("F_symbols")
        data.pop("R_symbols")
        category = UMTC.from_functions(data, F=f_symbol, R=r_symbol)
        self.assertEqual(calls, [])
        e, m, psi = [1, 0], [0, 1], [1, 1]
        self.assertEqual(category.F(e, m, e, m, psi, psi), 1)
        self.assertEqual([call[0] for call in calls], ["F"])
        self.assertEqual(category.R(e, m, psi), -1)
        self.assertEqual([call[0] for call in calls], ["F", "R"])

    def test_selection_rules_do_not_evaluate_native_functions(self):
        def forbidden(*args):
            self.fail("An inadmissible entry reached the native function")

        category = UMTC.from_functions(toric_code(), F=forbidden, R=forbidden)
        vacuum, e, m, psi = (0, 0), (1, 0), (0, 1), (1, 1)
        self.assertEqual(category.F(e, m, e, m, e, psi), 0)
        self.assertEqual(category.F(e, m, e, m, psi, psi, alpha=1), 0)
        self.assertEqual(category.F_matrix(e, m, e, vacuum), ())
        self.assertEqual(category.R(e, m, vacuum), 0)
        self.assertEqual(category.R(e, m, psi, mu=1), 0)
        self.assertEqual(category.R(e, m, psi, nu=1), 0)
        self.assertEqual(category.R_matrix(e, m, vacuum), ())
        with self.assertRaises(DataError):
            category.R((2, 0), m, psi)
        with self.assertRaises(DataError):
            category.F(e, m, e, m, psi, psi, alpha=-1)

    def test_formula_evaluation_is_lazy_and_respects_selection_rules(self):
        data = toric_code()
        data["F_symbols"] = {"expression": "1 / 0"}
        data["R_symbols"] = {"expression": "1 / 0"}
        category = UMTC.from_dict(data)
        vacuum, e, m, psi = (0, 0), (1, 0), (0, 1), (1, 1)
        self.assertEqual(category.F(e, m, e, m, e, psi), 0)
        self.assertEqual(category.R(e, m, vacuum), 0)
        with self.assertRaisesRegex(DataError, "F"):
            category.F(e, m, e, m, psi, psi)
        with self.assertRaisesRegex(DataError, "R"):
            category.R(e, m, psi)

    def test_corrupt_formula_and_native_symbols_fail_equations(self):
        for symbol, checker in (("F_symbols", check_pentagon), ("R_symbols", check_hexagon)):
            data = toric_code()
            data[symbol] = {"expression": "-1"}
            with self.subTest(symbol=symbol, mode="formula"):
                self.assertIs(checker(UMTC.from_dict(data)), False)
        category = UMTC.from_functions(
            toric_code(), F=lambda a, b, c, d, e, f: -1,
            R=lambda a, b, c: (-1) ** (a[0] * b[1]),
        )
        self.assertIs(check_pentagon(category), False)
        category = UMTC.from_functions(
            toric_code(), F=lambda a, b, c, d, e, f: 1,
            R=lambda a, b, c: 0,
        )
        self.assertIs(check_hexagon(category), False)

    def test_invalid_formula_is_a_data_error_on_load(self):
        for expression in ("__import__('os').getcwd()", "a.__class__", "missing + 1", "1 +"):
            data = toric_code()
            data["F_symbols"] = {"expression": expression}
            with self.subTest(expression=expression):
                with self.assertRaises(DataError):
                    UMTC.from_dict(data)

    def test_formula_definitions_require_exact_expression_fields_and_string_values(self):
        for symbol in ("F_symbols", "R_symbols"):
            for definition in ({}, {"expression": 1}, {"expression": None},
                               {"expression": "1", "default": 1}, "1"):
                data = toric_code()
                data[symbol] = definition
                with self.subTest(symbol=symbol, definition=definition):
                    with self.assertRaises(DataError):
                        UMTC.from_dict(data)

    def test_native_inputs_require_callables_with_compatible_arities(self):
        valid_f = lambda a, b, c, d, e, f: 1
        valid_r = lambda a, b, c: 1
        for f_symbol, r_symbol in ((1, valid_r), (valid_f, 1),
                                   (lambda a: 1, valid_r),
                                   (valid_f, lambda a, b: 1)):
            with self.subTest(F=f_symbol, R=r_symbol):
                with self.assertRaises(DataError):
                    UMTC.from_functions(toric_code(), F=f_symbol, R=r_symbol)

    def test_native_type_errors_are_reported_without_retrying_the_callback(self):
        calls = []

        def broken(*args, **kwargs):
            calls.append((args, kwargs))
            raise TypeError("error inside the callback")

        category = UMTC.from_functions(toric_code(), F=broken, R=broken)
        vacuum = (0, 0)
        for symbol, args in ((category.F, (vacuum,) * 6), (category.R, (vacuum,) * 3)):
            calls.clear()
            with self.assertRaisesRegex(DataError, "error inside the callback") as raised:
                symbol(*args)
            self.assertEqual(len(calls), 1)
            self.assertIsInstance(raised.exception.__cause__, TypeError)

    def test_native_arithmetic_errors_and_nonfinite_values_have_symbol_context(self):
        for callback in (lambda *args: 1 / 0, lambda *args: math.inf, lambda *args: "bad"):
            category = UMTC.from_functions(toric_code(), F=callback, R=callback)
            vacuum = (0, 0)
            with self.assertRaisesRegex(DataError, "F"):
                category.F(*(vacuum,) * 6)
            with self.assertRaisesRegex(DataError, "R"):
                category.R(*(vacuum,) * 3)

    def test_native_functions_preserve_all_multiplicity_indices(self):
        data = _a4_data()
        table = UMTC.from_dict(data)
        seen_f, seen_r = set(), set()

        def f_symbol(a, b, c, d, e, f, alpha, beta, mu, nu):
            seen_f.add((alpha, beta, mu, nu))
            return table.F(a, b, c, d, e, f, alpha, beta, mu, nu)

        def r_symbol(a, b, c, mu, nu):
            seen_r.add((mu, nu))
            return table.R(a, b, c, mu, nu)

        native = UMTC.from_functions(data, F=f_symbol, R=r_symbol)
        self.assert_same_symbols(table, native)
        self.assertEqual(seen_f, set(itertools.product(range(2), repeat=4)))
        self.assertEqual(seen_r, set(itertools.product(range(2), repeat=2)))
        self.assertAlmostEqual(native.R("v", "v", "v", 0, 1), -1j)
        # These data form Rep(A4), a coherent symmetric fusion category.
        self.assertIs(check_pentagon(native), True)
        self.assertIs(check_hexagon(native), True)

    def test_native_functions_accept_keyword_multiplicity_indices(self):
        table = UMTC.from_dict(_a4_data())

        def f_symbol(a, b, c, d, e, f, *, alpha=0, beta=0, mu=0, nu=0):
            return table.F(a, b, c, d, e, f, alpha, beta, mu, nu)

        def r_symbol(a, b, c, *, mu=0, nu=0):
            return table.R(a, b, c, mu, nu)

        native = UMTC.from_functions(_a4_data(), F=f_symbol, R=r_symbol)
        self.assert_same_symbols(table, native)

    def test_native_varargs_do_not_swallow_keyword_multiplicity_indices(self):
        data = _a4_data()
        table = UMTC.from_dict(data)
        seen_f, seen_r = set(), set()

        def f_symbol(a, b, c, d, e, f, *extra, alpha=0, beta=0, mu=0, nu=0):
            self.assertEqual(extra, ())
            seen_f.add((alpha, beta, mu, nu))
            return table.F(a, b, c, d, e, f, alpha, beta, mu, nu)

        def r_symbol(a, b, c, *extra, mu=0, nu=0):
            self.assertEqual(extra, ())
            seen_r.add((mu, nu))
            return table.R(a, b, c, mu, nu)

        native = UMTC.from_functions(data, F=f_symbol, R=r_symbol)
        self.assert_same_symbols(table, native)
        self.assertEqual(seen_f, set(itertools.product(range(2), repeat=4)))
        self.assertEqual(seen_r, set(itertools.product(range(2), repeat=2)))

    def test_formulas_receive_named_multiplicity_indices_in_basis_order(self):
        data = _a4_data()
        data["symbol_format"] = "functions"
        data["F_symbols"] = {"expression": "1000 * alpha + 100 * beta + 10 * mu + nu"}
        data["R_symbols"] = {"expression": "10 * mu + nu"}
        category = UMTC.from_dict(data)
        self.assertEqual(category.R_matrix("v", "v", "v"), ((0, 1), (10, 11)))
        matrix = category.F_matrix("v", "v", "v", "v")
        for row, (_, alpha, beta) in enumerate(category.F_left_basis("v", "v", "v", "v")):
            for col, (_, mu, nu) in enumerate(category.F_right_basis("v", "v", "v", "v")):
                self.assertEqual(matrix[row][col], 1000 * alpha + 100 * beta + 10 * mu + nu)

    def test_native_kwargs_do_not_swallow_positional_multiplicity_indices(self):
        data = _a4_data()
        table = UMTC.from_dict(data)

        def f_symbol(a, b, c, d, e, f, i=0, j=0, k=0, l=0, **options):
            self.assertEqual(options, {})
            return table.F(a, b, c, d, e, f, i, j, k, l)

        def r_symbol(a, b, c, row=0, column=0, **options):
            self.assertEqual(options, {})
            return table.R(a, b, c, row, column)

        native = UMTC.from_functions(data, F=f_symbol, R=r_symbol)
        self.assert_same_symbols(table, native)

    def test_multiplicity_functions_must_accept_the_index_arguments(self):
        for f_symbol, r_symbol in (
            (lambda a, b, c, d, e, f: 1, lambda a, b, c, mu, nu: 1),
            (lambda a, b, c, d, e, f, alpha, beta, mu, nu: 1, lambda a, b, c: 1),
        ):
            with self.assertRaisesRegex(DataError, "multiplicit"):
                UMTC.from_functions(_a4_data(), F=f_symbol, R=r_symbol)


if __name__ == "__main__":
    unittest.main()
