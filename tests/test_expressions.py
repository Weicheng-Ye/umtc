"""The JSON formula grammar and its evaluation limits."""

import cmath
import math
import unittest

from umtc.expressions import compile_action, compile_expression


class ExpressionTests(unittest.TestCase):
    def test_action_expressions_preserve_label_types(self):
        action = compile_action("[a[1], a[0]] if g == 'X' else a")
        self.assertEqual(action("X", (1, 0)), (0, 1))
        self.assertIs(type(action("X", (1, 0))), tuple)
        self.assertEqual(compile_action("'tau' if a == '1' else '1'")("X", "1"), "tau")
        self.assertEqual(compile_action("(a + 1) % 2")("X", 0), 1)
        self.assertIs(type(compile_action("a")("1", 9007199254740993)), int)

    def test_action_results_do_not_relax_numeric_symbol_contracts(self):
        for expression in ("a", "'tau'", "(0, 1)"):
            with self.subTest(expression=expression), self.assertRaises(ValueError):
                compile_expression(expression, ("a",))((0, 1))
        for expression in ("a == a", "not a", "1.0", "1j", "[0, 1.0]"):
            with self.subTest(expression=expression), self.assertRaises(ValueError):
                compile_action(expression)("X", (1, 0))

    def test_integer_results_can_be_preserved_without_complex_rounding(self):
        expression = compile_expression("9007199254740993", (), preserve_numeric_type=True)
        result = expression()
        self.assertIs(type(result), int)
        self.assertEqual(result, 2 ** 53 + 1)

    def test_toric_code_functions(self):
        f = compile_expression("1", ("a", "b", "c", "d", "e", "f"))
        r = compile_expression("(-1) ** (a[0] * b[1])", ("a", "b", "c"))
        self.assertEqual(f(*([(0, 0)] * 6)), 1 + 0j)
        self.assertEqual(r((1, 0), (0, 1), (1, 1)), -1 + 0j)
        self.assertEqual(r((0, 1), (1, 0), (1, 1)), 1 + 0j)

    def test_complex_mathematical_calls(self):
        expressions = {
            "exp(1j * pi / 3)": cmath.exp(1j * math.pi / 3),
            "sqrt(-1)": 1j,
            "conj(complex(2, 3))": 2 - 3j,
            "sin(pi / 2) + cos(0)": 2,
            "abs(3 + 4j)": 5,
            "complex(2)": 2,
        }
        for expression, expected in expressions.items():
            with self.subTest(expression=expression):
                actual = compile_expression(expression, ())()
                self.assertIs(type(actual), complex)
                self.assertAlmostEqual(actual, expected)

    def test_multiplicity_conditional_and_short_circuiting(self):
        f = compile_expression("1 if alpha == mu and beta == nu else 0",
                               ("alpha", "beta", "mu", "nu"))
        self.assertEqual(f(0, 1, 0, 1), 1)
        self.assertEqual(f(0, 1, 1, 1), 0)
        for expression in ("1 if not (a != b) else 0", "1 if a == b or 1 / 0 else 0"):
            self.assertEqual(compile_expression(expression, ("a", "b"))((0,), (0,)), 1)
        self.assertEqual(compile_expression("1 if a == 0 else 1 / a", ("a",))(0), 1)
        self.assertEqual(compile_expression("a or 1", ("a",))(0), 1)

    def test_integer_operators_indices_and_label_comparisons(self):
        f = compile_expression("((a[+0] ^ b[-1]) & 3) | 4", ("a", "b"))
        self.assertEqual(f((1,), (0,)), 5)
        self.assertEqual(compile_expression("+(a // 2) + a % 2", ("a",))(5), 3)
        f = compile_expression("1 if 0 <= a[0] < 2 and a == (0, 1) else 0", ("a",))
        self.assertEqual(f((0, 1)), 1)
        self.assertEqual(f((1, 1)), 0)
        self.assertEqual(compile_expression("1 if a == [0, 1] else 0", ("a",))([0, 1]), 1)
        self.assertEqual(compile_expression("1 if a == b else 0", ("a", "b"))("tau", "tau"), 1)
        f = compile_expression("-1 if a == 'tau' and e == '1' else 1", ("a", "e"))
        self.assertEqual(f("tau", "1"), -1)
        self.assertEqual(f("tau", "tau"), 1)

    def test_unsupported_syntax_is_rejected_before_evaluation(self):
        expressions = (
            "__import__('os')", "import os", "a.real", "a.__class__",
            "[x for x in a]", "lambda: 1", "unknown + 1", "eval(a)",
            "sqrt(x=1)", "sqrt(1, 2)", "complex()", "sqrt(*a)",
            "a[0:1]", "a[a[0]]", "a[0][0]", "a[1.0]", "a[True]",
            "{'a': 1}", "a << 2", "~a", "a is a", "0 in a",
            "True", "None", "1e309", "nan", "1; 2", "",
            "(1 if a == 0 else unknown)", "(1 if a == 0 else a.real)",
        )
        for expression in expressions:
            with self.subTest(expression=expression), self.assertRaises(ValueError):
                compile_expression(expression, ("a",))

    def test_argument_contract(self):
        for arguments in (["a"], ("a", "a"), ("pi",), ("sqrt",), ("bad-name",), ("if",), (1,)):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                compile_expression("1", arguments)
        f = compile_expression("1", ("a",))
        for args in ((), (0, 1)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                f(*args)

    def test_evaluation_failures_are_value_errors(self):
        cases = (
            ("1 / a", 0), ("a[1]", (0,)), ("a[-2]", (0,)),
            ("a[0]", "1"), ("a[0]", 1), ("a ** -1", 0),
            ("exp(a)", 1000), ("a + 1", (0,)), ("a * 1000", (0,)),
            ("a * 1000", "label"), ("a | 1", 1.5), ("a // 2", 1j),
            ("1 if a < 1 else 0", 1j), ("complex(a, a)", "label"),
            ("a", float("nan")), ("a", complex(0, float("inf"))),
            ("a", True), ("a == 1", 1), ("not a", 0), ("a", "label"),
            ("'1'", 0), ("'tau' * a", 2), ("['tau']", 0),
            ("a", (0,)), ("a", [[0]]), ("a", [True]),
            ("a", 10 ** 1000), ("a", [0] * 257), ("a", "a" * 4097),
        )
        for expression, argument in cases:
            with self.subTest(expression=expression, argument_type=type(argument)), self.assertRaises(ValueError):
                compile_expression(expression, ("a",))(argument)

    def test_work_and_magnitude_limits(self):
        for expression in ("9 ** (9 ** 9)", "2 ** 1025", "2 ** -1025", "1e100 * 1e100"):
            with self.subTest(expression=expression), self.assertRaises(ValueError):
                compile_expression(expression, ())()
        expressions = ("1" * 4097, "+" * 34 + "1", "+".join(["1"] * 100), "1e101")
        for expression in expressions:
            with self.subTest(expression=expression[:50]), self.assertRaises(ValueError):
                compile_expression(expression, ())

    def test_custom_objects_are_rejected_without_invoking_them(self):
        class PretendNumber:
            def __mul__(self, other):
                raise AssertionError("Custom arithmetic must never run")

            def __eq__(self, other):
                raise AssertionError("Custom comparisons must never run")

        for expression in ("a * 2", "1 if a == 0 else 0"):
            with self.subTest(expression=expression), self.assertRaises(ValueError):
                compile_expression(expression, ("a",))(PretendNumber())


if __name__ == "__main__":
    unittest.main()
