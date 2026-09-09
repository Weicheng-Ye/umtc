"""GAP backend tests, including exact nonabelian multiplication and rho."""

from dataclasses import FrozenInstanceError, replace
import itertools
import shutil
import subprocess
import unittest
from unittest.mock import patch

from umtc.gap import (
    GAPError, GAPGroup, GAPUnavailableError, GAPZ2Homomorphism,
    _ConstructionParser, _construct_homomorphism, _run_gap,
    _validate_export,
)


class GAPInputTests(unittest.TestCase):
    def test_accepts_constructor_lists_categories_and_permutations(self):
        for expression in (
            " AbelianGroup( [2, 2] ) ", "CyclicGroup(IsPermGroup,3)",
            "Group( (1,2,3), (1,2)(3,4), () )", "Group([(1,2), (1,2,3)])",
            "DirectProduct(CyclicGroup(2),CyclicGroup(3))", "AlternatingGroup(4)",
            "DihedralGroup(IsPcGroup,8)", "SmallGroup(8,1)",
        ):
            with self.subTest(expression=expression):
                result = _ConstructionParser(expression).parse()
                self.assertNotIn(" ", result)

    def test_rejects_code_and_unsupported_constructions_before_launching_gap(self):
        expressions = (
            'CyclicGroup(2);Exec("touch /tmp/no")', 'Read("/tmp/data")',
            "CyclicGroup(2) # comment", "G:=CyclicGroup(2)", "FreeGroup(2)",
            'Group("x")', "CyclicGroup(2)+CyclicGroup(2)", "Group(1/0)",
            "CyclicGroup(2).elements", "CyclicGroup(2)[1]", "Exec(1)",
            "[CyclicGroup(2)]", "CyclicGroup(List([1,2],x->x))",
            "Group((0,1))", "Group((-1,2))", "Group((1,1))", "Group((1,x))",
            "Group((1,999999999999))",
            "Group((1,,2))", "CyclicGroup(2,,3)", "CyclicGroup(2", "CyclicGroup(2))",
            "CyclicGroup(1e8)", "CyclicGroup(1234567890123)", "", " ", None, 2,
        )
        with patch("umtc.gap.subprocess.run") as run:
            for expression in expressions:
                with self.subTest(expression=expression), self.assertRaises(GAPError):
                    GAPGroup.from_gap(expression)
            run.assert_not_called()

    def test_rejects_deep_or_oversized_input(self):
        for expression in (
            "DirectProduct(" * 40 + "CyclicGroup(2)" + ")" * 40,
            "CyclicGroup(" + "1" * 20000 + ")",
        ):
            with self.subTest(expression=expression[:40]), self.assertRaises(GAPError):
                GAPGroup.from_gap(expression)

    def test_rejects_invalid_generator_names_and_options_without_gap(self):
        options = [
            {"generator_names": names}
            for names in ("X", {"X": 1}, ["X", "X"], ["1"], ["X*T"], ["X T"], [2])
        ] + [
            {"timeout": 0}, {"timeout": -1}, {"timeout": True}, {"timeout": float("inf")},
            {"timeout": float("nan")}, {"max_order": 0}, {"max_order": True},
            {"timeout": 10**1000},
            {"executable": ""}, {"executable": 2},
        ]
        with patch("umtc.gap.subprocess.run") as run:
            for kwargs in options:
                with self.subTest(kwargs=kwargs), self.assertRaises(GAPError):
                    GAPGroup.from_gap("CyclicGroup(2)", **kwargs)
            run.assert_not_called()

    def test_missing_executable_is_distinct_and_actionable(self):
        with self.assertRaisesRegex(GAPUnavailableError, "install GAP"):
            GAPGroup.from_gap("CyclicGroup(2)", executable="/nonexistent/umtc-gap")

    def test_timeout_is_a_gap_error(self):
        with patch("umtc.gap.subprocess.run", side_effect=subprocess.TimeoutExpired("gap", 0.01)):
            with self.assertRaisesRegex(GAPError, "timeout"):
                _run_gap("", "gap", 0.01)

    def test_protocol_errors_and_gap_failures_are_rejected(self):
        samples = (
            (0, "", ""), (1, "", "Error, invalid group"),
            (0, "__UMTC_GAP_BEGIN__[]__UMTC_GAP_END__", "Syntax error: unexpected symbol"),
            (0, "__UMTC_GAP_BEGIN__[fail]__UMTC_GAP_END__", ""),
            (0, "__UMTC_GAP_BEGIN__[]__UMTC_GAP_END____UMTC_GAP_END__", ""),
        )
        for code, stdout, stderr in samples:
            with self.subTest(stdout=stdout, stderr=stderr):
                result = subprocess.CompletedProcess(["gap"], code, stdout, stderr)
                with patch("umtc.gap.subprocess.run", return_value=result), self.assertRaises(GAPError):
                    _run_gap("", "gap", 30)

    def test_launch_uses_no_shell_and_disables_user_gap_root(self):
        result = subprocess.CompletedProcess(["gap"], 0, "__UMTC_GAP_BEGIN__[]__UMTC_GAP_END__", "")
        with patch("umtc.gap.subprocess.run", return_value=result) as run:
            self.assertEqual(_run_gap("a script", "/custom/gap", 7), [])
        args, kwargs = run.call_args
        self.assertEqual(args[0][0], "/custom/gap")
        self.assertIn("-r", args[0])
        self.assertIn("--quitonbreak", args[0])
        self.assertEqual(args[0][args[0].index("-K") + 1], "256m")
        self.assertNotIn("-T", args[0])  # -T makes GAP continue after errors.
        self.assertEqual(kwargs["input"], "a script")
        self.assertEqual(kwargs["timeout"], 7)
        self.assertFalse(kwargs.get("shell", False))

    def test_export_requires_exact_valid_group_metadata(self):
        malformed = (
            None, [], [0, 1, [], [], []], [True, 1, [], [[1]], [1]],
            [1, 0, [], [[1]], [1]], [1, 1, [2], [[1]], [1]],
            [1, 1, [], [[True]], [1]], [1, 1, [], [], [1]],
            [2, 1, [2], [[1, 1], [1, 1]], [1, 2]],
            [2, 1, [2], [[1, 2], [2, 1]], [2, 1]],
        )
        for raw in malformed:
            with self.subTest(raw=raw), self.assertRaises(GAPError):
                _validate_export(raw, 256)
        self.assertEqual(_validate_export([1, 1, [], [[1]], [1]], 256), (1, 0, (), ((0,),), (0,)))


@unittest.skipUnless(shutil.which("gap"), "GAP executable is not installed")
class LiveGAPTests(unittest.TestCase):
    def test_toric_group_labels_exact_products_and_antiunitary_homomorphism(self):
        group = GAPGroup.from_gap("AbelianGroup([2,2])", ["X", "T"])
        self.assertEqual(group.elements, ("1", "X", "T", "X*T"))
        self.assertEqual(group.identity, "1")
        self.assertEqual(group.generators, ("X", "T"))
        self.assertEqual(group.generator_names, ("X", "T"))
        self.assertEqual(group.mul("X", "T"), "X*T")
        self.assertEqual(group.mul("X*T", "T"), "X")
        rho = group.homomorphism_to_z2([0, 1])
        self.assertIsInstance(rho, GAPZ2Homomorphism)
        self.assertIs(rho.source, group)
        self.assertEqual(rho.generator_images, (0, 1))
        self.assertEqual(rho.images, (0, 0, 1, 1))
        for g, h in itertools.product(group.elements, repeat=2):
            self.assertEqual(rho(group.mul(g, h)), (rho(g) + rho(h)) % 2)
            self.assertEqual(group.inverse(g), g)

    def test_nonabelian_s3_and_parity_map(self):
        group = GAPGroup.from_gap("Group((1,2,3),(1,2))", ["r", "s"])
        self.assertEqual(len(group.elements), 6)
        self.assertNotEqual(group.mul("r", "s"), group.mul("s", "r"))
        self.assertEqual(group.pow("r", 3), "1")
        self.assertEqual(group.pow("s", 2), "1")
        self.assertEqual(group.mul(group.mul("s", "r"), "s"), group.inverse("r"))
        self.assertEqual(group.pow("r", -1), "r*r")
        self.assertEqual(group.pow("r", -3), "1")
        self.assertEqual(group.pow("r", 0), "1")
        rho = group.homomorphism_to_z2([0, 1])
        for g, h, k in itertools.product(group.elements, repeat=3):
            self.assertEqual(group.mul(group.mul(g, h), k), group.mul(g, group.mul(h, k)))
            self.assertEqual(rho(group.mul(g, h)), (rho(g) + rho(h)) % 2)
        for g in group.elements:
            self.assertEqual(group.mul(g, group.inverse(g)), "1")
        with self.assertRaisesRegex(GAPError, "homomorphism"):
            group.homomorphism_to_z2([1, 0])

    def test_trivial_homomorphism_need_not_be_surjective(self):
        group = GAPGroup.from_gap("CyclicGroup(3)")
        self.assertEqual(group.elements, ("1", "g1", "g1*g1"))
        self.assertEqual(group.homomorphism_to_z2([0]).images, (0, 0, 0))
        with self.assertRaisesRegex(GAPError, "homomorphism"):
            group.homomorphism_to_z2([1])

    def test_trivial_group_and_identity_generators(self):
        group = GAPGroup.from_gap("Group(())", ["I"])
        self.assertEqual(group.elements, ("1",))
        self.assertEqual(group.generators, ("1",))
        self.assertEqual(group.homomorphism_to_z2([0]).images, (0,))
        with self.assertRaises(GAPError):
            group.homomorphism_to_z2([1])
        cyclic = GAPGroup.from_gap("CyclicGroup(1)")
        self.assertEqual(cyclic.elements, ("1",))
        self.assertEqual(cyclic.homomorphism_to_z2([0] * len(cyclic.generators)).images, (0,))

    def test_redundant_generators_do_not_make_ambiguous_element_labels(self):
        group = GAPGroup.from_gap("Group((1,2),(1,2),())", ["X", "Y", "I"])
        self.assertEqual(group.elements, ("1", "X"))
        self.assertEqual(group.generators, ("X", "X", "1"))
        self.assertEqual(group.homomorphism_to_z2([1, 1, 0]).images, (0, 1))
        for bits in ([1, 0, 0], [0, 0, 1]):
            with self.subTest(bits=bits), self.assertRaises(GAPError):
                group.homomorphism_to_z2(bits)

    def test_explicit_category_direct_product_and_smallgroup_constructors(self):
        for expression, order in (
            ("DirectProduct(CyclicGroup(2),CyclicGroup(3))", 6),
            ("SmallGroup(6,1)", 6), ("DihedralGroup(IsPermGroup,8)", 8),
            ("AlternatingGroup(4)", 12), ("SymmetricGroup(3)", 6),
        ):
            with self.subTest(expression=expression):
                group = GAPGroup.from_gap(expression)
                self.assertEqual(len(group.elements), order)

    def test_group_size_and_gap_construction_failures_are_clear(self):
        with self.assertRaisesRegex(GAPError, "max_order"):
            GAPGroup.from_gap("SymmetricGroup(5)", max_order=30)
        with self.assertRaises(GAPError):
            GAPGroup.from_gap("CyclicGroup(0)")
        with self.assertRaisesRegex(GAPError, "must be finite"):
            GAPGroup.from_gap("AbelianGroup([0])")
        with self.assertRaisesRegex(GAPError, "generators"):
            GAPGroup.from_gap("AbelianGroup([2,2])", ["X"])

    def test_bad_images_and_unknown_elements_are_rejected(self):
        group = GAPGroup.from_gap("CyclicGroup(2)")
        for bits in ([2], [-1], [0.0], [False], [], [0, 1], "0", None, {"g1": 0}):
            with self.subTest(bits=bits), self.assertRaises(GAPError):
                group.homomorphism_to_z2(bits)
        rho = group.homomorphism_to_z2([1])
        for operation in (
            lambda: group.mul("unknown", "1"), lambda: group.mul("1", "unknown"),
            lambda: group.inverse("unknown"), lambda: group.pow("unknown", 0),
            lambda: group.pow("1", 0.5), lambda: group.pow("1", True),
            lambda: rho("unknown"), lambda: rho([]),
        ):
            with self.assertRaises(GAPError):
                operation()

    def test_backend_objects_are_immutable_and_identical_requests_cached(self):
        group = GAPGroup.from_gap(" AbelianGroup( [2,2] ) ", ["X", "T"])
        self.assertIs(group, GAPGroup.from_gap("AbelianGroup([2,2])", ("X", "T")))
        rho = group.homomorphism_to_z2([0, 1])
        self.assertIs(rho, group.homomorphism_to_z2((0, 1)))
        with self.assertRaises(FrozenInstanceError):
            group.identity = "X"
        with self.assertRaises(TypeError):
            group._positions["X"] = 0
        with self.assertRaises(FrozenInstanceError):
            rho.images = (0, 0, 0, 0)
        with self.assertRaisesRegex(GAPError, "from_gap"):
            replace(group, identity="X")

    def test_native_constructors_cannot_bypass_group_or_homomorphism_invariants(self):
        with self.assertRaisesRegex(GAPError, "from_gap"):
            GAPGroup(expression="CyclicGroup(2)", generator_names=("T",), elements=("1", "T"),
                     generators=("T",), multiplication_table=(("1", "T"), ("T", "1")))
        group = GAPGroup.from_gap("CyclicGroup(2)", ["T"])
        for bits, images in (
            ((1,), (0, 2)), ((1,), (0, -1)), ((1,), (0, True)), ((1,), (0, 1.0)),
            ((1,), (0,)), ((1, 0), (0, 1)), ((0,), (0, 1)), ((True,), (0, 1)),
            ((1,), (1, 1)), ((1,), None), (None, (0, 1)),
        ):
            with self.subTest(bits=bits, images=images), self.assertRaises(GAPError):
                GAPZ2Homomorphism(group, bits, images)
        with self.assertRaises(GAPError):
            GAPZ2Homomorphism(None, (), ())
        rho = group.homomorphism_to_z2([1])
        with self.assertRaises(GAPError):
            replace(rho, images=(0, 2))
        native_images = [0, 1]
        native = GAPZ2Homomorphism(group, [1], native_images)
        native_images[1] = 2
        self.assertEqual(native.images, (0, 1))
        self.assertEqual(native.generator_images, (1,))

    def test_homomorphism_cache_preserves_exact_source_object(self):
        implicit_names = GAPGroup.from_gap("CyclicGroup(2)")
        explicit_names = GAPGroup.from_gap("CyclicGroup(2)", ["g1"])
        self.assertIsNot(implicit_names, explicit_names)
        self.assertEqual(implicit_names.elements, explicit_names.elements)
        implicit_rho = implicit_names.homomorphism_to_z2([1])
        explicit_rho = explicit_names.homomorphism_to_z2([1])
        self.assertIs(implicit_rho.source, implicit_names)
        self.assertIs(explicit_rho.source, explicit_names)
        self.assertIsNot(implicit_rho, explicit_rho)

    def test_rejects_inconsistent_reconstructed_group_or_rho_protocol(self):
        group = GAPGroup.from_gap("AbelianGroup([2,2])", ["P", "Q"])
        exported = group._gap_export
        size, identity, generators, table, inverses = exported
        gap_data = [
            size, identity + 1, [x + 1 for x in generators],
            [[x + 1 for x in row] for row in table], [x + 1 for x in inverses],
        ]
        for data in (
            [gap_data, [0, 0, 0, 0]],  # Wrong requested generator images.
            [gap_data, [0, 0, 1, 0]],  # Fails the homomorphism law.
            [gap_data, [0, 0, 1, 2]],  # Not a bit.
            [gap_data, []],
            [[1, 1, [], [[1]], [1]], [0]],  # Different reconstructed group.
            [],
        ):
            _construct_homomorphism.cache_clear()
            with self.subTest(data=data), patch("umtc.gap._run_gap", return_value=data):
                with self.assertRaises(GAPError):
                    group.homomorphism_to_z2([0, 1])


if __name__ == "__main__":
    unittest.main()
