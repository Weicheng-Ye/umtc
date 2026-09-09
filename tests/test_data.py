"""Structural loader checks, independent of coherence checks."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import tempfile
import unittest

from umtc.data import DataError, UMTC, load_json


def unit_data():
    return {
        "schema_version": 1,
        "name": "Unit category",
        "anyons": ["1"],
        "vacuum": "1",
        "fusion_rules": [{"a": "1", "b": "1", "c": "1", "multiplicity": 1}],
        "quantum_dimensions": [{"anyon": "1", "value": 1}],
        "topological_spins": [{"anyon": "1", "value": 1}],
        "F_symbols": [{"a": "1", "b": "1", "c": "1", "d": "1", "matrix": [[1]]}],
        "R_symbols": [{"a": "1", "b": "1", "c": "1", "matrix": [[1]]}],
    }


def with_symmetry(data):
    data["symmetry"] = {
        "elements": ["id"],
        "identity": "id",
        "multiplication_table": [["id"]],
        "antiunitary": [],
        "action": [{"g": "id", "images": ["1"]}],
        "U_symbols": [{"g": "id", "a": "1", "b": "1", "c": "1", "matrix": [[1]]}],
        "eta_symbols": [{"anyon": "1", "g": "id", "h": "id", "value": 1}],
    }
    return data


def relabel_unit(data, label):
    data["vacuum"] = label
    for name in ("fusion_rules", "quantum_dimensions", "topological_spins", "F_symbols", "R_symbols"):
        for record in data[name]:
            for key in ("a", "b", "c", "d", "anyon"):
                if key in record:
                    record[key] = label
    return data


class DataTests(unittest.TestCase):
    def test_unit_accessors(self):
        cat = UMTC.from_dict(unit_data())
        self.assertEqual(cat.anyons, ("1",))
        self.assertEqual(cat.vacuum, "1")
        self.assertEqual(cat.fusion("1", "1"), {"1": 1})
        self.assertEqual(cat.N("1", "1", "1"), 1)
        self.assertEqual(cat.quantum_dimension("1"), 1)
        self.assertEqual(cat.spin("1"), 1 + 0j)
        self.assertEqual(cat.total_quantum_dimension, 1)
        self.assertEqual(cat.F("1", "1", "1", "1", "1", "1"), 1)
        self.assertEqual(cat.R("1", "1", "1"), 1)
        self.assertIsNone(cat.symmetry)

    def test_tuple_and_integer_labels(self):
        formats = ([7], [[0]], {"type": "tuples", "labels": [[0]]}, {"type": "tuples", "moduli": [1]})
        for anyons in formats:
            label = 7 if anyons == [7] else [0]
            data = relabel_unit(unit_data(), label)
            data["anyons"] = anyons
            cat = UMTC.from_dict(data)
            expected = 7 if label == 7 else (0,)
            self.assertEqual(cat.anyons, (expected,))
            self.assertEqual(cat.N(expected, expected, expected), 1)

    def test_moduli_order_and_tuple_access(self):
        path = Path(__file__).resolve().parents[1] / "examples" / "toric_code.json"
        cat = UMTC.from_json(path)
        self.assertEqual(cat.anyons, ((0, 0), (0, 1), (1, 0), (1, 1)))
        self.assertEqual(cat.N((1, 0), (0, 1), (1, 1)), 1)
        self.assertEqual(cat.N((1, 0), (0, 1), (0, 0)), 0)
        self.assertEqual(cat.R((1, 0), (0, 1), (0, 0)), 0)
        self.assertEqual(cat.F((1, 0), (0, 1), (0, 0), (0, 0), (1, 1), (0, 1)), 0)
        self.assertEqual(cat.F_matrix((1, 0), (0, 1), (0, 0), (0, 0)), ())

    def test_fusion_multiplicity_basis_order(self):
        data = unit_data()
        data["fusion_rules"][0]["multiplicity"] = 2
        matrix = [[10 * i + j for j in range(4)] for i in range(4)]
        data["F_symbols"][0]["matrix"] = matrix
        data["R_symbols"][0]["matrix"] = [[1, 2], [3, 4]]
        cat = UMTC.from_dict(data)
        basis = (("1", 0, 0), ("1", 0, 1), ("1", 1, 0), ("1", 1, 1))
        self.assertEqual(cat.F_left_basis("1", "1", "1", "1"), basis)
        self.assertEqual(cat.F_right_basis("1", "1", "1", "1"), basis)
        self.assertEqual(cat.F("1", "1", "1", "1", "1", "1", 1, 0, 0, 1), 21)
        self.assertEqual(cat.R("1", "1", "1", 1, 0), 3)
        self.assertEqual(cat.R("1", "1", "1", 2, 0), 0)
        self.assertEqual(cat.F("1", "1", "1", "1", "1", "1", alpha=2), 0)

    def test_complex_values_and_input_independence(self):
        data = unit_data()
        data["F_symbols"][0]["matrix"] = [[{"re": 0.5, "im": -0.25}]]
        cat = UMTC.from_dict(data)
        data["F_symbols"][0]["matrix"][0][0]["re"] = 9
        self.assertEqual(cat.F_matrix("1", "1", "1", "1"), ((0.5 - 0.25j,),))

    def test_data_are_read_only(self):
        cat = UMTC.from_dict(with_symmetry(unit_data()))
        with self.assertRaises(FrozenInstanceError):
            cat.name = "changed"
        with self.assertRaises(TypeError):
            cat.fusion_rules[("1", "1", "1")] = 2
        with self.assertRaises(TypeError):
            cat.symmetry.action["id"] = ()
        with self.assertRaises(TypeError):
            cat.F_matrix("1", "1", "1", "1")[0][0] = 2

    def test_unknown_labels_never_mean_zero(self):
        cat = UMTC.from_dict(with_symmetry(unit_data()))
        calls = [lambda: cat.N("missing", "1", "1"),
                 lambda: cat.F("1", "1", "1", "1", "missing", "1"),
                 lambda: cat.R("1", "missing", "1"),
                 lambda: cat.quantum_dimension("missing"),
                 lambda: cat.symmetry.act("id", "missing"),
                 lambda: cat.symmetry.mul("missing", "id"),
                 lambda: cat.symmetry.U("missing", "1", "1", "1"),
                 lambda: cat.symmetry.eta("1", "id", "missing")]
        for call in calls:
            with self.subTest(call=call), self.assertRaises(DataError):
                call()

    def test_invalid_indices(self):
        cat = UMTC.from_dict(unit_data())
        for index in (-1, True, 0.5):
            with self.subTest(index=index), self.assertRaises(DataError):
                cat.R("1", "1", "1", index)

    def test_schema_version_and_required_fields(self):
        for version in (None, True, 1.0, 2):
            data = unit_data()
            data["schema_version"] = version
            with self.subTest(version=version), self.assertRaises(DataError):
                UMTC.from_dict(data)
        for name in ("schema_version", "anyons", "vacuum", "fusion_rules", "quantum_dimensions",
                     "topological_spins", "F_symbols", "R_symbols"):
            data = unit_data()
            del data[name]
            with self.subTest(field=name), self.assertRaises(DataError):
                UMTC.from_dict(data)

    def test_invalid_anyon_forms(self):
        for value in ([], [True], [1.5], ["1", "1"], [[0], [0, 0]],
                      {"type": "tuples", "moduli": [True]},
                      {"type": "tuples", "moduli": [0]},
                      {"type": "tuples", "moduli": [1], "labels": [[0]]},
                      {"type": "tuples", "labels": ["1"]}):
            data = unit_data()
            data["anyons"] = value
            with self.subTest(anyons=value), self.assertRaises(DataError):
                UMTC.from_dict(data)

    def test_duplicate_and_missing_records(self):
        for field in ("fusion_rules", "quantum_dimensions", "topological_spins", "F_symbols", "R_symbols"):
            data = unit_data()
            data[field].append(deepcopy(data[field][0]))
            with self.subTest(duplicate=field), self.assertRaisesRegex(DataError, "duplicate"):
                UMTC.from_dict(data)
        for field in ("quantum_dimensions", "topological_spins", "F_symbols", "R_symbols"):
            data = unit_data()
            data[field] = []
            with self.subTest(missing=field), self.assertRaisesRegex(DataError, "missing"):
                UMTC.from_dict(data)

    def test_invalid_fusion_multiplicity(self):
        for multiplicity in (0, -1, True, 1.0, "1"):
            data = unit_data()
            data["fusion_rules"][0]["multiplicity"] = multiplicity
            with self.subTest(multiplicity=multiplicity), self.assertRaises(DataError):
                UMTC.from_dict(data)

    def test_invalid_symbol_values(self):
        for value in (True, "exp(1j)", float("inf"), float("nan"),
                      {"re": 1}, {"re": 1, "im": float("inf")},
                      {"re": True, "im": 0}, {"re": 1, "im": 0, "extra": 1}):
            data = unit_data()
            data["F_symbols"][0]["matrix"] = [[value]]
            with self.subTest(value=value), self.assertRaises(DataError):
                UMTC.from_dict(data)

    def test_dimensions_require_finite_real_numbers(self):
        for value in (True, "1", {"re": 1, "im": 0}, float("nan"), 10 ** 1000):
            data = unit_data()
            data["quantum_dimensions"][0]["value"] = value
            with self.subTest(value=value), self.assertRaises(DataError):
                UMTC.from_dict(data)

    def test_matrix_shape(self):
        for field in ("F_symbols", "R_symbols"):
            for matrix in ([], [[1, 0]], [[1], [0]], [1], "identity"):
                data = unit_data()
                data[field][0]["matrix"] = matrix
                with self.subTest(field=field, matrix=matrix), self.assertRaises(DataError):
                    UMTC.from_dict(data)

    def test_wrong_but_structurally_valid_symbols_load(self):
        data = with_symmetry(unit_data())
        data["F_symbols"][0]["matrix"] = [[2]]
        data["R_symbols"][0]["matrix"] = [[3]]
        data["quantum_dimensions"][0]["value"] = -1
        data["symmetry"]["eta_symbols"][0]["value"] = 0
        cat = UMTC.from_dict(data)
        self.assertEqual(cat.F("1", "1", "1", "1", "1", "1"), 2)
        self.assertEqual(cat.symmetry.eta("1", "id", "id"), 0)

    def test_symmetry_accessors(self):
        sym = UMTC.from_dict(with_symmetry(unit_data())).symmetry
        self.assertEqual(sym.elements, ("id",))
        self.assertEqual(sym.mul("id", "id"), "id")
        self.assertEqual(sym.inverse("id"), "id")
        self.assertEqual(sym.act("id", "1"), "1")
        self.assertFalse(sym.is_antiunitary("id"))
        self.assertEqual(sym.U_matrix("id", "1", "1", "1"), ((1 + 0j,),))
        self.assertEqual(sym.U("id", "1", "1", "1"), 1)
        self.assertEqual(sym.eta("1", "id", "id"), 1)

    def test_symmetry_requires_full_coverage(self):
        for field in ("action", "U_symbols", "eta_symbols"):
            data = with_symmetry(unit_data())
            data["symmetry"][field] = []
            with self.subTest(field=field), self.assertRaisesRegex(DataError, "missing"):
                UMTC.from_dict(data)

    def test_symmetry_rejects_duplicates(self):
        for field in ("action", "U_symbols", "eta_symbols"):
            data = with_symmetry(unit_data())
            data["symmetry"][field] *= 2
            with self.subTest(field=field), self.assertRaisesRegex(DataError, "duplicate"):
                UMTC.from_dict(data)

    def test_symmetry_shapes_and_unknown_elements(self):
        mutations = [("multiplication_table", []), ("multiplication_table", [["missing"]]),
                     ("elements", ["id", "id"]), ("identity", "missing"),
                     ("antiunitary", ["missing"]), ("antiunitary", ["id", "id"]),
                     ("action", [{"g": "id", "images": []}]),
                     ("U_symbols", [{"g": "id", "a": "1", "b": "1", "c": "1", "matrix": []}])]
        for field, value in mutations:
            data = with_symmetry(unit_data())
            data["symmetry"][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(DataError):
                UMTC.from_dict(data)

    def test_file_loading_and_json_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "category.json"
            path.write_text(json.dumps(unit_data()), encoding="utf-8")
            self.assertEqual(load_json(path).name, "Unit category")
            path.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
            with self.assertRaisesRegex(DataError, "duplicate key"):
                UMTC.from_json(path)
            path.write_text('{"schema_version":', encoding="utf-8")
            with self.assertRaisesRegex(DataError, "Invalid JSON"):
                UMTC.from_json(path)


if __name__ == "__main__":
    unittest.main()
