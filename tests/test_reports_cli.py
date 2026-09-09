"""Public diagnostics and CLI behavior, including invalid numerical inputs."""

import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from umtc import UMTC, check_symmetry, coherence_report
from test_consistency import materialize_symbols


ROOT = Path(__file__).resolve().parents[1]


class ReportTests(unittest.TestCase):
    def data(self, name="fibonacci"):
        return json.loads((ROOT / "examples" / f"{name}.json").read_text())

    def test_invalid_tolerances_and_check_names(self):
        cat = UMTC.from_dict(self.data())
        for value in (-1, math.inf, math.nan, True):
            with self.assertRaises(ValueError):
                coherence_report(cat, atol=value)
        with self.assertRaises(ValueError):
            coherence_report(cat, checks=["typo"])
        with self.assertRaises(ValueError):
            coherence_report(cat, max_failures=-1)

    def test_report_limit_does_not_limit_equations_evaluated(self):
        data = self.data()
        for record in data["R_symbols"]:
            record["matrix"] = [[0]]
        cat = UMTC.from_dict(data)
        short = coherence_report(cat, checks="hexagon", max_failures=1)
        full = coherence_report(cat, checks=["hexagon"], max_failures=None)
        self.assertFalse(short.ok)
        self.assertEqual(len(short.failures), 1)
        self.assertGreater(full.failure_count, 1)
        self.assertEqual(short.failure_count, full.failure_count)
        self.assertEqual(short.checked, full.checked)
        self.assertEqual(len(full.failures), full.failure_count)

    def test_underflow_and_overflow_produce_failure_reports(self):
        for tiny_spin in (0, 1e-320):
            data = self.data()
            data["topological_spins"][1]["value"] = tiny_spin
            report = coherence_report(UMTC.from_dict(data))
            self.assertFalse(report.ok)
            self.assertTrue(any(f.equation == "modularity.defined" for f in report.failures))
        data = self.data()
        data["F_symbols"][0]["matrix"] = [[1e308]]
        report = coherence_report(UMTC.from_dict(data), checks=["unitarity"])
        self.assertFalse(report.ok)
        # Diagnostics are still standards-compliant JSON after arithmetic overflow.
        json.dumps(report.to_dict(), allow_nan=False)

    def test_invalid_group_is_exact_even_with_large_tolerance(self):
        data = materialize_symbols(self.data("toric_code_z2_z2t"))
        data["symmetry"]["multiplication_table"] = [["1"] * 4 for _ in range(4)]
        cat = UMTC.from_dict(data)
        self.assertFalse(check_symmetry(cat, atol=10))
        self.assertFalse(coherence_report(cat, checks=["symmetry"], atol=10).ok)

    def test_cli_success_and_json(self):
        result = subprocess.run([sys.executable, "-m", "umtc", str(ROOT / "examples" / "fibonacci.json"), "--json"],
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertTrue(report["ok"])
        self.assertGreater(report["checked"], 0)

    def test_cli_bad_data_and_bad_equations_have_distinct_exit_codes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            path.write_text("{}")
            result = subprocess.run([sys.executable, "-m", "umtc", str(path)], text=True, capture_output=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("schema_version", result.stderr)
            data = self.data()
            data["R_symbols"][-1]["matrix"] = [[0]]
            path.write_text(json.dumps(data))
            result = subprocess.run([sys.executable, "-m", "umtc", str(path)], text=True, capture_output=True)
            self.assertEqual(result.returncode, 1)
            self.assertIn("FAIL", result.stdout)
            self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
