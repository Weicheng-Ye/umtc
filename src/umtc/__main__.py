"""Command line validation: python -m umtc category.json."""

import argparse
import json
import sys

from . import DataError, coherence_report, load_json
from .checks import DEFAULT_CHECKS


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Check numerical UMTC and symmetry data")
    parser.add_argument("path", help="UMTC JSON file")
    parser.add_argument("--checks", nargs="+", choices=DEFAULT_CHECKS,
                        help="Subset of checks; defaults to all")
    parser.add_argument("--atol", type=float, default=1e-9)
    parser.add_argument("--rtol", type=float, default=1e-9)
    parser.add_argument("--max-failures", type=int, default=20)
    parser.add_argument("--json", action="store_true", dest="json_output")
    args = parser.parse_args(argv)
    try:
        category = load_json(args.path)
        report = coherence_report(category, checks=args.checks, atol=args.atol,
                                  rtol=args.rtol, max_failures=args.max_failures)
    except (DataError, OSError, ValueError) as error:
        print(f"umtc-check: {error}", file=sys.stderr)
        return 2
    if args.json_output:
        print(json.dumps(report.to_dict(), indent=2, allow_nan=False))
    else:
        print(f"{category.name or args.path}: {'PASS' if report.ok else 'FAIL'}")
        print(f"{report.checked} scalar equations; {report.failure_count} failures; "
              f"maximum residual {report.max_error:.3g}")
        for failure in report.failures:
            print(f"  {failure.equation} {dict(failure.labels)}: "
                  f"{failure.lhs} != {failure.rhs} (residual {failure.error:.3g})")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
