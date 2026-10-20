"""Run unittest checks with persistent JSON counts, logs, and explicit skips."""

import argparse
import contextlib
import json
from pathlib import Path
import sys
import unittest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    tests = args.project_dir.resolve() / "tests"
    with (output / "tests.log").open("w", encoding="utf-8") as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            suite = unittest.defaultTestLoader.discover(str(tests), pattern="test_*.py")
            result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    success = result.wasSuccessful() and result.testsRun > 0
    report = {"status": ("PASS_WITH_SKIPS" if result.skipped else "PASS") if success else "FAIL",
              "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
              "skipped": [{"test": str(test), "reason": reason} for test, reason in result.skipped],
              "failed_tests": [{"test": str(test), "traceback": detail}
                               for test, detail in result.failures + result.errors]}
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print((output / "tests.log").read_text(encoding="utf-8"))
    print(json.dumps(report, indent=2))
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
