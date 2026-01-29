#!/usr/bin/env python
"""
Run All Tests for timeseries_toolkit Package

This script executes the complete test suite and reports pass/fail for each module.
It also provides a summary of test coverage and any issues found.

Usage:
    python tests/run_all_tests.py

Options:
    --verbose, -v    Show detailed output
    --stop, -x       Stop on first failure
"""

import subprocess
import sys
import time
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class ModuleTestResult:
    """Result of testing a single module."""
    module_name: str
    passed: int
    failed: int
    skipped: int
    errors: int
    duration: float
    success: bool


def run_module_tests(module_path: str, verbose: bool = False) -> ModuleTestResult:
    """
    Run tests for a specific module.

    Args:
        module_path: Path to the test module
        verbose: Whether to show detailed output

    Returns:
        ModuleTestResult with test statistics
    """
    module_name = module_path.split('/')[-1].replace('test_', '').replace('.py', '')

    cmd = [sys.executable, '-m', 'pytest', module_path, '-v', '--tb=short']

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout per module
        )
        duration = time.time() - start_time

        output = result.stdout + result.stderr

        # Parse results from pytest output
        passed = failed = skipped = errors = 0

        for line in output.split('\n'):
            if 'passed' in line and ('failed' in line or 'error' in line or 'skipped' in line or line.strip().endswith('passed')):
                # Parse summary line like "5 passed, 1 failed in 2.5s"
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == 'passed' or part == 'passed,':
                        passed = int(parts[i-1])
                    elif part == 'failed' or part == 'failed,':
                        failed = int(parts[i-1])
                    elif part == 'skipped' or part == 'skipped,':
                        skipped = int(parts[i-1])
                    elif part == 'error' or part == 'errors' or part == 'error,':
                        errors = int(parts[i-1])

        if verbose:
            print(output)

        success = failed == 0 and errors == 0

        return ModuleTestResult(
            module_name=module_name,
            passed=passed,
            failed=failed,
            skipped=skipped,
            errors=errors,
            duration=duration,
            success=success
        )

    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        return ModuleTestResult(
            module_name=module_name,
            passed=0,
            failed=0,
            skipped=0,
            errors=1,
            duration=duration,
            success=False
        )
    except Exception as e:
        duration = time.time() - start_time
        if verbose:
            print(f"Error running tests for {module_name}: {e}")
        return ModuleTestResult(
            module_name=module_name,
            passed=0,
            failed=0,
            skipped=0,
            errors=1,
            duration=duration,
            success=False
        )


def check_imports() -> List[Tuple[str, bool, str]]:
    """
    Check that all modules can be imported.

    Returns:
        List of (module_name, success, error_message) tuples
    """
    results = []

    imports_to_check = [
        ('timeseries_toolkit', 'import timeseries_toolkit'),
        ('preprocessing.fractional_diff', 'from timeseries_toolkit.preprocessing import fractional_diff'),
        ('preprocessing.filtering', 'from timeseries_toolkit.preprocessing import filtering'),
        ('preprocessing.imputation', 'from timeseries_toolkit.preprocessing import imputation'),
        ('models.kalman', 'from timeseries_toolkit.models import kalman'),
        ('validation.causality', 'from timeseries_toolkit.validation import causality'),
        ('validation.diagnostics', 'from timeseries_toolkit.validation import diagnostics'),
        ('utils.data_loader', 'from timeseries_toolkit.utils import data_loader'),
    ]

    # Optional imports
    optional_imports = [
        ('models.regime', 'from timeseries_toolkit.models.regime import RegimeDetector'),
        ('models.forecaster', 'from timeseries_toolkit.models.forecaster import GlobalBoostForecaster'),
    ]

    for name, import_stmt in imports_to_check:
        try:
            exec(import_stmt)
            results.append((name, True, ''))
        except ImportError as e:
            results.append((name, False, str(e)))
        except Exception as e:
            results.append((name, False, str(e)))

    for name, import_stmt in optional_imports:
        try:
            exec(import_stmt)
            results.append((name + ' (optional)', True, ''))
        except ImportError as e:
            results.append((name + ' (optional)', True, f'Skipped: {e}'))

    return results


def print_header(text: str, char: str = '='):
    """Print a formatted header."""
    print()
    print(char * 70)
    print(f" {text}")
    print(char * 70)


def print_result_row(name: str, status: str, detail: str = ''):
    """Print a formatted result row."""
    status_color = {
        'PASS': '\033[92m',  # Green
        'FAIL': '\033[91m',  # Red
        'SKIP': '\033[93m',  # Yellow
    }
    reset = '\033[0m'

    color = status_color.get(status, '')
    print(f"  {name:<35} [{color}{status:^6}{reset}] {detail}")


def main():
    """Main entry point for test runner."""
    verbose = '-v' in sys.argv or '--verbose' in sys.argv
    stop_on_fail = '-x' in sys.argv or '--stop' in sys.argv

    print_header("TIMESERIES_TOOLKIT TEST SUITE")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  Verbose: {verbose}")

    # Step 1: Check imports
    print_header("STEP 1: Import Checks", '-')

    import_results = check_imports()
    import_failures = 0

    for name, success, error in import_results:
        if success:
            if 'Skipped' in error:
                print_result_row(name, 'SKIP', error)
            else:
                print_result_row(name, 'PASS')
        else:
            print_result_row(name, 'FAIL', error)
            import_failures += 1

    if import_failures > 0:
        print(f"\n  WARNING: {import_failures} import(s) failed!")

    # Step 2: Run module tests
    print_header("STEP 2: Module Tests", '-')

    test_modules = [
        'tests/test_fractional_diff.py',
        'tests/test_filtering.py',
        'tests/test_imputation.py',
        'tests/test_kalman.py',
        'tests/test_regime.py',
        'tests/test_forecaster.py',
        'tests/test_causality.py',
        'tests/test_diagnostics.py',
        'tests/test_utils.py',
    ]

    all_results = []
    total_passed = 0
    total_failed = 0
    total_skipped = 0
    total_errors = 0

    for module_path in test_modules:
        print(f"\n  Testing: {module_path}...")
        result = run_module_tests(module_path, verbose=verbose)
        all_results.append(result)

        total_passed += result.passed
        total_failed += result.failed
        total_skipped += result.skipped
        total_errors += result.errors

        status = 'PASS' if result.success else 'FAIL'
        detail = f"{result.passed}p/{result.failed}f/{result.skipped}s in {result.duration:.1f}s"
        print_result_row(result.module_name, status, detail)

        if stop_on_fail and not result.success:
            print("\n  Stopping on first failure (--stop flag)")
            break

    # Step 3: Summary
    print_header("TEST SUMMARY")

    passed_modules = sum(1 for r in all_results if r.success)
    failed_modules = sum(1 for r in all_results if not r.success)

    print(f"\n  Modules Passed:  {passed_modules}/{len(all_results)}")
    print(f"  Total Tests:     {total_passed + total_failed + total_skipped + total_errors}")
    print(f"  - Passed:        {total_passed}")
    print(f"  - Failed:        {total_failed}")
    print(f"  - Skipped:       {total_skipped}")
    print(f"  - Errors:        {total_errors}")

    total_duration = sum(r.duration for r in all_results)
    print(f"\n  Total Duration:  {total_duration:.1f}s")

    # Final result
    print_header("FINAL RESULT")

    if failed_modules == 0 and total_errors == 0 and total_failed == 0:
        print("\033[92m  ALL TESTS PASSED!\033[0m")
        if import_failures > 0:
            print(f"  (Note: {import_failures} import check(s) failed - run 'pip install -e .' to install package)")
        return 0
    else:
        print(f"\033[91m  {failed_modules} module(s) failed, {total_failed} test(s) failed\033[0m")

        # List failed modules
        if failed_modules > 0:
            print("\n  Failed modules:")
            for r in all_results:
                if not r.success:
                    print(f"    - {r.module_name}: {r.failed} failed, {r.errors} errors")

        return 1


if __name__ == '__main__':
    sys.exit(main())
