"""
Unified Test Runner for Customer Churn Platform
Executes:
1. Preprocessing Pipeline Unit Tests (test_preprocessing.py)
2. API Endpoints & Recommender Tests (test_api_endpoints.py)
Generates comprehensive execution summary.
"""

import sys
import time
import unittest

import test_preprocessing
import test_api_endpoints


def run_full_suite():
    print("=" * 70, flush=True)
    print("CUSTOMER CHURN INTELLIGENCE PLATFORM - COMPLETE TEST SUITE", flush=True)
    print("=" * 70, flush=True)

    suite = unittest.TestSuite()
    loader = unittest.TestLoader()

    # Add Test Suites
    print("\n[Suite 1/2] Loading Preprocessing Pipeline Tests...", flush=True)
    suite.addTests(loader.loadTestsFromModule(test_preprocessing))

    print("[Suite 2/2] Loading API Endpoints & Recommender Tests...", flush=True)
    suite.addTests(loader.loadTestsFromModule(test_api_endpoints))

    total_tests = suite.countTestCases()
    print(f"\nTotal Unit Tests Discovered: {total_tests}", flush=True)
    print("-" * 70, flush=True)

    start_time = time.time()
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    elapsed = time.time() - start_time

    print("\n" + "=" * 70, flush=True)
    print("TEST SUITE EXECUTION SUMMARY", flush=True)
    print("=" * 70, flush=True)
    print(f"Total Tests Run: {result.testsRun}")
    print(f"Passed:         {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures:       {len(result.failures)}")
    print(f"Errors:         {len(result.errors)}")
    print(f"Execution Time: {elapsed:.2f} seconds")

    if result.wasSuccessful():
        print("\n>>> ALL TESTS PASSED SUCCESSFULLY! <<<")
        print("=" * 70, flush=True)
        return 0
    else:
        print("\n>>> TEST SUITE FAILED <<<")
        print("=" * 70, flush=True)
        return 1


if __name__ == '__main__':
    sys.exit(run_full_suite())
