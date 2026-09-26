#!/usr/bin/env python3
"""Unit tests for the pure (non-GPU) helpers in run_umap_hdbscan.py.

Runnable directly (`python test_run_umap_hdbscan_functions.py`) without pytest,
and also collectable by pytest if available. No GPU or cuML required.
"""

import os
import sys

import numpy as np

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"
    ),
)
import run_umap_hdbscan as R  # noqa: E402


def test_duplicate_stats_basic():
    a = np.array([[1, 2], [1, 2], [3, 4], [3, 4], [3, 4], [5, 6]], dtype=np.float32)
    d = R.duplicate_stats(a)
    assert d["n_rows"] == 6
    assert d["n_unique_rows"] == 3
    assert d["max_tie_group_size"] == 3
    assert d["n_tied_groups"] == 2
    assert d["n_rows_in_tied_groups"] == 5


def test_duplicate_stats_all_unique():
    a = np.arange(12, dtype=np.float32).reshape(6, 2)
    d = R.duplicate_stats(a)
    assert d["n_unique_rows"] == 6
    assert d["max_tie_group_size"] == 1
    assert d["n_tied_groups"] == 0


def test_relpath_strips_repo_root():
    p = R.REPO_ROOT + "/pipeline/x/y.parquet"
    assert R.relpath(p) == "pipeline/x/y.parquet"


def test_full_dataset_estimate_structure():
    meas = [
        {"op": "umap", "wall_clock_s": 10.0, "gpu_peak_gib": 2.0},
        {"op": "hdbscan", "wall_clock_s": 5.0, "gpu_peak_gib": 1.0},
    ]
    est = R.full_dataset_estimate(meas, meas[0], meas[1])
    assert est["target_n_rows"] == R.FULL_CORPUS_ROWS
    assert est["measured"]["n_rows"] == R.EXPECTED_ROWS
    # linear scaling factor ~8.87x; n log n a bit higher
    assert est["scale_ratio_linear"] > 8.0
    assert est["scale_ratio_n_log_n"] > est["scale_ratio_linear"]
    assert est["extrapolated_runtime_s"]["umap_linear"] > 10.0
    assert (
        est["extrapolated_runtime_s"]["umap_n_log_n"]
        >= est["extrapolated_runtime_s"]["umap_linear"]
    )
    # HDBSCAN quadratic upper case must exceed the linear lower bound
    assert (
        est["extrapolated_runtime_s"]["hdbscan_quadratic"]
        > est["extrapolated_runtime_s"]["hdbscan_linear_lower_bound"]
    )
    assert est["assumptions"] and est["not_reliably_estimable_from_one_run"]


def _run_all():
    tests = [
        v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
