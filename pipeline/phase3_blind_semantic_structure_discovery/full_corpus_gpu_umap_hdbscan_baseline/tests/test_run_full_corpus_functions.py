#!/usr/bin/env python3
"""Unit tests for the pure (non-GPU) helpers in run_full_corpus_umap_hdbscan.py.

Runnable directly (`python test_run_full_corpus_functions.py`) without pytest, and
also collectable by pytest if available. No GPU or cuML required.
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
import run_full_corpus_umap_hdbscan as R  # noqa: E402


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


def test_stage_failed_logic():
    assert R.stage_failed({"ok": False, "ceiling_breached": False}) is True
    assert R.stage_failed({"ok": True, "ceiling_breached": True}) is True
    assert R.stage_failed({"ok": True, "ceiling_breached": False}) is False


def _verif(quick, n, **overrides):
    emb5 = overrides.get("emb5d", np.zeros((n, 5), dtype=np.float32))
    c2d = overrides.get("coords2d", np.zeros((n, 2), dtype=np.float32))
    labels = overrides.get("labels", np.zeros((n,), dtype=np.int64))
    keys = overrides.get(
        "doc_keys", np.array([f"t1_{i}" for i in range(n)], dtype=object)
    )
    return R.build_verification(
        quick=quick,
        n=n,
        input_rows=overrides.get("input_rows", n),
        emb5d=emb5,
        coords2d=c2d,
        labels=labels,
        doc_keys=keys,
        input_all_finite=overrides.get("input_all_finite", True),
        row_order_ok=overrides.get("row_order_ok", True),
        emb_hash_before=overrides.get("emb_hash_before", "sha256:aaa"),
        emb_hash_after=overrides.get("emb_hash_after", "sha256:aaa"),
        idx_hash_before=overrides.get("idx_hash_before", "sha256:bbb"),
        idx_hash_after=overrides.get("idx_hash_after", "sha256:bbb"),
    )


def test_verification_quick_mode_pass():
    v = _verif(quick=True, n=8)
    # the "== 902,144" checks are marked not-applicable in quick mode
    assert v["input_rows_equals_902144"] == "n/a (quick mode)"
    assert v["umap5d_rows_equals_902144"] == "n/a (quick mode)"
    assert v["hdbscan_rows_equals_902144"] == "n/a (quick mode)"
    # internal-consistency + alignment + finiteness all pass
    assert v["input_rows_equal_working_set"] is True
    assert v["every_doc_has_exactly_one_label"] is True
    assert v["doc_keys_unique_no_dupes_introduced"] is True
    assert v["embeddings_unchanged"] is True
    assert v["index_unchanged"] is True
    assert v["all_boolean_checks_pass"] is True


def test_verification_full_mode_marks_902144_checks_bool_and_fails_when_short():
    # full (non-quick) mode with n != 902,144 -> the 902,144 checks are booleans (False)
    v = _verif(quick=False, n=8)
    assert v["input_rows_equals_902144"] is False
    assert v["umap5d_rows_equals_902144"] is False
    assert v["umap2d_rows_equals_902144"] is False
    assert v["hdbscan_rows_equals_902144"] is False
    # internal consistency still holds against the working set
    assert v["input_rows_equal_working_set"] is True
    # but the overall gate fails because the 902,144 checks are False
    assert v["all_boolean_checks_pass"] is False


def test_verification_detects_changed_input():
    v = _verif(quick=True, n=5, emb_hash_after="sha256:CHANGED")
    assert v["embeddings_unchanged"] is False
    assert v["all_boolean_checks_pass"] is False


def test_verification_detects_duplicate_doc_keys():
    keys = np.array(["t1_a", "t1_a", "t1_b", "t1_c", "t1_d"], dtype=object)
    v = _verif(quick=True, n=5, doc_keys=keys)
    assert v["doc_keys_unique_no_dupes_introduced"] is False
    assert v["n_unique_doc_keys"] == 4
    assert v["all_boolean_checks_pass"] is False


def test_verification_detects_nonfinite_coords():
    bad = np.zeros((5, 5), dtype=np.float32)
    bad[0, 0] = np.inf
    v = _verif(quick=True, n=5, emb5d=bad)
    assert v["umap5d_all_finite"] is False
    assert v["all_boolean_checks_pass"] is False


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
