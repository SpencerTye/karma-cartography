#!/usr/bin/env python3
"""Unit tests for the pure helpers in run_top_topics_umap.py.

Runnable directly (`python test_top_topics_functions.py`) without pytest. No GPU or cuML
required.
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
import run_top_topics_umap as T  # noqa: E402


def test_top_cluster_ids_orders_by_size_and_skips_noise():
    labels = np.array([-1, -1, -1, -1, 3, 3, 3, 7, 7, 2, 5, 5, 5, 5])
    assert T.top_cluster_ids(labels, 3, exclude=()) == [5, 3, 7]


def test_top_cluster_ids_excludes_placeholder():
    labels = np.array([160] * 9 + [1] * 4 + [2] * 3)
    assert T.top_cluster_ids(labels, 2) == [1, 2]


def test_top_cluster_ids_breaks_ties_by_smaller_id():
    labels = np.array([9, 9, 4, 4, 6, 6])
    assert T.top_cluster_ids(labels, 3, exclude=()) == [4, 6, 9]


def test_select_rows_returns_matching_positions_in_order():
    labels = np.array([0, 5, -1, 5, 2, 0])
    assert T.select_rows(labels, [5, 0]).tolist() == [0, 1, 3, 5]


def test_check_alignment_passes_and_fails():
    keys = np.array(["t1_a", "t3_b", "t1_c"], dtype=object)
    T.check_alignment(keys, keys.copy())
    for bad in (keys[::-1], keys[:2]):
        try:
            T.check_alignment(keys, bad)
        except ValueError:
            continue
        raise AssertionError("misaligned keys were not rejected")


def test_md_tag():
    assert T.md_tag(0.0) == "md0.0"
    assert T.md_tag(0.5) == "md0.5"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
