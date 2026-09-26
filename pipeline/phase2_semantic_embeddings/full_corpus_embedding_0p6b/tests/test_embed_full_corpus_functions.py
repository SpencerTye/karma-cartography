#!/usr/bin/env python
"""Unit test of the pure functions of embed_full_corpus.py (no data, no Docker, no GPU)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import embed_full_corpus as e  # noqa: E402


def approx(a, b, t=1e-9):
    return abs(a - b) <= t


def main():
    # eta_from_timed: plain rates and remaining time
    r = e.eta_from_timed(100, 1000, 10.0, 50, 500)
    assert approx(r["documents_per_s"], 10.0)
    assert approx(r["tokens_per_s"], 100.0)
    assert approx(r["eta_s_by_tokens"], 5.0)
    assert approx(r["eta_s_by_docs"], 5.0)
    # zero elapsed -> guarded to None
    r0 = e.eta_from_timed(1, 1, 0.0, 1, 1)
    assert r0["tokens_per_s"] is None and r0["eta_s_by_tokens"] is None

    # fmt_hms
    assert e.fmt_hms(None) is None
    assert e.fmt_hms(0) == "0:00:00"
    assert e.fmt_hms(3661) == "1:01:01"

    # plan_requests: greedy token packing by budget (TEI's non-padded rule)
    reqs = e.plan_requests([100, 100, 100, 100], 250)
    assert (
        reqs[0]["begin"] == 0 and reqs[0]["end"] == 2 and reqs[0]["n_tokens"] == 200
    ), reqs
    assert (
        reqs[1]["begin"] == 2 and reqs[1]["end"] == 4 and reqs[1]["n_tokens"] == 200
    ), reqs
    assert sum(x["n_documents"] for x in reqs) == 4
    # the whole stream is covered exactly once, in order
    covered = []
    for x in reqs:
        covered.extend(range(x["begin"], x["end"]))
    assert covered == list(range(4)), covered
    # a single document larger than the budget forms its own batch (should not occur in this run)
    reqs2 = e.plan_requests([500, 10], 250)
    assert reqs2[0]["n_documents"] == 1 and reqs2[0]["oversize"] is True, reqs2

    # EXPECTED_ROWS anchor is the canonical corpus size
    assert e.EXPECTED_ROWS == 902144

    print("test_embed_full_corpus_functions: all checks passed")


if __name__ == "__main__":
    main()
