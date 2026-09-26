"""Unit test of the pure functions in scripts/throughput_benchmark.py.

Runs on synthetic inputs only (no data, no Docker, no GPU, about a second). Requires the WSL2
venv (numpy, pyyaml):

    ~/venvs/socialmediadgg-phase2/bin/python "tests/test_throughput_benchmark_functions.py"

Prints "test_throughput_benchmark_functions: all checks passed" on success; an AssertionError
names the failing check otherwise.
"""

import importlib.util
import math
import pathlib
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
SCRIPT = HERE.parent / "scripts" / "throughput_benchmark.py"


def load_module():
    spec = importlib.util.spec_from_file_location("throughput_benchmark", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_units(m):
    assert m.mib(1048576) == 1.0, "mib"
    assert m.gib(22548578304) == 21.0, "gib of the ceiling"


def check_rank(m):
    idx = m.rank_candidates([5, 9, 9, 1], [1, 2, 3, 4], ["a", "b", "c", "d"], 3)
    assert idx == [2, 1, 0], idx


def cands(counts):
    return [{"doc_key": f"d{i}", "n_tokens": n} for i, n in enumerate(counts)]


def check_fill_batch(m):
    c = cands(
        [5823, 4863, 3123, 2703, 2703, 2525, 2418, 2228, 700, 512, 300, 200, 100, 12]
    )
    sel, total = m.fill_batch(c, 16000)
    assert [x["n_tokens"] for x in sel] == [
        5823,
        4863,
        3123,
        700,
        512,
        300,
        200,
        100,
        12,
    ], sel
    assert total == 15633, total
    sel, total = m.fill_batch(c, 10686)
    assert [x["n_tokens"] for x in sel] == [5823, 4863] and total == 10686, (
        "exact fill stops early"
    )
    assert m.fill_batch([], 1000) == ([], 0), "no candidates"
    assert m.fill_batch(c, 5) == ([], 0), "budget below every count"


def check_order(m):
    keys = [f"t1_{i:03d}" for i in range(20)]
    o1 = m.benchmark_order(keys, {3, 7}, 20260910)
    o2 = m.benchmark_order(keys, {3, 7}, 20260910)
    assert list(o1) == list(o2), "same seed, same order"
    assert sorted(int(i) for i in o1) == [i for i in range(20) if i not in (3, 7)], (
        "a permutation of the included rows"
    )
    assert list(o1) != sorted(o1), "shuffled"
    o3 = m.benchmark_order(keys, {3, 7}, 1)
    assert list(o3) != list(o1), "another seed, another order"
    # the position of a document depends only on (seed, doc_key), not on the row order
    rev = list(reversed(keys))
    o4 = m.benchmark_order(rev, {19 - 3, 19 - 7}, 20260910)
    assert [rev[i] for i in o4] == [keys[i] for i in o1], "row-order independent"


def packed(batches):
    return [
        (b["begin"], b["end"], b["n_documents"], b["n_tokens"], b["oversize"])
        for b in batches
    ]


def check_pack(m):
    n = [3, 4, 2, 5, 1, 1, 6]
    assert packed(m.pack_greedy(n, 0, 7, 4)) == [
        (0, 2, 2, 7, False),
        (2, 4, 2, 7, False),
        (4, 6, 2, 2, False),
        (6, 7, 1, 6, False),
    ], packed(m.pack_greedy(n, 0, 7, 4))
    assert packed(m.pack_greedy(n, 2, 7, 2)) == [
        (2, 4, 2, 7, False),
        (4, 6, 2, 2, False),
    ], "start inside the stream"
    assert packed(m.pack_greedy([9, 1, 2], 0, 5, 2)) == [
        (0, 1, 1, 9, True),
        (1, 3, 2, 3, False),
    ], "a document longer than the budget forms its own batch"
    assert packed(m.pack_greedy([1, 1], 0, 5, 3)) == [(0, 2, 2, 2, False)], (
        "stops at the end of the stream"
    )


def check_forward_passes(m):
    assert m.plan_forward_passes([5, 3, 3, 2, 1], 8) == [[0], [1, 2], [3, 4]]
    assert m.plan_forward_passes([2, 9, 2], 8) == [[1], [0, 2]], (
        "a document longer than the budget runs alone"
    )
    assert m.plan_forward_passes([4, 4, 4, 4], 8) == [[0, 1], [2, 3]]
    assert m.plan_forward_passes([], 8) == []
    assert m.padded_tokens([5, 3, 3, 2, 1], [[0], [1, 2], [3, 4]]) == 15


def check_timing(m):
    s = m.timing_stats([1.0, 2.0, 3.0, 4.0])
    assert s["n"] == 4 and s["total_s"] == 10.0 and s["mean_s"] == 2.5, s
    assert s["median_s"] == 2.5 and s["min_s"] == 1.0 and s["max_s"] == 4.0, s
    assert abs(s["std_s"] - math.sqrt(5.0 / 3.0)) < 1e-12, "sample standard deviation"
    one = m.timing_stats([2.0])
    assert one["std_s"] is None and one["median_s"] == 2.0, one


def check_steady_state(m):
    s = m.steady_state(
        [
            {"n_documents": 10, "n_tokens": 100, "time_s": 1.0},
            {"n_documents": 30, "n_tokens": 300, "time_s": 3.0},
        ]
    )
    assert s["total_documents"] == 40 and s["total_tokens"] == 400, s
    assert s["total_s"] == 4.0 and s["mean_s"] == 2.0, s
    assert s["documents_per_s"] == 10.0 and s["tokens_per_s"] == 100.0, s
    assert s["max_documents_in_batch"] == 30 and s["max_tokens_in_batch"] == 300, s
    assert s["min_documents_in_batch"] == 10 and s["min_tokens_in_batch"] == 100, s


def pt(budget, tps, failed=False, error=None):
    return {
        "budget": budget,
        "tokens_per_s": tps,
        "failed_at_ceiling": failed,
        "error": error,
    }


def check_sweep(m):
    nx = m.sweep_next
    assert nx([], 92160) == (1024, "start")
    assert nx([], 512) == (512, "start"), "start capped at the GPU budget"
    assert nx([], 92160, start=256) == (256, "start")
    assert nx([pt(1024, 100.0)], 92160) == (2048, "double")
    assert nx([pt(1024, 100.0), pt(2048, 120.0)], 92160) == (4096, "double")
    v, r = nx([pt(1024, 100.0), pt(2048, 120.0), pt(4096, 124.0)], 92160)
    assert v is None and r.startswith("no material improvement"), (v, r)
    v, r = nx([pt(1024, 100.0), pt(2048, 90.0)], 92160)
    assert v is None and r.startswith("no material improvement"), (
        "worse throughput stops",
        v,
        r,
    )
    v, r = nx([pt(1024, 100.0), pt(2048, 105.0)], 92160)
    assert v == 4096, ("exactly +5% continues", v, r)
    v, r = nx([pt(65536, 100.0)], 92160)
    assert v == 92160 and "GPU budget" in r, (v, r)
    v, r = nx([pt(65536, 100.0), pt(92160, 200.0)], 92160)
    assert v is None and "GPU budget" in r, (v, r)
    v, r = nx([pt(1024, 100.0), pt(2048, None, failed=True)], 92160)
    assert v is None and "RAM ceiling" in r and "2048" in r, (v, r)
    v, r = nx([pt(1024, 100.0), pt(2048, None, error="boom")], 92160)
    assert v is None and "boom" in r, (v, r)
    ch = m.sweep_choice
    assert ch([pt(1024, 100.0), pt(2048, 120.0), pt(4096, 124.0)])["budget"] == 4096
    assert ch([pt(1024, 100.0), pt(2048, None, failed=True)])["budget"] == 1024
    assert ch([pt(1024, 100.0), pt(2048, 100.0)])["budget"] == 1024, (
        "a tie goes to the smaller budget"
    )
    assert ch([]) is None and ch([pt(1024, None, error="x")]) is None


PROM_TEXT = """# HELP te_batch_next_size Summary
# TYPE te_batch_next_size histogram
te_batch_next_size_bucket{le="1"} 2
te_batch_next_size_bucket{le="2"} 2
te_batch_next_size_bucket{le="+Inf"} 3
te_batch_next_size_sum 9
te_batch_next_size_count 3
te_queue_size 0
te_request_failure{err="inference"} 1
"""


def check_prometheus(m):
    metrics = m.parse_prometheus(PROM_TEXT)
    assert metrics["te_queue_size"] == [({}, 0.0)], metrics["te_queue_size"]
    assert metrics["te_request_failure"] == [({"err": "inference"}, 1.0)], metrics
    h = m.histogram(metrics, "te_batch_next_size")
    assert h == {
        "buckets": {"1": 2.0, "2": 2.0, "+Inf": 3.0},
        "sum": 9.0,
        "count": 3.0,
    }, h
    later = m.parse_prometheus(
        PROM_TEXT.replace("} 3", "} 5")
        .replace("_sum 9", "_sum 20")
        .replace("_count 3", "_count 5")
    )
    d = m.histogram_delta(h, m.histogram(later, "te_batch_next_size"))
    assert d == {
        "buckets": {"1": 0.0, "2": 0.0, "+Inf": 2.0},
        "sum": 11.0,
        "count": 2.0,
    }, d
    assert m.histogram(metrics, "te_missing") == {
        "buckets": {},
        "sum": 0.0,
        "count": 0.0,
    }
    assert m.counter_value(metrics, "te_request_failure", {"err": "inference"}) == 1.0
    assert m.counter_value(metrics, "te_request_failure", {"err": "other"}) == 0.0


def check_verify_interval(m):
    # one empty poll, the two blockers (1 document each) and a batch of 7 documents
    size = {
        "buckets": {"1": 3, "2": 3, "4": 3, "8": 4, "+Inf": 4},
        "sum": 9,
        "count": 4,
    }
    tokens = {
        "buckets": {"1": 1, "4096": 1, "8192": 3, "16384": 4, "+Inf": 4},
        "sum": 5823 + 4863 + 9000,
        "count": 4,
    }
    expected = [(1, 5823), (1, 4863), (7, 9000)]
    ok, checks, info = m.verify_interval(size, tokens, expected)
    assert ok, checks
    assert info["n_empty_polls"] == 1 and info["real_batches"] == 3, info
    # the 7 documents split into batches of 3 and 4
    size2 = {
        "buckets": {"1": 3, "2": 3, "4": 5, "8": 5, "+Inf": 5},
        "sum": 9,
        "count": 5,
    }
    tokens2 = {
        "buckets": {"1": 1, "4096": 3, "8192": 5, "16384": 5, "+Inf": 5},
        "sum": 5823 + 4863 + 9000,
        "count": 5,
    }
    ok2, checks2, _ = m.verify_interval(size2, tokens2, expected)
    assert not ok2 and checks2["n_real_batches"] is False, checks2
    # right structure, wrong token sum
    ok3, checks3, _ = m.verify_interval(
        size, dict(tokens, sum=tokens["sum"] - 1), expected
    )
    assert not ok3 and checks3["token_sum"] is False, checks3
    # a later interval: one batch of 2,350 documents and 92,100 tokens plus one empty poll
    size4 = {
        "buckets": {"1": 1, "2048": 1, "4096": 2, "+Inf": 2},
        "sum": 2350,
        "count": 2,
    }
    tokens4 = {
        "buckets": {"1": 1, "65536": 1, "131072": 2, "+Inf": 2},
        "sum": 92100,
        "count": 2,
    }
    ok4, checks4, info4 = m.verify_interval(size4, tokens4, [(2350, 92100)])
    assert ok4 and info4["n_empty_polls"] == 1, (checks4, info4)
    # the same totals in the wrong bucket (a batch of 1,000 documents plus one of 1,350)
    size5 = {
        "buckets": {"1": 1, "1024": 2, "2048": 3, "4096": 3, "+Inf": 3},
        "sum": 2350,
        "count": 3,
    }
    ok5, _, _ = m.verify_interval(size5, tokens4, [(2350, 92100)])
    assert not ok5, "two batches are not one"


def check_cap(m):
    ceiling = 22548578304
    assert (
        m.compute_cap_bytes(ceiling, 1000 * 2**20, 835 * 2**20)
        == ceiling - 1835 * 2**20
    )
    assert m.compute_cap_bytes(100, 80, 50) == 0, "never negative"


def t(n, success):
    return {"max_batch_tokens": n, "success": success}


def check_next_value(m):
    nv = m.next_test_value
    assert nv([], 32768, 512) == (32768, "start")
    assert nv([t(32768, True)], 32768, 512) == (65536, "bracket")
    v, p = nv([t(32768, True), t(65536, True), t(131072, False)], 32768, 512)
    assert (v, p) == (98304, "bisect"), (v, p)
    v, p = nv([t(122880, True), t(123392, False)], 32768, 512)
    assert v is None and "resolution" in p, (v, p)
    v, p = nv([t(32768, False)], 32768, 512)
    assert v is None and "no permitted" in p, (v, p)
    v, p = nv([t(81920, True), t(82944, False)], 40960, 512)
    assert (v, p) == (82432, "bisect"), (v, p)


def check_classify(m):
    c = m.classify_outcome
    base = dict(
        started=True,
        below_minimum=False,
        refused=False,
        batch_ok=True,
        verified=True,
        watchdog=False,
        peak_over=False,
    )
    assert c(**base) == ("success", True, False)
    assert c(**dict(base, verified=False)) == (
        "inconclusive_unverified_batch",
        False,
        False,
    )
    assert c(**dict(base, peak_over=True)) == ("device_peak_over_ceiling", False, True)
    assert c(**dict(base, watchdog=True, peak_over=True)) == (
        "watchdog_kill",
        False,
        True,
    )
    assert c(**dict(base, refused=True, batch_ok=False, verified=False)) == (
        "batch_refused_at_cap",
        False,
        True,
    )
    assert c(
        **dict(base, started=False, refused=True, batch_ok=False, verified=False)
    ) == ("startup_refused_at_cap", False, True)
    assert c(
        **dict(base, started=False, below_minimum=True, batch_ok=False, verified=False)
    ) == ("refused_below_minimum", False, False)
    assert c(**dict(base, started=False, batch_ok=False, verified=False)) == (
        "startup_error",
        False,
        False,
    )
    assert c(**dict(base, batch_ok=False, verified=False)) == (
        "batch_error",
        False,
        False,
    )


LOG_LINES = [
    '{"timestamp":"2026-09-10T04:22:52.925Z","level":"INFO","message":"Starting model backend"}',
    '{"timestamp":"2026-09-10T04:22:53.720Z","level":"INFO","message":"Starting FlashQwen3 model on Cuda(CudaDevice(DeviceId(1)))"}',
    "[vram_cap] loaded: real=/x cap_bytes=1 trampolines=687 missing=0 intercepted=19 report=(none)",
    '{"timestamp":"2026-09-10T04:23:23.879Z","level":"INFO","message":"Warming up model"}',
    '{"timestamp":"2026-09-10T04:23:36.493Z","level":"INFO","message":"Ready"}',
]


def check_log_events(m):
    ev = m.parse_tei_log_events(LOG_LINES)
    assert abs(ev["load_s"] - 30.159) < 0.01 and abs(ev["warmup_s"] - 12.614) < 0.01, ev
    assert ev["model_variant"] == "FlashQwen3", ev
    errors = m.tei_log_errors(
        LOG_LINES
        + [
            '{"timestamp":"t","level":"ERROR","message":"Could not start Candle backend: DriverError(CUDA_ERROR_OUT_OF_MEMORY, \\"out of memory\\")"}',
            "Error: Could not create backend",
        ]
    )
    assert len(errors) == 2 and "OUT_OF_MEMORY" in errors[0], errors


def check_samples(m):
    samples = [(0.0, 10), (1.0, 50), (2.0, 20), (3.0, 30)]
    s = m.summarise_samples(samples, 0.5, 2.5)
    assert s == {"peak": 50, "t_peak": 1.0, "first": 50, "last": 20, "n_samples": 2}, s
    empty = m.summarise_samples(samples, 5.0, 6.0)
    assert empty["peak"] is None and empty["n_samples"] == 0, empty
    assert m.downsample_max(samples, 2.0) == [(0.0, 50), (2.0, 30)]


def check_peak_sampler(m):
    values = iter([10, 50, 20, 30])
    seen = []
    sampler = m.PeakSampler(
        lambda: next(values, 30), interval_s=0.001, on_sample=seen.append
    )
    sampler.start()
    time.sleep(0.1)
    result = sampler.stop()
    assert (
        result["peak"] == 50 and result["first"] == 10 and result["n_samples"] >= 4
    ), result
    assert seen[:2] == [10, 50], seen[:2]


def check_cosine(m):
    a = np.array([[1.0, 0.0], [0.6, 0.8], [1.0, 1.0]], dtype=np.float32)
    b = np.array([[1.0, 0.0], [0.8, 0.6], [-1.0, -1.0]], dtype=np.float32)
    assert np.allclose(m.cosine_rows(a, b), [1.0, 0.96, -1.0], atol=1e-6)
    s = m.agreement_summary(np.array([1.0, 0.9999, 0.995, 0.98]))
    assert s["n"] == 4 and s["n_nonfinite"] == 0, s
    assert abs(s["min"] - 0.98) < 1e-12 and s["n_below_0_999"] == 2, s
    assert s["n_below_0_99"] == 1 and abs(s["median"] - 0.99745) < 1e-9, s
    s2 = m.agreement_summary(np.array([1.0, np.nan]))
    assert s2["n_nonfinite"] == 1 and s2["min"] == 1.0, s2


def check_cgroup_parsers(m):
    d = m.parse_cpu_stat(
        "usage_usec 5000000\nuser_usec 4000000\nsystem_usec 1000000\nnr_periods 0\n"
    )
    assert d == {
        "usage_usec": 5000000,
        "user_usec": 4000000,
        "system_usec": 1000000,
        "nr_periods": 0,
    }, d
    assert m.cpu_utilisation(4_000_000, 1.0, 8) == (4.0, 0.5)
    assert m.cpu_utilisation(4_000_000, 0.0, 8) == (None, None)
    st = m.parse_proc_status(
        "Name:\tpython\nVmHWM:\t  2048 kB\nVmRSS:\t  1024 kB\nRssAnon:\t   512 kB\nRssFile:\t   512 kB\nThreads:\t9\n"
    )
    assert st == {
        "VmHWM": 2048 * 1024,
        "VmRSS": 1048576,
        "RssAnon": 524288,
        "RssFile": 524288,
        "Threads": 9,
    }, st


def check_json_diff(m):
    a = {"version": "1.0", "model": {"vocab": {"a": 1}}, "post_processor": {"x": 1}}
    b = {
        "version": "1.0",
        "model": {"vocab": {"a": 1}},
        "post_processor": {"x": 2},
        "padding": None,
    }
    assert m.json_top_level_diff(a, b) == ["padding", "post_processor"]
    assert m.json_top_level_diff(a, a) == []


def check_cpu_exit(m):
    limit = 22548578304
    c = m.classify_cpu_exit
    assert c(0, 0, 10**9, limit) == ("success", False)
    assert c(-9, 1, 10**9, limit) == ("killed_at_ram_ceiling", True)
    assert c(137, 0, int(limit * 0.99), limit) == ("killed_at_ram_ceiling", True)
    assert c(-9, 0, 10**9, limit) == ("killed", False)
    assert c(1, 0, 10**9, limit) == ("error", False)


def check_ratio(m):
    assert m.speed_ratio(200.0, 100.0) == 2.0
    assert m.speed_ratio(None, 1.0) is None and m.speed_ratio(1.0, 0.0) is None


def check_manifest(m):
    import yaml

    man = m.build_manifest(
        output_rel="pipeline/x/results/a.csv",
        output_hash="sha256:abc",
        output_format="csv",
        rows=3,
        cols=2,
        inputs=[
            {
                "path": "in.parquet",
                "hash": "sha256:def",
                "format": "parquet",
                "role": "input",
            }
        ],
        script_rel="pipeline/x/scripts/s.py",
        script_hash="sha256:123",
        parameters={"vram_ceiling_bytes": 22548578304, "note": "a: b # c"},
        packages=[{"name": "httpx", "version": "0.28.1"}],
        git_commit="abc1234",
        timestamp="2026-09-10T00:00:00Z",
        notes="line1\nline2",
        os_name="Linux-6.18",
        python_version="3.12.3",
    )
    assert list(man.keys())[:4] == [
        "manifest_version",
        "output_file",
        "output_hash",
        "output_format",
    ], man
    assert man["output_rows"] == 3 and man["schema_diff"] == {
        "added": [],
        "removed": [],
        "renamed": [],
        "retyped": [],
    }
    assert yaml.safe_load(m.manifest_yaml(man)) == man, "yaml round trip"


def check_render_table(m):
    table = m.render_table(["a", "b"], [{"a": 1, "b": "x|y"}, {"a": None, "b": "z"}])
    lines = table.splitlines()
    assert lines[0] == "| a | b |" and lines[1] == "|---|---|", lines
    assert lines[2] == "| 1 | x\\|y |" and lines[3] == "|  | z |", lines


def main():
    m = load_module()
    check_units(m)
    check_rank(m)
    check_fill_batch(m)
    check_order(m)
    check_pack(m)
    check_forward_passes(m)
    check_timing(m)
    check_steady_state(m)
    check_sweep(m)
    check_prometheus(m)
    check_verify_interval(m)
    check_cap(m)
    check_next_value(m)
    check_classify(m)
    check_log_events(m)
    check_samples(m)
    check_peak_sampler(m)
    check_cosine(m)
    check_cgroup_parsers(m)
    check_json_diff(m)
    check_cpu_exit(m)
    check_ratio(m)
    check_manifest(m)
    check_render_table(m)
    print("test_throughput_benchmark_functions: all checks passed")


if __name__ == "__main__":
    main()
