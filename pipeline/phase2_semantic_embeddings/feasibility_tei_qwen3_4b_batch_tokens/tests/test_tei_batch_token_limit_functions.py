"""Unit test of the pure functions in scripts/tei_batch_token_limit.py.

Runs on synthetic inputs only (no data, no Docker, no GPU). Requires the WSL2 venv (pyyaml):

    ~/venvs/socialmediadgg-phase2/bin/python "tests/test_tei_batch_token_limit_functions.py"

Prints "test_tei_batch_token_limit_functions: all checks passed" on success; an AssertionError
names the failing check otherwise.
"""

import importlib.util
import pathlib
import time

HERE = pathlib.Path(__file__).resolve().parent
SCRIPT = HERE.parent / "scripts" / "tei_batch_token_limit.py"


def load_module():
    spec = importlib.util.spec_from_file_location("tei_bt", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cands(counts):
    return [{"doc_key": f"d{i}", "n_tokens": n} for i, n in enumerate(counts)]


def check_units(m):
    assert m.mib(1048576) == 1.0, "mib"
    assert m.gib(22548578304) == 21.0, "gib of the ceiling"


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
    sel, total = m.fill_batch(c, 100)
    assert [x["n_tokens"] for x in sel] == [100] and total == 100, (
        "single small document"
    )
    assert m.fill_batch([], 1000) == ([], 0), "no candidates"
    assert m.fill_batch(c, 5) == ([], 0), "budget below every count"


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


def check_verify(m):
    # batches formed: an empty poll, blocker (1), blocker (1), the test batch (7 documents)
    size = {
        "buckets": {"1": 3, "2": 3, "4": 3, "8": 4, "+Inf": 4},
        "sum": 9,
        "count": 4,
    }
    tokens = {
        "buckets": {"1": 1, "8192": 3, "16384": 4, "+Inf": 4},
        "sum": 2 * 5823 + 9000,
        "count": 4,
    }
    ok, checks, info = m.verify_single_batch(
        size, tokens, k=7, blocker_tokens=5823, batch_tokens=9000
    )
    assert ok, checks
    assert info["n_empty_polls"] == 1 and info["real_batches"] == 3, info
    # split: 0, 1, 1, 3, 4
    size2 = {
        "buckets": {"1": 3, "2": 3, "4": 5, "8": 5, "+Inf": 5},
        "sum": 9,
        "count": 5,
    }
    tokens2 = {
        "buckets": {"1": 1, "8192": 5, "+Inf": 5},
        "sum": 2 * 5823 + 9000,
        "count": 5,
    }
    ok2, checks2, _ = m.verify_single_batch(
        size2, tokens2, k=7, blocker_tokens=5823, batch_tokens=9000
    )
    assert not ok2 and checks2["three_real_batches"] is False, checks2
    # right structure, wrong token sum
    tokens3 = dict(tokens, sum=2 * 5823 + 8999)
    ok3, checks3, _ = m.verify_single_batch(
        size, tokens3, k=7, blocker_tokens=5823, batch_tokens=9000
    )
    assert not ok3 and checks3["token_sum"] is False, checks3


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
    assert nv([], 40960, 512) == (40960, "start")
    assert nv([t(40960, True)], 40960, 512) == (81920, "bracket")
    assert nv([t(40960, True), t(81920, True)], 40960, 512) == (163840, "bracket")
    v, p = nv([t(40960, True), t(81920, True), t(163840, False)], 40960, 512)
    assert (v, p) == (122880, "bisect"), (v, p)
    v, p = nv(
        [t(40960, True), t(81920, True), t(163840, False), t(122880, True)], 40960, 512
    )
    assert (v, p) == (143360, "bisect"), (v, p)
    v, p = nv([t(122880, True), t(123392, False)], 40960, 512)
    assert v is None and "resolution" in p, (v, p)
    v, p = nv([t(40960, False)], 40960, 512)
    assert v is None and "no permitted" in p, (v, p)
    v, p = nv([t(81920, True), t(82944, False)], 40960, 512)
    assert (v, p) == (82432, "bisect"), (v, p)
    # a failure below a later success is ignored for the bracket (recorded elsewhere)
    v, p = nv([t(40960, True), t(81920, False), t(122880, True)], 40960, 512)
    assert (v, p) == (245760, "bracket"), (v, p)
    for tested in (
        [t(40960, True), t(81920, True), t(163840, False)],
        [t(81920, True), t(82944, False)],
    ):
        v, _ = nv(tested, 40960, 512)
        safe = max(x["max_batch_tokens"] for x in tested if x["success"])
        fail = min(x["max_batch_tokens"] for x in tested if not x["success"])
        assert v % 512 == 0 and safe < v < fail, (v, safe, fail)


def check_classify(m):
    c = m.classify_outcome
    assert c(
        started=False,
        below_minimum=True,
        refused=False,
        batch_ok=False,
        verified=False,
        watchdog=False,
        peak_over=False,
    ) == ("refused_below_minimum", False, False)
    assert c(
        started=False,
        below_minimum=False,
        refused=True,
        batch_ok=False,
        verified=False,
        watchdog=False,
        peak_over=False,
    ) == ("startup_refused_at_cap", False, True)
    assert c(
        started=False,
        below_minimum=False,
        refused=False,
        batch_ok=False,
        verified=False,
        watchdog=False,
        peak_over=False,
    ) == ("startup_error", False, False)
    assert c(
        started=True,
        below_minimum=False,
        refused=True,
        batch_ok=False,
        verified=False,
        watchdog=False,
        peak_over=False,
    ) == ("batch_refused_at_cap", False, True)
    assert c(
        started=True,
        below_minimum=False,
        refused=False,
        batch_ok=False,
        verified=False,
        watchdog=False,
        peak_over=False,
    ) == ("batch_error", False, False)
    assert c(
        started=True,
        below_minimum=False,
        refused=False,
        batch_ok=True,
        verified=True,
        watchdog=False,
        peak_over=True,
    ) == ("device_peak_over_ceiling", False, True)
    assert c(
        started=True,
        below_minimum=False,
        refused=False,
        batch_ok=True,
        verified=True,
        watchdog=True,
        peak_over=True,
    ) == ("watchdog_kill", False, True)
    assert c(
        started=False,
        below_minimum=False,
        refused=False,
        batch_ok=False,
        verified=False,
        watchdog=True,
        peak_over=True,
    ) == ("watchdog_kill", False, True), (
        "a watchdog kill during start-up is a failure at the ceiling"
    )
    assert c(
        started=False,
        below_minimum=False,
        refused=True,
        batch_ok=False,
        verified=False,
        watchdog=True,
        peak_over=True,
    ) == ("startup_refused_at_cap", False, True), (
        "a refusal is the primary event even if the watchdog also fired"
    )
    assert c(
        started=True,
        below_minimum=False,
        refused=True,
        batch_ok=False,
        verified=False,
        watchdog=True,
        peak_over=True,
    ) == ("batch_refused_at_cap", False, True)
    assert c(
        started=False,
        below_minimum=False,
        refused=False,
        batch_ok=False,
        verified=False,
        watchdog=False,
        peak_over=True,
    ) == ("device_peak_over_ceiling", False, True)
    assert c(
        started=True,
        below_minimum=False,
        refused=False,
        batch_ok=True,
        verified=False,
        watchdog=False,
        peak_over=False,
    ) == ("inconclusive_unverified_batch", False, False)
    assert c(
        started=True,
        below_minimum=False,
        refused=False,
        batch_ok=True,
        verified=True,
        watchdog=False,
        peak_over=False,
    ) == ("success", True, False)


LOG_LINES = [
    '{"timestamp":"2026-09-10T04:22:52.925Z","level":"INFO","message":"Starting model backend"}',
    '{"timestamp":"2026-09-10T04:22:53.720Z","level":"INFO","message":"Starting FlashQwen3 model on Cuda(CudaDevice(DeviceId(1)))"}',
    "[vram_cap] loaded: real=/x cap_bytes=1 trampolines=687 missing=0 intercepted=19 report=(none)",
    '{"timestamp":"2026-09-10T04:23:23.879Z","level":"INFO","message":"Warming up model"}',
    '{"timestamp":"2026-09-10T04:23:36.493Z","level":"INFO","message":"Ready"}',
]


def check_log_events(m):
    ev = m.parse_tei_log_events(LOG_LINES)
    assert ev["model_start"] == "2026-09-10T04:22:53.720Z", ev
    assert (
        ev["warmup_start"] == "2026-09-10T04:23:23.879Z"
        and ev["ready"] == "2026-09-10T04:23:36.493Z"
    ), ev
    assert abs(ev["load_s"] - 30.159) < 0.01 and abs(ev["warmup_s"] - 12.614) < 0.01, ev
    assert ev["model_variant"] == "FlashQwen3", ev
    ev2 = m.parse_tei_log_events(LOG_LINES[:2])
    assert ev2["ready"] is None and ev2["warmup_s"] is None and ev2["load_s"] is None, (
        ev2
    )
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
    s_all = m.summarise_samples(samples, None, None)
    assert s_all["peak"] == 50 and s_all["n_samples"] == 4 and s_all["last"] == 30, (
        s_all
    )
    empty = m.summarise_samples(samples, 5.0, 6.0)
    assert empty["peak"] is None and empty["n_samples"] == 0, empty
    down = m.downsample_max(samples, 2.0)
    assert down == [(0.0, 50), (2.0, 30)], down


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
    assert len(sampler.samples) == result["n_samples"] and sampler.samples[1][1] == 50


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
    check_fill_batch(m)
    check_prometheus(m)
    check_verify(m)
    check_cap(m)
    check_next_value(m)
    check_classify(m)
    check_log_events(m)
    check_samples(m)
    check_peak_sampler(m)
    check_manifest(m)
    check_render_table(m)
    print("test_tei_batch_token_limit_functions: all checks passed")


if __name__ == "__main__":
    main()
