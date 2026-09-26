"""Unit test of the pure functions in scripts/cpu_vs_gpu_three_documents.py.

Runs on synthetic inputs only (no data, no model, no GPU, about a second). Requires the WSL2
venv (numpy, pyyaml):

    ~/venvs/socialmediadgg-phase2/bin/python "tests/test_cpu_vs_gpu_three_documents_functions.py"

Prints "test_cpu_vs_gpu_three_documents_functions: all checks passed" on success; an
AssertionError names the failing check otherwise.
"""

import importlib.util
import math
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
SCRIPT = HERE.parent / "scripts" / "cpu_vs_gpu_three_documents.py"


def load_module():
    spec = importlib.util.spec_from_file_location("cpu_vs_gpu_three_documents", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def expect_value_error(fn, why):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError(f"accepted: {why}")


CASE_ROWS = [
    {
        "model": "Qwen3-Embedding-4B",
        "case": "A",
        "doc_keys": "t1_a",
        "token_counts": "50",
    },
    {
        "model": "Qwen3-Embedding-4B",
        "case": "B",
        "doc_keys": "t1_a;t1_b;t1_c",
        "token_counts": "50;40;30",
    },
    {
        "model": "Qwen3-Embedding-4B",
        "case": "C",
        "doc_keys": "t1_a;t1_b;t1_c;t1_d",
        "token_counts": "50;40;30;20",
    },
    {
        "model": "Qwen3-Embedding-8B",
        "case": "B",
        "doc_keys": "t1_x;t1_y;t1_z",
        "token_counts": "9;8;7",
    },
]


def check_select_case_documents(m):
    docs = m.select_case_documents(CASE_ROWS, case="B", model="Qwen3-Embedding-4B")
    assert [d["doc_key"] for d in docs] == ["t1_a", "t1_b", "t1_c"], docs
    assert [d["n_tokens"] for d in docs] == [50, 40, 30], docs
    assert [d["position"] for d in docs] == [1, 2, 3], docs
    b = CASE_ROWS[1]
    expect_value_error(
        lambda: m.select_case_documents(
            [r for r in CASE_ROWS if r["case"] != "B"],
            case="B",
            model="Qwen3-Embedding-4B",
        ),
        "case missing for the model",
    )
    expect_value_error(
        lambda: m.select_case_documents(
            CASE_ROWS + [b], case="B", model="Qwen3-Embedding-4B"
        ),
        "case listed twice",
    )
    expect_value_error(
        lambda: m.select_case_documents(
            [dict(b, doc_keys="t1_a;t1_a;t1_c")], case="B", model="Qwen3-Embedding-4B"
        ),
        "duplicated document",
    )
    expect_value_error(
        lambda: m.select_case_documents(
            [dict(b, token_counts="50;40")], case="B", model="Qwen3-Embedding-4B"
        ),
        "token counts do not match the keys",
    )


def check_allocator_cap(m):
    cap, frac = m.allocator_cap(22548578304, 1_500_000_000, 25769803776)
    assert cap == 22548578304 - 1_500_000_000, cap
    assert math.isclose(frac, cap / 25769803776, rel_tol=0, abs_tol=1e-15), frac
    assert m.allocator_cap(1000, 2000, 10000) == (0, 0.0), "no room under the ceiling"


def check_format_factor(m):
    assert m.format_factor(50.0) == "50.0", m.format_factor(50.0)
    assert m.format_factor(123.456) == "123.5", m.format_factor(123.456)
    assert m.format_factor(3.0) == "3.00", m.format_factor(3.0)
    assert m.format_factor(1.234) == "1.23", m.format_factor(1.234)


def check_speed_comparison(m):
    c = m.speed_comparison("GPU", 2.0, "CPU", 100.0)
    assert math.isclose(c["factor"], 50.0), c
    assert (c["faster"], c["slower"]) == ("GPU", "CPU"), c
    assert c["statement"] == "GPU is 50.0 times faster than CPU", c["statement"]
    c = m.speed_comparison("4B", 30.0, "0.6B", 10.0)
    assert math.isclose(c["factor"], 3.0) and (c["faster"], c["slower"]) == (
        "0.6B",
        "4B",
    ), c
    assert c["statement"] == "4B is 3.00 times slower than 0.6B", c["statement"]
    c = m.speed_comparison("a", 5.0, "b", 5.0)
    assert c["factor"] == 1.0 and c["statement"] == "a and b are equally fast", c
    assert m.speed_comparison("a", None, "b", 1.0) is None, "missing time"
    assert m.speed_comparison("a", 1.0, "b", 0.0) is None, "zero time"


def check_ratio_rows(m):
    summaries = {
        ("0.6B", "cpu"): {"mean_s": 60.0, "median_s": 66.0},
        ("0.6B", "gpu"): {"mean_s": 0.5, "median_s": 0.6},
        ("4B", "cpu"): {"mean_s": 400.0, "median_s": 396.0},
        ("4B", "gpu"): {"mean_s": 2.5, "median_s": 2.4},
    }
    rows = m.ratio_rows(summaries)
    assert [r["comparison"] for r in rows] == [
        "GPU vs CPU, Qwen3-Embedding-0.6B",
        "GPU vs CPU, Qwen3-Embedding-4B",
        "4B vs 0.6B, CPU",
        "4B vs 0.6B, GPU",
    ], rows
    assert rows[0]["statement_mean"] == "GPU is 120.0 times faster than CPU", rows[0]
    assert rows[0]["statement_median"] == "GPU is 110.0 times faster than CPU", rows[0]
    assert rows[1]["statement_mean"] == "GPU is 160.0 times faster than CPU", rows[1]
    assert rows[2]["statement_mean"] == "4B is 6.67 times slower than 0.6B", rows[2]
    assert rows[2]["statement_median"] == "4B is 6.00 times slower than 0.6B", rows[2]
    assert rows[3]["statement_mean"] == "4B is 5.00 times slower than 0.6B", rows[3]
    assert math.isclose(rows[3]["factor_mean"], 5.0), rows[3]
    missing = m.ratio_rows({k: v for k, v in summaries.items() if k != ("4B", "gpu")})
    assert missing[1]["statement_mean"] == "not measured", missing[1]
    assert missing[3]["factor_mean"] is None, missing[3]
    assert missing[0]["statement_mean"] == "GPU is 120.0 times faster than CPU", (
        missing[0]
    )


def batch(phase, index, time_s, t_start, n_documents=3, n_tokens=120):
    return {
        "phase": phase,
        "index": index,
        "time_s": time_s,
        "n_documents": n_documents,
        "n_tokens": n_tokens,
        "t_start": t_start,
        "t_end": t_start + time_s,
    }


def check_condition_summary(m):
    batches = [
        batch("warmup", 1, 9.0, 0.0),
        batch("warmup", 2, 5.0, 9.0),
        batch("timed", 1, 2.0, 20.0),
        batch("timed", 2, 4.0, 22.0),
        batch("timed", 3, 3.0, 26.0),
    ]
    s = m.condition_summary(batches)
    assert (s["n_warmup"], s["n_timed"]) == (2, 3), s
    assert s["warmup_times_s"] == [9.0, 5.0], s
    assert s["timed_times_s"] == [2.0, 4.0, 3.0], s
    assert s["mean_s"] == 3.0 and s["median_s"] == 3.0, s
    assert (s["min_s"], s["max_s"], s["total_s"]) == (2.0, 4.0, 9.0), s
    assert math.isclose(s["documents_per_s"], 1.0), s
    assert math.isclose(s["tokens_per_s"], 40.0), s
    assert (s["documents_per_batch"], s["tokens_per_batch"]) == (3, 120), s
    assert s["timed_window"] == (20.0, 29.0), s
    expect_value_error(
        lambda: m.condition_summary(
            batches + [batch("timed", 4, 3.0, 29.0, n_tokens=119)]
        ),
        "timed batches with different token counts",
    )
    expect_value_error(
        lambda: m.condition_summary(batches[:2]),
        "no timed batch",
    )


def check_max_abs_diff_to_first(m):
    base = np.arange(6, dtype=np.float32).reshape(2, 3)
    diffs = m.max_abs_diff_to_first([base, base.copy(), base + np.float32(0.5)])
    assert diffs == [0.0, 0.0, 0.5], diffs
    assert m.max_abs_diff_to_first([]) == [], "no batches"


def check_document_agreement(m):
    a = np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]], dtype=np.float32)
    b = np.array([[1.0, 0.0], [0.0, -1.0], [0.8, 0.6]], dtype=np.float32)
    rows = m.document_agreement(["d1", "d2", "d3"], a, b)
    assert [r["doc_key"] for r in rows] == ["d1", "d2", "d3"], rows
    assert [r["position"] for r in rows] == [1, 2, 3], rows
    assert math.isclose(rows[0]["cosine"], 1.0) and rows[0]["max_abs_diff"] == 0.0, (
        rows[0]
    )
    assert math.isclose(rows[1]["cosine"], -1.0), rows[1]
    assert math.isclose(rows[1]["max_abs_diff"], 2.0), rows[1]
    assert math.isclose(rows[2]["cosine"], 0.96, rel_tol=1e-6), rows[2]
    assert math.isclose(rows[2]["max_abs_diff"], 0.2, rel_tol=1e-6), rows[2]
    expect_value_error(
        lambda: m.document_agreement(["d1", "d2"], a, b), "keys and rows differ"
    )
    expect_value_error(
        lambda: m.document_agreement(["d1", "d2", "d3"], a, b[:, :1]), "shapes differ"
    )


def check_attention_kernels(m):
    names = [
        "aten::linear",
        "aten::scaled_dot_product_attention",
        "aten::_scaled_dot_product_flash_attention_for_cpu",
        "fmha_cutlassF_bf16_aligned_64x64_rf_sm80(PyTorchMemEffAttention::AttentionKernel<...>)",
        "aten::_efficient_attention_forward",
        "aten::scaled_dot_product_attention",
        "ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_tn",
    ]
    assert m.attention_kernels(names) == [
        "aten::_efficient_attention_forward",
        "aten::_scaled_dot_product_flash_attention_for_cpu",
        "aten::scaled_dot_product_attention",
        "fmha_cutlassF_bf16_aligned_64x64_rf_sm80(PyTorchMemEffAttention::AttentionKernel<...>)",
    ], m.attention_kernels(names)
    assert m.attention_kernels(["aten::mm", "aten::add"]) == [], "no attention kernel"


EPOCH_TICKS = (
    621355968000000000  # .NET DateTime ticks (100 ns since 0001-01-01) at 1970-01-01
)


def check_windows_ticks(m):
    assert m.windows_ticks_to_unix_ns(EPOCH_TICKS) == 0, "epoch"
    assert m.windows_ticks_to_unix_ns(EPOCH_TICKS + 10_000_000) == 1_000_000_000, (
        "one second"
    )
    t = 638930000000000123
    assert m.windows_ticks_to_unix_ns(t) == (t - EPOCH_TICKS) * 100, (
        "exact integer arithmetic"
    )


def check_parse_marker_listing(m):
    text = (
        f"001_start {EPOCH_TICKS}\r\n"
        f"001_end {EPOCH_TICKS + 12345678}\n"
        "\n"
        "a line without a number\n"
        "002_start x\n"
    )
    got = m.parse_marker_listing(text)
    assert got == {"001_start": 0, "001_end": 1234567800}, got


def check_attach_host_times(m):
    batches = [
        {"seq": 1, "time_s": 1.0},
        {"seq": 2, "time_s": 2.0},
        {"seq": 3, "time_s": 3.0},
    ]
    marks = {
        "001_start": 10_000_000_000,
        "001_end": 11_040_000_000,
        "002_start": 12_000_000_000,
        "002_end": 14_080_000_000,
        "003_start": 15_000_000_000,
    }
    out = m.attach_host_times(batches, marks)
    assert out[0]["host_time_s"] == 1.04, out[0]
    assert math.isclose(out[0]["host_over_vm"], 1.04), out[0]
    assert (out[0]["host_start_unix_ns"], out[0]["host_end_unix_ns"]) == (
        10_000_000_000,
        11_040_000_000,
    ), out[0]
    assert out[1]["host_time_s"] == 2.08, out[1]
    assert out[2]["host_time_s"] is None and out[2]["host_over_vm"] is None, (
        "end marker missing"
    )
    assert batches[0].get("host_time_s") is None, "the input rows are not modified"


def check_condition_summary_time_key(m):
    batches = [
        dict(batch("warmup", 1, 9.0, 0.0), host_time_s=9.5),
        dict(batch("timed", 1, 2.0, 20.0), host_time_s=2.1),
        dict(batch("timed", 2, 4.0, 22.0), host_time_s=4.2),
    ]
    s = m.condition_summary(batches, time_key="host_time_s")
    assert s["time_key"] == "host_time_s", s
    assert s["warmup_times_s"] == [9.5] and s["timed_times_s"] == [2.1, 4.2], s
    assert math.isclose(s["mean_s"], 3.15) and math.isclose(s["median_s"], 3.15), s
    assert math.isclose(s["documents_per_s"], 6 / 6.3), s
    assert m.condition_summary(batches)["timed_times_s"] == [2.0, 4.0], (
        "default: time_s"
    )
    expect_value_error(
        lambda: m.condition_summary(
            [dict(b, host_time_s=None) for b in batches], time_key="host_time_s"
        ),
        "a timed batch without a time",
    )


def check_clock_ratio(m):
    assert math.isclose(m.clock_ratio({"mean_s": 104.0}, {"mean_s": 100.0}), 1.04)
    assert m.clock_ratio(None, {"mean_s": 1.0}) is None, "missing summary"
    assert m.clock_ratio({"mean_s": 1.0}, {"mean_s": 0.0}) is None, "zero time"


def check_qpc_ticks(m):
    assert m.qpc_ticks_to_ns(10_000_000, 10_000_000) == 1_000_000_000, (
        "one second at 10 MHz"
    )
    assert m.qpc_ticks_to_ns(123, 10_000_000) == 12_300, "100 ns per tick"
    assert m.qpc_ticks_to_ns(3_579_545, 3_579_545) == 1_000_000_000, "another frequency"


def check_cristian_duration(m):
    # Host readings 2 s apart; each round trip 0.2 ms, half of which lies outside the batch.
    got = m.cristian_duration_s(10_000_000_000, 12_000_000_000, 200_000, 200_000)
    assert math.isclose(got, 1.9998, rel_tol=0, abs_tol=1e-12), got
    assert m.cristian_duration_s(None, 1, 0, 0) is None, "missing reading"


def check_pick_primary(m):
    a, b = {"mean_s": 1.0}, {"mean_s": 2.0}
    assert m.pick_primary([("qpc", None), ("marker", a), ("vm", b)]) == ("marker", a)
    assert m.pick_primary([("qpc", b), ("marker", a)]) == ("qpc", b)
    assert m.pick_primary([("qpc", None), ("vm", None)]) == (None, None)


def main():
    m = load_module()
    check_select_case_documents(m)
    check_allocator_cap(m)
    check_format_factor(m)
    check_speed_comparison(m)
    check_ratio_rows(m)
    check_condition_summary(m)
    check_max_abs_diff_to_first(m)
    check_document_agreement(m)
    check_attention_kernels(m)
    check_windows_ticks(m)
    check_parse_marker_listing(m)
    check_attach_host_times(m)
    check_condition_summary_time_key(m)
    check_clock_ratio(m)
    check_qpc_ticks(m)
    check_cristian_duration(m)
    check_pick_primary(m)
    check_partial_outcome(m)
    print("test_cpu_vs_gpu_three_documents_functions: all checks passed")


def check_partial_outcome(m):
    rec = {"batches": [{"phase": "warmup"}, {"phase": "warmup"}, {"phase": "warmup"}]}
    assert m.partial_outcome(rec) == (
        "stopped before completion after 3 batches (3 untimed warm-up, 0 timed); no result"
    ), m.partial_outcome(rec)
    assert m.partial_outcome({}) == (
        "stopped before completion after 0 batches (0 untimed warm-up, 0 timed); no result"
    ), m.partial_outcome({})


if __name__ == "__main__":
    main()
