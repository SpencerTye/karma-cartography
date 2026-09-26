"""Unit test of the pure functions in scripts/phase2_feasibility_qwen3_worst_case.py.

Runs on synthetic inputs only (no data, no model, no GPU). Requires the WSL2 venv
(torch and pyyaml are used by two of the checks):

    ~/venvs/socialmediadgg-phase2/bin/python "tests/test_phase2_feasibility_functions.py"

Prints "test_phase2_feasibility_functions: all checks passed" on success; an AssertionError
names the failing check otherwise.
"""

import importlib.util
import pathlib
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
SCRIPT = HERE.parent / "scripts" / "phase2_feasibility_qwen3_worst_case.py"


def load_module():
    spec = importlib.util.spec_from_file_location("feas", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_units(feas):
    assert feas.mib(1048576) == 1.0, "mib"
    assert feas.mib(1572864) == 1.5, "mib rounding"
    assert feas.gib(1073741824) == 1.0, "gib"
    assert feas.gib(22548578304) == 21.0, "gib of the ceiling"


def check_limits(feas):
    assert feas.fits_supported_length(32768, 32768) is True, "fits at the limit"
    assert feas.fits_supported_length(32769, 32768) is False, "does not fit one over"
    assert feas.within_ceiling(100, 100) is True, "within at equality"
    assert feas.within_ceiling(101, 100) is False, "not within one over"


def check_padding(feas):
    padded, pads = feas.padding_per_document([7, 3, 5])
    assert padded == 7 and pads == [0, 4, 2], (padded, pads)
    padded, pads = feas.padding_per_document([11])
    assert padded == 11 and pads == [0], (padded, pads)


def check_rank_longest(feas):
    n_tokens = np.array([5, 9, 9, 1, 9, 7])
    n_chars = np.array([10, 30, 40, 1, 40, 5])
    keys = np.array(["t1_b", "t1_c", "t1_a", "t3_x", "t1_d", "t1_e"])
    # 9-token documents first; among them 40 characters before 30; among equal
    # characters the smaller doc_key first; then the 7-token document.
    assert feas.rank_longest(n_tokens, n_chars, keys, 4) == [2, 4, 1, 5]
    assert feas.rank_longest(n_tokens, n_chars, keys, 10) == [2, 4, 1, 5, 0, 3]
    assert feas.rank_longest(n_tokens, n_chars, keys, 1) == [2]


def check_summary(feas):
    s = feas.summarise_token_lengths(np.array([1, 2, 3, 4, 100]), limit=50)
    assert s["n_documents"] == 5 and s["min"] == 1 and s["max"] == 100, s
    assert s["sum"] == 110 and s["n_over_limit"] == 1 and s["limit"] == 50, s
    assert s["p50"] == 3, s
    assert s["n_over_8192"] == 0 and s["n_over_40960"] == 0, s


def check_peak_sampler(feas):
    values = iter([10, 50, 20, 30])
    sampler = feas.PeakSampler(lambda: next(values, 30), interval_s=0.001)
    sampler.start()
    time.sleep(0.1)
    result = sampler.stop()
    assert result["peak"] == 50, result
    assert result["n_samples"] >= 4, result
    assert result["first"] == 10, result


def check_manifest(feas):
    import yaml

    m = feas.build_manifest(
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
                "rows": 5,
                "cols": 4,
                "role": "input",
            }
        ],
        script_rel="pipeline/x/scripts/s.py",
        script_hash="sha256:123",
        parameters={"vram_ceiling_bytes": 22548578304, "note": "a: b # c"},
        packages=[{"name": "torch", "version": "2.8.0"}],
        git_commit="abc1234",
        timestamp="2026-09-10T00:00:00Z",
        notes="line1\nline2",
        os_name="Linux-6.18",
        python_version="3.12.3",
    )
    keys = list(m.keys())
    assert keys[:4] == [
        "manifest_version",
        "output_file",
        "output_hash",
        "output_format",
    ], keys
    assert m["output_rows"] == 3 and m["output_cols"] == 2, m
    assert m["transformation"]["script_hash"] == "sha256:123", m
    assert m["software"]["packages"][0]["name"] == "torch", m
    assert m["schema_diff"] == {
        "added": [],
        "removed": [],
        "renamed": [],
        "retyped": [],
    }, m
    text = feas.manifest_yaml(m)
    assert yaml.safe_load(text) == m, "yaml round trip"
    m2 = feas.build_manifest(
        output_rel="a.json",
        output_hash="sha256:x",
        output_format="json",
        rows=None,
        cols=None,
        inputs=[],
        script_rel="s.py",
        script_hash="sha256:y",
        parameters={},
        packages=[],
        git_commit="",
        timestamp="t",
        notes="n",
        os_name="o",
        python_version="p",
    )
    assert "output_rows" not in m2 and "output_cols" not in m2, m2
    assert "git_commit" not in m2["transformation"], m2


def check_last_token_pool(feas):
    import torch

    h = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    left_mask = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]])
    out = feas.last_token_pool(h, left_mask)
    assert torch.equal(out, h[:, -1]), "left padding takes the last position"
    right_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]])
    out2 = feas.last_token_pool(h, right_mask)
    assert torch.equal(out2[0], h[0, 1]) and torch.equal(out2[1], h[1, 3]), (
        "right padding"
    )


def check_side_by_side(feas):
    def model_result(name, dim, base, peak_a):
        return {
            "model_name": name,
            "embedding_dimension": dim,
            "baseline_after_load": {
                "device_used_bytes": base,
                "allocated_bytes": base - 100,
                "reserved_bytes": base - 50,
                "process_attributable_bytes": base - 10,
            },
            "cache": {
                "config_use_cache": True,
                "measured_use_cache": False,
                "measured_past_key_values": "None",
                "default_probe_creates_cache": True,
                "default_probe_summary": "DynamicCache, 36 layers",
            },
            "ceiling": {
                "ceiling_bytes": 1000,
                "all_measured_cases_within_ceiling": True,
            },
            "attention": {
                "config_attn_implementation": "flash_attention_2",
                "registry_function": "flash_attention_forward",
                "kernels_observed": "flash_fwd_kernel",
            },
            "cases": {
                "A": {
                    "success": True,
                    "input_shape": [1, 5],
                    "peak_device_used_bytes": peak_a,
                    "incremental_device_bytes": peak_a - base,
                    "peak_rss_bytes": 7,
                    "forward_wall_s": 1.5,
                    "within_ceiling": True,
                    "failure_reason": "",
                },
                "B": {
                    "success": False,
                    "input_shape": [3, 5],
                    "peak_device_used_bytes": 999,
                    "incremental_device_bytes": 999 - base,
                    "peak_rss_bytes": 8,
                    "forward_wall_s": 0.2,
                    "within_ceiling": True,
                    "failure_reason": "OutOfMemoryError at the allocator cap",
                },
                "C": {
                    "success": True,
                    "input_shape": [8, 5],
                    "peak_device_used_bytes": 900,
                    "incremental_device_bytes": 900 - base,
                    "peak_rss_bytes": 9,
                    "forward_wall_s": 3.0,
                    "within_ceiling": True,
                    "failure_reason": "",
                },
            },
            "top_documents": [{"rank": 1, "n_tokens": 5}, {"rank": 2, "n_tokens": 4}],
        }

    table = feas.side_by_side_table(
        {
            "4B": model_result("Qwen3-Embedding-4B", 2560, 500, 600),
            "8B": model_result("Qwen3-Embedding-8B", 4096, 700, 800),
        }
    )
    assert table.startswith("| Measure | Qwen3-Embedding-4B | Qwen3-Embedding-8B |"), (
        table.splitlines()[0]
    )
    assert "| Embedding dimension | 2560 | 4096 |" in table, table
    assert "[3, 5]" in table and "OutOfMemoryError" in table, table
    assert "| Case B: ran within the VRAM ceiling | yes | yes |" in table, table
    assert "| Case A: wall-clock of the forward pass (s) | 1.5 | 1.5 |" in table, table
    assert "failed after 0.20 s" in table, table
    lines = [ln for ln in table.splitlines() if ln.startswith("|")]
    assert all(ln.count("|") == 4 for ln in lines), "every row has three cells"


def main():
    feas = load_module()
    check_units(feas)
    check_limits(feas)
    check_padding(feas)
    check_rank_longest(feas)
    check_summary(feas)
    check_peak_sampler(feas)
    check_manifest(feas)
    check_last_token_pool(feas)
    check_side_by_side(feas)
    print("test_phase2_feasibility_functions: all checks passed")


if __name__ == "__main__":
    main()
