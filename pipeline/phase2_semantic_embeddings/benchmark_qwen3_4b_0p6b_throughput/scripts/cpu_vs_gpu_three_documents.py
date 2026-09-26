#!/usr/bin/env python
"""Phase 2 benchmark, corrected: CPU versus GPU on the same three documents.

Compares four conditions directly: Qwen3-Embedding-0.6B on the CPU, 0.6B on the GPU (RTX 3090),
Qwen3-Embedding-4B on the CPU and 4B on the GPU. Every condition embeds the same three real
Reddit documents, in the same order, in every batch: the three documents of Case B of the Phase 2
worst-case feasibility test (`feasibility_qwen3_embedding_worst_case/results/cases_4B.csv`). Each
condition loads its model once, runs untimed warm-up batches, then timed batches, all of the same
three documents.

All four conditions run the same code (transformers + PyTorch, `AutoModel`, sdpa attention with
keys and values repeated to the query heads, `use_cache=False`, left padding, the model card's
last-token pooling, L2 normalisation, no truncation, no prompt); only the device and the numeric
precision differ (CPU float32, GPU bfloat16, both recorded). Every worker runs in the same kind of
systemd scope (21 GiB memory limit, no swap, 8 threads, one batch at a time). GPU runs keep the
21 GiB ceiling on the total memory of the card (other processes included): PyTorch's allocator is
capped at the ceiling minus the device total after the CUDA context exists, and a 20 ms NVML
watchdog kills the worker if the device total goes over the ceiling.

Nothing is extrapolated; no batch-size search, no memory-limit search, no full-corpus embedding.

Stages (run inside the WSL2 venv; see ../README.md):
  run --model M --device D   one condition (M = 0.6B or 4B, D = cpu or gpu)
  report                     the comparison table, the speed ratios, the precision agreement and
                             CPU_VS_GPU_REPORT.md from the four runs
  worker                     internal: the process started by run inside the scope
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import throughput_benchmark as tb  # noqa: E402  (the helpers of the original benchmark)

SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_SHA256 = tb._sha256_of(SCRIPT_PATH)

CASE_FILE = tb.PRIOR_FEAS_DIR / "results" / "cases_4B.csv"
CASE_NAME = "B"
CASE_MODEL = "Qwen3-Embedding-4B"
EXPECTED_DOC_KEYS = ["t1_oxogwqy", "t1_oxwqnx9", "t1_ozvubg1"]
EXPECTED_TOKENS = [5823, 4863, 3123]
DEVICES = ("cpu", "gpu")
PRECISION = {"cpu": "float32", "gpu": "bfloat16"}
CONDITIONS = [("0.6B", "cpu"), ("0.6B", "gpu"), ("4B", "cpu"), ("4B", "gpu")]
RATIO_SPECS = [
    (
        "GPU vs CPU, Qwen3-Embedding-0.6B",
        ("0.6B", "gpu"),
        "GPU",
        ("0.6B", "cpu"),
        "CPU",
    ),
    ("GPU vs CPU, Qwen3-Embedding-4B", ("4B", "gpu"), "GPU", ("4B", "cpu"), "CPU"),
    ("4B vs 0.6B, CPU", ("4B", "cpu"), "4B", ("0.6B", "cpu"), "0.6B"),
    ("4B vs 0.6B, GPU", ("4B", "gpu"), "4B", ("0.6B", "gpu"), "0.6B"),
]

# ----------------------------------------------------------------------------------------------
# Pure functions (unit-tested by ../tests/test_cpu_vs_gpu_three_documents_functions.py)
# ----------------------------------------------------------------------------------------------


def select_case_documents(rows, case=CASE_NAME, model=CASE_MODEL):
    """The documents of one case of the feasibility test's case table, in the case's order, as
    [{"position", "doc_key", "n_tokens"}]. Refuses a missing or repeated case, a duplicated
    document and a token list that does not match the key list."""
    hits = [r for r in rows if r.get("model") == model and r.get("case") == case]
    if len(hits) != 1:
        raise ValueError(f"case {case} of {model}: {len(hits)} rows (expected 1)")
    keys = [k for k in str(hits[0]["doc_keys"]).split(";") if k]
    counts = [int(c) for c in str(hits[0]["token_counts"]).split(";") if c]
    if len(keys) != len(counts):
        raise ValueError(
            f"case {case}: {len(keys)} doc_keys but {len(counts)} token counts"
        )
    if len(set(keys)) != len(keys):
        raise ValueError(f"case {case}: a document appears more than once: {keys}")
    return [
        {"position": i, "doc_key": k, "n_tokens": n}
        for i, (k, n) in enumerate(zip(keys, counts), start=1)
    ]


def allocator_cap(ceiling_bytes, used_after_context, total_bytes):
    """(cap, fraction) for torch.cuda.set_per_process_memory_fraction: the ceiling minus what the
    device holds once the CUDA context exists (other processes and the context included)."""
    cap = max(0, int(ceiling_bytes) - int(used_after_context))
    return cap, (cap / int(total_bytes) if cap > 0 else 0.0)


def format_factor(x):
    """A speed factor: one decimal from 10 up, two decimals below 10."""
    return f"{x:.1f}" if x >= 10 else f"{x:.2f}"


def speed_comparison(label_a, time_a, label_b, time_b):
    """How many times faster or slower A is than B, from the time each takes for the same work;
    None when a time is missing or not positive."""
    if time_a is None or time_b is None or time_a <= 0 or time_b <= 0:
        return None
    if time_a == time_b:
        return {
            "faster": None,
            "slower": None,
            "factor": 1.0,
            "statement": f"{label_a} and {label_b} are equally fast",
        }
    if time_a < time_b:
        factor = time_b / time_a
        return {
            "faster": label_a,
            "slower": label_b,
            "factor": factor,
            "statement": f"{label_a} is {format_factor(factor)} times faster than {label_b}",
        }
    factor = time_a / time_b
    return {
        "faster": label_b,
        "slower": label_a,
        "factor": factor,
        "statement": f"{label_a} is {format_factor(factor)} times slower than {label_b}",
    }


def ratio_rows(summaries):
    """The four requested comparisons from the per-condition summaries ({(model, device):
    {"mean_s", "median_s"}}), on the mean batch time (equivalent to documents or tokens per
    second: every batch holds the same documents) and on the median batch time."""
    rows = []
    for name, key_a, label_a, key_b, label_b in RATIO_SPECS:
        row = {
            "comparison": name,
            "a": f"{key_a[0]} {key_a[1].upper()}",
            "b": f"{key_b[0]} {key_b[1].upper()}",
        }
        for basis in ("mean", "median"):
            t_a = (summaries.get(key_a) or {}).get(f"{basis}_s")
            t_b = (summaries.get(key_b) or {}).get(f"{basis}_s")
            c = speed_comparison(label_a, t_a, label_b, t_b)
            row[f"a_{basis}_s"] = t_a
            row[f"b_{basis}_s"] = t_b
            row[f"factor_{basis}"] = c["factor"] if c else None
            row[f"statement_{basis}"] = c["statement"] if c else "not measured"
        rows.append(row)
    return rows


def condition_summary(batches, time_key="time_s"):
    """Statistics of one condition's timed batches on the clock in `time_key` (warm-up batches
    listed, never counted): total, mean, median, minimum, maximum and standard deviation of the
    batch time, documents and tokens per second over the total timed time, and the timed window.
    Every timed batch must hold the same documents and tokens and have a time."""
    warm = [b for b in batches if b["phase"] == "warmup"]
    timed = [b for b in batches if b["phase"] == "timed"]
    if not timed:
        raise ValueError("no timed batch")
    if any(b.get(time_key) is None for b in timed):
        raise ValueError(f"a timed batch has no {time_key}")
    shapes = {(int(b["n_documents"]), int(b["n_tokens"])) for b in timed}
    if len(shapes) != 1:
        raise ValueError(
            f"timed batches differ in documents or tokens: {sorted(shapes)}"
        )
    ((n_documents, n_tokens),) = shapes
    s = tb.steady_state([dict(b, time_s=float(b[time_key])) for b in timed])
    s.update(
        {
            "time_key": time_key,
            "n_warmup": len(warm),
            "n_timed": len(timed),
            "warmup_times_s": [
                float(b[time_key]) if b.get(time_key) is not None else None
                for b in warm
            ],
            "timed_times_s": [float(b[time_key]) for b in timed],
            "documents_per_batch": n_documents,
            "tokens_per_batch": n_tokens,
            "timed_window": (
                min(float(b["t_start"]) for b in timed),
                max(float(b["t_end"]) for b in timed),
            ),
        }
    )
    return s


def max_abs_diff_to_first(embs):
    """Largest absolute difference between each batch's embeddings and the first batch's."""
    if not embs:
        return []
    first = np.asarray(embs[0], dtype=np.float64)
    return [
        float(np.max(np.abs(np.asarray(e, dtype=np.float64) - first))) for e in embs
    ]


def document_agreement(doc_keys, a, b):
    """Per document: the cosine and the largest absolute difference between its embedding in a
    and in b (a document is compared only with itself)."""
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        raise ValueError(f"embedding shapes differ: {a.shape} vs {b.shape}")
    if len(doc_keys) != a.shape[0]:
        raise ValueError(f"{len(doc_keys)} doc_keys for {a.shape[0]} embeddings")
    cos = tb.cosine_rows(a, b)
    diff = np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)), axis=1)
    return [
        {"position": i, "doc_key": k, "cosine": float(c), "max_abs_diff": float(d)}
        for i, (k, c, d) in enumerate(zip(doc_keys, cos, diff), start=1)
    ]


_ATTENTION_NAME = re.compile(r"attention|fmha|flash|sdp", re.IGNORECASE)


def attention_kernels(names):
    """The distinct attention operators and kernels among profiler event names, sorted."""
    return sorted({n for n in names if _ATTENTION_NAME.search(n)})


_NET_TICKS_AT_UNIX_EPOCH = 621355968000000000


def windows_ticks_to_unix_ns(ticks):
    """A .NET DateTime tick count (100 ns since 0001-01-01 UTC) to nanoseconds since 1970."""
    return (int(ticks) - _NET_TICKS_AT_UNIX_EPOCH) * 100


def parse_marker_listing(text):
    """PowerShell's 'name ticks' lines to {name: Unix ns}; any other line is ignored."""
    out = {}
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) == 2 and parts[1].isdigit():
            out[parts[0]] = windows_ticks_to_unix_ns(parts[1])
    return out


def attach_host_times(batches, marks):
    """Copies of the batch rows with each batch's time on the Windows clock (the difference of
    the Windows modification times of its start and end markers; None when one is missing) and
    its ratio to the VM clock's time."""
    out = []
    for b in batches:
        seq = int(b["seq"])
        start = marks.get(f"{seq:03d}_start")
        end = marks.get(f"{seq:03d}_end")
        host = (end - start) / 1e9 if start is not None and end is not None else None
        vm = b.get("time_s")
        out.append(
            dict(
                b,
                host_start_unix_ns=start,
                host_end_unix_ns=end,
                host_time_s=host,
                host_over_vm=(host / vm) if host is not None and vm else None,
            )
        )
    return out


def clock_ratio(host_summary, vm_summary):
    """Mean timed batch time on the Windows clock over the same on the VM's clock."""
    if not host_summary or not vm_summary:
        return None
    h, v = host_summary.get("mean_s"), vm_summary.get("mean_s")
    if h is None or not v:
        return None
    return h / v


def qpc_ticks_to_ns(ticks, freq):
    """Windows QueryPerformanceCounter ticks to nanoseconds (integer arithmetic)."""
    return int(ticks) * 1_000_000_000 // int(freq)


def cristian_duration_s(h_start_ns, h_end_ns, rtt_start_ns, rtt_end_ns):
    """Seconds between two host-clock readings taken just before and just after a block, each
    taken at the midpoint of a round trip: half of each round trip lies outside the block."""
    if h_start_ns is None or h_end_ns is None:
        return None
    return ((h_end_ns - h_start_ns) - (rtt_start_ns + rtt_end_ns) / 2) / 1e9


def pick_primary(candidates):
    """The first (clock, summary) that has a summary, in the given order of preference."""
    for name, summary in candidates:
        if summary:
            return name, summary
    return None, None


# ----------------------------------------------------------------------------------------------
# Worker: one condition, inside the scope (model loaded once; warm-up, then timed batches)
# ----------------------------------------------------------------------------------------------


def torch_record(torch, gpu):
    """The PyTorch configuration the worker actually ran with."""
    rec = {
        "torch_version": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "parallel_info": torch.__config__.parallel_info(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "tokenizers_parallelism": os.environ.get("TOKENIZERS_PARALLELISM"),
        "cpu_affinity_count": len(os.sched_getaffinity(0)),
    }
    if gpu:
        props = torch.cuda.get_device_properties(0)
        m = torch.backends.cuda.matmul
        rec.update(
            {
                "cuda_device": props.name,
                "cuda_capability": f"{props.major}.{props.minor}",
                "cuda_total_memory_bytes": int(props.total_memory),
                "torch_cuda_version": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(),
                "sdp_flash_enabled": torch.backends.cuda.flash_sdp_enabled(),
                "sdp_mem_efficient_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
                "sdp_math_enabled": torch.backends.cuda.math_sdp_enabled(),
                "sdp_cudnn_enabled": torch.backends.cuda.cudnn_sdp_enabled(),
                "matmul_allow_tf32": m.allow_tf32,
                "matmul_allow_bf16_reduced_precision_reduction": m.allow_bf16_reduced_precision_reduction,
            }
        )
    return rec


HOST_CLOCK_SCRIPT = (
    "[Console]::Out.WriteLine([System.Diagnostics.Stopwatch]::Frequency); [Console]::Out.Flush(); "
    "while ($null -ne ($l = [Console]::In.ReadLine())) { "
    "[Console]::Out.WriteLine([System.Diagnostics.Stopwatch]::GetTimestamp()); [Console]::Out.Flush() }"
)


class HostClock:
    """The Windows host's performance counter (QPC), read through a PowerShell process over WSL
    interop (the VM's own clock runs slow). read() returns (host ns, round trip in VM ns)."""

    def __init__(self):
        self.p = subprocess.Popen(
            [
                POWERSHELL,
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                HOST_CLOCK_SCRIPT,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.freq = int(self.p.stdout.readline())
        for _ in range(20):
            self.read()

    def read(self):
        g0 = time.perf_counter_ns()
        self.p.stdin.write("t\n")
        self.p.stdin.flush()
        ticks = int(self.p.stdout.readline())
        g1 = time.perf_counter_ns()
        return qpc_ticks_to_ns(ticks, self.freq), g1 - g0

    def close(self):
        try:
            self.p.stdin.close()
            self.p.wait(timeout=30)
        except Exception:
            self.p.kill()


def host_reading(clock):
    """One reading of the host clock, or (None, None) without a clock or on a failed read."""
    if clock is None:
        return None, None
    try:
        return clock.read()
    except Exception:
        return None, None


def write_marker(directory, name):
    """Writes one small file on a Windows drive. Windows stamps its modification time with the
    host's clock, which times the batch independently of the VM's clock (the VM's clock runs
    slow under CPU load). Does nothing without a directory."""
    if directory is None:
        return
    with open(Path(directory) / name, "wb") as f:
        f.write(b"1")


def stage_worker(args):
    """Runs inside the systemd scope: loads the model once on the condition's device and in its
    precision, embeds the three documents n_warmup times (untimed) and then n_timed times
    (timed), and rewrites its record after every batch. The first warm-up batch runs under the
    PyTorch profiler to record which attention kernels ran (it is untimed on every condition)."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    plan = tb.read_json(args.plan)
    out_path = Path(args.worker_out)
    gpu = plan["device"] == "gpu"
    import torch
    import torch.nn.functional as F

    torch.set_num_threads(int(plan["threads"]))
    import transformers.integrations.sdpa_attention as sdpa_mod
    from transformers import AutoModel, AutoTokenizer

    # The attention path of the CPU runs of the original benchmark, on both devices: keys and
    # values repeated to the query heads instead of enable_gqa (same arithmetic; see
    # throughput_benchmark.stage_cpu_worker).
    sdpa_mod.use_gqa_in_sdpa = lambda *a, **k: False
    pid = os.getpid()
    cg = tb.cgroup_dir_of_pid(pid)

    def cpu_usage():
        return tb.cgroup_keyed(cg, "cpu.stat").get("usage_usec") if cg else None

    def cg_text(name):
        return (tb.read_text(Path(cg) / name) or "").strip() if cg else None

    rec = {
        "stage": "worker",
        "model": plan["model"],
        "device": plan["device"],
        "precision_requested": plan["precision"],
        "pid": pid,
        "cgroup": cg,
        "memory_max": cg_text("memory.max"),
        "memory_swap_max": cg_text("memory.swap.max"),
        "torch": torch_record(torch, gpu),
        "use_gqa_in_sdpa_forced_false": True,
        "t_process_start": time.time(),
        "status": "starting",
        "batches": [],
    }
    tb.write_json(out_path, rec)
    embs = []
    try:
        if gpu != torch.cuda.is_available():
            raise SystemExit(
                f"device {plan['device']} but torch.cuda.is_available() = {torch.cuda.is_available()}"
            )
        device = torch.device("cuda:0" if gpu else "cpu")
        dtype = getattr(torch, plan["precision"])
        if gpu:
            nvml = tb.Nvml(0)
            before = nvml.used()
            torch.zeros(1, device=device)
            torch.cuda.synchronize()
            after, total = nvml.used(), nvml.total()
            nvml.shutdown()
            cap, frac = allocator_cap(plan["ceiling_bytes"], after, total)
            rec["ceiling"] = {
                "ceiling_bytes": plan["ceiling_bytes"],
                "device_total_bytes": total,
                "device_used_before_context_bytes": before,
                "device_used_after_context_bytes": after,
                "cuda_context_bytes": after - before,
                "allocator_cap_bytes": cap,
                "allocator_fraction": frac,
                "enforcement": "torch.cuda.set_per_process_memory_fraction(cap / total) once the CUDA context exists; the orchestrator's 20 ms NVML watchdog kills the worker if the device total exceeds the ceiling",
            }
            if cap <= 0:
                raise SystemExit(
                    f"no room under the ceiling: device used {tb.mib(after)} MiB after the CUDA context"
                )
            torch.cuda.set_per_process_memory_fraction(frac, 0)
        snap = str(tb.snapshot_path(plan["hub_dir"], plan["model"]))
        tb.reset_high_water_mark()
        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(snap, padding_side="left")
        model = AutoModel.from_pretrained(snap, dtype=dtype, attn_implementation="sdpa")
        model = model.to(device).eval()
        if gpu:
            torch.cuda.synchronize()
        t1 = time.time()
        st = tb.proc_status(pid)
        first = next(model.parameters())
        rec["load"] = {
            "t_start": t0,
            "t_end": t1,
            "wall_s": t1 - t0,
            "param_dtype": str(first.dtype),
            "param_device": str(first.device),
            "n_parameters": int(sum(p.numel() for p in model.parameters())),
            "attn_implementation": getattr(model.config, "_attn_implementation", None),
            "tokenizer_class": type(tok).__name__,
            "padding_side": tok.padding_side,
            "vm_hwm_bytes": st.get("VmHWM"),
            "rss_anon_bytes": st.get("RssAnon"),
            "gpu_allocated_bytes": int(torch.cuda.memory_allocated()) if gpu else None,
            "gpu_reserved_bytes": int(torch.cuda.memory_reserved()) if gpu else None,
        }
        rec["status"] = "running"
        tb.write_json(out_path, rec)
        texts = [d["text"] for d in plan["documents"]]
        expected = [int(d["n_tokens"]) for d in plan["documents"]]
        marker_dir = Path(plan["marker_dir"]) if plan.get("marker_dir") else None
        host_clock, host_clock_error = None, None
        try:
            host_clock = HostClock()
        except Exception as e:
            host_clock_error = repr(e)
        rec["host_clock"] = {
            "method": "Windows QueryPerformanceCounter through a PowerShell process over WSL interop; each reading at the midpoint of its round trip",
            "frequency_hz": host_clock.freq if host_clock else None,
            "error": host_clock_error,
        }
        n_warmup, n_timed = int(plan["n_warmup"]), int(plan["n_timed"])
        activities = [torch.profiler.ProfilerActivity.CPU]
        if gpu:
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        for seq in range(1, n_warmup + n_timed + 1):
            phase = "warmup" if seq <= n_warmup else "timed"
            profiled = seq == 1
            if gpu:
                torch.cuda.reset_peak_memory_stats()
            tb.reset_high_water_mark()
            prof_ctx = (
                torch.profiler.profile(activities=activities)
                if profiled
                else contextlib.nullcontext()
            )
            u_a, p_a, wall_a = cpu_usage(), os.times(), time.time()
            with prof_ctx as prof:
                write_marker(marker_dir, f"{seq:03d}_start")
                hs = host_reading(host_clock)
                t_0 = time.perf_counter()
                enc = tok(texts, padding=True, truncation=False, return_tensors="pt")
                t_1 = time.perf_counter()
                enc = {k: v.to(device) for k, v in enc.items()}
                if gpu:
                    torch.cuda.synchronize()
                t_2 = time.perf_counter()
                with torch.inference_mode():
                    out = model(**enc, use_cache=False)
                    vec = F.normalize(
                        tb.last_token_pool(
                            out.last_hidden_state, enc["attention_mask"]
                        ),
                        p=2,
                        dim=1,
                    )
                if gpu:
                    torch.cuda.synchronize()
                t_3 = time.perf_counter()
                emb = vec.float().cpu().numpy()
                t_4 = time.perf_counter()
                he = host_reading(host_clock)
                write_marker(marker_dir, f"{seq:03d}_end")
            wall_b, u_b, p_b = time.time(), cpu_usage(), os.times()
            counts = [int(x) for x in enc["attention_mask"].sum(dim=1).tolist()]
            s = tb.proc_status(pid)
            wall = t_4 - t_0
            usage = (u_b - u_a) if u_a is not None and u_b is not None else None
            row = {
                "seq": seq,
                "phase": phase,
                "index": seq if phase == "warmup" else seq - n_warmup,
                "profiled": profiled,
                "n_documents": len(texts),
                "n_tokens": sum(counts),
                "doc_tokens": ";".join(str(c) for c in counts),
                "counts_equal_expected": counts == expected,
                "padded_shape": "x".join(str(x) for x in enc["input_ids"].shape),
                "padded_tokens": int(enc["input_ids"].numel()),
                "time_s": wall,
                "tokenize_s": t_1 - t_0,
                "to_device_s": t_2 - t_1,
                "forward_pool_normalize_s": t_3 - t_2,
                "to_host_s": t_4 - t_3,
                "embeddings_finite": bool(np.isfinite(emb).all()),
                "embedding_dim": int(emb.shape[1]),
                "cpu_usage_usec": usage,
                "busy_cpus": (usage / 1e6 / wall) if usage is not None else None,
                "busy_cpus_process": ((p_b.user + p_b.system) - (p_a.user + p_a.system))
                / wall,
                "vm_hwm_bytes": s.get("VmHWM"),
                "rss_anon_bytes": s.get("RssAnon"),
                "threads": s.get("Threads"),
                "scope_memory_peak_bytes": tb.cgroup_int(cg, "memory.peak")
                if cg
                else None,
                "gpu_max_allocated_bytes": int(torch.cuda.max_memory_allocated())
                if gpu
                else None,
                "gpu_max_reserved_bytes": int(torch.cuda.max_memory_reserved())
                if gpu
                else None,
                "t_start": wall_a,
                "t_end": wall_b,
                "perf_t0": t_0,
                "perf_t4": t_4,
                "qpc_start_ns": hs[0],
                "qpc_end_ns": he[0],
                "qpc_rtt_start_ns": hs[1],
                "qpc_rtt_end_ns": he[1],
                "qpc_time_s": cristian_duration_s(hs[0], he[0], hs[1], he[1]),
            }
            if profiled:
                names = [e.key for e in prof.key_averages()]
                rec["profile_first_warmup_batch"] = {
                    "attention_ops_and_kernels": attention_kernels(names),
                    "n_distinct_events": len(set(names)),
                }
            rec["batches"].append(row)
            if phase == "timed":
                embs.append(emb)
            del out, vec, enc
            tb.write_json(out_path, rec)
            if counts != expected:
                raise SystemExit(
                    f"token counts {counts} differ from the stored counts {expected}"
                )
        if host_clock is not None:
            host_clock.close()
        rec["repeat_max_abs_diff_to_first_timed"] = max_abs_diff_to_first(embs)
        if args.worker_embeddings:
            np.save(args.worker_embeddings, np.stack(embs).astype(np.float32))
            rec["embeddings_file"] = str(args.worker_embeddings)
        rec["status"] = "complete"
    except torch.OutOfMemoryError as e:
        rec["status"] = "refused_at_allocator_cap"
        rec["error"] = repr(e)[:3000]
        tb.write_json(out_path, rec)
        raise SystemExit(3)
    except SystemExit as e:
        rec["status"] = "stopped"
        rec["error"] = str(e)[:3000]
        tb.write_json(out_path, rec)
        raise
    except Exception:
        rec["status"] = "error"
        rec["error"] = traceback.format_exc()[-4000:]
        tb.write_json(out_path, rec)
        raise
    rec["t_end"] = time.time()
    rec["final_status"] = tb.proc_status(pid)
    tb.write_json(out_path, rec)


# ----------------------------------------------------------------------------------------------
# Orchestration: inputs, the scope, the samplers and the GPU watchdog
# ----------------------------------------------------------------------------------------------


def verify_input(path, project_dir):
    """Hashes an input and compares it with its sidecar; refuses to run on a mismatch."""
    path = Path(path)
    sidecar = path.with_name(path.name + ".manifest.yml")
    measured = tb.sha256_file(path)
    expected = tb.read_sidecar_hash(sidecar) if sidecar.exists() else ""
    if measured != expected:
        raise SystemExit(
            f"{path} hash {measured} differs from its sidecar {expected}: refusing to run on unverified data"
        )
    return {
        "path": tb.rel_path(path, project_dir),
        "sha256": measured,
        "hash_equal_to_sidecar": True,
    }


def load_documents(project_dir):
    """The three documents: keys and token counts from Case B of the feasibility test's case
    table (in its order), cross-checked against the stored token table, and their canonical
    text, unchanged."""
    import pyarrow.parquet as pq

    canonical = tb.canonical_paths(project_dir)[0]
    token_table = tb.token_table_paths()[0]
    checks = {
        "case_table": verify_input(CASE_FILE, project_dir),
        "canonical_table": verify_input(canonical, project_dir),
        "token_table": verify_input(token_table, project_dir),
    }
    with open(CASE_FILE, encoding="utf-8", newline="") as f:
        docs = select_case_documents(list(csv.DictReader(f)))
    keys = [d["doc_key"] for d in docs]
    if keys != EXPECTED_DOC_KEYS or [d["n_tokens"] for d in docs] != EXPECTED_TOKENS:
        raise SystemExit(
            f"Case {CASE_NAME} lists {docs}, not {EXPECTED_DOC_KEYS} / {EXPECTED_TOKENS}"
        )
    tt = pq.read_table(
        token_table,
        columns=["doc_key", "record_type", "n_chars", "n_tokens_4B"],
        filters=[("doc_key", "in", keys)],
    ).to_pylist()
    ct = pq.read_table(
        canonical,
        columns=["doc_key", tb.CANONICAL_TEXT_FIELD],
        filters=[("doc_key", "in", keys)],
    ).to_pylist()
    for rows, label in ((tt, "token table"), (ct, "canonical table")):
        found = sorted(r["doc_key"] for r in rows)
        if found != sorted(keys):
            raise SystemExit(
                f"{label}: expected one row for each of {keys}, found {found}"
            )
    tokens = {r["doc_key"]: r for r in tt}
    texts = {r["doc_key"]: r[tb.CANONICAL_TEXT_FIELD] for r in ct}
    for d in docs:
        t, text = tokens[d["doc_key"]], texts[d["doc_key"]]
        if int(t["n_tokens_4B"]) != d["n_tokens"] or len(text) != int(t["n_chars"]):
            raise SystemExit(
                f"{d['doc_key']}: case table, token table and text disagree"
            )
        d.update(
            {
                "record_type": t["record_type"],
                "n_chars": len(text),
                "text_sha256": "sha256:"
                + hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "text": text,
            }
        )
    return docs, checks


def wait_for_scope(proc, limit):
    """The worker's transient scope (cgroup v2 directory) once its memory limit is in place."""
    for _ in range(400):
        c = tb.cgroup_dir_of_pid(proc.pid)
        if c and c.endswith(".scope") and tb.cgroup_int(c, "memory.max") == limit:
            return c
        if proc.poll() is not None:
            return None
        time.sleep(0.025)
    return None


def launch_worker(args, plan_path, rec_path, emb_path, log_path, gpu):
    """Starts the worker in a transient systemd scope (MemoryMax = the RAM ceiling, no swap,
    the same threads on both devices; the GPU is hidden from CPU runs) and samples the scope's
    memory, the worker's memory, the VM's memory and the GPU's total memory until it exits. On
    GPU runs the GPU samples come every 20 ms and a sample above the ceiling kills the worker."""
    limit = args.ram_ceiling_mib * 1048576
    ceiling = args.ceiling_mib * 1048576
    cmd = [
        "systemd-run",
        "--user",
        "--scope",
        "--quiet",
        "-p",
        f"MemoryMax={limit}",
        "-p",
        "MemorySwapMax=0",
        "--",
        "env",
        f"CUDA_VISIBLE_DEVICES={'0' if gpu else ''}",
        f"OMP_NUM_THREADS={args.threads}",
        f"MKL_NUM_THREADS={args.threads}",
        "TOKENIZERS_PARALLELISM=true",
        "HF_HUB_OFFLINE=1",
        "PYTHONUNBUFFERED=1",
        sys.executable,
        str(SCRIPT_PATH),
        "--stage",
        "worker",
        "--plan",
        str(plan_path),
        "--worker-out",
        str(rec_path),
        "--worker-embeddings",
        str(emb_path),
    ]
    nvml = tb.Nvml()
    gpu_before = nvml.used()
    watchdog = {"armed": gpu, "fired": False, "first_over": None, "n_samples_over": 0}
    logf = open(log_path, "w", encoding="utf-8")
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)

    def on_gpu_sample(used):
        if not gpu or used <= ceiling:
            return
        watchdog["n_samples_over"] += 1
        if not watchdog["fired"]:
            watchdog["fired"] = True
            watchdog["first_over"] = {"t": time.time(), "used_bytes": used}
            try:
                os.kill(proc.pid, 9)
            except ProcessLookupError:
                pass

    cg = wait_for_scope(proc, limit)
    last = {"memory_peak": None, "oom_kill": 0}

    def scope_current():
        v = tb.cgroup_int(cg, "memory.current") if cg else None
        p = tb.cgroup_int(cg, "memory.peak") if cg else None
        if p is not None:
            last["memory_peak"] = p
        ev = tb.cgroup_keyed(cg, "memory.events") if cg else {}
        if ev.get("oom_kill"):
            last["oom_kill"] = ev["oom_kill"]
        return v

    interval = args.sample_ms / 1000.0
    gpu_interval = args.nvml_interval_ms / 1000.0 if gpu else max(interval, 0.1)
    samplers = {
        "scope_memory": tb.PeakSampler(scope_current, interval_s=interval).start(),
        "worker_rss": tb.PeakSampler(
            lambda: tb.proc_status(proc.pid).get("VmRSS"), interval_s=interval
        ).start(),
        "worker_anon": tb.PeakSampler(
            lambda: tb.proc_status(proc.pid).get("RssAnon"), interval_s=interval
        ).start(),
        "vm_used": tb.PeakSampler(
            tb.vm_used_bytes, interval_s=max(interval, 0.1)
        ).start(),
        "gpu_used": tb.PeakSampler(
            nvml.used, interval_s=gpu_interval, on_sample=on_gpu_sample
        ).start(),
    }
    timed_out = False
    try:
        rc = proc.wait(timeout=args.worker_timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        rc = proc.wait()
        timed_out = True
    t1 = time.time()
    for s in samplers.values():
        s.stop()
    logf.close()
    nvml.shutdown()
    scope_peak = max(
        x
        for x in (last["memory_peak"], samplers["scope_memory"].peak, 0)
        if x is not None
    )
    gpu_peak = samplers["gpu_used"].peak
    if watchdog["fired"]:
        outcome, at_ceiling = "watchdog_kill_gpu_ceiling", True
    elif gpu and gpu_peak is not None and gpu_peak > ceiling:
        outcome, at_ceiling = "gpu_peak_over_ceiling", True
    else:
        outcome, at_ceiling = tb.classify_cpu_exit(
            rc, last["oom_kill"], scope_peak, limit
        )
    return {
        "command": cmd,
        "returncode": rc,
        "outcome": outcome,
        "failure_at_ceiling": at_ceiling,
        "timed_out": timed_out,
        "t_start": t0,
        "t_end": t1,
        "wall_s": t1 - t0,
        "scope_cgroup": cg,
        "memory_limit_bytes": limit,
        "scope_memory_peak_bytes": scope_peak,
        "scope_oom_kill_events": last["oom_kill"],
        "gpu_ceiling_bytes": ceiling,
        "gpu_used_before_start_bytes": gpu_before,
        "gpu_sample_interval_ms": round(gpu_interval * 1000),
        "watchdog": watchdog,
        "samplers": samplers,
    }


# ----------------------------------------------------------------------------------------------
# Stage: run (one condition)
# ----------------------------------------------------------------------------------------------

IMPLEMENTATION = (
    "transformers AutoModel with attn_implementation=sdpa and use_gqa_in_sdpa forced to False "
    "(keys and values repeated to the query heads), use_cache=False, left padding (the three "
    "documents padded to the longest), the model card's last-token pooling, L2 normalisation "
    "(torch.nn.functional.normalize), no truncation, no prompt, no quantisation; the same code "
    "on both devices"
)
BATCH_TIME_DEFINITION = (
    "wall time on the Windows host's performance counter (QPC) from the three texts to their "
    "normalised embeddings in host memory as float32 (tokenisation, copy to the device, forward "
    "pass, pooling, normalisation and the copy back; the GPU synchronised), read through a "
    "PowerShell helper immediately before and after the batch with half of each round trip "
    "subtracted; Windows marker-file times and the VM's own clock (perf_counter, which runs "
    "slow) are recorded beside it"
)
BATCH_COLUMNS = [
    "seq",
    "phase",
    "index",
    "profiled",
    "n_documents",
    "n_tokens",
    "doc_tokens",
    "counts_equal_expected",
    "padded_shape",
    "padded_tokens",
    "qpc_time_s",
    "host_time_s",
    "time_s",
    "host_over_vm",
    "tokenize_s",
    "to_device_s",
    "forward_pool_normalize_s",
    "to_host_s",
    "embeddings_finite",
    "embedding_dim",
    "busy_cpus",
    "busy_cpus_process",
    "threads",
    "vm_hwm_bytes",
    "rss_anon_bytes",
    "scope_memory_peak_bytes",
    "gpu_max_allocated_bytes",
    "gpu_max_reserved_bytes",
    "gpu_total_peak_bytes",
    "t_start",
    "t_end",
    "host_start_unix_ns",
    "host_end_unix_ns",
    "perf_t0",
    "perf_t4",
    "qpc_start_ns",
    "qpc_end_ns",
    "qpc_rtt_start_ns",
    "qpc_rtt_end_ns",
]
POWERSHELL = "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"


def read_host_marker_times(directory):
    """The Windows modification times of the marker files, read on the Windows side (PowerShell
    through WSL interop, so no clock of the VM is involved), as {name: Unix ns}, and a record of
    the read."""
    rec = {
        "method": "PowerShell Get-ChildItem LastWriteTimeUtc.Ticks through WSL interop",
        "error": None,
    }
    try:
        win = subprocess.run(
            ["wslpath", "-w", str(directory)],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout.strip()
        literal = win.replace("'", "''")
        command = (
            f"Get-ChildItem -LiteralPath '{literal}' -File | "
            "ForEach-Object { $_.Name + ' ' + $_.LastWriteTimeUtc.Ticks }"
        )
        out = subprocess.run(
            [POWERSHELL, "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True,
            text=True,
            timeout=300,
        )
        marks = parse_marker_listing(out.stdout)
        rec.update(
            {
                "windows_path": win,
                "returncode": out.returncode,
                "n_markers": len(marks),
                "stderr": out.stderr[-500:],
            }
        )
        if out.returncode != 0:
            rec["error"] = f"PowerShell exit {out.returncode}"
        return marks, rec
    except Exception as e:
        rec["error"] = repr(e)
        return {}, rec


DOC_FIELDS = (
    "position",
    "doc_key",
    "record_type",
    "n_chars",
    "n_tokens",
    "text_sha256",
)


def run_dir(args, model_key, device):
    return Path(args.results_dir) / f"run_{tb.MODELS[model_key]['slug']}_{device}"


def run_parameters(args, model_key, device):
    spec = tb.MODELS[model_key]
    return {
        "model": f"{spec['repo']} @ {spec['revision']} (local snapshot, HF_HUB_OFFLINE=1)",
        "device": device,
        "precision": PRECISION[device],
        "implementation": IMPLEMENTATION,
        "documents": f"Case {CASE_NAME} of {tb.rel_path(CASE_FILE, args.project_dir)}: "
        + ", ".join(EXPECTED_DOC_KEYS)
        + " (every batch, in this order)",
        "n_warmup": args.n_warmup,
        "n_timed": args.n_timed,
        "batch_time": BATCH_TIME_DEFINITION,
        "timing_clock": "Windows performance counter (QPC) read through a PowerShell helper over WSL interop immediately before and after each batch, half of each round trip subtracted; Windows marker-file modification times and the VM's perf_counter recorded beside it",
        "threads": args.threads,
        "scope": "systemd-run --user --scope -p MemoryMax=<bytes> -p MemorySwapMax=0; one worker process, one batch at a time, the model loaded once",
        "memory_max_bytes": args.ram_ceiling_mib * 1048576,
        "memory_swap_max_bytes": 0,
        "gpu_ceiling_bytes": args.ceiling_mib * 1048576,
        "gpu_ceiling_scope": "total memory of the card as NVML reports it, other processes (the Windows desktop) included",
        "gpu_enforcement": "GPU runs: torch.cuda.set_per_process_memory_fraction((ceiling - device total after the CUDA context) / total) plus a 20 ms NVML watchdog that kills the worker above the ceiling; CPU runs: CUDA hidden, GPU memory sampled",
        "nvml_interval_ms": args.nvml_interval_ms,
        "sample_ms": args.sample_ms,
        "quick_run": bool(args.quick),
    }


def stage_run(args):
    """One condition: verifies the inputs, writes the worker's plan (the three texts) outside the
    repository, runs the worker in its scope and writes run.json, batches.csv and their sidecars."""
    model_key, device = args.model, args.device
    if model_key not in tb.MODELS or device not in DEVICES:
        raise SystemExit("--stage run needs --model 0.6B|4B and --device cpu|gpu")
    gpu = device == "gpu"
    spec = tb.MODELS[model_key]
    cond = f"{model_key} {device.upper()}"
    out = run_dir(args, model_key, device)
    out.mkdir(parents=True, exist_ok=True)
    names = (
        "run.json",
        "batches.csv",
        "worker_record.json",
        "worker.log",
        "timed_embeddings.npy",
        "host_clock_markers.csv",
    )
    paths = {n: out / n for n in names}
    tb.remove_outputs(paths.values())
    marker_dir = out / "host_clock_markers"
    if marker_dir.exists():
        for p in marker_dir.iterdir():
            p.unlink()
    marker_dir.mkdir(parents=True, exist_ok=True)
    sc = tb.Sidecars(
        args.project_dir, run_parameters(args, model_key, device), "", SCRIPT_PATH
    )
    sc.script_hash = SCRIPT_SHA256
    docs, checks = load_documents(args.project_dir)
    work = Path(args.work_dir).expanduser()
    work.mkdir(parents=True, exist_ok=True)
    plan_path = work / f"plan_{spec['slug']}_{device}.json"
    tb.write_json(
        plan_path,
        {
            "model": model_key,
            "device": device,
            "precision": PRECISION[device],
            "threads": args.threads,
            "n_warmup": args.n_warmup,
            "n_timed": args.n_timed,
            "ceiling_bytes": args.ceiling_mib * 1048576,
            "hub_dir": args.hub_dir,
            "marker_dir": str(marker_dir),
            "documents": [
                {k: d[k] for k in ("position", "doc_key", "n_tokens", "text")}
                for d in docs
            ],
        },
    )
    tb.log(
        f"{cond} ({PRECISION[device]}): load, then {args.n_warmup} warm-up and {args.n_timed} "
        f"timed batches of {', '.join(EXPECTED_DOC_KEYS)}"
    )
    launch = launch_worker(
        args,
        plan_path,
        paths["worker_record.json"],
        paths["timed_embeddings.npy"],
        paths["worker.log"],
        gpu,
    )
    rec_file = paths["worker_record.json"]
    worker = tb.read_json(rec_file) if rec_file.exists() else {}
    batches = worker.get("batches", [])
    marks, host_read = read_host_marker_times(marker_dir)
    batches = attach_host_times(batches, marks)
    s = launch["samplers"]
    for b in batches:
        b["gpu_total_peak_bytes"] = tb.summarise_samples(
            s["gpu_used"].samples, b["t_start"], b["t_end"]
        )["peak"]
    refused = worker.get("status") == "refused_at_allocator_cap"
    complete = worker.get("status") == "complete" and launch["outcome"] == "success"

    def summary_on(key):
        timed = [b for b in batches if b["phase"] == "timed"]
        ok = complete and timed and all(b.get(key) is not None for b in timed)
        return condition_summary(batches, key) if ok else None

    summary_qpc = summary_on("qpc_time_s")
    summary_host = summary_on("host_time_s")
    summary_vm = summary_on("time_s")
    summary_clock, summary = pick_primary(
        [
            ("windows_qpc", summary_qpc),
            ("windows_file_mtime", summary_host),
            ("vm_perf_counter", summary_vm),
        ]
    )
    window = summary["timed_window"] if summary else (None, None)
    gpu_values = [v for _, v in s["gpu_used"].samples]
    ceiling = launch["gpu_ceiling_bytes"]

    def batch_max(key):
        values = [b[key] for b in batches if b.get(key) is not None]
        return max(values) if values else None

    run = {
        "stage": "run",
        "timestamp": tb.utc_now(),
        "condition": cond,
        "model": model_key,
        "model_repo": spec["repo"],
        "model_revision": spec["revision"],
        "device": device,
        "precision_requested": PRECISION[device],
        "precision_used": (worker.get("load") or {}).get("param_dtype"),
        "complete": complete,
        "outcome": "refused_at_allocator_cap" if refused else launch["outcome"],
        "failure_at_ceiling": bool(launch["failure_at_ceiling"] or refused),
        "worker_status": worker.get("status"),
        "worker_error": worker.get("error"),
        "documents": [{k: d[k] for k in DOC_FIELDS} for d in docs],
        "input_checks": checks,
        "n_warmup": args.n_warmup,
        "n_timed": args.n_timed,
        "batch_time_definition": BATCH_TIME_DEFINITION,
        "summary": summary,
        "summary_clock": summary_clock,
        "summary_windows_qpc": summary_qpc,
        "summary_windows_file_mtime": summary_host,
        "summary_vm_clock": summary_vm,
        "clock_ratio_windows_over_vm": clock_ratio(
            summary_qpc or summary_host, summary_vm
        ),
        "clock_ratio_file_mtime_over_qpc": clock_ratio(summary_host, summary_qpc),
        "host_clock_helper": worker.get("host_clock"),
        "host_clock_read": host_read,
        "configuration": {
            "implementation": IMPLEMENTATION,
            "torch": worker.get("torch"),
            "scope_memory_max": worker.get("memory_max"),
            "scope_memory_swap_max": worker.get("memory_swap_max"),
            "worker_processes": 1,
            "batches_at_a_time": 1,
            "worker_os_threads_max": batch_max("threads"),
        },
        "load": worker.get("load"),
        "gpu_ceiling_enforcement": worker.get("ceiling"),
        "gpu_memory": {
            "ceiling_bytes": ceiling,
            "device_used_before_start_bytes": launch["gpu_used_before_start_bytes"],
            "peak_total_bytes_whole_run": s["gpu_used"].peak,
            "peak_total_bytes_timed_batches": tb.summarise_samples(
                s["gpu_used"].samples, *window
            )["peak"]
            if summary
            else None,
            "min_total_bytes_whole_run": min(gpu_values, default=None),
            "all_samples_at_or_below_ceiling": all(v <= ceiling for v in gpu_values),
            "n_samples": s["gpu_used"].n_samples,
            "sample_interval_ms": launch["gpu_sample_interval_ms"],
            "watchdog": launch["watchdog"],
            "allocator_peak_allocated_bytes": batch_max("gpu_max_allocated_bytes"),
            "allocator_peak_reserved_bytes": batch_max("gpu_max_reserved_bytes"),
        },
        "ram": {
            "scope_memory_limit_bytes": launch["memory_limit_bytes"],
            "scope_memory_peak_bytes": launch["scope_memory_peak_bytes"],
            "scope_oom_kill_events": launch["scope_oom_kill_events"],
            "worker_anon_peak_bytes_whole_run": s["worker_anon"].peak,
            "worker_anon_peak_bytes_timed_batches": tb.summarise_samples(
                s["worker_anon"].samples, *window
            )["peak"]
            if summary
            else None,
            "worker_rss_peak_bytes_whole_run": s["worker_rss"].peak,
            "vm_used_peak_bytes": s["vm_used"].peak,
        },
        "profile_first_warmup_batch": worker.get("profile_first_warmup_batch"),
        "repeat_max_abs_diff_to_first_timed": worker.get(
            "repeat_max_abs_diff_to_first_timed"
        ),
        "launch": {k: v for k, v in launch.items() if k != "samplers"},
        "quick_run": bool(args.quick),
    }
    tb.write_json(paths["run.json"], run)
    tb.write_csv(paths["batches.csv"], batches, BATCH_COLUMNS)
    tb.write_csv(
        paths["host_clock_markers.csv"],
        [{"marker": k, "host_unix_ns": v} for k, v in sorted(marks.items())],
        ["marker", "host_unix_ns"],
    )
    if marks and not host_read.get("error"):
        for p in marker_dir.iterdir():
            p.unlink()
        marker_dir.rmdir()
    inputs = [
        sc.input_entry(
            CASE_FILE,
            "the three documents: Case B, in order",
            checks["case_table"]["sha256"],
        ),
        sc.input_entry(
            tb.canonical_paths(args.project_dir)[0],
            "canonical Phase 1 table (the texts), read-only",
            checks["canonical_table"]["sha256"],
        ),
        sc.input_entry(
            tb.token_table_paths()[0],
            "stored token counts (cross-check)",
            checks["token_table"]["sha256"],
        ),
    ]
    for name in names:
        if paths[name].exists():
            sc.write(
                paths[name],
                inputs,
                notes=f"CPU-versus-GPU comparison, condition {cond}: {name}",
            )
    if summary:
        times = ", ".join(f"{t:.3f}" for t in summary["timed_times_s"])
        tb.log(
            f"{cond}: timed batches {times} s; mean {summary['mean_s']:.3f} s, median "
            f"{summary['median_s']:.3f} s; {summary['documents_per_s']:.4f} documents/s, "
            f"{summary['tokens_per_s']:.1f} tokens/s; precision {run['precision_used']}"
        )
        tb.log(
            f"{cond}: clock {summary_clock}; mean Windows / VM {_f(run['clock_ratio_windows_over_vm'], 4)}; "
            f"file times / performance counter {_f(run['clock_ratio_file_mtime_over_qpc'], 4)}; "
            f"helper {(worker.get('host_clock') or {}).get('error') or 'ok'}; markers read "
            f"{host_read.get('n_markers')}, error {host_read.get('error')}"
        )
        if gpu:
            g = run["gpu_memory"]
            tb.log(
                f"{cond}: peak total GPU memory {tb.mib(g['peak_total_bytes_whole_run'])} MiB "
                f"(before start {tb.mib(g['device_used_before_start_bytes'])} MiB; ceiling "
                f"{args.ceiling_mib} MiB; every sample at or below: {g['all_samples_at_or_below_ceiling']})"
            )
    else:
        tb.log(
            f"{cond}: NOT complete: outcome {run['outcome']}, worker status "
            f"{worker.get('status')}: {str(worker.get('error'))[:400]}"
        )
        raise SystemExit(2)


# ----------------------------------------------------------------------------------------------
# Stage: report (the comparison of the four conditions)
# ----------------------------------------------------------------------------------------------


def load_runs(args):
    runs = {}
    for key in CONDITIONS:
        p = run_dir(args, *key) / "run.json"
        runs[key] = tb.read_json(p) if p.exists() else None
    return runs


def partial_outcome(partial):
    """The outcome of a worker that was stopped before its run was recorded."""
    batches = (partial or {}).get("batches", [])
    timed = sum(1 for b in batches if b.get("phase") == "timed")
    return (
        f"stopped before completion after {len(batches)} batches "
        f"({len(batches) - timed} untimed warm-up, {timed} timed); no result"
    )


def comparison_row(key, run, partial=None):
    """One row of the comparison table (a condition that did not complete keeps its outcome)."""
    model_key, device = key
    row = {
        "condition": f"{model_key} {device.upper()}",
        "model": tb.MODELS[model_key]["repo"],
        "device": device.upper(),
    }
    if not run:
        row["outcome"] = partial_outcome(partial) if partial else "not run"
        return row
    s = run.get("summary") or {}
    vm = run.get("summary_vm_clock") or {}
    g = run.get("gpu_memory") or {}
    ram = run.get("ram") or {}
    torch_rec = (run.get("configuration") or {}).get("torch") or {}

    def mib_or_none(v):
        return tb.mib(v) if v is not None else None

    row.update(
        {
            "outcome": run["outcome"],
            "precision_used": run.get("precision_used"),
            "n_warmup_untimed": run["n_warmup"],
            "n_timed": run["n_timed"],
            "documents_per_batch": s.get("documents_per_batch"),
            "tokens_per_batch": s.get("tokens_per_batch"),
            "timed_batch_times_s": "; ".join(
                f"{x:.4f}" for x in s.get("timed_times_s", [])
            ),
            "mean_batch_s": s.get("mean_s"),
            "median_batch_s": s.get("median_s"),
            "min_batch_s": s.get("min_s"),
            "max_batch_s": s.get("max_s"),
            "std_batch_s": s.get("std_s"),
            "documents_per_s": s.get("documents_per_s"),
            "tokens_per_s": s.get("tokens_per_s"),
            "timing_clock": run.get("summary_clock"),
            "vm_clock_mean_batch_s": vm.get("mean_s"),
            "vm_clock_median_batch_s": vm.get("median_s"),
            "vm_clock_timed_batch_times_s": "; ".join(
                f"{x:.4f}" for x in vm.get("timed_times_s", [])
            ),
            "clock_ratio_windows_over_vm": run.get("clock_ratio_windows_over_vm"),
            "file_mtime_mean_batch_s": (
                run.get("summary_windows_file_mtime") or {}
            ).get("mean_s"),
            "clock_ratio_file_mtime_over_qpc": run.get(
                "clock_ratio_file_mtime_over_qpc"
            ),
            "peak_total_gpu_memory_mib": mib_or_none(
                g.get("peak_total_bytes_whole_run")
            )
            if device == "gpu"
            else None,
            "gpu_total_before_start_mib": mib_or_none(
                g.get("device_used_before_start_bytes")
            ),
            "gpu_total_min_during_run_mib": mib_or_none(
                g.get("min_total_bytes_whole_run")
            ),
            "gpu_total_max_during_run_mib": mib_or_none(
                g.get("peak_total_bytes_whole_run")
            ),
            "gpu_all_samples_at_or_below_ceiling": g.get(
                "all_samples_at_or_below_ceiling"
            ),
            "torch_threads": torch_rec.get("torch_threads"),
            "torch_interop_threads": torch_rec.get("torch_interop_threads"),
            "cpu_affinity_count": torch_rec.get("cpu_affinity_count"),
            "scope_memory_max_bytes": (run.get("configuration") or {}).get(
                "scope_memory_max"
            ),
            "worker_anon_peak_gib": tb.gib(ram["worker_anon_peak_bytes_whole_run"])
            if ram.get("worker_anon_peak_bytes_whole_run")
            else None,
            "scope_memory_peak_gib": tb.gib(ram["scope_memory_peak_bytes"])
            if ram.get("scope_memory_peak_bytes")
            else None,
        }
    )
    return row


def stage_report(args):
    """Combines the four runs: documents.csv, timings_all_batches.csv, comparison.csv,
    speed_ratios.csv, precision_agreement.csv, environment.json (+ sidecars) and the report."""
    res = Path(args.results_dir)
    res.mkdir(parents=True, exist_ok=True)
    runs = load_runs(args)
    done = {k: r for k, r in runs.items() if r and r.get("complete")}
    if not done:
        raise SystemExit(f"no completed run under {res}")
    doc_lists = {json_key(r["documents"]) for r in done.values()}
    if len(doc_lists) != 1:
        raise SystemExit("the completed runs did not embed the same documents")
    documents = next(iter(done.values()))["documents"]
    partials = {}
    for k in CONDITIONS:
        rec_path = run_dir(args, *k) / "worker_record.json"
        if not runs.get(k) and rec_path.exists():
            partials[k] = tb.read_json(rec_path)
    comparison = [comparison_row(k, runs.get(k), partials.get(k)) for k in CONDITIONS]
    ratios = ratio_rows({k: r["summary"] for k, r in done.items()})
    agreement = []
    for model_key in ("0.6B", "4B"):
        if (model_key, "gpu") not in done or (model_key, "cpu") not in done:
            continue
        g = np.load(run_dir(args, model_key, "gpu") / "timed_embeddings.npy")
        c = np.load(run_dir(args, model_key, "cpu") / "timed_embeddings.npy")
        for r in document_agreement(EXPECTED_DOC_KEYS, g[0], c[0]):
            agreement.append(
                {
                    "model": tb.MODELS[model_key]["repo"],
                    **r,
                    "gpu_precision": done[(model_key, "gpu")]["precision_used"],
                    "cpu_precision": done[(model_key, "cpu")]["precision_used"],
                    "compared": "first timed batch of each device",
                }
            )
    timings = []
    for key in CONDITIONS:
        p = run_dir(args, *key) / "batches.csv"
        if p.exists():
            with open(p, encoding="utf-8", newline="") as f:
                source = list(csv.DictReader(f))
            counted = bool((runs.get(key) or {}).get("complete"))
        elif key in partials:
            source = partials[key].get("batches", [])
            counted = False
        else:
            continue
        for b in source:
            timings.append(
                {
                    "condition": f"{key[0]} {key[1].upper()}",
                    "phase": b["phase"],
                    "index": b["index"],
                    "counted_in_statistics": counted and b["phase"] == "timed",
                    "profiled": b["profiled"],
                    "windows_qpc_time_s": b.get("qpc_time_s"),
                    "windows_file_time_s": b.get("host_time_s"),
                    "vm_clock_time_s": b["time_s"],
                    "tokenize_s": b["tokenize_s"],
                    "to_device_s": b["to_device_s"],
                    "forward_pool_normalize_s": b["forward_pool_normalize_s"],
                    "to_host_s": b["to_host_s"],
                    "n_documents": b["n_documents"],
                    "n_tokens": b["n_tokens"],
                    "doc_tokens": b["doc_tokens"],
                    "padded_shape": b["padded_shape"],
                    "busy_cpus": b["busy_cpus"],
                    "gpu_total_peak_bytes": b.get("gpu_total_peak_bytes"),
                }
            )
    env = {
        "timestamp": tb.utc_now(),
        "host": tb.host_record(),
        "nvidia_smi": tb.nvidia_smi_text(),
        "python": sys.version,
        "packages": tb.package_versions(),
        "torch_per_condition": {
            f"{k[0]} {k[1].upper()}": (r.get("configuration") or {}).get("torch")
            for k, r in runs.items()
            if r
        },
    }
    outputs = {
        "documents.csv": documents,
        "timings_all_batches.csv": timings,
        "comparison.csv": comparison,
        "speed_ratios.csv": ratios,
        "precision_agreement.csv": agreement,
    }
    sc = tb.Sidecars(args.project_dir, report_parameters(args), "", SCRIPT_PATH)
    sc.script_hash = SCRIPT_SHA256
    inputs = [
        sc.input_entry(
            run_dir(args, *k) / "run.json", f"run record, {k[0]} {k[1].upper()}"
        )
        for k in CONDITIONS
        if runs.get(k)
    ]
    for name, rows in outputs.items():
        fields = list(rows[0].keys()) if rows else ["none"]
        if name == "comparison.csv":
            fields = list(dict.fromkeys(f for r in rows for f in r))
        tb.write_csv(res / name, rows, fields)
        sc.write(res / name, inputs, notes=f"CPU-versus-GPU comparison: {name}")
    tb.write_json(res / "environment.json", env)
    sc.write(res / "environment.json", inputs, notes="environment at report time")
    report = build_report(
        args, runs, documents, comparison, ratios, agreement, partials
    )
    Path(args.report_path).write_text(report, encoding="utf-8")
    sc.write(
        Path(args.report_path), inputs, notes="generated report (never edited by hand)"
    )
    tb.log(f"report written: {args.report_path}")
    for r in ratios:
        tb.log(
            f"  {r['comparison']}: {r['statement_mean']} (median basis: {r['statement_median']})"
        )


def json_key(value):
    import json

    return json.dumps(value, sort_keys=True)


def _f(x, nd=3):
    return "" if x is None else f"{x:,.{nd}f}"


def _mib(n_bytes):
    return "" if n_bytes is None else f"{n_bytes / 1048576:,.1f}"


def _prec(dtype_text):
    return str(dtype_text).replace("torch.", "") if dtype_text else dtype_text


def _worker_batches(args, key):
    p = run_dir(args, *key) / "worker_record.json"
    return tb.read_json(p).get("batches", []) if p.exists() else []


def build_report(args, runs, documents, comparison, ratios, agreement, partials=None):
    """The Markdown report: only what the four runs recorded."""
    L = []
    add = L.append
    rel = lambda p: tb.rel_path(p, args.project_dir)  # noqa: E731
    add(
        "# Phase 2: CPU versus GPU on the same three documents (Qwen3-Embedding-0.6B and 4B)"
    )
    add("")
    add(
        f"Generated {tb.utc_now()} by `{rel(SCRIPT_PATH)}` (stage `report`). Every number is "
        "measured in these runs; nothing is estimated or extrapolated."
    )
    if args.quick:
        add("")
        add("**SMOKE TEST WITH REDUCED SETTINGS: NOT A RESULT.**")
    add("")
    add("## What was compared")
    add("")
    add(
        "Four conditions: Qwen3-Embedding-0.6B on the CPU, 0.6B on the GPU (RTX 3090), "
        "Qwen3-Embedding-4B on the CPU and 4B on the GPU. Every batch of every condition is the "
        "same three real Reddit documents in the same order. Each condition loads its model "
        "once, runs untimed warm-up batches, then timed batches. All four run the same code: "
        f"{IMPLEMENTATION}. Batch time: {BATCH_TIME_DEFINITION}."
    )
    add("")
    add("## The three documents")
    add("")
    add(
        f"Case {CASE_NAME} of `{rel(CASE_FILE)}` (the three longest documents of the corpus, "
        "already used by the Phase 2 feasibility test). Token counts include the appended "
        "end-of-text token; both models' tokenizers give identical token ids "
        "(`results/prepare.json`), and every batch's counts were checked against these."
    )
    add("")
    add(
        tb.render_table(
            [
                "position",
                "doc_key",
                "record_type",
                "n_chars",
                "n_tokens",
                "text_sha256",
            ],
            documents,
        )
    )
    add("")
    add("## Comparison")
    add("")
    simple = []
    for row in comparison:
        simple.append(
            {
                "Condition": row["condition"],
                "Precision": _prec(row.get("precision_used")) or row.get("outcome"),
                "Mean batch time (s)": _f(row.get("mean_batch_s"), 4),
                "Median batch time (s)": _f(row.get("median_batch_s"), 4),
                "Documents/s": _f(row.get("documents_per_s"), 4),
                "Tokens/s": _f(row.get("tokens_per_s"), 1),
                "Peak total GPU memory (MiB)": _f(
                    row.get("peak_total_gpu_memory_mib"), 1
                )
                if row["device"] == "GPU"
                else "not used",
            }
        )
    add(tb.render_table(list(simple[0].keys()), simple))
    add("")
    for r in ratios:
        add(
            f"- {r['comparison']}: **{r['statement_mean']}** (mean batch time; median basis: "
            f"{r['statement_median']})."
        )
    add("")
    add("## Clocks")
    add("")
    add(
        "Every time above is measured on the Windows host's performance counter: just before "
        "and just after each batch the worker asks a small PowerShell process on Windows for the "
        "counter, and half of each round trip is subtracted. As a second Windows reading, the "
        "worker also writes a marker file before and after each batch and Windows stamps each "
        "file with its time (the file writes add a few milliseconds per batch). The VM's own "
        "clock (the timer inside Linux) is recorded beside both; the table shows how far it was "
        "off in each run, which is why no result uses it. A first pass timed on the VM's clock "
        "alone is kept, superseded, in `results/cpu_vs_gpu_three_documents_first_pass_vm_clock/`."
    )
    add("")
    clocks = [
        {
            "Condition": row["condition"],
            "Timing clock": row.get("timing_clock"),
            "Mean batch time, Windows performance counter (s)": _f(
                row.get("mean_batch_s"), 4
            )
            if row.get("timing_clock") == "windows_qpc"
            else "",
            "Mean batch time, Windows file times (s)": _f(
                row.get("file_mtime_mean_batch_s"), 4
            ),
            "Mean batch time, VM clock (s)": _f(row.get("vm_clock_mean_batch_s"), 4),
            "Windows / VM": _f(row.get("clock_ratio_windows_over_vm"), 4),
            "File times / performance counter": _f(
                row.get("clock_ratio_file_mtime_over_qpc"), 4
            ),
        }
        for row in comparison
    ]
    add(tb.render_table(list(clocks[0].keys()), clocks))
    add("")
    add("## Each condition")
    for key, row in zip(CONDITIONS, comparison):
        run = runs.get(key)
        add("")
        add(f"### {row['condition']}")
        add("")
        if not run or not run.get("complete"):
            err = (run or {}).get("worker_error")
            add(
                f"Not complete: {row.get('outcome')}"
                + (f"; {err}" if err else "")
                + "."
            )
            partial = (partials or {}).get(key)
            if partial:
                shown = [
                    f"{b['phase']} {b['index']}: Windows "
                    + (_f(b.get("qpc_time_s"), 3) or "?")
                    + f" s, VM clock {_f(b.get('time_s'), 3)} s"
                    for b in partial.get("batches", [])
                ]
                add("")
                add(
                    "Batches the worker recorded before it was stopped (untimed warm-up batches "
                    "are excluded from every statistic): " + "; ".join(shown) + "."
                )
            continue
        s, g, ram = run["summary"], run["gpu_memory"], run["ram"]
        t = run["configuration"]["torch"] or {}
        batches = _worker_batches(args, key)
        timed = [b for b in batches if b["phase"] == "timed"]
        busy = [b["busy_cpus"] for b in timed if b.get("busy_cpus") is not None]
        add(
            f"- Model: `{run['model_repo']}` @ `{run['model_revision']}`; device: {key[1].upper()}"
        )
        add(
            f"- Precision actually used: {_prec(run['precision_used'])} (requested {run['precision_requested']})"
        )
        add(
            "- Documents (every batch, in order): "
            + ", ".join(
                f"`{d['doc_key']}` ({d['n_tokens']:,} tokens)" for d in run["documents"]
            )
            + f"; {s['tokens_per_batch']:,} tokens per batch, padded to "
            + (timed[0]["padded_shape"] if timed else "?")
        )
        add(
            f"- Untimed warm-up batches: {s['n_warmup']} (times, excluded from every statistic: "
            + ", ".join(_f(x, 4) or "?" for x in s["warmup_times_s"])
            + " s; the first ran under the profiler)"
        )
        clock = run.get("summary_clock")
        label = {
            "windows_qpc": "Windows performance counter",
            "windows_file_mtime": "Windows file times (the performance counter could not be read)",
            "vm_perf_counter": "VM clock (Windows could not be read)",
        }.get(clock, str(clock))
        add(
            f"- Timed batches ({s['n_timed']}), {label}: "
            + ", ".join(f"{x:.4f}" for x in s["timed_times_s"])
            + " s"
        )
        fm = run.get("summary_windows_file_mtime") or {}
        if clock == "windows_qpc" and fm:
            add(
                "- The same timed batches on the Windows file times: "
                + ", ".join(f"{x:.4f}" for x in fm["timed_times_s"])
                + f" s (mean file times / performance counter {_f(run.get('clock_ratio_file_mtime_over_qpc'), 4)})"
            )
        vm = run.get("summary_vm_clock") or {}
        if clock != "vm_perf_counter" and vm:
            add(
                "- The same timed batches on the VM's clock: "
                + ", ".join(f"{x:.4f}" for x in vm["timed_times_s"])
                + f" s (mean Windows / VM {_f(run.get('clock_ratio_windows_over_vm'), 4)})"
            )
        add(
            f"- Mean {s['mean_s']:.4f} s, median {s['median_s']:.4f} s, minimum {s['min_s']:.4f} s, "
            f"maximum {s['max_s']:.4f} s, standard deviation {_f(s['std_s'], 4)} s"
        )
        add(
            f"- Documents per second {s['documents_per_s']:.4f}; tokens per second {s['tokens_per_s']:,.1f}"
        )
        if key[1] == "gpu":
            ce = run.get("gpu_ceiling_enforcement") or {}
            add(
                f"- Peak total GPU memory (whole card, other processes included): "
                f"{_mib(g['peak_total_bytes_whole_run'])} MiB over the whole run (load, warm-up "
                f"and timed batches), {_mib(g['peak_total_bytes_timed_batches'])} MiB during the "
                f"timed batches; {_mib(g['device_used_before_start_bytes'])} MiB before the worker "
                f"started; ceiling {_mib(g['ceiling_bytes'])} MiB; every sample "
                f"({g['n_samples']:,}, every {g['sample_interval_ms']} ms) at or below the "
                f"ceiling: {g['all_samples_at_or_below_ceiling']}; PyTorch allocator cap "
                f"{_mib(ce.get('allocator_cap_bytes'))} MiB, its peak reservation "
                f"{_mib(g['allocator_peak_reserved_bytes'])} MiB"
            )
        else:
            add(
                f"- CPU configuration actually used: one worker process, one batch at a time; "
                f"PyTorch compute threads {t.get('torch_threads')}, inter-op threads "
                f"{t.get('torch_interop_threads')}, OMP_NUM_THREADS={t.get('omp_num_threads')}, "
                f"MKL_NUM_THREADS={t.get('mkl_num_threads')}, CPUs available to the process "
                f"{t.get('cpu_affinity_count')} (the WSL2 VM's logical CPUs), CPU capability "
                f"{t.get('cpu_capability')}; scope MemoryMax {run['configuration']['scope_memory_max']} "
                f"bytes, swap {run['configuration']['scope_memory_swap_max']}; CUDA visible: "
                f"{t.get('cuda_available')}"
            )
            if busy:
                add(
                    f"- Busy CPUs during the timed batches (scope CPU time / wall time, uncorrected, "
                    f"about +-5% on this VM): {min(busy):.2f} to {max(busy):.2f}"
                )
            add(
                f"- GPU total during the run (desktop only): {_f(row.get('gpu_total_min_during_run_mib'), 1)} "
                f"to {_f(row.get('gpu_total_max_during_run_mib'), 1)} MiB"
            )
        add(
            f"- Memory: worker anonymous memory peak {_f(tb.gib(ram['worker_anon_peak_bytes_whole_run']) if ram['worker_anon_peak_bytes_whole_run'] else None)} GiB; "
            f"scope peak (page cache included) {_f(tb.gib(ram['scope_memory_peak_bytes']) if ram['scope_memory_peak_bytes'] else None)} GiB "
            f"of {tb.gib(ram['scope_memory_limit_bytes'])} GiB"
        )
        prof = run.get("profile_first_warmup_batch") or {}
        add(
            "- Attention operators and kernels recorded in the profiled warm-up batch: "
            + ", ".join(f"`{k}`" for k in prof.get("attention_ops_and_kernels", []))
        )
        diffs = run.get("repeat_max_abs_diff_to_first_timed") or []
        add(
            f"- Largest difference of each timed batch's embeddings from the first timed batch's: {diffs}"
        )
    add("")
    add("## Precision")
    add("")
    add(
        "The CPU runs compute in float32 (32-bit numbers, about 7 significant decimal digits); the GPU runs "
        "in bfloat16 (16-bit numbers, 2 to 3 significant decimal digits, the format the model files are "
        "published in and the one the GPU's tensor cores run fast; this CPU has no fast 16-bit "
        "arithmetic). The same document therefore gets a slightly different embedding on each "
        "device. The table compares each document's GPU embedding with its CPU embedding "
        "(a document is only compared with itself; cosine 1 = identical direction)."
    )
    add("")
    if agreement:
        shown = [
            {
                "model": a["model"],
                "position": a["position"],
                "doc_key": a["doc_key"],
                "cosine (GPU vs CPU)": f"{a['cosine']:.6f}",
                "largest absolute difference": f"{a['max_abs_diff']:.5f}",
                "GPU precision": _prec(a["gpu_precision"]),
                "CPU precision": _prec(a["cpu_precision"]),
            }
            for a in agreement
        ]
        add(tb.render_table(list(shown[0].keys()), shown))
    else:
        add("Not available (a model lacks a completed CPU or GPU run).")
    add("")
    add("## Deviations from a like-for-like comparison")
    add("")
    for d in deviations(runs, partials):
        add(f"- {d}")
    if args.note:
        add(f"- {args.note}")
    add("")
    add("## Files")
    add("")
    add(f"- Script: `{rel(SCRIPT_PATH)}` (helpers from `{rel(Path(tb.__file__))}`)")
    add(
        f"- Unit test: `{rel(tb.BENCH_DIR / 'tests' / 'test_cpu_vs_gpu_three_documents_functions.py')}`"
    )
    add(
        f"- Results: `{rel(args.results_dir)}/` (per condition `run_<model>_<device>/`: run.json, batches.csv, worker_record.json, worker.log, host_clock_markers.csv; combined: comparison.csv, speed_ratios.csv, precision_agreement.csv, timings_all_batches.csv, documents.csv, environment.json; every file has a `.manifest.yml` sidecar; the embeddings `.npy` are git-ignored)"
    )
    add(
        "- The first pass of this comparison, timed on the VM's clock alone and superseded, is kept "
        "in `results/cpu_vs_gpu_three_documents_first_pass_vm_clock/` (with the Windows-clock check "
        "that found the problem in `vm_clock_check/`). Earlier measurements of the original "
        "benchmark stay in `results/`; none is reused here."
    )
    return "\n".join(L) + "\n"


def deviations(runs, partials=None):
    """The differences between the conditions, from their records."""
    out = []
    prec = {
        f"{k[0]} {k[1].upper()}": r.get("precision_used") for k, r in runs.items() if r
    }
    out.append(
        "Precision differs by device: "
        + ", ".join(f"{c} {_prec(p)}" for c, p in prec.items())
        + ". This is the only setting of the embedding task that differs; its effect on the "
        "embeddings is measured under Precision."
    )
    kernels = {
        f"{k[0]} {k[1].upper()}": (r.get("profile_first_warmup_batch") or {}).get(
            "attention_ops_and_kernels"
        )
        for k, r in runs.items()
        if r
    }
    out.append(
        "The code is the same, but PyTorch runs device-specific kernels for the same operations "
        "(attention recorded in the profiled warm-up batch: "
        + "; ".join(f"{c}: {', '.join(v or [])}" for c, v in kernels.items())
        + ")."
    )
    out.append(
        "The hardware differs by design: the CPU conditions use the 8 logical CPUs of the WSL2 VM "
        "(a Ryzen 9 3900X host); the GPU conditions use the RTX 3090 with the same CPU-side "
        "envelope (the same scope, memory limit and threads)."
    )
    for k, r in runs.items():
        if r and k[1] == "gpu":
            out.append(
                f"{k[0]} GPU: the Windows desktop held {_mib(r['gpu_memory']['device_used_before_start_bytes'])} MiB "
                "of the card when the worker started; it counts inside the 21 GiB ceiling."
            )
        if not r:
            p = (partials or {}).get(k)
            out.append(
                f"{k[0]} {k[1].upper()}: "
                + (partial_outcome(p) if p else "not run")
                + "; the comparisons that need it are not measured."
            )
        elif not r.get("complete"):
            out.append(f"{k[0]} {k[1].upper()}: did not complete ({r.get('outcome')}).")
    ratios = {
        f"{k[0]} {k[1].upper()}": r.get("clock_ratio_windows_over_vm")
        for k, r in runs.items()
        if r and r.get("complete")
    }
    fallback = [
        f"{k[0]} {k[1].upper()}"
        for k, r in runs.items()
        if r and r.get("complete") and r.get("summary_clock") != "windows_qpc"
    ]
    out.append(
        "Clock: the reported times are on the Windows host's clock, the same clock for all four "
        "conditions. The ratio of the Windows-clock to the VM-clock mean batch time was "
        + ", ".join(f"{c} {_f(x, 4)}" for c, x in ratios.items() if x is not None)
        + (
            f"; the Windows performance counter could not be read for {', '.join(fallback)}, "
            "whose times come from the next clock available (Windows file times, then the VM's clock)"
            if fallback
            else ""
        )
        + "."
    )
    out.append(
        "Timings come from a shared desktop machine (the Windows desktop and other programs ran "
        "at the same time)."
    )
    return out


def report_parameters(args):
    return {
        "conditions": [f"{m} {d.upper()}" for m, d in CONDITIONS],
        "ratio_basis": "mean timed batch time (equal to the ratio of documents or tokens per second, since every batch holds the same documents); the median basis alongside",
        "precision_agreement": "cosine and largest absolute difference between the GPU and the CPU embedding of the same document (first timed batch of each device)",
        "timing_clock": "Windows performance counter (QPC helper); Windows marker-file times and the VM's clock alongside",
        "note": args.note,
        "quick_run": bool(args.quick),
    }


# ----------------------------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------------------------

STAGES = ["run", "report", "worker"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--stage", choices=STAGES, required=True)
    p.add_argument("--model", choices=list(tb.MODELS))
    p.add_argument("--device", choices=list(DEVICES))
    p.add_argument("--n-warmup", type=int, default=3, help="untimed warm-up batches")
    p.add_argument("--n-timed", type=int, default=5, help="timed batches")
    p.add_argument(
        "--threads", type=int, default=8, help="PyTorch/OMP/MKL threads (both devices)"
    )
    p.add_argument(
        "--ceiling-mib",
        type=int,
        default=21504,
        help="GPU ceiling: total memory of the card, other processes included",
    )
    p.add_argument(
        "--ram-ceiling-mib",
        type=int,
        default=21504,
        help="MemoryMax of the worker's scope (both devices)",
    )
    p.add_argument(
        "--nvml-interval-ms", type=int, default=20, help="GPU sampling on GPU runs"
    )
    p.add_argument("--sample-ms", type=int, default=100, help="memory sampling")
    p.add_argument("--worker-timeout-s", type=int, default=4 * 3600)
    p.add_argument(
        "--quick", action="store_true", help="marks the outputs as a smoke test"
    )
    p.add_argument(
        "--note",
        default="",
        help="a line added to the report's deviations (for example an operator decision)",
    )
    p.add_argument("--project-dir", default=str(tb.DEFAULT_PROJECT_DIR))
    p.add_argument(
        "--results-dir",
        default=str(tb.BENCH_DIR / "results" / "cpu_vs_gpu_three_documents"),
    )
    p.add_argument("--report-path", default=str(tb.BENCH_DIR / "CPU_VS_GPU_REPORT.md"))
    p.add_argument(
        "--work-dir",
        default="~/cpu_vs_gpu_work",
        help="where the worker's plan (with the texts) is written; outside the repository",
    )
    p.add_argument("--hub-dir", default="~/.cache/huggingface/hub")
    p.add_argument("--plan", help=argparse.SUPPRESS)
    p.add_argument("--worker-out", help=argparse.SUPPRESS)
    p.add_argument("--worker-embeddings", help=argparse.SUPPRESS)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args.project_dir = str(Path(args.project_dir).resolve())
    args.results_dir = str(Path(args.results_dir).expanduser().resolve())
    args.report_path = str(Path(args.report_path).expanduser().resolve())
    args.hub_dir = str(Path(args.hub_dir).expanduser())
    if args.stage == "worker":
        stage_worker(args)
    elif args.stage == "run":
        stage_run(args)
    else:
        stage_report(args)


if __name__ == "__main__":
    main()
