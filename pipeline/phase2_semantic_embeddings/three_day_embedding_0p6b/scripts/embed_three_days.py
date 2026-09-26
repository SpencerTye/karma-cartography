#!/usr/bin/env python
"""Three-day real embedding run: Qwen3-Embedding-0.6B on the RTX 3090 via TEI (float16).

Embeds *every* document of three complete UTC days (2026-07-15, -16, -17) of the canonical
Phase 1 table with the established Phase 2 TEI setup, keeps the embeddings linked to their
original doc_key, and measures the run: an early time-to-completion estimate once enough real
batches have run, then the actual totals at completion.

This is NOT the benchmark. It reuses the benchmark's TEI client, NVML sampler, provenance manifest
writer and token packer (imported from ../../benchmark_qwen3_4b_0p6b_throughput/scripts/
throughput_benchmark.py) but launches TEI WITHOUT the vram_cap shim and imposes no allocator
cap: max_batch_tokens is fixed at 327,680 in advance. A short real-batch memory
check runs first; the budget is reduced only if the measured whole-card GPU memory actually
exceeds 19 GiB.

Stages (run inside the WSL2 venv):
  prepare   verify inputs, build the ordered three-day subset, record the environment
  embed     launch TEI (no shim), untimed warm-up + memory check, embed every document,
            emit the first ETA, run to completion, save embeddings + measurements, verify
  report    render REPORT.md from the recorded evidence
  all       prepare, embed, report
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import time
from pathlib import Path

import numpy as np

# --- locate the project and import the established benchmark helpers ----------------------------
SCRIPT_PATH = Path(__file__).resolve()
RUN_DIR = SCRIPT_PATH.parents[1]  # three_day_embedding_0p6b/
PHASE2_DIR = SCRIPT_PATH.parents[2]  # phase2_semantic_embeddings/
PROJECT_DIR = SCRIPT_PATH.parents[4]  # repository root 
BENCH_SCRIPTS = PHASE2_DIR / "benchmark_qwen3_4b_0p6b_throughput" / "scripts"
sys.path.insert(0, str(BENCH_SCRIPTS))
import throughput_benchmark as tb  # noqa: E402

# --- constants ----------------------------------------------------------------------------------
MODEL_KEY = "0.6B"
MAX_BATCH_TOKENS = 327680  # fixed in advance; not tuned here
FALLBACK_BATCH_TOKENS = (
    262144  # single documented fallback, used only if the check exceeds
)
MEMORY_CEILING_BYTES = (
    19 * 1024**3
)  # 19 GiB: reduce the budget only if the measured peak exceeds
CLIENT_REQUEST_TOKENS = 327680  # each client request is up to one full GPU batch
N_WARMUP_BATCHES = 3  # untimed real-data warm-up batches (embeddings still kept)
DAYS = ("2026-07-15", "2026-07-16", "2026-07-17")
PORT = 8080
CONTAINER = "tei_three_day_0p6b"
HUB_DIR = os.path.expanduser("~/.cache/huggingface/hub")
EMBED_DIM_EXPECTED = 1024
READY_TIMEOUT_S = 300
REQUEST_TIMEOUT_S = 1200

DEFAULT_RESULTS_DIR = RUN_DIR / "results"


# ================================================================================================
# Pure functions (unit-tested by ../tests/test_embed_three_days_functions.py)
# ================================================================================================
def epoch_bounds(days):
    """Inclusive-lower / exclusive-upper epoch-second bounds spanning the given UTC calendar days
    (which must be contiguous). Returns (lo, hi)."""
    ds = sorted(dt.date.fromisoformat(d) for d in days)
    lo = int(
        dt.datetime(
            ds[0].year, ds[0].month, ds[0].day, tzinfo=dt.timezone.utc
        ).timestamp()
    )
    last = ds[-1] + dt.timedelta(days=1)
    hi = int(
        dt.datetime(last.year, last.month, last.day, tzinfo=dt.timezone.utc).timestamp()
    )
    return lo, hi


def eta_from_timed(
    timed_docs, timed_tokens, elapsed_s, docs_remaining, tokens_remaining
):
    """Rates and time-to-completion from the timed work so far and the work still remaining."""
    dps = timed_docs / elapsed_s if elapsed_s and elapsed_s > 0 else None
    tps = timed_tokens / elapsed_s if elapsed_s and elapsed_s > 0 else None
    return {
        "documents_per_s": dps,
        "tokens_per_s": tps,
        "eta_s_by_tokens": (tokens_remaining / tps) if tps else None,
        "eta_s_by_docs": (docs_remaining / dps) if dps else None,
    }


def fmt_hms(seconds):
    """Whole-second H:MM:SS (or None)."""
    if seconds is None:
        return None
    s = int(round(seconds))
    return f"{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def plan_requests(n_tokens, budget):
    """Pack the ordered stream of per-document token counts into consecutive client requests,
    each with a real-token sum within `budget` (TEI's own non-padded rule). Reuses the
    benchmark's packer; covers the whole stream."""
    return tb.pack_greedy(list(n_tokens), 0, int(budget), len(n_tokens))


# ================================================================================================
# Data
# ================================================================================================
def canonical_parquet():
    return tb.canonical_paths(PROJECT_DIR)[0]


def token_parquet():
    return tb.token_table_paths()[0]


def verify_input(path):
    """Hash a file and compare to the output_hash recorded in its provenance sidecar."""
    sidecar = path.with_name(path.name + ".manifest.yml")
    measured = tb.sha256_file(path)
    expected = tb.read_sidecar_hash(sidecar) if sidecar.exists() else ""
    return {
        "path": tb.rel_path(path, PROJECT_DIR),
        "size_bytes": path.stat().st_size,
        "sha256_measured": measured,
        "sha256_in_sidecar": expected,
        "hash_equal_to_sidecar": bool(expected) and measured == expected,
    }


def build_subset():
    """Read the canonical table, restrict to the three UTC days, order by (created_utc, doc_key),
    attach each document's Qwen token count. Returns a dict of parallel lists plus summary facts.
    Reads only; writes nothing to Phase 1."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    day_arr = pa.array([dt.date.fromisoformat(d) for d in DAYS], type=pa.date32())
    t = pq.read_table(
        canonical_parquet(),
        columns=[
            "doc_key",
            "record_type",
            "created_utc",
            "created_date_utc",
            "text_n_chars",
        ],
    )
    mask = pc.is_in(t.column("created_date_utc"), value_set=day_arr)
    sub = t.filter(mask).sort_by(
        [("created_utc", "ascending"), ("doc_key", "ascending")]
    )

    doc_key = sub.column("doc_key").to_pylist()
    record_type = sub.column("record_type").to_pylist()
    created_utc = [int(x) for x in sub.column("created_utc").to_pylist()]
    created_date = [str(d) for d in sub.column("created_date_utc").to_pylist()]
    text_n_chars = [
        int(x) if x is not None else 0 for x in sub.column("text_n_chars").to_pylist()
    ]

    tok = pq.read_table(token_parquet(), columns=["doc_key", "n_tokens_4B"])
    tokmap = dict(
        zip(
            tok.column("doc_key").to_pylist(),
            (int(x) for x in tok.column("n_tokens_4B").to_pylist()),
        )
    )
    missing = [k for k in doc_key if k not in tokmap]
    if missing:
        raise SystemExit(
            f"{len(missing)} subset doc_keys missing from the token table, e.g. {missing[:5]}"
        )
    n_tokens = [tokmap[k] for k in doc_key]

    lo, hi = epoch_bounds(DAYS)
    n_epoch = int(sum(1 for c in created_utc if lo <= c < hi))
    per_day, per_rt = {}, {}
    for d, rt in zip(created_date, record_type):
        per_day[d] = per_day.get(d, 0) + 1
        per_rt[rt] = per_rt.get(rt, 0) + 1

    summary = {
        "days": list(DAYS),
        "epoch_lo_inclusive": lo,
        "epoch_hi_exclusive": hi,
        "n_documents": len(doc_key),
        "n_documents_by_epoch_filter": n_epoch,
        "date_and_epoch_filters_agree": len(doc_key) == n_epoch,
        "per_day": per_day,
        "per_record_type": per_rt,
        "total_tokens_n_tokens_4B": int(sum(n_tokens)),
        "min_tokens_per_doc": int(min(n_tokens)),
        "max_tokens_per_doc": int(max(n_tokens)),
        "min_created_utc": min(created_utc),
        "max_created_utc": max(created_utc),
        "min_created_utc_iso": dt.datetime.fromtimestamp(
            min(created_utc), dt.timezone.utc
        ).isoformat(),
        "max_created_utc_iso": dt.datetime.fromtimestamp(
            max(created_utc), dt.timezone.utc
        ).isoformat(),
        "n_empty_text_docs": int(sum(1 for c in text_n_chars if c == 0)),
        "text_rule": tb.CANONICAL_TEXT_RULE,
        "order": "created_utc ascending, then doc_key ascending",
    }
    return {
        "doc_key": doc_key,
        "record_type": record_type,
        "created_utc": created_utc,
        "created_date_utc": created_date,
        "n_tokens": n_tokens,
        "text_n_chars": text_n_chars,
        "summary": summary,
    }


def read_texts_in_order(doc_keys):
    """Canonical `text` for the given doc_keys, in that exact order (read-only)."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    t = pq.read_table(canonical_parquet(), columns=["doc_key", "text"])
    keyset = pa.array(doc_keys)
    t = t.filter(pc.is_in(t.column("doc_key"), value_set=keyset))
    textmap = dict(zip(t.column("doc_key").to_pylist(), t.column("text").to_pylist()))
    return [textmap[k] for k in doc_keys]


# ================================================================================================
# TEI container (no shim, no cap)
# ================================================================================================
def launch_tei(port, max_batch_tokens):
    snapshot = tb.snapshot_dir_in_container(MODEL_KEY)
    cmd = [
        "docker",
        "run",
        "-d",
        "--name",
        CONTAINER,
        "--gpus",
        "all",
        "--network",
        "host",
        "-v",
        f"{HUB_DIR}:/data:ro",
        "-e",
        "HF_HUB_OFFLINE=1",
        tb.IMAGE,
        "--model-id",
        snapshot,
        "--port",
        str(port),
        "--max-batch-tokens",
        str(int(max_batch_tokens)),
        *tb.TEI_ARGS_FIXED,
    ]
    out = tb.run(cmd, timeout=120)
    if out.returncode != 0:
        raise RuntimeError(f"docker run failed: {out.stderr.strip()}")
    return cmd


def teardown():
    try:
        tb.docker_remove(CONTAINER)
    except Exception:
        pass


# ================================================================================================
# Stages
# ================================================================================================
def stage_prepare(args):
    results = Path(args.results_dir)
    results.mkdir(parents=True, exist_ok=True)
    tb.log("verifying inputs against their sidecars")
    checks = {
        "canonical": verify_input(canonical_parquet()),
        "token_table": verify_input(token_parquet()),
    }
    for label, c in checks.items():
        if not c["hash_equal_to_sidecar"]:
            tb.log(
                f"  WARNING: {label} hash does not match its sidecar ({c['sha256_measured']} vs {c['sha256_in_sidecar']})"
            )

    tb.log("building the ordered three-day subset (read-only)")
    data = build_subset()
    s = data["summary"]
    tb.log(
        f"  {s['n_documents']} documents, {s['total_tokens_n_tokens_4B']} tokens; "
        f"per day {s['per_day']}; types {s['per_record_type']}; empties {s['n_empty_text_docs']}"
    )

    # subset.parquet (row index -> doc_key and metadata; the embedding row order)
    import pyarrow as pa
    import pyarrow.parquet as pq

    n = s["n_documents"]
    table = pa.table(
        {
            "row": pa.array(list(range(n)), type=pa.int64()),
            "doc_key": pa.array(data["doc_key"]),
            "record_type": pa.array(data["record_type"]),
            "created_utc": pa.array(data["created_utc"], type=pa.int64()),
            "created_date_utc": pa.array(data["created_date_utc"]),
            "n_tokens_4B": pa.array(data["n_tokens"], type=pa.int64()),
            "text_n_chars": pa.array(data["text_n_chars"], type=pa.int64()),
        }
    )
    subset_path = results / "subset.parquet"
    pq.write_table(table, subset_path)
    tb.write_json(results / "subset_summary.json", {"input_checks": checks, **s})

    # environment
    env = {
        "timestamp": tb.utc_now(),
        "hub_dir": HUB_DIR,
        "image": tb.IMAGE,
        "max_batch_tokens_planned": MAX_BATCH_TOKENS,
        "model": MODEL_KEY,
    }
    try:
        env["tei_image"] = tb.docker_image_record()
    except Exception as e:
        env["tei_image_error"] = repr(e)
    try:
        env["nvidia_smi"] = tb.nvidia_smi_text()
    except Exception as e:
        env["nvidia_smi_error"] = repr(e)
    env["packages"] = tb.package_versions()
    tb.write_json(results / "environment.json", env)

    sc = tb.Sidecars(
        PROJECT_DIR,
        {
            "days": list(DAYS),
            "text_field": tb.CANONICAL_TEXT_FIELD,
            "order": s["order"],
        },
        "Three-day embedding run: ordered subset and environment.",
        script_path=SCRIPT_PATH,
    )
    inputs = [
        sc.input_entry(
            canonical_parquet(), "canonical Phase 1 table, read-only", with_shape=True
        ),
        sc.input_entry(
            token_parquet(),
            "per-document Qwen token counts, read-only",
            with_shape=True,
        ),
    ]
    for p in (
        subset_path,
        results / "subset_summary.json",
        results / "environment.json",
    ):
        sc.write(p, inputs, notes=f"prepare: {p.name}")
    tb.log(f"prepare done -> {tb.rel_path(subset_path, PROJECT_DIR)}")
    return s


def _load_subset(results):
    import pyarrow.parquet as pq

    t = pq.read_table(results / "subset.parquet")
    return {
        "doc_key": t.column("doc_key").to_pylist(),
        "record_type": t.column("record_type").to_pylist(),
        "created_utc": [int(x) for x in t.column("created_utc").to_pylist()],
        "created_date_utc": t.column("created_date_utc").to_pylist(),
        "n_tokens": [int(x) for x in t.column("n_tokens_4B").to_pylist()],
    }


def stage_embed(args):
    results = Path(args.results_dir)
    if not (results / "subset.parquet").exists():
        raise SystemExit("run prepare first: subset.parquet is missing")
    sub = _load_subset(results)
    doc_key, n_tokens = sub["doc_key"], sub["n_tokens"]
    n_docs = len(doc_key)
    total_tokens = int(sum(n_tokens))
    tb.log(
        f"embedding {n_docs} documents ({total_tokens} tokens) at max_batch_tokens={args.max_batch_tokens}"
    )

    tb.log("reading canonical texts in row order (read-only)")
    texts = read_texts_in_order(doc_key)
    if len(texts) != n_docs:
        raise SystemExit("text/row count mismatch")

    requests = plan_requests(n_tokens, args.client_request_tokens)
    tb.log(
        f"planned {len(requests)} client requests (<= {args.client_request_tokens} tokens each); "
        f"{args.n_warmup} untimed warm-up batches"
    )

    ctx = {"port": PORT}
    nvml = tb.Nvml()
    device_name, driver = nvml.name(), nvml.driver()
    baseline_before = nvml.used()
    tb.log(
        f"GPU {device_name} (driver {driver}); baseline before start {tb.mib(baseline_before)} MiB"
    )

    budget = args.max_batch_tokens
    embeddings = None
    emb_dim = None
    batch_rows = []
    warmup_docs = warmup_tokens = 0
    reduced = False

    teardown()
    sampler = tb.PeakSampler(nvml.used, interval_s=0.05).start()
    run_started = time.time()
    try:
        docker_cmd = launch_tei(PORT, budget)
        tb.log(f"TEI starting (max_batch_tokens={budget}); waiting for /health")
        status, wait_s, state = tb.wait_ready(ctx, CONTAINER, READY_TIMEOUT_S)
        if status != "ready":
            logs = tb.docker_logs(CONTAINER)
            raise RuntimeError(
                f"TEI did not become ready ({status} after {wait_s:.1f}s):\n{logs[-2000:]}"
            )
        info = tb.get_info(ctx)
        log_events = tb.parse_tei_log_events(tb.docker_logs(CONTAINER).splitlines())
        peak_at_ready = sampler.peak or nvml.used()
        tb.log(
            f"TEI ready in {wait_s:.1f}s (load {log_events.get('load_s')}s, warmup {log_events.get('warmup_s')}s); "
            f"dtype={info.get('model_dtype')} pooling={info.get('model_type')} "
            f"max_batch_tokens={info.get('max_batch_tokens')}; device now {tb.mib(nvml.used())} MiB, "
            f"peak so far {tb.mib(peak_at_ready)} MiB"
        )

        # info sanity
        info_ok = {
            "dtype_float16": info.get("model_dtype") == "float16",
            "auto_truncate_false": info.get("auto_truncate") is False,
            "max_batch_tokens_equals_requested": int(info.get("max_batch_tokens", -1))
            == int(budget),
            "pooling_last_token": (info.get("model_type") or {})
            .get("embedding", {})
            .get("pooling")
            == "last_token",
        }
        for k, v in info_ok.items():
            if not v:
                tb.log(f"  WARNING: /info check failed: {k}")

        def run_batch(req, phase):
            nonlocal embeddings, emb_dim
            begin, end = req["begin"], req["end"]
            res = tb.post_embed(
                ctx, texts[begin:end], REQUEST_TIMEOUT_S, keep_body=True
            )
            if res["status"] != 200 or res["body"] is None:
                raise RuntimeError(
                    f"/embed failed for rows [{begin}:{end}] status={res['status']} "
                    f"error={res.get('error')} exc={res.get('exception')}"
                )
            emb = tb.embeddings_from_body(res["body"])
            res["body"] = None
            if emb.shape[0] != (end - begin):
                raise RuntimeError(
                    f"embedding count {emb.shape[0]} != documents {end - begin}"
                )
            if embeddings is None:
                emb_dim = int(emb.shape[1])
                embeddings = np.empty((n_docs, emb_dim), dtype=np.float32)
            embeddings[begin:end] = emb
            x_inf_ms = tb.header_int(res["headers"], "x-inference-time")
            x_tokens = tb.header_int(res["headers"], "x-compute-tokens")
            rec = {
                "seq": len(batch_rows) + 1,
                "phase": phase,
                "begin": begin,
                "end": end,
                "n_documents": end - begin,
                "n_tokens_planned": req["n_tokens"],
                "x_compute_tokens": x_tokens,
                "x_inference_ms": x_inf_ms,
                "wall_s": res["wall_s"],
                "device_used_mib": tb.mib(nvml.used()),
                "peak_mib": tb.mib(sampler.peak or 0),
            }
            batch_rows.append(rec)
            return rec

        # ---- untimed warm-up batches (embeddings kept; times excluded from throughput) ----
        n_warm = min(args.n_warmup, len(requests))
        for i in range(n_warm):
            rec = run_batch(requests[i], "warmup")
            warmup_docs += rec["n_documents"]
            warmup_tokens += rec["n_tokens_planned"]
            tb.log(
                f"  warm-up {i + 1}/{n_warm}: {rec['n_documents']} docs, {rec['n_tokens_planned']} tokens, "
                f"{rec['wall_s']:.2f}s wall, TEI {rec['x_inference_ms']}ms; peak {rec['peak_mib']} MiB"
            )

        # ---- memory check ----
        peak_after_warmup = sampler.peak or nvml.used()
        tb.log(
            f"MEMORY CHECK: whole-card peak through warm-up = {tb.mib(peak_after_warmup)} MiB "
            f"({tb.gib(peak_after_warmup)} GiB); ceiling 19 GiB = {tb.mib(MEMORY_CEILING_BYTES)} MiB"
        )
        if peak_after_warmup > MEMORY_CEILING_BYTES and not reduced:
            tb.log(
                f"  measured peak EXCEEDS 19 GiB; reducing to fallback {FALLBACK_BATCH_TOKENS} and restarting"
            )
            teardown()
            reduced = True
            budget = FALLBACK_BATCH_TOKENS
            embeddings = None
            batch_rows.clear()
            warmup_docs = warmup_tokens = 0
            requests = plan_requests(n_tokens, min(args.client_request_tokens, budget))
            launch_tei(PORT, budget)
            status, wait_s, state = tb.wait_ready(ctx, CONTAINER, READY_TIMEOUT_S)
            if status != "ready":
                raise RuntimeError(f"TEI (fallback) did not become ready: {status}")
            info = tb.get_info(ctx)
            n_warm = min(args.n_warmup, len(requests))
            for i in range(n_warm):
                rec = run_batch(requests[i], "warmup")
                warmup_docs += rec["n_documents"]
                warmup_tokens += rec["n_tokens_planned"]
            peak_after_warmup = sampler.peak or nvml.used()
            tb.log(
                f"  fallback peak = {tb.mib(peak_after_warmup)} MiB ({tb.gib(peak_after_warmup)} GiB)"
            )
        else:
            tb.log("  within 19 GiB -> proceeding at the fixed budget")

        # ---- timed run ----
        first_eta = None
        timed_docs = timed_tokens = 0
        t_timed_start = time.perf_counter()
        for i in range(n_warm, len(requests)):
            rec = run_batch(requests[i], "timed")
            timed_docs += rec["n_documents"]
            timed_tokens += rec["n_tokens_planned"]
            elapsed = time.perf_counter() - t_timed_start
            timed_batches = i - n_warm + 1
            if first_eta is None and timed_batches >= 3:
                embedded_docs = warmup_docs + timed_docs
                embedded_tokens = warmup_tokens + timed_tokens
                docs_remaining = n_docs - embedded_docs
                tokens_remaining = total_tokens - embedded_tokens
                rates = eta_from_timed(
                    timed_docs, timed_tokens, elapsed, docs_remaining, tokens_remaining
                )
                first_eta = {
                    "at_timed_batch": timed_batches,
                    "elapsed_timed_s": elapsed,
                    "embedded_documents": embedded_docs,
                    "embedded_tokens": embedded_tokens,
                    "documents_remaining": docs_remaining,
                    "tokens_remaining": tokens_remaining,
                    "pct_documents_done": 100.0 * embedded_docs / n_docs,
                    "pct_tokens_done": 100.0 * embedded_tokens / total_tokens,
                    **rates,
                    "eta_hms_by_tokens": fmt_hms(rates["eta_s_by_tokens"]),
                }
                tb.log(
                    f"FIRST ETA after {timed_batches} timed batches "
                    f"({first_eta['pct_tokens_done']:.1f}% of tokens done): "
                    f"{rates['tokens_per_s']:.0f} tokens/s, {rates['documents_per_s']:.1f} docs/s; "
                    f"remaining ~{fmt_hms(rates['eta_s_by_tokens'])} "
                    f"({tokens_remaining} tokens, {docs_remaining} docs)"
                )
            else:
                tb.log(
                    f"  timed {timed_batches}: {rec['n_documents']} docs, {rec['n_tokens_planned']} tokens, "
                    f"{rec['wall_s']:.2f}s wall, TEI {rec['x_inference_ms']}ms"
                )
        timed_wall_s = time.perf_counter() - t_timed_start
        overall_peak = sampler.peak or nvml.used()
    finally:
        sampler.stop()
        run_wall_s = time.time() - run_started
        # keep the container's final logs before removing it
        try:
            final_logs = tb.docker_logs(CONTAINER)
        except Exception:
            final_logs = ""
        teardown()
        nvml.shutdown()

    if embeddings is None:
        raise RuntimeError("no embeddings produced")

    # ---- save embeddings + index ----
    (results).mkdir(parents=True, exist_ok=True)
    emb_path = results / "embeddings.npy"
    np.save(emb_path, embeddings)
    import pyarrow as pa
    import pyarrow.parquet as pq

    idx = pa.table(
        {
            "row": pa.array(list(range(n_docs)), type=pa.int64()),
            "doc_key": pa.array(doc_key),
            "record_type": pa.array(sub["record_type"]),
            "created_utc": pa.array(sub["created_utc"], type=pa.int64()),
            "created_date_utc": pa.array(sub["created_date_utc"]),
            "n_tokens_4B": pa.array(n_tokens, type=pa.int64()),
        }
    )
    idx_path = results / "embedding_index.parquet"
    pq.write_table(idx, idx_path)

    # ---- measurements ----
    timed_rows = [r for r in batch_rows if r["phase"] == "timed"]
    final_timed_docs = sum(r["n_documents"] for r in timed_rows)
    final_timed_tokens = sum(r["n_tokens_planned"] for r in timed_rows)
    tei_inf_s = sum((r["x_inference_ms"] or 0) for r in batch_rows) / 1000.0
    tei_compute_tokens = sum((r["x_compute_tokens"] or 0) for r in batch_rows)
    sustained = {
        "documents_per_s": final_timed_docs / timed_wall_s
        if timed_wall_s > 0
        else None,
        "tokens_per_s": final_timed_tokens / timed_wall_s if timed_wall_s > 0 else None,
    }

    # ---- verification ----
    finite = bool(np.isfinite(embeddings).all())
    embedded_keys = set(doc_key)
    dates_ok = set(sub["created_date_utc"]) <= set(DAYS)
    verification = {
        "n_expected": n_docs,
        "n_embedded_rows": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "all_rows_finite": finite,
        "unique_doc_keys": len(embedded_keys) == n_docs,
        "every_doc_key_has_embedding": embeddings.shape[0] == n_docs
        and len(embedded_keys) == n_docs,
        "only_the_three_dates": dates_ok,
        "distinct_dates_embedded": sorted(set(sub["created_date_utc"])),
    }
    tb.log(
        f"VERIFY: rows={verification['n_embedded_rows']} dim={verification['embedding_dim']} "
        f"finite={finite} unique_keys={verification['unique_doc_keys']} dates_ok={dates_ok}"
    )

    run_rec = {
        "stage": "embed",
        "timestamp": tb.utc_now(),
        "model": MODEL_KEY,
        "image": tb.IMAGE,
        "precision": info.get("model_dtype"),
        "date_range": {
            "days": list(DAYS),
            "epoch_lo_inclusive": epoch_bounds(DAYS)[0],
            "epoch_hi_exclusive": epoch_bounds(DAYS)[1],
            "min_created_utc_iso": dt.datetime.fromtimestamp(
                min(sub["created_utc"]), dt.timezone.utc
            ).isoformat(),
            "max_created_utc_iso": dt.datetime.fromtimestamp(
                max(sub["created_utc"]), dt.timezone.utc
            ).isoformat(),
        },
        "n_documents": n_docs,
        "total_tokens_n_tokens_4B": total_tokens,
        "max_batch_tokens_requested": MAX_BATCH_TOKENS,
        "max_batch_tokens_used": budget,
        "reduced_from_fixed_budget": reduced,
        "client_request_tokens": args.client_request_tokens,
        "n_warmup_batches": args.n_warmup,
        "n_client_requests": len(requests),
        "info_checks": info_ok,
        "docker_run_command": docker_cmd
        if not reduced
        else "see logs (relaunched at fallback budget)",
        "memory": {
            "device_name": device_name,
            "driver": driver,
            "baseline_before_start_bytes": baseline_before,
            "baseline_before_start_mib": tb.mib(baseline_before),
            "peak_at_ready_bytes": peak_at_ready,
            "peak_at_ready_mib": tb.mib(peak_at_ready),
            "peak_after_warmup_bytes": peak_after_warmup,
            "peak_after_warmup_mib": tb.mib(peak_after_warmup),
            "peak_after_warmup_gib": tb.gib(peak_after_warmup),
            "overall_peak_bytes": overall_peak,
            "overall_peak_mib": tb.mib(overall_peak),
            "overall_peak_gib": tb.gib(overall_peak),
            "ceiling_19gib_bytes": MEMORY_CEILING_BYTES,
            "overall_peak_under_19gib": overall_peak <= MEMORY_CEILING_BYTES,
            "tei_own_peak_mib": tb.mib(max(0, overall_peak - baseline_before)),
        },
        "warmup": {"n_documents": warmup_docs, "n_tokens": warmup_tokens},
        "timed": {
            "n_batches": len(timed_rows),
            "n_documents": final_timed_docs,
            "n_tokens": final_timed_tokens,
            "wall_s": timed_wall_s,
            "wall_hms": fmt_hms(timed_wall_s),
            "documents_per_s": sustained["documents_per_s"],
            "tokens_per_s": sustained["tokens_per_s"],
            "tei_backend_inference_s_all_batches": tei_inf_s,
            "tei_backend_tokens_per_s": (total_tokens / tei_inf_s)
            if tei_inf_s > 0
            else None,
        },
        "tei_compute_tokens_total": tei_compute_tokens,
        "first_eta": first_eta,
        "run_wall_s_ready_to_done": run_wall_s,
        "verification": verification,
        "embeddings_file": "embeddings.npy",
        "embedding_index_file": "embedding_index.parquet",
        "tei_log_events": log_events,
    }
    tb.write_json(results / "run.json", run_rec)
    tb.write_csv(
        results / "batches.csv",
        batch_rows,
        [
            "seq",
            "phase",
            "begin",
            "end",
            "n_documents",
            "n_tokens_planned",
            "x_compute_tokens",
            "x_inference_ms",
            "wall_s",
            "device_used_mib",
            "peak_mib",
        ],
    )
    tb.write_json(results / "verification.json", verification)
    (results / "container_final.log").write_text(final_logs, encoding="utf-8")

    # ---- sidecars ----
    sc = tb.Sidecars(
        PROJECT_DIR,
        {
            "days": list(DAYS),
            "max_batch_tokens": budget,
            "client_request_tokens": args.client_request_tokens,
            "precision": info.get("model_dtype"),
            "normalize": True,
            "truncate": False,
            "n_warmup_batches": args.n_warmup,
            "order": "created_utc asc, doc_key asc",
        },
        "Three-day embedding run: embeddings, index and measurements.",
        script_path=SCRIPT_PATH,
    )
    inputs = [
        sc.input_entry(
            results / "subset.parquet",
            "ordered three-day subset (prepare stage)",
            with_shape=True,
        ),
        sc.input_entry(
            canonical_parquet(), "canonical Phase 1 table, read-only", with_shape=True
        ),
    ]
    for p in (
        emb_path,
        idx_path,
        results / "run.json",
        results / "batches.csv",
        results / "verification.json",
    ):
        sc.write(p, inputs, notes=f"embed: {p.name}")

    tb.log(
        f"DONE: {verification['n_embedded_rows']} embeddings (dim {verification['embedding_dim']}) "
        f"saved; timed {sustained['tokens_per_s']:.0f} tokens/s over {fmt_hms(timed_wall_s)}"
    )
    return run_rec


def stage_report(args):
    results = Path(args.results_dir)
    run = tb.read_json(results / "run.json")
    tmd, mem, eta, ver = (
        run["timed"],
        run["memory"],
        run["first_eta"],
        run["verification"],
    )
    dr = run["date_range"]

    def g(x):
        return "—" if x is None else x

    tei_tps = tmd["tei_backend_tokens_per_s"]
    tei_tps_str = f"{tei_tps:.0f}" if tei_tps is not None else "—"

    lines = []
    lines.append(
        "# Three-day embedding run — Qwen3-Embedding-0.6B on the RTX 3090 (TEI, float16)\n"
    )
    lines.append(
        f"Generated {run['timestamp']}. Model {run['model']} via `{run['image']}`; "
        f"precision **{run['precision']}**, last-token pooling, L2-normalized, no truncation. "
        "This is a real embedding run, not a benchmark; the canonical Phase 1 table was read only.\n"
    )

    lines.append("## Required evidence\n")
    rows = [
        (
            "Three-day date range (UTC)",
            f"{', '.join(dr['days'])}  "
            f"[{dr['min_created_utc_iso']} … {dr['max_created_utc_iso']}]; "
            f"epoch [{dr['epoch_lo_inclusive']}, {dr['epoch_hi_exclusive']})",
        ),
        ("Documents in the subset", f"{run['n_documents']:,}"),
        ("Total Qwen tokens (n_tokens_4B)", f"{run['total_tokens_n_tokens_4B']:,}"),
        ("Untimed warm-up batches", run["n_warmup_batches"]),
        (
            "Batch-token setting used (max_batch_tokens)",
            f"{run['max_batch_tokens_used']:,}"
            + (
                f"  (reduced from {run['max_batch_tokens_requested']:,})"
                if run["reduced_from_fixed_budget"]
                else "  (as fixed)"
            ),
        ),
        (
            "Observed total GPU memory (whole-card peak)",
            f"{mem['overall_peak_mib']:,} MiB = {mem['overall_peak_gib']} GiB "
            f"(baseline before start {mem['baseline_before_start_mib']:,} MiB; "
            f"TEI's own peak ~{mem['tei_own_peak_mib']:,} MiB; under 19 GiB: {mem['overall_peak_under_19gib']})",
        ),
        ("Elapsed timed processing time", f"{tmd['wall_hms']} ({tmd['wall_s']:.1f} s)"),
    ]
    if eta:
        rows += [
            (
                "Documents / tokens done at first ETA",
                f"{eta['embedded_documents']:,} docs / {eta['embedded_tokens']:,} tokens "
                f"({eta['pct_tokens_done']:.1f}% of tokens)",
            ),
            (
                "First estimated time to completion",
                f"~{eta['eta_hms_by_tokens']} remaining ({g(eta['tokens_remaining']):,} tokens, "
                f"{g(eta['documents_remaining']):,} docs left)",
            ),
            (
                "Measured docs/s and tokens/s at first ETA",
                f"{eta['documents_per_s']:.1f} docs/s, {eta['tokens_per_s']:.0f} tokens/s",
            ),
        ]
    rows += [
        (
            "Final documents embedded",
            f"{ver['n_embedded_rows']:,} (dim {ver['embedding_dim']})",
        ),
        (
            "Final tokens processed",
            f"{run['total_tokens_n_tokens_4B']:,} planned; TEI x-compute-tokens {run['tei_compute_tokens_total']:,}",
        ),
        (
            "Actual completed wall-clock processing time (timed region)",
            f"{tmd['wall_hms']} ({tmd['wall_s']:.1f} s)",
        ),
        (
            "Final sustained docs/s and tokens/s (end-to-end)",
            f"{tmd['documents_per_s']:.1f} docs/s, {tmd['tokens_per_s']:.0f} tokens/s",
        ),
        (
            "TEI backend-only rate (GPU ceiling, all batches)",
            f"{tei_tps_str} tokens/s over {tmd['tei_backend_inference_s_all_batches']:.1f} s of inference",
        ),
        (
            "Every expected doc_key has an embedding",
            f"{ver['every_doc_key_has_embedding']} ({ver['n_embedded_rows']:,}/{ver['n_expected']:,}, "
            f"unique keys: {ver['unique_doc_keys']}, all finite: {ver['all_rows_finite']})",
        ),
        (
            "No additional dates embedded",
            f"{ver['only_the_three_dates']} (distinct dates: {', '.join(ver['distinct_dates_embedded'])})",
        ),
    ]
    lines.append(
        tb.render_table(["Item", "Value"], [{"Item": a, "Value": b} for a, b in rows])
    )

    lines.append("\n## How the estimate compares with the actual\n")
    if eta and eta.get("eta_s_by_tokens") is not None:
        predicted_total = eta["elapsed_timed_s"] + eta["eta_s_by_tokens"]
        lines.append(
            f"- First ETA (made at {eta['pct_tokens_done']:.1f}% of tokens): predicted **{fmt_hms(predicted_total)}** "
            f"of total timed processing ({fmt_hms(eta['elapsed_timed_s'])} elapsed + "
            f"~{fmt_hms(eta['eta_s_by_tokens'])} remaining).\n"
        )
    lines.append(
        f"- Actual total timed processing: **{tmd['wall_hms']}** ({tmd['wall_s']:.1f} s) over "
        f"{tmd['n_batches']} timed batches.\n"
    )

    lines.append("\n## Method\n")
    lines.append(
        f"- TEI `{run['image']}`, `--model-id` the local 0.6B snapshot, `--max-batch-tokens "
        f"{run['max_batch_tokens_used']}`, `--auto-truncate false`, no `--max-batch-requests`; "
        "**no vram_cap shim and no allocator cap.**\n"
    )
    lines.append(
        f"- Documents ordered by (created_utc, doc_key); packed into {run['n_client_requests']} client "
        f"requests of up to {run['client_request_tokens']:,} tokens each; each request POSTs its texts to "
        "`/embed` with `truncate=false, normalize=true`; embeddings return as float32.\n"
    )
    lines.append(
        f"- {run['n_warmup_batches']} untimed warm-up batches (their embeddings are kept) precede the timed "
        "region; TEI also runs its own start-up warm-up at the full budget. Memory was sampled whole-card "
        "via NVML every 50 ms.\n"
    )
    lines.append(
        "\nOutputs in `results/`: `embeddings.npy` (float32, git-ignored), `embedding_index.parquet` "
        "(row → doc_key), `subset.parquet`, `run.json`, `batches.csv`, `verification.json`, "
        "`environment.json`, each with a `.manifest.yml` sidecar.\n"
    )

    report_path = RUN_DIR / "REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    tb.log(f"report -> {tb.rel_path(report_path, PROJECT_DIR)}")
    return report_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--stage", required=True, choices=["prepare", "embed", "report", "all"]
    )
    ap.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    ap.add_argument("--max-batch-tokens", type=int, default=MAX_BATCH_TOKENS)
    ap.add_argument("--client-request-tokens", type=int, default=CLIENT_REQUEST_TOKENS)
    ap.add_argument("--n-warmup", type=int, default=N_WARMUP_BATCHES)
    args = ap.parse_args()
    if args.stage in ("prepare", "all"):
        stage_prepare(args)
    if args.stage in ("embed", "all"):
        stage_embed(args)
    if args.stage in ("report", "all"):
        stage_report(args)


if __name__ == "__main__":
    main()
