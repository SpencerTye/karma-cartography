#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# RAPIDS cuML environment for the FULL-CORPUS GPU UMAP + HDBSCAN baseline
# (Phase 3). Reuses the SAME isolated uv venv built for the three-day
# feasibility run (~/venvs/rapids-umap-hdbscan); does NOT touch the existing
# Phase 2 (socialmediadgg-phase2) or FAISS (faiss-bench) venvs.
#
# Idempotent: if the venv already exists it is reused as-is (the full-corpus
# run must use the identical software stack as the validated three-day run).
# This script simply (re-)captures the exact resolved environment into this
# deliverable's environment/ directory.
#
# Overridable via env vars:
#   RAPIDS_VENV   target venv path        (default ~/venvs/rapids-umap-hdbscan)
#   ENV_OUT_DIR   environment capture dir (default this deliverable's environment/)
# ---------------------------------------------------------------------------
set -uo pipefail

VENV="${RAPIDS_VENV:-$HOME/venvs/rapids-umap-hdbscan}"
ENV_OUT_DIR="${ENV_OUT_DIR:-$REPO/pipeline/phase3_blind_semantic_structure_discovery/full_corpus_gpu_umap_hdbscan_baseline/environment}"
UV="$HOME/.local/bin/uv"

mkdir -p "$ENV_OUT_DIR"
LOG="$ENV_OUT_DIR/env_setup.log"
: > "$LOG"
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG"; }

log "uv: $("$UV" --version 2>&1)"
log "target venv: $VENV"

if [ ! -x "$VENV/bin/python" ]; then
  log "venv missing; creating (Python 3.12) + installing the RAPIDS cu12 stack ..."
  "$UV" venv "$VENV" --python 3.12 2>&1 | tee -a "$LOG"
  PYBIN="$VENV/bin/python"
  set -x
  UV_INDEX_STRATEGY=unsafe-best-match "$UV" pip install --python "$PYBIN" \
    --extra-index-url=https://pypi.nvidia.com \
    "cuml-cu12" matplotlib "nvidia-ml-py" psutil pyyaml pyarrow numpy pandas 2>&1 | tee -a "$LOG"
  CORE_RC=${PIPESTATUS[0]}
  set +x
  log "core install exit code: $CORE_RC"
  [ "$CORE_RC" -ne 0 ] && { log "CORE INSTALL FAILED — stopping."; exit "$CORE_RC"; }
  UV_INDEX_STRATEGY=unsafe-best-match "$UV" pip install --python "$PYBIN" hdbscan 2>&1 | tee -a "$LOG" \
    || log "WARNING: hdbscan CPU package failed; tree .to_pandas() export may be limited"
else
  log "venv already exists; reusing the validated three-day stack (no changes)"
fi

PYBIN="$VENV/bin/python"

# ---- capture exact versions & GPU/CUDA environment ----
log "freezing package versions ..."
"$UV" pip freeze --python "$PYBIN" > "$ENV_OUT_DIR/packages.txt" 2>>"$LOG"

log "nvidia-smi snapshot ..."
nvidia-smi > "$ENV_OUT_DIR/nvidia_smi.txt" 2>&1 || echo "nvidia-smi failed" > "$ENV_OUT_DIR/nvidia_smi.txt"

log "probing cuml / cupy / cuda versions ..."
"$PYBIN" - <<'PY' > "$ENV_OUT_DIR/cuml_env.json" 2>>"$LOG"
import json, platform
info = {"python_version": platform.python_version(), "platform": platform.platform()}
try:
    import cuml; info["cuml"] = cuml.__version__
except Exception as e:
    info["cuml_error"] = repr(e)
try:
    import cupy; info["cupy"] = cupy.__version__
    info["cupy_cuda_runtime_version"] = cupy.cuda.runtime.runtimeGetVersion()
    info["cupy_cuda_driver_version"] = cupy.cuda.runtime.driverGetVersion()
    props = cupy.cuda.runtime.getDeviceProperties(0)
    info["gpu_name"] = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
    info["gpu_compute_capability"] = f'{props["major"]}.{props["minor"]}'
except Exception as e:
    info["cupy_error"] = repr(e)
for mod in ("numpy", "pandas", "pyarrow", "matplotlib", "hdbscan", "sklearn", "scipy", "pynvml"):
    try:
        m = __import__(mod); info[mod] = getattr(m, "__version__", "unknown")
    except Exception as e:
        info[mod + "_error"] = repr(e)
print(json.dumps(info, indent=2))
PY

log "=== cuml_env.json ==="
cat "$ENV_OUT_DIR/cuml_env.json" | tee -a "$LOG"
log "DONE. venv=$VENV"
