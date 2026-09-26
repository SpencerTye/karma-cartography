#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Isolated RAPIDS cuML environment for the three-day GPU UMAP + HDBSCAN
# feasibility test (Phase 3). Creates a fresh uv venv; does NOT touch the
# existing Phase 2 (socialmediadgg-phase2) or FAISS (faiss-bench) venvs.
#
# Reproducible: pip wheels from PyPI + the NVIDIA Python Package Index.
# Records exact resolved versions to <environment>/packages.txt.
#
# Overridable via env vars:
#   RAPIDS_VENV   target venv path        (default ~/venvs/rapids-umap-hdbscan)
#   ENV_OUT_DIR   environment capture dir (default the deliverable environment/)
# ---------------------------------------------------------------------------
set -uo pipefail

VENV="${RAPIDS_VENV:-$HOME/venvs/rapids-umap-hdbscan}"
ENV_OUT_DIR="${ENV_OUT_DIR:-$REPO/pipeline/phase3_blind_semantic_structure_discovery/three_day_gpu_umap_hdbscan_feasibility/environment}"
UV="$HOME/.local/bin/uv"

mkdir -p "$ENV_OUT_DIR"
LOG="$ENV_OUT_DIR/env_setup.log"
: > "$LOG"
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG"; }

log "uv: $("$UV" --version 2>&1)"
log "target venv: $VENV"

if [ ! -x "$VENV/bin/python" ]; then
  log "creating venv (Python 3.12) ..."
  "$UV" venv "$VENV" --python 3.12 2>&1 | tee -a "$LOG"
else
  log "venv already exists; reusing"
fi

PYBIN="$VENV/bin/python"

# ---- core install: RAPIDS cuML (CUDA 12 wheels) + support libs ----
# unsafe-best-match lets uv resolve deps across PyPI + the NVIDIA index the
# way pip's --extra-index-url does (RAPIDS deps are split across both indexes).
log "installing cuml-cu12 + support libraries (this pulls the RAPIDS cu12 stack) ..."
set -x
UV_INDEX_STRATEGY=unsafe-best-match "$UV" pip install --python "$PYBIN" \
  --extra-index-url=https://pypi.nvidia.com \
  "cuml-cu12" matplotlib "nvidia-ml-py" psutil pyyaml pyarrow numpy pandas 2>&1 | tee -a "$LOG"
CORE_RC=${PIPESTATUS[0]}
set +x
log "core install exit code: $CORE_RC"
if [ "$CORE_RC" -ne 0 ]; then
  log "CORE INSTALL FAILED — stopping."
  exit "$CORE_RC"
fi

# ---- optional: CPU hdbscan package (only to materialize tree diagnostics
# via cuML's *_tree_ .to_pandas(); does NOT affect the GPU clustering) ----
log "installing CPU hdbscan (for condensed/linkage/MST export helpers) ..."
UV_INDEX_STRATEGY=unsafe-best-match "$UV" pip install --python "$PYBIN" hdbscan 2>&1 | tee -a "$LOG" \
  || log "WARNING: hdbscan CPU package failed to install; tree .to_pandas() export may be limited"

# ---- capture exact versions & GPU/CUDA environment ----
log "freezing package versions ..."
"$UV" pip freeze --python "$PYBIN" > "$ENV_OUT_DIR/packages.txt" 2>>"$LOG"

log "nvidia-smi snapshot ..."
nvidia-smi > "$ENV_OUT_DIR/nvidia_smi.txt" 2>&1 || echo "nvidia-smi failed" > "$ENV_OUT_DIR/nvidia_smi.txt"

log "probing cuml / cupy / cuda versions ..."
"$PYBIN" - <<'PY' > "$ENV_OUT_DIR/cuml_env.json" 2>>"$LOG"
import json, platform, sys
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
