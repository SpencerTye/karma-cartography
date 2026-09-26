#!/usr/bin/env bash
# Builds the WSL2 (Ubuntu-24.04) Python venv for the Phase 2 feasibility test and downloads both
# models at their pinned revisions. Pinned: torch 2.8.0 (CUDA 12.8), flash-attn 2.8.3.post1; see ../README.md.
# Run inside WSL2: bash "$REPO/pipeline/phase2_semantic_embeddings/feasibility_qwen3_embedding_worst_case/scripts/setup_wsl2_env.sh"
# Env (optional): PHASE2_VENV (default ~/venvs/socialmediadgg-phase2), HF_HOME (default ~/.cache/huggingface).
# Needs about 30 GB free on the Windows drive that holds the distro's ext4.vhdx (4 GB libraries,
# 23 GB weights); df / inside WSL2 does not show the host's free space.
set -euo pipefail

VENV="${PHASE2_VENV:-$HOME/venvs/socialmediadgg-phase2}"
export PATH="$HOME/.local/bin:$PATH"

echo "== uv =="
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

echo "== venv: $VENV =="
if [ ! -x "$VENV/bin/python" ]; then
  uv venv --python 3.12 "$VENV"
fi
PY="$VENV/bin/python"
"$PY" --version

echo "== torch 2.8.0 =="
uv pip install --python "$PY" "torch==2.8.0"
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())"

echo "== flash-attn 2.8.3.post1 (official prebuilt wheel) =="
ABI=$("$PY" -c "import torch; print('TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE')")
PYTAG=$("$PY" -c "import sys; print('cp%d%d' % (sys.version_info[0], sys.version_info[1]))")
WHL="flash_attn-2.8.3.post1%2Bcu12torch2.8cxx11abi${ABI}-${PYTAG}-${PYTAG}-linux_x86_64.whl"
URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/${WHL}"
echo "wheel: $URL"
uv pip install --python "$PY" "$URL"

echo "== libraries =="
uv pip install --python "$PY" transformers accelerate safetensors huggingface_hub sentence-transformers pyarrow numpy psutil nvidia-ml-py pyyaml

echo "== versions =="
"$PY" - <<'PYEOF'
import importlib.metadata as m
for name in ["torch", "flash_attn", "transformers", "tokenizers", "accelerate", "safetensors",
             "huggingface_hub", "sentence-transformers", "pyarrow", "numpy", "psutil",
             "nvidia-ml-py", "pyyaml"]:
    try:
        print(f"{name}=={m.version(name)}")
    except m.PackageNotFoundError:
        print(f"{name} NOT INSTALLED")
import torch
print("torch.version.cuda", torch.version.cuda, "| cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("device", p.name, "| total_memory_bytes", p.total_memory, "| sm", p.major, p.minor,
          "| bf16 supported", torch.cuda.is_bf16_supported())
PYEOF

echo "== model snapshots (pinned revisions) =="
"$PY" - <<'PYEOF'
from huggingface_hub import snapshot_download
pins = {
    "Qwen/Qwen3-Embedding-4B": "5cf2132abc99cad020ac570b19d031efec650f2b",
    "Qwen/Qwen3-Embedding-8B": "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af",
}
for repo, rev in pins.items():
    path = snapshot_download(repo_id=repo, revision=rev)
    print(repo, rev, "->", path)
PYEOF

echo "== snapshot integrity (every LFS blob file's sha256 must equal its file name) =="
CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
for d in "$CACHE"/hub/models--Qwen--Qwen3-Embedding-4B/blobs "$CACHE"/hub/models--Qwen--Qwen3-Embedding-8B/blobs; do
  for f in "$d"/*; do
    n=$(basename "$f")
    case "$n" in *.incomplete) echo "INCOMPLETE $f"; continue ;; esac
    if [ ${#n} -eq 64 ]; then
      h=$(sha256sum "$f" | cut -d' ' -f1)
      if [ "$h" = "$n" ]; then echo "OK $n"; else echo "MISMATCH $n (got $h)"; fi
    fi
  done
done

echo "setup_wsl2_env.sh: done"
