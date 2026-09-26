#!/usr/bin/env bash
# Builds the vram_cap shim (a drop-in libcuda.so.1) against the real driver library.
# Usage: build_vram_cap.sh <source dir> <output dir> [real libcuda path]
set -euo pipefail
SRC_DIR="$1"
OUT_DIR="$2"
REAL="${3:-/usr/lib/wsl/lib/libcuda.so.1}"
mkdir -p "$OUT_DIR"
nm -D --defined-only "$REAL" | awk '$2 ~ /^[TW]$/ {print $3}' | sort -u > "$OUT_DIR/libcuda_exports.txt"
python3 "$SRC_DIR/gen_trampolines.py" "$OUT_DIR/libcuda_exports.txt" "$SRC_DIR/vram_cap.c" \
  "$OUT_DIR/trampolines.S" "$OUT_DIR/trampolines.h"
gcc -shared -fPIC -O2 -Wall -Wextra -Wno-unused-parameter -I"$OUT_DIR" \
  -o "$OUT_DIR/libcuda.so.1" "$SRC_DIR/vram_cap.c" "$OUT_DIR/trampolines.S" \
  -ldl -lpthread -Wl,-soname,libcuda.so.1 -Wl,-z,noexecstack
ln -sf libcuda.so.1 "$OUT_DIR/libcuda.so"
n_shim=$(nm -D --defined-only "$OUT_DIR/libcuda.so.1" | awk '$2 ~ /^[TW]$/ && $3 ~ /^cu/' | wc -l)
n_real=$(wc -l < "$OUT_DIR/libcuda_exports.txt")
echo "shim exports ${n_shim} cu* functions; real library exports ${n_real}"
comm -23 "$OUT_DIR/libcuda_exports.txt" <(nm -D --defined-only "$OUT_DIR/libcuda.so.1" | awk '$2 ~ /^[TW]$/ {print $3}' | sort -u) > "$OUT_DIR/missing_exports.txt" || true
echo "real exports missing from the shim: $(wc -l < "$OUT_DIR/missing_exports.txt")"
readelf -d "$OUT_DIR/libcuda.so.1" | grep -i soname
sha256sum "$OUT_DIR/libcuda.so.1" "$SRC_DIR/vram_cap.c" "$SRC_DIR/gen_trampolines.py"
gcc --version | head -1
echo "build_vram_cap.sh: done -> $OUT_DIR"
