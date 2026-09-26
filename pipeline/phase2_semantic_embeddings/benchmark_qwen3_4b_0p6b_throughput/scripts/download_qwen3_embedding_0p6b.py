#!/usr/bin/env python
"""Download Qwen/Qwen3-Embedding-0.6B at a pinned revision into the WSL2 Hugging Face cache and
verify every file against the hashes the Hub API reported for that revision.

Run inside the WSL2 venv (see ../README.md):

    ~/venvs/socialmediadgg-phase2/bin/python download_qwen3_embedding_0p6b.py

Large files (LFS) are checked by SHA256; small files by their git blob id (SHA1 of
"blob <size>\\0" + content). Writes ../environment/model_snapshot_0p6b.json and exits non-zero if
any file differs.
"""

import datetime as dt
import hashlib
import json
import sys
import time
from pathlib import Path

from huggingface_hub import snapshot_download

REPO = "Qwen/Qwen3-Embedding-0.6B"
REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
# From https://huggingface.co/api/models/Qwen/Qwen3-Embedding-0.6B?blobs=true (2026-09-10).
EXPECTED_LFS_SHA256 = {
    "model.safetensors": "0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd",
    "tokenizer.json": "def76fb086971c7867b829c23a26261e38d9d74e02139253b38aeb9df8b4b50a",
}
EXPECTED_GIT_BLOB = {
    ".gitattributes": "52373fe24473b1aa44333d318f578ae6bf04b49b",
    "1_Pooling/config.json": "b6291baabbc39f9792d6759883a47d7d5bd0fbcf",
    "README.md": "00a82ea60dfb1d8aed899c16e2b4f12673cd2f79",
    "config.json": "cef2749ee93607b8f9a58ec72f4f6bfaf874e71d",
    "config_sentence_transformers.json": "76aef3ade63553ebb698fe3c2a3264040ed093f8",
    "generation_config.json": "d46f1983345269c582611bbedb3ca0a13f8e5f7b",
    "merges.txt": "31349551d90c7606f325fe0f11bbb8bd5fa0d7c7",
    "modules.json": "952a9b81c0bfd99800fabf352f69c7ccd46c5e43",
    "tokenizer_config.json": "7345216a0785dc7086e8c245b2a9d3896ce2b756",
    "vocab.json": "4783fe10ac3adce15ac8f358ef5462739852c569",
}
OUT = Path(__file__).resolve().parents[1] / "environment" / "model_snapshot_0p6b.json"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_blob_sha1(path):
    data = Path(path).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def main():
    t0 = time.time()
    snapshot = Path(snapshot_download(repo_id=REPO, revision=REVISION))
    download_s = time.time() - t0
    files = []
    all_ok = True
    for name, expected in sorted(EXPECTED_LFS_SHA256.items()):
        p = snapshot / name
        got = sha256_file(p) if p.exists() else "missing"
        ok = got == expected
        all_ok &= ok
        files.append(
            {
                "file": name,
                "check": "sha256",
                "expected": expected,
                "measured": got,
                "size_bytes": p.stat().st_size if p.exists() else None,
                "equal": ok,
            }
        )
    for name, expected in sorted(EXPECTED_GIT_BLOB.items()):
        p = snapshot / name
        got = git_blob_sha1(p) if p.exists() else "missing"
        ok = got == expected
        all_ok &= ok
        files.append(
            {
                "file": name,
                "check": "git_blob_sha1",
                "expected": expected,
                "measured": got,
                "size_bytes": p.stat().st_size if p.exists() else None,
                "equal": ok,
            }
        )
    record = {
        "repo": REPO,
        "revision": REVISION,
        "snapshot_path": str(snapshot),
        "download_or_cache_check_s": round(download_s, 3),
        "all_files_verified": all_ok,
        "files": files,
        "timestamp": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in record.items() if k != "files"}, indent=1))
    for f in files:
        print(f"  {f['file']}: {f['check']} equal={f['equal']}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
