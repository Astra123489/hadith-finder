"""First-run model downloader.

Fetches the three recommended open models (Apache-2.0 / MIT) from
HuggingFace over HTTPS with resume support and exact-size verification.
Models are stored under <library dir>/models/ by default (override with
HADITH_MODELS_DIR). Progress is streamed as JSON-RPC notifications with
op="download".

Integrity: each file is checked against its exact expected byte size
(recorded from the HuggingFace LFS pointer at authoring time). No
fabricated hashes — sizes catch truncation, HTML error pages and partial
writes.
"""

from __future__ import annotations

import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

MODELS_ENV = "HADITH_MODELS_DIR"
LIB_ENV = "HADITH_LIBRARY"

MODEL_CATALOG: dict[str, dict] = {
    "llm": {
        "name": "Qwen3-8B-Instruct (Q4_K_M)",
        "purpose": "Answers your questions, grounded only in the highlighted sources",
        "license": "Apache-2.0",
        "dir": "llm",
        "total_bytes": 5027783488,
        "files": [
            {
                "name": "Qwen3-8B-Q4_K_M.gguf",
                "url": "https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf",
                "size": 5027783488,
            },
        ],
    },
    "embedder": {
        "name": "BGE-M3 (int8 ONNX)",
        "purpose": "Finds the right passages — cross-lingual Arabic ↔ English, even when you don't know the wording",
        "license": "MIT",
        "dir": "bge-m3",
        "total_bytes": 586777351,
        "files": [
            {
                "name": "model.onnx",
                "url": "https://huggingface.co/Xenova/bge-m3/resolve/main/onnx/model_quantized.onnx",
                "size": 569694530,
            },
            {
                "name": "tokenizer.json",
                "url": "https://huggingface.co/Xenova/bge-m3/resolve/main/tokenizer.json",
                "size": 17082821,
            },
        ],
    },
    "reranker": {
        "name": "bge-reranker-v2-m3 (int8 ONNX)",
        "purpose": "Double-checks the top candidates before answering — the precision booster",
        "license": "MIT",
        "dir": "bge-reranker-v2-m3",
        "total_bytes": 587069662,
        "files": [
            {
                "name": "model.onnx",
                "url": "https://huggingface.co/woxpas-ai/bge-reranker-v2-m3-onnx/resolve/main/onnx/model_quantized.onnx",
                "size": 569986762,
            },
            {
                "name": "tokenizer.json",
                "url": "https://huggingface.co/woxpas-ai/bge-reranker-v2-m3-onnx/resolve/main/tokenizer.json",
                "size": 17082900,
            },
        ],
    },
}


def models_root() -> Path:
    explicit = os.environ.get(MODELS_ENV)
    if explicit:
        return Path(explicit)
    lib = os.environ.get(LIB_ENV)
    if lib:
        return Path(lib).expanduser().resolve().parent / "models"
    return Path.home() / ".hadith-finder" / "models"


def model_dir(model_id: str) -> Path:
    return models_root() / MODEL_CATALOG[model_id]["dir"]


def default_llm_path() -> Path:
    return model_dir("llm") / "Qwen3-8B-Q4_K_M.gguf"


def _file_ok(path: Path, expected_size: int) -> bool:
    try:
        return path.is_file() and path.stat().st_size == expected_size
    except OSError:
        return False


def status() -> dict:
    root = models_root()
    out = {}
    total = sum(m["total_bytes"] for m in MODEL_CATALOG.values())
    ready_models = 0
    for model_id, spec in MODEL_CATALOG.items():
        present_bytes = 0
        files = []
        for f in spec["files"]:
            target = root / spec["dir"] / f["name"]
            ok = _file_ok(target, f["size"])
            if target.is_file() and not ok:
                present_bytes += min(target.stat().st_size, f["size"])
            elif ok:
                present_bytes += f["size"]
            files.append({"name": f["name"], "size": f["size"], "installed": ok})
        ready = all(f["installed"] for f in files)
        ready_models += ready
        out[model_id] = {
            "name": spec["name"],
            "purpose": spec["purpose"],
            "license": spec["license"],
            "dir": str(root / spec["dir"]),
            "ready": ready,
            "bytes_installed": present_bytes,
            "total_bytes": spec["total_bytes"],
            "files": files,
        }
    out["all_ready"] = ready_models == len(MODEL_CATALOG)
    out["total_bytes"] = total
    return out


def _download_file(url: str, target: Path, size: int, notify_cb, chunk: int = 1 << 20) -> None:
    part = target.with_suffix(target.suffix + ".part")
    existing = part.stat().st_size if part.is_file() else 0
    headers = {}
    mode = "wb"
    if existing > 0 and existing < size:
        headers["Range"] = f"bytes={existing}-"
        mode = "ab"
    elif existing >= size:
        existing = 0  # corrupt partial — start over

    req = urllib.request.Request(url, headers=headers)
    downloaded = existing
    last_emit = 0.0
    try:
        with urllib.request.urlopen(req, timeout=60) as resp, open(part, mode) as out:
            if existing and resp.status != 206:
                out.seek(0)  # server ignored Range — rewrite from scratch
                downloaded = 0
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                out.write(buf)
                downloaded += len(buf)
                now = time.monotonic()
                if now - last_emit > 0.25:
                    last_emit = now
                    notify_cb(downloaded, size)
    except Exception:
        # keep .part for resume on next attempt
        raise

    final_size = part.stat().st_size
    if final_size != size:
        raise RuntimeError(
            f"download incomplete for {target.name}: expected {size} bytes, got {final_size}. "
            "Check your connection and retry — the download resumes."
        )
    part.replace(target)


def download_model(model_id: str, notify) -> dict:
    if model_id not in MODEL_CATALOG:
        raise ValueError(f"unknown model: {model_id!r} (expected one of {sorted(MODEL_CATALOG)})")
    spec = MODEL_CATALOG[model_id]
    dest_dir = models_root() / spec["dir"]
    dest_dir.mkdir(parents=True, exist_ok=True)
    files_total = len(spec["files"])
    for index, f in enumerate(spec["files"]):
        target = dest_dir / f["name"]
        if _file_ok(target, f["size"]):
            continue
        url, size, name = f["url"], f["size"], f["name"]

        def cb(downloaded: int, total: int, _name=name, _i=index):
            notify(
                op="download", model=model_id, file=_name,
                downloaded=downloaded, total=total,
                file_index=_i, file_total=files_total,
            )

        _download_file(url, target, size, cb)
    # final integrity pass
    missing = [f["name"] for f in spec["files"] if not _file_ok(dest_dir / f["name"], f["size"])]
    if missing:
        raise RuntimeError(f"integrity check failed for: {', '.join(missing)}")
    return {"model": model_id, "ready": True, "dir": str(dest_dir)}


def erase_for_tests() -> None:
    shutil.rmtree(models_root(), ignore_errors=True)
