"""Embedding providers.

Default production model: BGE-M3 (MIT), used in this app for its dense
arm (multilingual, strong cross-lingual Arabic↔English). CLS pooling +
L2 normalization, run via ONNX Runtime (see docs/ARCHITECTURE.md).

HashEmbedder exists ONLY for tests and offline development — it needs no
model files and makes retrieval deterministic. Never used in production.
"""

from __future__ import annotations

import hashlib
import math
import os
import re

import numpy as np

MODEL_PATH_ENV = "HADITH_BGE_M3_ONNX"  # directory holding model.onnx + tokenizer.json


class Embedder:
    name = "abstract"
    dim = 0

    def embed(self, texts: list[str]) -> np.ndarray:  # L2-normalized rows
        raise NotImplementedError


class HashEmbedder(Embedder):
    """Deterministic feature-hashing embedder for tests (not semantic)."""

    name = "hash"
    dim = 128

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            toks = re.findall(r"\S+", (text or "").lower())
            for tok in toks:
                h = int(hashlib.blake2b(tok.encode("utf-8"), digest_size=8).hexdigest(), 16)
                slot = h % self.dim
                sign = 1.0 if (h >> 32) & 1 else -1.0
                out[i, slot] += sign
            n = np.linalg.norm(out[i])
            if n > 0:
                out[i] /= n
        return out


class OnnxM3Embedder(Embedder):
    """BGE-M3 dense embeddings via ONNX Runtime.

    Expects MODEL_PATH_ENV pointing at a directory containing `model.onnx`
    (or `onnx/model.onnx`) and `tokenizer.json` exported from BAAI/bge-m3.
    Install with:  pip install onnxruntime tokenizers
    """

    name = "bge-m3"
    dim = 1024

    def __init__(self, model_dir: str | None = None):
        self.model_dir = model_dir or os.environ.get(MODEL_PATH_ENV)
        if not self.model_dir:
            raise RuntimeError(
                "BGE-M3 embedding model not installed yet — open Settings → AI Models and "
                f"download it (~550 MB, one time). Set {MODEL_PATH_ENV} to override the location."
            )
        try:
            import onnxruntime as ort  # noqa: F401
            from tokenizers import Tokenizer  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "Missing embedding dependencies. Install with: pip install onnxruntime tokenizers"
            ) from e

        from pathlib import Path

        import onnxruntime as ort
        from tokenizers import Tokenizer

        root = Path(self.model_dir)
        model_file = root / "model.onnx"
        if not model_file.is_file():
            alt = root / "onnx" / "model.onnx"
            model_file = alt if alt.is_file() else model_file
        tok_file = root / "tokenizer.json"
        if not model_file.is_file() or not tok_file.is_file():
            raise RuntimeError(f"{root} must contain model.onnx and tokenizer.json")

        self._sess = ort.InferenceSession(str(model_file), providers=["CPUExecutionProvider"])
        self._tok = Tokenizer.from_file(str(tok_file))
        self._tok.enable_truncation(max_length=512)
        self._tok.enable_padding(pad_token="<pad>", pad_id=1)
        self.dim = 1024

    def embed(self, texts: list[str]) -> np.ndarray:
        batch = self._tok.encode_batch(texts)
        input_ids = np.array([e.ids for e in batch], dtype=np.int64)
        attn = np.array([e.attention_mask for e in batch], dtype=np.int64)
        feed = {"input_ids": input_ids, "attention_mask": attn}
        inputs = {i.name for i in self._sess.get_inputs()}
        if "token_type_ids" in inputs:
            feed["token_type_ids"] = np.zeros_like(input_ids)
        hidden = self._sess.run(None, feed)[0]  # (batch, seq, 1024)
        cls = hidden[:, 0, :].astype(np.float32)  # CLS pooling per BGE-M3 card
        norms = np.linalg.norm(cls, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return cls / norms


def get_embedder(kind: str | None = None) -> Embedder:
    """kind: 'auto'/'bge-m3' (the downloaded model, error if absent), 'hash' (tests only)."""
    from .downloader import models_root

    kind = (kind or os.environ.get("HADITH_EMBEDDER") or "auto").lower()
    default_dir = os.environ.get(MODEL_PATH_ENV) or str(models_root() / "bge-m3")
    if kind == "hash":
        return HashEmbedder()
    if kind in ("bge-m3", "auto"):
        return OnnxM3Embedder(default_dir)
    return OnnxM3Embedder(default_dir)
