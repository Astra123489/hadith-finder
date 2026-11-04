"""Cross-encoder reranking (multilingual).

Production: bge-reranker-v2-m3 via ONNX Runtime (MIT), set
HADITH_RERANKER_ONNX to the model directory. Absent a model, the
retrieval order from RRF is kept as-is (identity) — precision drops but
the app stays functional and says so in logs.
"""

from __future__ import annotations

import os

RERANKER_PATH_ENV = "HADITH_RERANKER_ONNX"


class Reranker:
    name = "identity"

    def rerank(self, query: str, texts: list[str]) -> list[float]:
        raise NotImplementedError


class IdentityReranker(Reranker):
    name = "identity"

    def rerank(self, query: str, texts: list[str]) -> list[float]:
        return [1.0 / (1 + i) for i in range(len(texts))]


class OnnxReranker(Reranker):
    name = "bge-reranker-v2-m3"

    def __init__(self, model_dir: str | None = None):
        self.model_dir = model_dir or os.environ.get(RERANKER_PATH_ENV)
        if not self.model_dir:
            raise RuntimeError(f"Set {RERANKER_PATH_ENV} to the reranker model directory.")
        from pathlib import Path

        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer

        root = Path(self.model_dir)
        model_file = root / "model.onnx"
        if not model_file.is_file():
            alt = root / "onnx" / "model.onnx"
            model_file = alt if alt.is_file() else model_file
        if not model_file.is_file() or not (root / "tokenizer.json").is_file():
            raise RuntimeError(f"{root} must contain model.onnx and tokenizer.json")

        self._np = np
        self._sess = ort.InferenceSession(str(model_file), providers=["CPUExecutionProvider"])
        self._tok = Tokenizer.from_file(str(root / "tokenizer.json"))
        self._tok.enable_truncation(max_length=512)
        self._input_names = {i.name for i in self._sess.get_inputs()}

    def rerank(self, query: str, texts: list[str]) -> list[float]:
        np, sess, tok = self._np, self._sess, self._tok
        out: list[float] = []
        for i in range(0, len(texts), 8):
            batch = tok.encode_batch([(query, t) for t in texts[i:i + 8]], is_pretokenized=False)
            maxlen = max(len(e.ids) for e in batch)
            input_ids = np.zeros((len(batch), maxlen), dtype=np.int64)
            attn = np.zeros_like(input_ids)
            for j, e in enumerate(batch):
                input_ids[j, :len(e.ids)] = e.ids
                attn[j, :len(e.attention_mask)] = e.attention_mask
            feed = {"input_ids": input_ids, "attention_mask": attn}
            if "token_type_ids" in self._input_names:
                feed["token_type_ids"] = np.zeros_like(input_ids)
            logits = sess.run(None, feed)[0]
            scores = logits.reshape(len(batch), -1)[:, -1].astype(np.float64)
            out.extend(scores.tolist())
        return out


def get_reranker(kind: str | None = None) -> Reranker:
    kind = (kind or os.environ.get("HADITH_RERANKER") or "auto").lower()
    if kind in ("identity", "none"):
        return IdentityReranker()
    from .downloader import models_root

    model_dir = os.environ.get(RERANKER_PATH_ENV) or str(models_root() / "bge-reranker-v2-m3")
    if kind in ("bge", "bge-reranker-v2-m3"):
        return OnnxReranker(model_dir)
    if kind == "auto":
        try:
            return OnnxReranker(model_dir)
        except Exception:
            return IdentityReranker()  # reranking is a booster, not a hard dependency
    return IdentityReranker()
