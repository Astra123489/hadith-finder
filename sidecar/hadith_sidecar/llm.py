"""LLM providers.

Production: llama.cpp via llama-cpp-python (MIT), model = Qwen3-8B-Instruct
Q4_K_M by default (Apache-2.0). The app ships with no weights; the model is
downloaded on first run into the user's data dir (see docs/ARCHITECTURE.md).

EchoProvider exists ONLY for tests: it cites the first excerpts it sees so
the citation-extraction path is exercised deterministically. Never used in
production.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

LLM_PATH_ENV = "HADITH_LLM_GGUF"
_CAREFUL_THINK_PREFIX = ""  # Qwen3 thinking enabled by default in chat template
_FAST_THINK_PREFIX = "<think>\n\n</think>\n\n"  # prefill empty thought = non-thinking


class LLMProvider:
    name = "abstract"

    def generate(self, system: str, user: str, *, max_tokens: int = 768,
                 temperature: float = 0.2, careful: bool = False) -> str:
        raise NotImplementedError


class EchoProvider(LLMProvider):
    name = "echo"

    def generate(self, system: str, user: str, *, max_tokens: int = 768,
                 temperature: float = 0.2, careful: bool = False) -> str:
        refs = re.findall(r"\[(S\d+)\]", user)
        if not refs:
            return "I could not find this in the loaded books."
        cited = ", ".join(f"[{r}]" for r in refs[:2])
        return f"Echo answer grounded in the excerpts {cited}."


class LlamaCppProvider(LLMProvider):
    name = "llama-cpp"

    def __init__(self, model_path: str | None = None, *, n_ctx: int = 8192):
        if not model_path:
            from .downloader import default_llm_path

            model_path = os.environ.get(LLM_PATH_ENV) or str(default_llm_path())
        self.model_path = model_path
        if not Path(self.model_path).is_file():
            raise RuntimeError(
                f"Qwen3 model not downloaded yet — open Settings → AI Models and download it "
                f"(~4.7 GB, one time). Expected at: {self.model_path}"
            )
        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise RuntimeError(
                "Missing LLM dependency. Install with: pip install llama-cpp-python"
            ) from e
        threads = max(1, (os.cpu_count() or 4) // 2)
        self._llm = Llama(
            model_path=self.model_path,
            n_ctx=n_ctx,
            n_threads=threads,
            n_gpu_layers=int(os.environ.get("HADITH_GPU_LAYERS", "0")),
            verbose=False,
        )

    def _format_qwen3(self, system: str, user: str, careful: bool) -> str:
        prefill = "" if careful else _FAST_THINK_PREFIX
        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n{prefill}"
        )

    def generate(self, system: str, user: str, *, max_tokens: int = 768,
                 temperature: float = 0.2, careful: bool = False) -> str:
        prompt = self._format_qwen3(system, user, careful)
        out = self._llm.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            repeat_penalty=1.05,
            stop=["<|im_end|>", "<|endoftext|>"],
        )
        text = out["choices"][0]["text"].strip()
        if not careful and text.startswith("<think>"):
            text = re.sub(r"^<think>.*?</think>\s*", "", text, flags=re.S).strip()
        return text


def get_llm(kind: str | None = None) -> LLMProvider:
    """kind: 'auto' (llama-cpp, else error), 'llama-cpp', 'echo' (tests only)."""
    kind = (kind or os.environ.get("HADITH_LLM") or "auto").lower()
    if kind == "echo":
        return EchoProvider()
    if kind in ("llama-cpp", "llama_cpp", "auto"):
        return LlamaCppProvider()
    raise ValueError(f"unknown LLM provider: {kind}")
