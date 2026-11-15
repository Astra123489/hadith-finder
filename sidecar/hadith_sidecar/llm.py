"""LLM providers.

Production: llama.cpp via llama-cpp-python (MIT), model = Qwen3-8B-Instruct
Q4_K_M by default (Apache-2.0). The app ships with no weights; the model is
downloaded on first run into the user's data dir (see docs/ARCHITECTURE.md).

The chat template comes from the GGUF file itself (llama.cpp reads its
metadata), so swapping in another instruct model does not silently produce
mangled prompts. Fast mode appends Qwen3's soft switch (`/no_think`);
Careful mode leaves reasoning on and streams the reasoning too.

EchoProvider exists ONLY for tests: it cites the first excerpts it sees so
the citation-extraction path is exercised deterministically. Never used in
production.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

LLM_PATH_ENV = "HADITH_LLM_GGUF"
_STOP = ["<|im_end|>", "<|endoftext|>"]


def resolve_model_path(stored: str | None = None, explicit: str | None = None) -> str:
    """Pick the GGUF to load.

    Order: explicit argument → $HADITH_LLM_GGUF → the path saved from Settings →
    any .gguf in <models>/llm (newest first) → the built-in download location.
    This is what lets a user point the app at a model they already have
    (LM Studio, Ollama's blobs, a hand-copied file) without moving it.
    """
    for candidate in (explicit, os.environ.get(LLM_PATH_ENV), stored):
        if candidate and Path(candidate).expanduser().is_file():
            return str(Path(candidate).expanduser())
    from .downloader import default_llm_path, model_dir

    folder = model_dir("llm")
    if folder.is_dir():
        found = sorted(folder.glob("*.gguf"), key=lambda p: p.stat().st_mtime, reverse=True)
        if found:
            return str(found[0])
    return str(default_llm_path())


def active_model_info(stored: str | None = None) -> dict:
    path = Path(resolve_model_path(stored))
    exists = path.is_file()
    return {
        "name": path.stem,
        "path": str(path),
        "exists": exists,
        "size": path.stat().st_size if exists else 0,
        "custom": bool(stored) and str(path) == str(Path(stored).expanduser()),
    }


class LLMProvider:
    name = "abstract"

    def generate(self, system: str, user: str, *, max_tokens: int = 512,
                 temperature: float = 0.2, careful: bool = False,
                 on_token=None) -> str:
        raise NotImplementedError


class EchoProvider(LLMProvider):
    name = "echo"

    def generate(self, system: str, user: str, *, max_tokens: int = 512,
                 temperature: float = 0.2, careful: bool = False,
                 on_token=None) -> str:
        refs = re.findall(r"\[(S\d+)\]", user)
        if not refs:
            return "I could not find this in the loaded books."
        cited = ", ".join(f"[{r}]" for r in refs[:2])
        text = f"Echo answer grounded in the excerpts {cited}."
        if on_token:
            on_token(text)
        return text


class LlamaCppProvider(LLMProvider):
    name = "llama-cpp"
    context_size = 6144

    def __init__(self, model_path: str | None = None, *, n_ctx: int | None = None,
                 stored: str | None = None):
        self.model_path = resolve_model_path(stored, model_path)
        if not Path(self.model_path).is_file():
            raise RuntimeError(
                "No language model found. Download the default model in Settings → AI Models, "
                f"or choose a local .gguf file. Looked for: {self.model_path}"
            )
        try:
            from llama_cpp import Llama, llama_supports_gpu_offload
        except ImportError as e:
            raise RuntimeError(
                "Missing LLM dependency. Install with: pip install llama-cpp-python"
            ) from e

        threads = max(1, (os.cpu_count() or 4) // 2)
        self.context_size = n_ctx or int(os.environ.get("HADITH_CTX") or 6144)
        self.gpu_layers = self._gpu_layers(llama_supports_gpu_offload)
        self._llm = Llama(
            model_path=self.model_path,
            n_ctx=self.context_size,
            n_threads=threads,
            n_batch=512,
            n_gpu_layers=self.gpu_layers,
            verbose=False,
        )
        self._chat_ok = True

    @staticmethod
    def _gpu_layers(supports) -> int:
        raw = (os.environ.get("HADITH_GPU_LAYERS") or "auto").strip().lower()
        if raw in ("auto", ""):
            try:
                return -1 if supports() else 0
            except Exception:
                return 0
        try:
            return int(raw)
        except ValueError:
            return 0

    def _messages(self, system: str, user: str, careful: bool) -> list[dict]:
        prompt = user if careful else user + "\n/no_think"
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]

    def _chatml(self, system: str, user: str, careful: bool) -> str:
        prefill = "" if careful else "\n\n"
        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n{prefill}"
        )

    def _completion(self, system: str, user: str, *, max_tokens: int,
                    temperature: float, careful: bool, stream: bool):
        """Chat completion via the model's own template, with a ChatML fallback."""
        if self._chat_ok:
            try:
                return self._llm.create_chat_completion(
                    messages=self._messages(system, user, careful),
                    max_tokens=max_tokens,
                    temperature=temperature,
                    repeat_penalty=1.05,
                    stop=_STOP,
                    stream=stream,
                )
            except Exception:
                self._chat_ok = False
        return self._llm.create_completion(
            prompt=self._chatml(system, user, careful),
            max_tokens=max_tokens,
            temperature=temperature,
            repeat_penalty=1.05,
            stop=_STOP,
            stream=stream,
        )

    @staticmethod
    def _text_of(out) -> str:
        """Text from either a chat-completion or a raw-completion response."""
        choice = out["choices"][0]
        if isinstance(choice, dict):
            if "message" in choice:
                return (choice["message"] or {}).get("content") or ""
            if "text" in choice:
                return choice["text"] or ""
        return ""

    @staticmethod
    def _delta_of(part) -> str:
        choice = part["choices"][0]
        if isinstance(choice, dict):
            if choice.get("delta") is not None:
                return (choice["delta"] or {}).get("content") or ""
            return choice.get("text") or ""
        return ""

    def generate(self, system: str, user: str, *, max_tokens: int = 512,
                 temperature: float = 0.2, careful: bool = False,
                 on_token=None) -> str:
        if on_token is None:
            out = self._completion(system, user, max_tokens=max_tokens,
                                   temperature=temperature, careful=careful, stream=False)
            text = self._text_of(out).strip()
            return self._strip_think(text, careful)

        pieces: list[str] = []
        pending = ""
        started = False
        for part in self._completion(system, user, max_tokens=max_tokens,
                                     temperature=temperature, careful=careful, stream=True):
            delta = self._delta_of(part)
            if not delta:
                continue
            if started:
                pending = delta
            else:
                pending += delta
                if "</think>" in pending:
                    pending = pending.split("</think>", 1)[1].lstrip()
                elif len(pending) > 60 and not pending.lstrip().startswith("<think"):
                    pass
                else:
                    continue
                started = True
            if pending:
                pieces.append(pending)
                try:
                    on_token(pending)
                except Exception:
                    pass
                pending = ""
        if pending:
            pieces.append(pending)
            try:
                on_token(pending)
            except Exception:
                pass
        return self._strip_think("".join(pieces).strip(), careful)

    @staticmethod
    def _strip_think(text: str, careful: bool) -> str:
        if careful:
            return text
        return re.sub(r"^.*?</think>\s*", "", text, flags=re.S).strip() if "</think>" in text else text


def get_llm(kind: str | None = None, *, model_path: str | None = None,
            stored: str | None = None) -> LLMProvider:
    """kind: 'auto' (llama-cpp, else error), 'llama-cpp', 'echo' (tests only)."""
    kind = (kind or os.environ.get("HADITH_LLM") or "auto").lower()
    if kind == "echo":
        return EchoProvider()
    if kind in ("llama-cpp", "llama_cpp", "auto"):
        return LlamaCppProvider(model_path=model_path, stored=stored)
    raise ValueError(f"unknown LLM provider: {kind}")
