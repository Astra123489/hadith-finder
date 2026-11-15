"""LLM providers.

Production: llama.cpp via llama-cpp-python (MIT), model = Gemma 4 E2B Instruct
Q8_0 by default (Apache-2.0). The app ships with no weights; the model is
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
import subprocess
from pathlib import Path

LLM_PATH_ENV = "HADITH_LLM_GGUF"
DEFAULT_CTX = 6144
MIN_CTX = 1024
MAX_CTX = 131072
_STOP = ["<|im_end|>", "<|endoftext|>"]


def _nvidia_smi() -> dict | None:
    """Device name and VRAM from nvidia-smi (absent on non-NVIDIA machines)."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=6,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        name, total, free = [p.strip() for p in out.stdout.strip().splitlines()[0].split(",")][:3]
        return {"device": name, "vram_total_mb": int(total), "vram_free_mb": int(free)}
    except Exception:
        return None


def gpu_status(layers: int | None = None) -> dict:
    """Whether the answer model can run on the GPU, and on which device.

    llama-cpp-python wheels are either CPU-only or GPU-enabled. A CPU-only
    wheel reports no offload support and llama.cpp then ignores n_gpu_layers,
    so the app surfaces this in Settings instead of leaving users guessing why
    answers are slow.
    """
    try:
        from llama_cpp import llama_supports_gpu_offload

        available = bool(llama_supports_gpu_offload())
    except Exception:
        available = False
    info = {
        "available": available,
        "device": None,
        "vram_total_mb": None,
        "vram_free_mb": None,
        "layers": layers,
    }
    if available:
        info.update(_nvidia_smi() or {})
        if info["device"] is None:
            info["device"] = "GPU"
    return info


def resolve_ctx(stored_ctx=None, explicit=None) -> int:
    """Context window (tokens) to load the model with.

    Order: explicit → saved setting → $HADITH_CTX → DEFAULT_CTX. A larger
    window means more prompt room (more/longer excerpts) at the cost of RAM,
    so this is a user-facing setting.
    """
    raw = explicit if explicit not in (None, "") else stored_ctx
    if raw in (None, ""):
        raw = os.environ.get("HADITH_CTX") or DEFAULT_CTX
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_CTX
    value = max(MIN_CTX, min(MAX_CTX, value))
    return (value // 256) * 256


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


def active_model_info(stored: str | None = None, ctx=None) -> dict:
    path = Path(resolve_model_path(stored))
    exists = path.is_file()
    return {
        "name": path.stem,
        "path": str(path),
        "exists": exists,
        "size": path.stat().st_size if exists else 0,
        "custom": bool(stored) and str(path) == str(Path(stored).expanduser()),
        "ctx": resolve_ctx(ctx),
        "ctx_default": DEFAULT_CTX,
        "ctx_min": MIN_CTX,
        "ctx_max": MAX_CTX,
        "gpu": gpu_status(),
    }


_THINK_OPEN = ("<|channel>thought", chr(60) + "think" + chr(62))
_THINK_CLOSE = ("<channel|>", chr(60) + chr(47) + "think" + chr(62))


class ThoughtStream:
    """Turn raw model deltas into clean (phase, text) pieces.

    Gemma 4 wraps its reasoning in a `<|channel>thought … <channel|>` block and
    other instruct models use ` thinking … </think>`. The markers arrive split
    across deltas (e.g. "<|chan" then "nel>thought"), so a trailing partial
    marker is held back until it can be classified instead of being shown as
    raw markup.
    """

    THINKING = "thinking"
    ANSWER = "answer"

    def __init__(self):
        self.phase = self.ANSWER
        self._buf = ""

    def feed(self, delta: str) -> list[tuple[str, str]]:
        self._buf += delta
        return self._drain(final=False)

    def flush(self) -> list[tuple[str, str]]:
        return self._drain(final=True)

    def _drain(self, final: bool) -> list[tuple[str, str]]:
        events: list[tuple[str, str]] = []
        while self._buf:
            markers = _THINK_CLOSE if self.phase == self.THINKING else _THINK_OPEN
            hit = None
            for m in markers:
                i = self._buf.find(m)
                if i >= 0 and (hit is None or i < hit[0]):
                    hit = (i, m)
            if hit:
                i, marker = hit
                if i:
                    events.append((self.phase, self._buf[:i]))
                self._buf = self._buf[i + len(marker):]
                self.phase = self.ANSWER if self.phase == self.THINKING else self.THINKING
                continue
            keep = 0
            if not final:
                for m in markers:
                    for k in range(min(len(m) - 1, len(self._buf)), 0, -1):
                        if self._buf.endswith(m[:k]):
                            keep = max(keep, k)
                            break
            emit = self._buf[: len(self._buf) - keep] if keep else self._buf
            if emit:
                events.append((self.phase, emit))
                self._buf = self._buf[len(emit):]
            break
        return events


class LLMProvider:
    name = "abstract"

    def generate(self, system: str, user: str, *, max_tokens: int = 512,
                 temperature: float = 0.2, on_token=None) -> str:
        raise NotImplementedError


class EchoProvider(LLMProvider):
    name = "echo"

    def generate(self, system: str, user: str, *, max_tokens: int = 512,
                 temperature: float = 0.2, on_token=None) -> str:
        refs = re.findall(r"\[(S\d+)\]", user)
        if not refs:
            return "I could not find this in the loaded books."
        cited = ", ".join(f"[{r}]" for r in refs[:2])
        text = f"Echo answer grounded in the excerpts {cited}."
        if on_token:
            on_token(text, ThoughtStream.ANSWER)
        return text


class LlamaCppProvider(LLMProvider):
    name = "llama-cpp"
    context_size = 6144

    def __init__(self, model_path: str | None = None, *, n_ctx: int | None = None,
                 stored: str | None = None, stored_ctx=None):
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
        self.context_size = resolve_ctx(stored_ctx, n_ctx)
        self.gpu_layers = self._gpu_layers(llama_supports_gpu_offload)
        kwargs = dict(
            model_path=self.model_path,
            n_ctx=self.context_size,
            n_threads=threads,
            n_batch=512,
            n_gpu_layers=self.gpu_layers,
            verbose=False,
        )
        # Flash attention cuts prompt-processing time on both CPU and GPU
        # builds; older builds reject the flag, so fall back quietly.
        try:
            self._llm = Llama(flash_attn=True, **kwargs)
        except (TypeError, ValueError):
            self._llm = Llama(**kwargs)
        except RuntimeError:
            # The GPU could not take the model (driver, VRAM, backend). Better a
            # slow answer on the CPU than a hard failure.
            if not self.gpu_layers:
                raise
            self.gpu_layers = 0
            self._llm = Llama(**dict(kwargs, n_gpu_layers=0))
        self.gpu = gpu_status(self.gpu_layers)
        self._chat_ok = True
        # Gemma 4 only reasons when its template is told to, and
        # llama-cpp-python 0.3.x cannot pass template kwargs, so its prompt is
        # built by hand below with the model's own think control token.
        arch = ""
        try:
            arch = str(self._llm.metadata.get("general.architecture") or "")
        except Exception:
            pass
        self.thinking = arch.startswith("gemma")

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

    def _messages(self, system: str, user: str) -> list[dict]:
        # The model's own chat template decides how thinking is enabled; the
        # answer model always reasons before answering, so no soft switch is
        # appended to the prompt here.
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def _chatml(self, system: str, user: str) -> str:
        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def _gemma_prompt(self, system: str, user: str) -> str:
        """Gemma 4 turn format with thinking turned on."""
        think = chr(60) + "|think|" + chr(62)
        nl = chr(10)
        return (
            "<|turn>system" + nl + think + system + "<turn|>" + nl
            + "<|turn>user" + nl + user + "<turn|>" + nl
            + "<|turn>model" + nl
        )

    def _completion(self, system: str, user: str, *, max_tokens: int,
                    temperature: float, stream: bool):
        """Chat completion via the model's own template, with a ChatML fallback."""
        if self.thinking:
            return self._llm.create_completion(
                prompt=self._gemma_prompt(system, user),
                max_tokens=max_tokens,
                temperature=temperature,
                repeat_penalty=1.05,
                stop=list(_STOP) + ["<turn|>"],
                stream=stream,
            )
        if self._chat_ok:
            try:
                return self._llm.create_chat_completion(
                    messages=self._messages(system, user),
                    max_tokens=max_tokens,
                    temperature=temperature,
                    repeat_penalty=1.05,
                    stop=_STOP,
                    stream=stream,
                )
            except Exception:
                self._chat_ok = False
        return self._llm.create_completion(
            prompt=self._chatml(system, user),
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
                 temperature: float = 0.2, on_token=None) -> str:
        if self.thinking:
            # Reasoning is emitted before the answer; give it room.
            max_tokens = max(max_tokens, 768)
        if on_token is None:
            out = self._completion(system, user, max_tokens=max_tokens,
                                   temperature=temperature, stream=False)
            raw = self._text_of(out)
            stream = ThoughtStream()
            events = stream.feed(raw) + stream.flush()
            answer = "".join(t for phase, t in events if phase == ThoughtStream.ANSWER)
            return self._strip_think(answer.strip() or raw.strip())

        stream = ThoughtStream()
        answer: list[str] = []

        def emit(events):
            for phase, text in events:
                if phase == ThoughtStream.ANSWER:
                    answer.append(text)
                try:
                    on_token(text, phase)
                except Exception:
                    pass

        for part in self._completion(system, user, max_tokens=max_tokens,
                                     temperature=temperature, stream=True):
            delta = self._delta_of(part)
            if delta:
                emit(stream.feed(delta))
        emit(stream.flush())
        return self._strip_think("".join(answer).strip())

    @staticmethod
    def _strip_think(text: str) -> str:
        """Keep only the answer when the model writes its reasoning first.

        Thinking is always on and the streamed tokens show it; this only tidies
        the final text before citations are parsed out of it. Gemma 4 marks the
        reasoning channel with <channel|>, other instruct models close a reasoning
        block with </think>.
        """
        for marker in ('<channel|>', '</think>'):
            if marker in text:
                text = text.rsplit(marker, 1)[1]
        return re.sub(r"<\|channel>thought\s*", "", text).strip()


def get_llm(kind: str | None = None, *, model_path: str | None = None,
            stored: str | None = None, stored_ctx=None) -> LLMProvider:
    """kind: 'auto' (llama-cpp, else error), 'llama-cpp', 'echo' (tests only)."""
    kind = (kind or os.environ.get("HADITH_LLM") or "auto").lower()
    if kind == "echo":
        return EchoProvider()
    if kind in ("llama-cpp", "llama_cpp", "auto"):
        return LlamaCppProvider(model_path=model_path, stored=stored, stored_ctx=stored_ctx)
    raise ValueError(f"unknown LLM provider: {kind}")
