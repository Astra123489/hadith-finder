"""API keys, kept out of the library file.

A key is encrypted with Windows DPAPI (per-user, per-machine) and written to a
small JSON file next to `library.haddb` rather than into the SQLite library
itself, so a copied or backed-up library never carries someone's credentials.
Only the ciphertext is stored; the plaintext exists in memory while a request
is in flight.
"""

from __future__ import annotations

import base64
import ctypes
import json
import sys
from ctypes import wintypes
from pathlib import Path


class _Blob(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]


def _blob(data: bytes) -> tuple[_Blob, ctypes.Array]:
    # Keep the buffer alive for as long as the blob points at it.
    buffer = ctypes.create_string_buffer(data)
    return _Blob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_char))), buffer


def _require_windows() -> None:
    if not sys.platform.startswith("win"):
        raise RuntimeError("API keys are stored with Windows DPAPI; this build is Windows-only")


def protect(text: str) -> str:
    """Encrypt for the current Windows user and return base64 ciphertext."""
    _require_windows()
    data, _keep = _blob(text.encode("utf-8"))
    out = _Blob()
    ok = ctypes.windll.crypt32.CryptProtectData(
        ctypes.byref(data), None, None, None, None, 0, ctypes.byref(out)
    )
    if not ok:
        raise OSError("CryptProtectData failed")
    try:
        return base64.b64encode(ctypes.string_at(out.pbData, out.cbData)).decode("ascii")
    finally:
        ctypes.windll.kernel32.LocalFree(out.pbData)


def unprotect(encoded: str) -> str:
    """Decrypt a value produced by protect() on the same Windows account."""
    _require_windows()
    data, _keep = _blob(base64.b64decode(encoded))
    out = _Blob()
    ok = ctypes.windll.crypt32.CryptUnprotectData(
        ctypes.byref(data), None, None, None, None, 0, ctypes.byref(out)
    )
    if not ok:
        raise OSError("CryptUnprotectData failed")
    try:
        return ctypes.string_at(out.pbData, out.cbData).decode("utf-8")
    finally:
        ctypes.windll.kernel32.LocalFree(out.pbData)


def store_path(library: str | Path) -> Path:
    return Path(library).with_suffix(Path(library).suffix + ".keys.json")


def _read(library: str | Path) -> dict:
    path = store_path(library)
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write(library: str | Path, data: dict) -> None:
    path = store_path(library)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not data:
        try:
            path.unlink()
        except OSError:
            pass
        return
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def save(library: str | Path, name: str, value: str) -> None:
    value = (value or "").strip()
    if not value:
        clear(library, name)
        return
    data = _read(library)
    data[name] = protect(value)
    _write(library, data)


def load(library: str | Path, name: str) -> str | None:
    raw = _read(library).get(name)
    if not raw:
        return None
    try:
        return unprotect(raw)
    except Exception:
        # A key written by another Windows account cannot be read; treat it as
        # absent rather than failing the whole app.
        return None


def clear(library: str | Path, name: str) -> None:
    data = _read(library)
    if data.pop(name, None) is not None:
        _write(library, data)


def has(library: str | Path, name: str) -> bool:
    return bool(_read(library).get(name))
