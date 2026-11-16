"""Entry point:  python -m hadith_sidecar [library.haddb]

Library path resolution order: argv[1] → $HADITH_LIBRARY → per-user default
(%LOCALAPPDATA%/HadithFinder/library.haddb on Windows, else ~/.hadith-finder/library.haddb).
"""

from __future__ import annotations

import datetime
import os
import sys
import threading
from pathlib import Path

from . import __version__, db
from .embeddings import get_embedder
from .llm import get_llm
from .qa import ask as qa_ask
from .rerank import get_reranker

_state: dict = {
    "conn": None,
    "embedder": None,
    "llm": None,
    "reranker": None,
    "llm_gen": 0,
    # Conversations, owned here so the prompt and the transcript always agree.
    # Loaded from the library file on first use; one of them is active.
    "chats": None,
    "active_chat": None,
}

# Loading llama.cpp weights takes seconds to minutes; one loader at a time.
_llm_lock = threading.Lock()


def _default_library() -> Path:
    if sys.platform.startswith("win"):
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local"))
        return base / "HadithFinder" / "library.haddb"
    return Path.home() / ".hadith-finder" / "library.haddb"


def _conn():
    if _state["conn"] is None:
        path = Path(_state["library"]).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        _state["conn"] = db.connect(path)
    return _state["conn"]


def _embedder():
    if _state["embedder"] is None:
        _state["embedder"] = get_embedder()
    return _state["embedder"]


def _llm_settings() -> dict:
    conn = _conn()
    return {
        "stored": db.get_meta(conn, "llm_path"),
        "stored_ctx": db.get_meta(conn, "llm_ctx"),
        "stored_device": db.get_meta(conn, "llm_device"),
        "stored_kv": db.get_meta(conn, "llm_kv"),
    }


def _llm():
    if _state["llm"] is not None:
        return _state["llm"]
    settings = _llm_settings()
    with _llm_lock:
        if _state["llm"] is None:
            _state["llm"] = get_llm(**settings)
        return _state["llm"]


def _preload_llm() -> None:
    """Load the selected model at launch when the user asked for it.

    Runs on a daemon thread so the window is usable immediately; the Ask view
    receives progress events and the first question finds the model warm.
    """
    if db.get_meta(_conn(), "llm_auto_load") != "1":
        return
    settings = _llm_settings()
    generation = _state["llm_gen"]

    def worker() -> None:
        from . import protocol

        protocol.notify(op="llm", stage="loading")
        try:
            with _llm_lock:
                if _state["llm_gen"] != generation or _state["llm"] is not None:
                    return
                llm = get_llm(**settings)
                _state["llm"] = llm
            protocol.notify(
                op="llm",
                stage="ready",
                model=Path(getattr(llm, "model_path", "")).stem,
                gpu_layers=getattr(llm, "gpu_layers", 0),
                device=getattr(llm, "device", "auto"),
                kv=getattr(llm, "kv_cache", "f16"),
                kv_applied=getattr(llm, "kv_applied", False),
            )
        except Exception as exc:  # a missing model must not kill the app
            protocol.notify(op="llm", stage="error", message=str(exc))

    threading.Thread(target=worker, name="llm-preload", daemon=True).start()


def _reranker():
    if _state["reranker"] is None:
        _state["reranker"] = get_reranker()
    return _state["reranker"]


def dispatch(method: str, params: dict):
    handlers = {
        "ping": _ping,
        "import_book": _import_book,
        "rebuild_book": _rebuild_book,
        "reindex_book": _reindex_book,
        "list_books": _list_books,
        "delete_book": _delete_book,
        "stats": _stats,
        "ask": _ask,
        "thread_state": _thread_state,
        "list_chats": _list_chats,
        "new_chat": _new_chat,
        "open_chat": _open_chat,
        "rename_chat": _rename_chat,
        "delete_chat": _delete_chat,
        "reset_thread": _reset_thread,
        "compact_thread": _compact_thread,
        "rewind_thread": _rewind_thread,
        "suggest_followups": _suggest_followups,
        "models_status": _models_status,
        "set_llm": _set_llm,
        "list_llms": _list_llms,
        "download_model": _download_model,
        "list_bookmarks": _list_bookmarks,
        "add_bookmark": _add_bookmark,
        "delete_bookmark": _delete_bookmark,
        "update_bookmark": _update_bookmark,
        "daily_hadith": _daily_hadith,
        "shutdown": _shutdown,
    }
    if method not in handlers:
        raise KeyError(method)
    return handlers[method](params)


def _ping(_params):
    return {
        "pong": True,
        "version": __version__,
        "library": str(_state["library"]),
        "ocr": _ocr_status(),
        "onnx": _onnx_status(),
    }


def _onnx_status() -> dict:
    from .embeddings import ort_providers

    try:
        import onnxruntime as ort

        available = ort.get_available_providers()
    except Exception:
        available = []
    return {"providers": ort_providers(), "available": available}


def _ocr_status() -> dict:
    from . import ocr

    return {"mode": ocr.mode(), "available": ocr.available()}


def _import_book(params):
    path = params.get("path")
    if not path:
        raise ValueError("params.path is required")
    from . import protocol
    from .ingest import ingest_pdf

    return ingest_pdf(
        _conn(), path,
        title=params.get("title"), author=params.get("author"),
        progress=protocol.notify,
    )


def _rebuild_book(params):
    book_id = params.get("book_id")
    if book_id is None:
        raise ValueError("params.book_id is required")
    from . import protocol
    from .ingest import rebuild_book

    return rebuild_book(_conn(), int(book_id), progress=protocol.notify)


def _reindex_book(params):
    conn = _conn()
    book_id = params.get("book_id")
    batch = int(params.get("batch") or 16)
    if book_id is None:
        raise ValueError("params.book_id is required")
    book_id = int(book_id)
    embedder = _embedder()

    # Model change → drop stale vectors so dimensions never mix (applies to all books).
    stored = db.get_meta(conn, "embedder")
    model_changed = bool(stored) and stored != embedder.name
    if model_changed:
        with conn:
            conn.execute("DELETE FROM embeddings")
    if stored != embedder.name:
        db.set_meta(conn, "embedder", embedder.name)
        conn.commit()

    all_rows = list(conn.execute(
        "SELECT c.* FROM chunks c LEFT JOIN embeddings e ON e.chunk_id = c.id "
        "WHERE e.chunk_id IS NULL ORDER BY c.book_id, c.chunk_index"
    ))
    # Scope: after a model switch everything needs re-embedding; otherwise just this book's gaps.
    todo = all_rows if model_changed else [r for r in all_rows if r["book_id"] == book_id]
    return _embed_rows(conn, todo, embedder, book_id=book_id, batch=batch)


def _embed_rows(conn, todo, embedder, *, book_id: int, batch: int = 16) -> dict:
    from . import protocol

    total = len(todo)
    done = 0
    if total == 0:
        return {"book_id": book_id, "embedded": 0, "total": 0}
    protocol.notify(op="reindex", book_id=book_id, done=0, total=total)
    for i in range(0, total, batch):
        rows = todo[i:i + batch]
        vecs = embedder.embed([r["norm_text"] for r in rows])
        with conn:
            for r, v in zip(rows, vecs):
                db.set_embedding(conn, r["id"], v)
        done += len(rows)
        protocol.notify(op="reindex", book_id=book_id, done=done, total=total)
    return {"book_id": book_id, "embedded": total, "total": total}


def _list_books(_params):
    return {"books": db.list_books(_conn())}


def _delete_book(params):
    book_id = params.get("book_id")
    if book_id is None:
        raise ValueError("params.book_id is required")
    ok = db.delete_book(_conn(), int(book_id))
    _conn().commit()
    return {"deleted": ok}


def _stats(_params):
    conn = _conn()
    books = len(db.list_books(conn))
    chunks = conn.execute("SELECT COUNT(*) c FROM chunks").fetchone()["c"]
    embedded = conn.execute("SELECT COUNT(*) c FROM embeddings").fetchone()["c"]
    return {"books": books, "chunks": chunks, "embedded": embedded}


def _chats() -> dict:
    """Every conversation, loaded from the library on first use."""
    store = _state.get("chats")
    if store is None:
        store = _state["chats"] = {}
        conn = _conn()
        for row in db.list_chats(conn):
            chat = db.load_chat(conn, row["id"])
            if chat:
                store[row["id"]] = chat
    return store


def _create_chat() -> dict:
    from .thread import new_conversation

    import uuid

    chat = new_conversation(chat_id=f"c{uuid.uuid4().hex[:10]}")
    _chats()[chat["id"]] = chat
    _state["active_chat"] = chat["id"]
    _persist_chat(chat)
    return chat


def _persist_chat(chat: dict) -> None:
    """Save the transcript and remember which chat was open."""
    conn = _conn()
    db.save_chat(conn, chat)
    db.set_meta(conn, "active_chat", chat["id"])
    conn.commit()


def _thread():
    """The active conversation, opening or creating one as needed."""
    store = _chats()
    active = _state.get("active_chat")
    if active in store:
        return store[active]
    remembered = db.get_meta(_conn(), "active_chat")
    if remembered in store:
        _state["active_chat"] = remembered
        return store[remembered]
    if store:
        newest = max(store.values(), key=lambda c: c.get("updated_at") or "")
        _state["active_chat"] = newest["id"]
        return newest
    return _create_chat()


def _chat_list() -> list[dict]:
    # Before the first request the active chat is only known from the library.
    active = _state.get("active_chat") or db.get_meta(_conn(), "active_chat")
    return [{**row, "active": row["id"] == active} for row in db.list_chats(_conn())]


def _chat_snapshot(chat: dict) -> dict:
    from . import thread as thread_mod

    llm = _state["llm"]  # never load weights just to measure the window
    return thread_mod.snapshot(chat, thread_mod.usage(llm, chat, _context_size(llm)))


def _context_size(llm=None) -> int:
    """The window the conversation has to work with.

    A loaded model knows its own size; otherwise the saved setting (or the
    default) is what the next load will use.
    """
    from .llm import resolve_ctx

    size = getattr(llm, "context_size", None) or resolve_ctx(
        db.get_meta(_conn(), "llm_ctx")
    )
    return max(256, int(size))


def _ask(params):
    query = (params.get("query") or "").strip()
    if len(query) < 2:
        raise ValueError("params.query must be at least 2 characters")
    book_ids = params.get("book_ids")
    if book_ids is not None:
        book_ids = [int(b) for b in book_ids]
    ask_id = params.get("ask_id")
    from . import protocol
    from . import thread as thread_mod
    from .qa import MAX_EXCERPT_CHARS

    def on_token(text: str, phase: str = "answer") -> None:
        protocol.notify(op="token", ask_id=ask_id, text=text, phase=phase)

    def on_thread(**fields) -> None:
        protocol.notify(op="thread", **fields)

    conv = _thread()
    if params.get("new_thread"):
        conv = _create_chat()

    llm = _llm()
    ctx = _context_size(llm)
    # Make room before asking: fold the oldest turns away until the prompt fits
    # comfortably, so a long conversation never overflows the window.
    for _ in range(8):
        if not thread_mod.needs_compaction(llm, conv, ctx):
            break
        if not thread_mod.compact(conv, llm, notify=on_thread)["compacted"]:
            break

    result = qa_ask(
        _conn(), query,
        book_ids=book_ids,
        embedder=_embedder(),
        llm=llm,
        reranker=_reranker(),
        on_token=on_token if ask_id else None,
        ask_id=ask_id,
        context=thread_mod.history_block(conv),
        excerpt_chars=thread_mod.excerpt_chars(llm, conv, ctx, MAX_EXCERPT_CHARS),
    )
    thread_mod.add_turn(conv, query, result)

    # Answering may have filled the window; compact now so the next follow-up
    # has room without waiting for a failure.
    excerpt_tokens = result.get("excerpt_tokens", 0) or 0
    info = thread_mod.usage(llm, conv, ctx, excerpt_tokens=excerpt_tokens)
    if info["used"] > info["compact_at"]:
        thread_mod.compact(conv, llm, notify=on_thread)
        info = thread_mod.usage(llm, conv, ctx, excerpt_tokens=excerpt_tokens)
    _persist_chat(conv)
    result["thread"] = thread_mod.snapshot(conv, info)
    result["chats"] = _chat_list()
    return result


def _list_chats(_params):
    # Opening the list also settles which chat is open, so "active" is real.
    _thread()
    return {"chats": _chat_list(), "active": _state.get("active_chat")}


def _thread_state(_params):
    """The open conversation as it stands, so a view can redraw without asking."""
    conv = _thread()
    return {"thread": _chat_snapshot(conv), "chats": _chat_list()}


def _new_chat(_params):
    """Start a fresh conversation. An untouched one is reused, not duplicated."""
    conv = _thread()
    if conv["turns"]:
        conv = _create_chat()
    return {"thread": _chat_snapshot(conv), "chats": _chat_list()}


def _open_chat(params):
    chat_id = params.get("id")
    if not chat_id:
        raise ValueError("params.id is required")
    chat = _chats().get(str(chat_id))
    if chat is None:
        raise ValueError(f"no such chat: {chat_id}")
    _state["active_chat"] = chat["id"]
    _persist_chat(chat)
    return {"thread": _chat_snapshot(chat), "chats": _chat_list()}


def _rename_chat(params):
    """Give a conversation the name the reader wants."""
    chat_id = params.get("id")
    title = " ".join(str(params.get("title") or "").split())
    if not chat_id:
        raise ValueError("params.id is required")
    if not title:
        raise ValueError("a conversation needs a name")
    chat = _chats().get(str(chat_id))
    if chat is None:
        raise ValueError(f"no such chat: {chat_id}")
    chat["title"] = title[:120]
    conn = _conn()
    db.save_chat(conn, chat)
    conn.commit()
    return {"chats": _chat_list()}


def _delete_chat(params):
    chat_id = params.get("id")
    if not chat_id:
        raise ValueError("params.id is required")
    chat_id = str(chat_id)
    conn = _conn()
    deleted = db.delete_chat(conn, chat_id)
    conn.commit()
    _chats().pop(chat_id, None)
    if _state.get("active_chat") == chat_id:
        _state["active_chat"] = None
    conv = _thread()
    _persist_chat(conv)
    return {"deleted": deleted, "thread": _chat_snapshot(conv), "chats": _chat_list()}


def _reset_thread(_params):
    """Start a fresh conversation."""
    conv = _create_chat()
    return {"reset": True, "thread": _chat_snapshot(conv), "chats": _chat_list()}


def _compact_thread(_params):
    """Summarise the older turns on the user's command."""
    from . import protocol
    from . import thread as thread_mod

    def on_thread(**fields) -> None:
        protocol.notify(op="thread", **fields)

    conv = _thread()
    llm = _llm()
    ctx = _context_size(llm)
    outcome = thread_mod.compact(conv, llm, notify=on_thread, force=True)
    _persist_chat(conv)
    info = thread_mod.usage(llm, conv, ctx)
    return {
        "compacted": outcome["compacted"],
        "dropped": outcome["dropped"],
        "thread": thread_mod.snapshot(conv, info),
    }


def _rewind_thread(params):
    """Drop a question and everything after it, so it can be retried or edited."""
    from . import thread as thread_mod

    conv = _thread()
    outcome = thread_mod.rewind(conv, params.get("turn_id"))
    _persist_chat(conv)
    llm = _state["llm"]  # never load weights just to measure the window
    info = thread_mod.usage(llm, conv, _context_size(llm))
    return {
        "dropped": outcome["dropped"],
        "question": outcome["question"],
        "thread": thread_mod.snapshot(conv, info),
        "chats": _chat_list(),
    }


def _suggest_followups(params):
    """Questions the last answer left open, phrased by the model when it can."""
    from . import thread as thread_mod

    conv = _thread()
    if not conv["turns"]:
        return {"suggestions": []}
    try:
        count = int(params.get("count") or 3)
    except (TypeError, ValueError):
        count = 3
    return {"suggestions": thread_mod.suggest(conv, _llm(), count)}


def _llm_info() -> dict:
    from .llm import active_model_info

    conn = _conn()
    info = active_model_info(
        db.get_meta(conn, "llm_path"),
        db.get_meta(conn, "llm_ctx"),
        device=db.get_meta(conn, "llm_device"),
        kv=db.get_meta(conn, "llm_kv"),
        auto_load=db.get_meta(conn, "llm_auto_load") == "1",
    )
    llm = _state["llm"]
    info["runtime"] = {
        "loaded": llm is not None,
        "model": Path(getattr(llm, "model_path", "")).stem if llm is not None else "",
        "gpu_layers": getattr(llm, "gpu_layers", 0),
        "device": getattr(llm, "device", "auto"),
        "kv": getattr(llm, "kv_cache", "f16"),
        "kv_applied": getattr(llm, "kv_applied", False),
        "flash_attn": getattr(llm, "flash_attn", False),
    }
    return info


def _models_status(_params):
    from .downloader import status

    out = status()
    out["ocr"] = _ocr_status()
    out["onnx"] = _onnx_status()
    out["llm_active"] = _llm_info()
    return out


def _list_llms(_params):
    """Answer models the user can switch to from the ask screen."""
    from .downloader import MODEL_CATALOG, default_llm_path, model_dir
    from .llm import resolve_model_path

    conn = _conn()
    stored = db.get_meta(conn, "llm_path")
    active = Path(resolve_model_path(stored))
    default = default_llm_path()
    entries = []
    seen = set()

    def add(path, label=None, is_default=False):
        p = Path(str(path)).expanduser()
        key = str(p).lower()
        if key in seen:
            return
        seen.add(key)
        exists = p.is_file()
        entries.append({
            "path": str(p),
            "name": label or p.stem,
            "size": p.stat().st_size if exists else 0,
            "installed": exists,
            "default": is_default,
            "selected": key == str(active).lower(),
        })

    add(default, MODEL_CATALOG["llm"]["name"], True)
    folder = model_dir("llm")
    if folder.is_dir():
        for gguf in sorted(folder.glob("*.gguf"), key=lambda q: q.name.lower()):
            add(gguf)
    if stored:
        add(stored)
    if not any(e["selected"] for e in entries):
        add(active)
    return {"llms": entries, "active": str(active)}


def _set_llm(params):
    """Configure the answer model: a local GGUF path and/or its context window."""
    from .llm import MAX_CTX, MIN_CTX

    conn = _conn()
    changed = False

    if "path" in params:
        path = params.get("path")
        if path is None or str(path).strip() == "":
            conn.execute("DELETE FROM meta WHERE key = 'llm_path'")
        else:
            candidate = Path(str(path)).expanduser()
            if not candidate.is_file():
                raise FileNotFoundError(f"no such model file: {candidate}")
            if candidate.suffix.lower() != ".gguf":
                raise ValueError(f"not a GGUF model file: {candidate.name}")
            db.set_meta(conn, "llm_path", str(candidate))
        changed = True

    if "ctx" in params:
        ctx = params.get("ctx")
        if ctx is None or str(ctx).strip() == "":
            conn.execute("DELETE FROM meta WHERE key = 'llm_ctx'")
        else:
            try:
                value = int(ctx)
            except (TypeError, ValueError):
                raise ValueError("context length must be a whole number of tokens")
            if not (MIN_CTX <= value <= MAX_CTX):
                raise ValueError(
                    f"context length must be between {MIN_CTX} and {MAX_CTX} tokens"
                )
            db.set_meta(conn, "llm_ctx", str((value // 256) * 256))
        changed = True

    if "device" in params:
        from .llm import DEVICES, resolve_device

        raw = params.get("device")
        if raw is None or str(raw).strip() == "":
            conn.execute("DELETE FROM meta WHERE key = 'llm_device'")
        else:
            value = str(raw).strip().lower()
            if value not in DEVICES:
                raise ValueError(f"device must be one of: {', '.join(DEVICES)}")
            db.set_meta(conn, "llm_device", resolve_device(value))
        changed = True

    if "kv" in params:
        from .llm import DEFAULT_KV, KV_CACHE_TYPES, resolve_kv

        raw = params.get("kv")
        if raw is None or str(raw).strip() == "":
            db.set_meta(conn, "llm_kv", DEFAULT_KV)
        else:
            value = str(raw).strip().lower()
            if value not in KV_CACHE_TYPES:
                raise ValueError(
                    f"kv cache must be one of: {', '.join(KV_CACHE_TYPES)}"
                )
            db.set_meta(conn, "llm_kv", resolve_kv(value))
        changed = True

    if "auto_load" in params:
        raw = params.get("auto_load")
        enabled = raw in (True, 1, "1", "true", "True", "yes")
        db.set_meta(conn, "llm_auto_load", "1" if enabled else "0")
        changed = True

    if changed:
        conn.commit()
        _state["llm"] = None  # reload on next question
        _state["llm_gen"] += 1
    return {"llm_active": _llm_info()}


def _list_bookmarks(_params):
    return {"bookmarks": db.list_bookmarks(_conn())}


def _add_bookmark(params):
    data = params.get("bookmark")
    if not isinstance(data, dict):
        raise ValueError("params.bookmark must be an object")
    conn = _conn()
    bookmark_id = db.add_bookmark(conn, data)
    conn.commit()
    return {"id": bookmark_id, "bookmarks": db.list_bookmarks(conn)}


def _delete_bookmark(params):
    bookmark_id = params.get("id")
    if bookmark_id is None:
        raise ValueError("params.id is required")
    conn = _conn()
    deleted = db.delete_bookmark(conn, int(bookmark_id))
    conn.commit()
    return {"deleted": deleted, "bookmarks": db.list_bookmarks(conn)}


def _update_bookmark(params):
    bookmark_id = params.get("id")
    if bookmark_id is None:
        raise ValueError("params.id is required")
    conn = _conn()
    updated = db.set_bookmark_description(conn, int(bookmark_id), params.get("description"))
    conn.commit()
    return {"updated": updated, "bookmarks": db.list_bookmarks(conn)}


def _daily_hadith(params):
    """The narration of the day from the bundled canonical collections."""
    from . import crossref

    raw = (params.get("date") or "").strip()
    if raw:
        try:
            day = datetime.date.fromisoformat(raw).isoformat()
        except ValueError:
            raise ValueError("date must be an ISO date (YYYY-MM-DD)")
    else:
        day = datetime.date.today().isoformat()
    return {"hadith": crossref.daily(_conn(), day)}


def _download_model(params):
    model_id = params.get("model")
    if not model_id:
        raise ValueError("params.model is required (llm | embedder | reranker)")
    from . import protocol
    from .downloader import download_model

    result = download_model(model_id, protocol.notify)
    # Pick up the newly installed model without restarting the app.
    _state["llm"] = None
    _state["embedder"] = None
    _state["reranker"] = None
    return result


def _shutdown(_params):
    conn = _state["conn"]
    if conn is not None:
        conn.close()
    sys.exit(0)


def main() -> int:
    _state["library"] = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("HADITH_LIBRARY") or str(_default_library())
    lib_path = Path(_state["library"]).expanduser().resolve()
    os.environ.setdefault("HADITH_LIBRARY", str(lib_path))
    os.environ.setdefault("HADITH_MODELS_DIR", str(lib_path.parent / "models"))
    from . import protocol
    return protocol.run(dispatch, on_ready=_preload_llm)


if __name__ == "__main__":
    raise SystemExit(main())
