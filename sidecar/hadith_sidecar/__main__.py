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
from .qa import continue_answer as qa_continue
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


def _api_info() -> dict:
    """The user's own OpenAI-compatible endpoint, without ever the key."""
    from . import secrets as secret_store

    conn = _conn()
    enabled = db.get_meta(conn, "api_mode") == "1"
    base_url = db.get_meta(conn, "api_base_url") or ""
    model = db.get_meta(conn, "api_model") or ""
    has_key = secret_store.has(_state["library"], "openai")
    return {
        "enabled": enabled,
        "base_url": base_url,
        "model": model,
        "has_key": has_key,
        "ready": bool(enabled and base_url and model and has_key),
    }


def _llm_settings() -> dict:
    conn = _conn()
    settings = {
        "stored": db.get_meta(conn, "llm_path"),
        "stored_ctx": db.get_meta(conn, "llm_ctx"),
        "stored_device": db.get_meta(conn, "llm_device"),
        "stored_kv": db.get_meta(conn, "llm_kv"),
    }
    if db.get_meta(conn, "api_mode") == "1":
        from . import secrets as secret_store

        settings.update({
            "kind": "openai",
            "api_base_url": db.get_meta(conn, "api_base_url") or "",
            "api_model": db.get_meta(conn, "api_model") or "",
            "api_key": secret_store.load(_state["library"], "openai") or "",
        })
    return settings


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
        "continue_answer": _continue_answer,
        "thread_state": _thread_state,
        "list_chats": _list_chats,
        "new_chat": _new_chat,
        "open_chat": _open_chat,
        "rename_chat": _rename_chat,
        "delete_chat": _delete_chat,
        "reset_thread": _reset_thread,
        "compact_thread": _compact_thread,
        "rewind_thread": _rewind_thread,
        "read_narration": _read_narration,
        "locate_narration": _locate_narration,
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
        "add_folder": _add_folder,
        "rename_folder": _rename_folder,
        "delete_folder": _delete_folder,
        "move_chat": _move_chat,
        "daily_notify": _daily_notify,
        "api_config": _api_config,
        "set_api_config": _set_api_config,
        "set_api_key": _set_api_key,
        "clear_api_key": _clear_api_key,
        "test_api": _test_api,
        "set_daily_notify": _set_daily_notify,
        "mark_daily_notify": _mark_daily_notify,
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


def _create_chat(folder_id: str | None = None) -> dict:
    from .thread import new_conversation

    import uuid

    chat = new_conversation(chat_id=f"c{uuid.uuid4().hex[:10]}")
    chat["folder_id"] = folder_id
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


def _folder_list() -> list[dict]:
    return db.list_folders(_conn())


def _add_folder(params):
    """A named drawer for conversations."""
    import uuid

    name = " ".join(str(params.get("name") or "").split()) or "New folder"
    conn = _conn()
    folder_id = f"f{uuid.uuid4().hex[:10]}"
    db.insert_folder(
        conn, folder_id, name[:60], datetime.datetime.now(datetime.timezone.utc).isoformat()
    )
    conn.commit()
    return {"folders": _folder_list(), "folder": folder_id}


def _rename_folder(params):
    folder_id = str(params.get("id") or "").strip()
    name = " ".join(str(params.get("name") or "").split())
    if not folder_id:
        raise ValueError("params.id is required")
    if not name:
        raise ValueError("a folder needs a name")
    conn = _conn()
    if not db.rename_folder(conn, folder_id, name[:60]):
        raise ValueError(f"no such folder: {folder_id}")
    conn.commit()
    return {"folders": _folder_list()}


def _delete_folder(params):
    """Removing a folder never removes the conversations inside it."""
    folder_id = str(params.get("id") or "").strip()
    if not folder_id:
        raise ValueError("params.id is required")
    conn = _conn()
    db.delete_folder(conn, folder_id)
    conn.commit()
    for chat in _chats().values():
        if chat.get("folder_id") == folder_id:
            chat["folder_id"] = None
    return {"folders": _folder_list(), "chats": _chat_list(), "folders": _folder_list()}


def _move_chat(params):
    """File a conversation under a folder, or back out of one."""
    chat_id = str(params.get("id") or "").strip()
    raw = params.get("folder_id")
    folder_id = str(raw).strip() if raw not in (None, "") else None
    if not chat_id:
        raise ValueError("params.id is required")
    chat = _chats().get(chat_id)
    if chat is None:
        raise ValueError(f"no such chat: {chat_id}")
    conn = _conn()
    if folder_id is not None and folder_id not in {f["id"] for f in db.list_folders(conn)}:
        raise ValueError(f"no such folder: {folder_id}")
    chat["folder_id"] = folder_id
    db.save_chat(conn, chat)
    db.set_chat_folder(conn, chat_id, folder_id)
    conn.commit()
    return {"chats": _chat_list(), "folders": _folder_list(), "folders": _folder_list()}


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


def _last_citations(conv: dict) -> tuple[str | None, list[int]]:
    """The passages the reader is looking at: "this chapter", "these two"."""
    for turn in reversed(conv.get("turns") or []):
        cites = turn.get("citations") or []
        if not cites:
            continue
        chapter = next((str(c["chapter"]) for c in cites if c.get("chapter")), None)
        chunk_ids: list[int] = []
        for cite in cites:
            cid = cite.get("chunk_id")
            if cid is not None and int(cid) not in chunk_ids:
                chunk_ids.append(int(cid))
        return chapter, chunk_ids
    return None, []


def _ask(params):
    query = (params.get("query") or "").strip()
    if len(query) < 2:
        raise ValueError("params.query must be at least 2 characters")
    book_ids = params.get("book_ids")
    if book_ids is not None:
        book_ids = [int(b) for b in book_ids]
    pinned_chunk_ids = params.get("pinned_chunk_ids")
    if pinned_chunk_ids is not None:
        pinned_chunk_ids = [int(c) for c in pinned_chunk_ids]
    ask_id = params.get("ask_id")
    from . import protocol
    from . import thread as thread_mod
    from .intent import classify
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

    # Read the question once, here: it decides how many excerpts the window
    # can afford and what shape the answer should take.
    history_chapter, history_chunk_ids = _last_citations(conv)
    intent = classify(query, history_chapter=history_chapter)

    result = qa_ask(
        _conn(), query,
        book_ids=book_ids,
        embedder=_embedder(),
        llm=llm,
        reranker=_reranker(),
        on_token=on_token if ask_id else None,
        ask_id=ask_id,
        context=thread_mod.history_block(conv),
        excerpt_chars=thread_mod.excerpt_chars(
            llm, conv, ctx, MAX_EXCERPT_CHARS, top_k=intent.top_k
        ),
        history_chapter=history_chapter,
        history_chunk_ids=history_chunk_ids,
        pinned_chunk_ids=pinned_chunk_ids,
        intent=intent,
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
    return {"chats": _chat_list(), "folders": _folder_list(), "active": _state.get("active_chat")}


def _continue_answer(params):
    """Write the rest of an answer the token limit cut off.

    The turn keeps its question and its citations, so the continuation stays
    grounded in the same passages and [S#] keeps meaning what it did.
    """
    from . import thread as thread_mod

    conv = _thread()
    turn_id = params.get("turn_id")
    turn = next((t for t in conv["turns"] if str(t["id"]) == str(turn_id)), None)
    if turn is None:
        raise ValueError(f"no such turn: {turn_id}")
    if not turn.get("truncated"):
        raise ValueError("this answer was not cut off")
    if (turn.get("intent") or {}).get("kind") == "ruling":
        # A ruling must come whole from the gate that checked its citations.
        raise ValueError("a ruling answer is not continued; ask the question again")

    llm = _llm()
    result = qa_continue(
        question=turn["question"],
        partial=turn["answer"],
        citations=turn["citations"] or [],
        llm=llm,
        intent=turn.get("intent"),
    )
    thread_mod.append_answer(
        conv, turn["id"], result["answer"], truncated=result["truncated"]
    )
    _persist_chat(conv)
    info = thread_mod.usage(llm, conv, _context_size(llm))
    result["turn"] = turn
    result["thread"] = thread_mod.snapshot(conv, info)
    result["chats"] = _chat_list()
    return result


def _thread_state(_params):
    """The open conversation as it stands, so a view can redraw without asking."""
    conv = _thread()
    return {"thread": _chat_snapshot(conv), "chats": _chat_list(), "folders": _folder_list()}


def _new_chat(params):
    """Start a conversation in the folder asked for, or outside every folder.

    An untouched conversation is reused only when it already lives where the
    new one was asked for. Pressing "+" inside a folder must not hand back the
    blank chat filed there, and a new chat must never land in a folder the
    reader did not choose.
    """
    folder_id = params.get("folder_id") or None
    if folder_id is not None:
        folder_id = str(folder_id)
        if not any(f["id"] == folder_id for f in _folder_list()):
            raise ValueError(f"no such folder: {folder_id}")
    conv = _thread()
    if conv["turns"] or (conv.get("folder_id") or None) != folder_id:
        conv = _create_chat(folder_id)
    return {"thread": _chat_snapshot(conv), "chats": _chat_list(), "folders": _folder_list()}


def _open_chat(params):
    chat_id = params.get("id")
    if not chat_id:
        raise ValueError("params.id is required")
    chat = _chats().get(str(chat_id))
    if chat is None:
        raise ValueError(f"no such chat: {chat_id}")
    _state["active_chat"] = chat["id"]
    _persist_chat(chat)
    return {"thread": _chat_snapshot(chat), "chats": _chat_list(), "folders": _folder_list()}


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
    return {"chats": _chat_list(), "folders": _folder_list()}


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
    return {"deleted": deleted, "thread": _chat_snapshot(conv), "chats": _chat_list(), "folders": _folder_list()}


def _reset_thread(_params):
    """Start a fresh conversation."""
    conv = _create_chat()
    return {"reset": True, "thread": _chat_snapshot(conv), "chats": _chat_list(), "folders": _folder_list()}


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


def _read_narration(params):
    """The full text behind a "same narration" cross-reference."""
    from . import crossref

    collection = str(params.get("collection") or "").strip()
    number = params.get("number")
    if not collection or number is None:
        raise ValueError("params.collection and params.number are required")
    found = crossref.narration(_conn(), collection, int(number))
    if found is None:
        raise ValueError(f"no such narration: {collection} {number}")
    return {"narration": found}


def _locate_narration(params):
    """Point a cross-reference at the page it occupies in the imported book."""
    from . import crossref

    collection = str(params.get("collection") or "").strip()
    number = params.get("number")
    if not collection or number is None:
        raise ValueError("params.collection and params.number are required")
    conn = _conn()
    if crossref.narration(conn, collection, int(number)) is None:
        raise ValueError(f"no such narration: {collection} {number}")
    return {"place": crossref.locate(conn, collection, int(number))}


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
        "chats": _chat_list(), "folders": _folder_list(),
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
    info["api"] = _api_info()
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


def _api_config(_params):
    return {"api": _api_info(), "llm_active": _llm_info()}


def _set_api_config(params):
    """Switch answers to a user endpoint and remember where it lives."""
    conn = _conn()
    if "enabled" in params:
        db.set_meta(conn, "api_mode", "1" if params.get("enabled") else "0")
    if "base_url" in params:
        url = str(params.get("base_url") or "").strip().rstrip("/")
        if url and not url.lower().startswith(("http://", "https://")):
            raise ValueError("the endpoint must start with http:// or https://")
        db.set_meta(conn, "api_base_url", url)
    if "model" in params:
        db.set_meta(conn, "api_model", str(params.get("model") or "").strip())
    conn.commit()
    _state["llm"] = None  # rebuild the provider on the next question
    _state["llm_gen"] += 1
    return {"api": _api_info(), "llm_active": _llm_info()}


def _set_api_key(params):
    """Store the key with DPAPI; it is never returned to the UI."""
    from . import secrets as secret_store

    key = str(params.get("key") or "").strip()
    if not key:
        raise ValueError("an API key is required")
    secret_store.save(_state["library"], "openai", key)
    _state["llm"] = None
    _state["llm_gen"] += 1
    return {"api": _api_info()}


def _clear_api_key(_params):
    from . import secrets as secret_store

    secret_store.clear(_state["library"], "openai")
    _state["llm"] = None
    _state["llm_gen"] += 1
    return {"api": _api_info()}


def _test_api(params):
    """One tiny request, so Settings can say whether the key works."""
    from . import secrets as secret_store
    from .llm import test_endpoint

    conn = _conn()
    base = str(params.get("base_url") or db.get_meta(conn, "api_base_url") or "").strip()
    model = str(params.get("model") or db.get_meta(conn, "api_model") or "").strip()
    key = str(params.get("key") or "").strip()
    if not key:
        key = secret_store.load(_state["library"], "openai") or ""
    if not base:
        raise ValueError("an endpoint URL is required")
    if not model:
        raise ValueError("a model name is required")
    return {"test": test_endpoint(base, model, key)}


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
    api = _api_info()
    if api["enabled"]:
        # The picker offers whichever writer is actually answering; switching
        # back to a local file is one click away, in the same list.
        for entry in entries:
            entry["selected"] = False
        entries.insert(0, {
            "path": "api:openai",
            "name": f"Your endpoint · {api['model'] or 'unnamed model'}",
            "size": 0,
            "installed": bool(api["model"]),
            "default": False,
            "selected": True,
            "kind": "api",
        })
        return {"llms": entries, "active": "api:openai"}
    return {"llms": entries, "active": str(active)}


def _set_llm(params):
    """Configure the answer model: a local GGUF path and/or its context window."""
    from .llm import MAX_CTX, MIN_CTX

    conn = _conn()
    changed = False

    if "path" in params:
        path = params.get("path")
        if str(path or "").strip() == "api:openai":
            # Choosing the endpoint in the model picker switches to it; the
            # local model choice is remembered and comes back when reselected.
            db.set_meta(conn, "api_mode", "1")
            conn.commit()
            _state["llm"] = None
            _state["llm_gen"] += 1
            return {"llm_active": _llm_info()}
        if db.get_meta(conn, "api_mode") == "1" and path is not None:
            db.set_meta(conn, "api_mode", "0")
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


NOTIFY_TIME = "07:00"


def _parse_time(raw) -> str:
    text = str(raw or "").strip()
    parts = text.split(":")
    if len(parts) != 2 or not all(p.isdigit() for p in parts):
        raise ValueError("time must be HH:MM, 24-hour")
    hour, minute = (int(p) for p in parts)
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError("time must be HH:MM, 24-hour")
    return f"{hour:02d}:{minute:02d}"


def _notify_state(conn) -> dict:
    """The daily-hadith reminder: when it should land, and whether it is due.

    A reminder is due once the day's time has passed and that day has not been
    shown yet, so a machine that was off at 07:00 still gets it when it wakes
    up rather than missing the day entirely.
    """
    enabled = db.get_meta(conn, "daily_notify_enabled") == "1"
    time = db.get_meta(conn, "daily_notify_time") or NOTIFY_TIME
    last = db.get_meta(conn, "daily_notify_last") or None
    now = datetime.datetime.now()
    today = now.date().isoformat()
    due = False
    if enabled and last != today:
        hour, minute = (int(part) for part in time.split(":"))
        due = now >= now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    state = {
        "enabled": enabled,
        "time": time,
        "last_shown": last,
        "due": due,
        "date": today,
        "hadith": None,
    }
    if due:
        from . import crossref

        state["hadith"] = crossref.daily(conn, today)
    return state


def _daily_notify(params):
    return _notify_state(_conn())


def _set_daily_notify(params):
    conn = _conn()
    if "enabled" in params:
        db.set_meta(conn, "daily_notify_enabled", "1" if params.get("enabled") else "0")
    if "time" in params:
        db.set_meta(conn, "daily_notify_time", _parse_time(params.get("time")))
    conn.commit()
    return _notify_state(conn)


def _mark_daily_notify(params):
    conn = _conn()
    raw = str(params.get("date") or "").strip()
    if raw:
        try:
            day = datetime.date.fromisoformat(raw).isoformat()
        except ValueError:
            raise ValueError("date must be an ISO date (YYYY-MM-DD)")
    else:
        day = datetime.date.today().isoformat()
    db.set_meta(conn, "daily_notify_last", day)
    conn.commit()
    return _notify_state(conn)


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
