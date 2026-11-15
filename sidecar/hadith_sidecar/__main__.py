"""Entry point:  python -m hadith_sidecar [library.haddb]

Library path resolution order: argv[1] → $HADITH_LIBRARY → per-user default
(%LOCALAPPDATA%/HadithFinder/library.haddb on Windows, else ~/.hadith-finder/library.haddb).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from . import __version__, db
from .embeddings import get_embedder
from .llm import get_llm
from .qa import ask as qa_ask
from .rerank import get_reranker

_state: dict = {"conn": None, "embedder": None, "llm": None, "reranker": None}


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


def _llm():
    if _state["llm"] is None:
        stored = db.get_meta(_conn(), "llm_path")
        ctx = db.get_meta(_conn(), "llm_ctx")
        _state["llm"] = get_llm(stored=stored, stored_ctx=ctx)
    return _state["llm"]


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
        "models_status": _models_status,
        "set_llm": _set_llm,
        "list_llms": _list_llms,
        "download_model": _download_model,
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
    }


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


def _ask(params):
    query = (params.get("query") or "").strip()
    if len(query) < 2:
        raise ValueError("params.query must be at least 2 characters")
    book_ids = params.get("book_ids")
    if book_ids is not None:
        book_ids = [int(b) for b in book_ids]
    ask_id = params.get("ask_id")

    def on_token(text: str) -> None:
        from . import protocol

        protocol.notify(op="token", ask_id=ask_id, text=text)

    return qa_ask(
        _conn(), query,
        book_ids=book_ids,
        embedder=_embedder(),
        llm=_llm(),
        reranker=_reranker(),
        on_token=on_token if ask_id else None,
        ask_id=ask_id,
    )


def _llm_info() -> dict:
    from .llm import active_model_info

    return active_model_info(db.get_meta(_conn(), "llm_path"), db.get_meta(_conn(), "llm_ctx"))


def _models_status(_params):
    from .downloader import status

    out = status()
    out["ocr"] = _ocr_status()
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

    if changed:
        conn.commit()
        _state["llm"] = None  # reload on next question
    return {"llm_active": _llm_info()}


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
    return protocol.run(dispatch)


if __name__ == "__main__":
    raise SystemExit(main())
