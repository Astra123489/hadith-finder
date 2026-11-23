"""SQLite storage: books, chunks, FTS5 index, embeddings, gradings.

The whole library lives in one `.haddb` file. FTS5 keeps the keyword arm of
hybrid retrieval dependency-free; dense vectors are scanned in NumPy
(a multi-book library is at most a few ×10^4 chunks — no need for a
vector DB server).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

SCHEMA_VERSION = "2"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS books(
  id INTEGER PRIMARY KEY,
  title TEXT NOT NULL,
  author TEXT,
  language TEXT NOT NULL DEFAULT 'mixed',
  file_path TEXT NOT NULL,
  sha256 TEXT NOT NULL UNIQUE,
  page_count INTEGER NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS chunks(
  id INTEGER PRIMARY KEY,
  book_id INTEGER NOT NULL REFERENCES books(id) ON DELETE CASCADE,
  chunk_index INTEGER NOT NULL,
  page_start INTEGER NOT NULL,
  page_end INTEGER NOT NULL,
  hadith_ref TEXT,
  chapter TEXT,
  text TEXT NOT NULL,
  norm_text TEXT NOT NULL,
  spans_json TEXT NOT NULL,
  UNIQUE(book_id, chunk_index)
);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
  norm_text,
  content='chunks',
  content_rowid='id'
);
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
  INSERT INTO chunks_fts(rowid, norm_text) VALUES (new.id, new.norm_text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
  INSERT INTO chunks_fts(chunks_fts, rowid, norm_text) VALUES ('delete', old.id, old.norm_text);
END;
CREATE TABLE IF NOT EXISTS embeddings(
  chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
  dim INTEGER NOT NULL,
  vector BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS gradings(
  id INTEGER PRIMARY KEY,
  chunk_id INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
  grading TEXT NOT NULL,
  raw TEXT NOT NULL,
  UNIQUE(chunk_id, grading, raw)
);
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS bookmarks(
  id INTEGER PRIMARY KEY,
  kind TEXT NOT NULL DEFAULT 'library',
  title TEXT NOT NULL,
  chapter TEXT,
  reference TEXT,
  collection TEXT,
  number INTEGER,
  page_start INTEGER,
  page_end INTEGER,
  excerpt TEXT NOT NULL,
  gradings TEXT,
  spans_json TEXT,
  book_id INTEGER,
  chunk_id INTEGER,
  description TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS chats(
  id TEXT PRIMARY KEY,
  title TEXT NOT NULL DEFAULT '',
  turns INTEGER NOT NULL DEFAULT 0,
  data TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_book ON chunks(book_id);
CREATE INDEX IF NOT EXISTS idx_gradings_chunk ON gradings(chunk_id);
CREATE INDEX IF NOT EXISTS idx_bookmarks_created ON bookmarks(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_chats_updated ON chats(updated_at DESC);
CREATE TABLE IF NOT EXISTS chat_folders(
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  created_at TEXT NOT NULL
);
"""


def connect(path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.executescript(_SCHEMA)
    # Libraries made before folders existed gain the column in place.
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(chats)").fetchall()}
    if "folder_id" not in columns:
        conn.execute("ALTER TABLE chats ADD COLUMN folder_id TEXT")
    conn.execute(
        "INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', ?)",
        (SCHEMA_VERSION,),
    )
    return conn


# ---------- books ----------

def insert_book(conn, *, title: str, author: str | None, language: str,
                file_path: str, sha256: str, page_count: int) -> int:
    cur = conn.execute(
        "INSERT INTO books(title, author, language, file_path, sha256, page_count) "
        "VALUES (?,?,?,?,?,?)",
        (title, author, language, file_path, sha256, page_count),
    )
    return int(cur.lastrowid)


def find_book_by_sha(conn, sha256: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM books WHERE sha256 = ?", (sha256,)).fetchone()


def get_book(conn, book_id: int) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM books WHERE id = ?", (book_id,)).fetchone()


def list_books(conn) -> list[dict]:
    rows = conn.execute(
        """
        SELECT b.*,
               COUNT(c.id)              AS chunk_count,
               (SELECT COUNT(*) FROM embeddings e
                  JOIN chunks c2 ON c2.id = e.chunk_id WHERE c2.book_id = b.id) AS embedded_count
        FROM books b LEFT JOIN chunks c ON c.book_id = b.id
        GROUP BY b.id ORDER BY b.created_at DESC
        """
    ).fetchall()
    return [dict(r) for r in rows]


def delete_book(conn, book_id: int) -> bool:
    cur = conn.execute("DELETE FROM books WHERE id = ?", (book_id,))
    return cur.rowcount > 0


# ---------- chunks ----------

def insert_chunk(conn, *, book_id: int, chunk_index: int, page_start: int,
                 page_end: int, hadith_ref: str | None, chapter: str | None,
                 text: str, norm_text: str, spans: dict[int, list[list[float]]]) -> int:
    spans_json = json.dumps({str(k): v for k, v in spans.items()}, ensure_ascii=False)
    cur = conn.execute(
        """INSERT INTO chunks(book_id, chunk_index, page_start, page_end,
                               hadith_ref, chapter, text, norm_text, spans_json)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (book_id, chunk_index, page_start, page_end, hadith_ref, chapter,
         text, norm_text, spans_json),
    )
    return int(cur.lastrowid)


def delete_chunks(conn, book_id: int) -> int:
    """Remove a book's chunks (and, by cascade, embeddings and gradings)."""
    cur = conn.execute("DELETE FROM chunks WHERE book_id = ?", (book_id,))
    return cur.rowcount


def get_chunks(conn, ids: list[int]) -> list[sqlite3.Row]:
    if not ids:
        return []
    q = f"SELECT * FROM chunks WHERE id IN ({','.join('?' * len(ids))})"
    rows = conn.execute(q, ids).fetchall()
    by_id = {r["id"]: r for r in rows}
    return [by_id[i] for i in ids if i in by_id]


def iter_book_chunks(conn, book_id: int):
    yield from conn.execute(
        "SELECT * FROM chunks WHERE book_id = ? ORDER BY chunk_index", (book_id,)
    )


def count_missing_embeddings(conn, book_id: int) -> int:
    row = conn.execute(
        "SELECT COUNT(*) c FROM chunks c LEFT JOIN embeddings e ON e.chunk_id = c.id"
        " WHERE c.book_id = ? AND e.chunk_id IS NULL",
        (book_id,),
    ).fetchone()
    return int(row["c"])


def load_vectors(conn, book_ids: list[int] | None = None):
    """Return (ids: list[int], matrix: np.ndarray float32, dim: int)."""
    import numpy as np

    sql = "SELECT e.chunk_id, e.dim, e.vector FROM embeddings e JOIN chunks c ON c.id = e.chunk_id"
    params: list = []
    if book_ids:
        sql += f" WHERE c.book_id IN ({','.join('?' * len(book_ids))})"
        params = list(book_ids)
    rows = conn.execute(sql, params).fetchall()
    if not rows:
        return [], np.zeros((0, 0), dtype=np.float32), 0
    dim = int(rows[0]["dim"])
    ids, vecs = [], []
    for r in rows:
        ids.append(int(r["chunk_id"]))
        vecs.append(np.frombuffer(r["vector"], dtype=np.float32, count=dim))
    return ids, np.vstack(vecs), dim


def set_embedding(conn, chunk_id: int, vector) -> None:
    import numpy as np

    v = np.ascontiguousarray(vector, dtype=np.float32)
    conn.execute(
        "INSERT INTO embeddings(chunk_id, dim, vector) VALUES (?,?,?) "
        "ON CONFLICT(chunk_id) DO UPDATE SET dim=excluded.dim, vector=excluded.vector",
        (chunk_id, int(v.shape[0]), v.tobytes()),
    )


def count_chunks(conn, book_id: int) -> int:
    row = conn.execute("SELECT COUNT(*) c FROM chunks WHERE book_id = ?", (book_id,)).fetchone()
    return int(row["c"])


def list_chapters(conn, book_ids: list[int] | None = None) -> list[str]:
    """Every chapter heading in scope, for resolving "the chapter on …"."""
    sql = "SELECT DISTINCT chapter FROM chunks WHERE chapter IS NOT NULL AND chapter != ''"
    params: list = []
    if book_ids is not None:
        if not book_ids:
            return []
        sql += f" AND book_id IN ({','.join('?' * len(book_ids))})"
        params += list(book_ids)
    return [str(r["chapter"]) for r in conn.execute(sql, params).fetchall()]


def chapter_chunks(conn, chapter: str, book_ids: list[int] | None = None) -> list[sqlite3.Row]:
    """One chapter in reading order — what a chapter summary is built from."""
    sql = "SELECT * FROM chunks WHERE chapter = ?"
    params: list = [chapter]
    if book_ids is not None:
        if not book_ids:
            return []
        sql += f" AND book_id IN ({','.join('?' * len(book_ids))})"
        params += list(book_ids)
    sql += " ORDER BY book_id, chunk_index"
    return conn.execute(sql, params).fetchall()


# ---------- gradings ----------

def insert_gradings(conn, chunk_id: int, found: list[tuple[str, str]]) -> None:
    conn.executemany(
        "INSERT OR IGNORE INTO gradings(chunk_id, grading, raw) VALUES (?,?,?)",
        [(chunk_id, g, r) for g, r in found],
    )


def gradings_for(conn, chunk_id: int) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT grading, raw FROM gradings WHERE chunk_id = ?", (chunk_id,)
    ).fetchall()


# ---------- bookmarks ----------

_BOOKMARK_FIELDS = (
    "kind", "title", "chapter", "reference", "collection", "number",
    "page_start", "page_end", "excerpt", "gradings", "spans_json",
    "book_id", "chunk_id", "description",
)


def add_bookmark(conn, data: dict) -> int:
    """Save a narration the user wants to keep, with their own note."""
    excerpt = " ".join(str(data.get("excerpt") or "").split())
    if not excerpt:
        raise ValueError("a bookmark needs an excerpt")
    values = {key: data.get(key) for key in _BOOKMARK_FIELDS}
    values["excerpt"] = excerpt
    values["kind"] = values["kind"] or "library"
    values["title"] = values["title"] or "Untitled source"
    gradings = values["gradings"]
    if isinstance(gradings, (list, tuple)):
        values["gradings"] = json.dumps(list(gradings))
    spans = values["spans_json"]
    if isinstance(spans, (dict, list)):
        values["spans_json"] = json.dumps(spans)
    columns = ", ".join(values)
    marks = ", ".join("?" for _ in values)
    cur = conn.execute(
        f"INSERT INTO bookmarks({columns}) VALUES ({marks})", tuple(values.values())
    )
    return int(cur.lastrowid)


def list_bookmarks(conn) -> list[dict]:
    rows = conn.execute("SELECT * FROM bookmarks ORDER BY id DESC").fetchall()
    out = []
    for r in rows:
        item = dict(r)
        try:
            item["gradings"] = json.loads(item["gradings"] or "[]")
        except (TypeError, ValueError):
            item["gradings"] = []
        try:
            item["spans"] = json.loads(item.pop("spans_json") or "{}")
        except (TypeError, ValueError):
            item["spans"] = {}
        out.append(item)
    return out


def delete_bookmark(conn, bookmark_id: int) -> bool:
    cur = conn.execute("DELETE FROM bookmarks WHERE id = ?", (int(bookmark_id),))
    return cur.rowcount > 0


def set_bookmark_description(conn, bookmark_id: int, description: str | None) -> bool:
    cur = conn.execute(
        "UPDATE bookmarks SET description = ? WHERE id = ?",
        ((description or "").strip() or None, int(bookmark_id)),
    )
    return cur.rowcount > 0


# ---------- chats ----------

def save_chat(conn, chat: dict) -> None:
    """Upsert one conversation; the transcript itself lives in `data` as JSON."""
    turns = chat.get("turns") or []
    conn.execute(
        "INSERT INTO chats(id, title, turns, data, created_at, updated_at, folder_id) "
        "VALUES (?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET "
        "title=excluded.title, turns=excluded.turns, data=excluded.data, "
        "updated_at=excluded.updated_at, folder_id=excluded.folder_id",
        (
            chat["id"],
            chat.get("title") or "",
            len(turns),
            json.dumps(chat, ensure_ascii=False),
            chat.get("created_at") or "",
            chat.get("updated_at") or chat.get("created_at") or "",
            chat.get("folder_id"),
        ),
    )


def load_chat(conn, chat_id: str) -> dict | None:
    row = conn.execute("SELECT data FROM chats WHERE id = ?", (chat_id,)).fetchone()
    if row is None:
        return None
    try:
        return json.loads(row["data"])
    except (TypeError, ValueError):
        return None


def list_chats(conn) -> list[dict]:
    """Chat list without the transcripts, newest first."""
    rows = conn.execute(
        "SELECT id, title, turns, created_at, updated_at, folder_id FROM chats "
        "ORDER BY updated_at DESC, id DESC"
    ).fetchall()
    return [dict(r) for r in rows]


# ---------- chat folders ----------

def list_folders(conn) -> list[dict]:
    rows = conn.execute(
        "SELECT id, name, created_at FROM chat_folders ORDER BY created_at, name"
    ).fetchall()
    return [dict(r) for r in rows]


def insert_folder(conn, folder_id: str, name: str, created_at: str) -> None:
    conn.execute(
        "INSERT INTO chat_folders(id, name, created_at) VALUES (?,?,?)",
        (folder_id, name, created_at),
    )


def rename_folder(conn, folder_id: str, name: str) -> bool:
    cur = conn.execute("UPDATE chat_folders SET name = ? WHERE id = ?", (name, folder_id))
    return cur.rowcount > 0


def delete_folder(conn, folder_id: str) -> bool:
    """Removing a folder keeps its conversations, unfiled."""
    conn.execute(
        "UPDATE chats SET folder_id = NULL WHERE folder_id = ?", (folder_id,)
    )
    cur = conn.execute("DELETE FROM chat_folders WHERE id = ?", (folder_id,))
    return cur.rowcount > 0


def set_chat_folder(conn, chat_id: str, folder_id: str | None) -> bool:
    cur = conn.execute(
        "UPDATE chats SET folder_id = ? WHERE id = ?", (folder_id, chat_id)
    )
    return cur.rowcount > 0


def delete_chat(conn, chat_id: str) -> bool:
    cur = conn.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
    return cur.rowcount > 0


def set_meta(conn, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta(key, value) VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )


def get_meta(conn, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else None
