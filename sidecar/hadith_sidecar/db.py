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
CREATE INDEX IF NOT EXISTS idx_chunks_book ON chunks(book_id);
CREATE INDEX IF NOT EXISTS idx_gradings_chunk ON gradings(chunk_id);
"""


def connect(path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.executescript(_SCHEMA)
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


def set_meta(conn, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta(key, value) VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )


def get_meta(conn, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else None
