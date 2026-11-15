"""Hybrid retrieval: FTS5 (BM25) + dense cosine, fused with RRF.

Corpus is small enough (≲ 50k chunks) to scan vectors in NumPy, so no
vector-DB server is required. The keyword arm catches exact names and
rare terms; the dense arm catches paraphrases and cross-lingual
Arabic↔English matches.

Either arm may be unavailable — a fresh book has no vectors yet, and a
model switch can leave vectors of the wrong width behind. Retrieval then
degrades to the arm that works instead of returning nothing or crashing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import db
from .normalize import fts_tokens, normalize_query

RRF_K = 60


@dataclass
class Retrieved:
    chunk_id: int
    score: float
    fts_rank: int | None
    vec_rank: int | None


def _fts_search(conn, query: str, book_ids: list[int] | None, limit: int) -> list[int]:
    toks = fts_tokens(query)
    if not toks:
        return []
    match = " OR ".join(f'"{t}"' for t in toks)
    sql = "SELECT c.id FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.rowid WHERE chunks_fts MATCH ?"
    params: list = [match]
    if book_ids is not None:
        sql += f" AND c.book_id IN ({','.join('?' * len(book_ids))})"
        params += book_ids
    sql += " ORDER BY bm25(chunks_fts) LIMIT ?"
    params.append(limit)
    try:
        rows = conn.execute(sql, params).fetchall()
    except Exception:
        return []  # malformed MATCH — degrade to vector-only
    return [int(r["id"]) for r in rows]


def _vec_search(conn, query: str, embedder, book_ids: list[int] | None, limit: int) -> list[int]:
    ids, matrix, dim = db.load_vectors(conn, book_ids)
    if not ids or dim == 0:
        return []
    q = np.asarray(embedder.embed([normalize_query(query)])[0], dtype=np.float32)
    if q.shape[0] != dim:
        # Stale vectors from a different embedder: ignore the dense arm rather
        # than raising a shape error at the user.
        return []
    sims = matrix @ q  # rows are L2-normalized; q from providers is normalized
    order = np.argsort(-sims)[:limit]
    return [int(ids[i]) for i in order]


def hybrid_search(conn, query: str, embedder, *, book_ids: list[int] | None = None,
                  pool: int = 30) -> list[Retrieved]:
    if book_ids is not None and len(book_ids) == 0:
        return []  # an empty selection means "no books", never "all books"

    fts_ids = _fts_search(conn, query, book_ids, pool * 2)
    vec_ids = _vec_search(conn, query, embedder, book_ids, pool * 2)

    fused: dict[int, dict] = {}
    for rank, cid in enumerate(fts_ids):
        e = fused.setdefault(cid, {"fts_rank": None, "vec_rank": None})
        e["fts_rank"] = rank + 1
    for rank, cid in enumerate(vec_ids):
        e = fused.setdefault(cid, {"fts_rank": None, "vec_rank": None})
        e["vec_rank"] = rank + 1

    out: list[Retrieved] = []
    for cid, e in fused.items():
        score = 0.0
        if e["fts_rank"] is not None:
            score += 1.0 / (RRF_K + e["fts_rank"])
        if e["vec_rank"] is not None:
            score += 1.0 / (RRF_K + e["vec_rank"])
        out.append(Retrieved(cid, score, e["fts_rank"], e["vec_rank"]))
    out.sort(key=lambda r: r.score, reverse=True)
    return out[:pool]
