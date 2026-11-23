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


def _chapter_ids(conn, chapter: str, book_ids: list[int] | None) -> set[int]:
    """Chunk ids inside one chapter, so a scope never leaks into the rest of the book."""
    sql = "SELECT id FROM chunks WHERE chapter = ?"
    params: list = [chapter]
    if book_ids is not None:
        if not book_ids:
            return set()
        sql += f" AND book_id IN ({','.join('?' * len(book_ids))})"
        params += list(book_ids)
    return {int(r["id"]) for r in conn.execute(sql, params).fetchall()}


def _fts_search(conn, query: str, book_ids: list[int] | None, limit: int,
                allowed: set[int] | None = None) -> list[int]:
    toks = fts_tokens(query)
    if not toks or (allowed is not None and not allowed):
        return []
    match = " OR ".join(f'"{t}"' for t in toks)
    sql = "SELECT c.id FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.rowid WHERE chunks_fts MATCH ?"
    params: list = [match]
    if book_ids is not None:
        sql += f" AND c.book_id IN ({','.join('?' * len(book_ids))})"
        params += book_ids
    sql += " ORDER BY bm25(chunks_fts) LIMIT ?"
    params.append(limit if allowed is None else max(limit, len(allowed)))
    try:
        rows = conn.execute(sql, params).fetchall()
    except Exception:
        return []  # malformed MATCH — degrade to vector-only
    out = [int(r["id"]) for r in rows]
    if allowed is not None:
        out = [cid for cid in out if cid in allowed]
    return out[:limit]


def _vec_search(conn, query: str, embedder, book_ids: list[int] | None, limit: int,
                allowed: set[int] | None = None) -> list[int]:
    if allowed is not None and not allowed:
        return []
    ids, matrix, dim = db.load_vectors(conn, book_ids)
    if not ids or dim == 0:
        return []
    q = np.asarray(embedder.embed([normalize_query(query)])[0], dtype=np.float32)
    if q.shape[0] != dim:
        # Stale vectors from a different embedder: ignore the dense arm rather
        # than raising a shape error at the user.
        return []
    sims = matrix @ q  # rows are L2-normalized; q from providers is normalized
    if allowed is not None:
        mask = np.fromiter((cid in allowed for cid in ids), dtype=bool, count=len(ids))
        if not mask.any():
            return []
        sims = np.where(mask, sims, -np.inf)
    order = np.argsort(-sims)[:limit]
    return [int(ids[i]) for i in order if sims[i] != -np.inf]


def hybrid_search(conn, query: str, embedder, *, book_ids: list[int] | None = None,
                  pool: int = 30, chapter: str | None = None,
                  queries: tuple[str, ...] | list[str] = ()) -> list[Retrieved]:
    """Fuse keyword and dense results, optionally over several phrasings.

    ``queries`` are extra phrasings of the same question (the topical core of
    "bring me ten on patience" is just "patience"); each is weighted below the
    question itself so it widens recall without taking over the ranking.
    """
    if book_ids is not None and len(book_ids) == 0:
        return []  # an empty selection means "no books", never "all books"

    variants = [query, *(q for q in queries if q and q != query)]
    allowed = _chapter_ids(conn, chapter, book_ids) if chapter else None
    if allowed is not None and not allowed:
        return []  # the chapter has no indexed chunks in this scope

    fused: dict[int, dict] = {}
    for index, variant in enumerate(variants):
        weight = 1.0 if index == 0 else 0.6
        for rank, cid in enumerate(
                _fts_search(conn, variant, book_ids, pool * 2, allowed)):
            e = fused.setdefault(cid, {"score": 0.0, "fts_rank": None, "vec_rank": None})
            e["score"] += weight / (RRF_K + rank + 1)
            if e["fts_rank"] is None:
                e["fts_rank"] = rank + 1
        for rank, cid in enumerate(
                _vec_search(conn, variant, embedder, book_ids, pool * 2, allowed)):
            e = fused.setdefault(cid, {"score": 0.0, "fts_rank": None, "vec_rank": None})
            e["score"] += weight / (RRF_K + rank + 1)
            if e["vec_rank"] is None:
                e["vec_rank"] = rank + 1

    out = [
        Retrieved(cid, float(e["score"]), e["fts_rank"], e["vec_rank"])
        for cid, e in fused.items()
    ]
    out.sort(key=lambda r: r.score, reverse=True)
    return out[:pool]
