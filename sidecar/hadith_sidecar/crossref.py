"""Cross-collection cross-reference: "the same matn in another collection".

The app ships a compact corpus of the five canonical collections
(`data/crossref.json.gz`, built by `tools/build_crossref.py`). On first use
it is loaded into the library database as `xref` + `xref_fts`, and each
answer citation is matched against it.

Matching is deliberately conservative, because a wrong "also in Bukhari"
claim is worse than none:

  * candidate generation: FTS5 over the citation's rarest content words;
  * scoring: token containment between the citation and the canonical matn,
    after stripping isnad boilerplate ("narrated", "messenger of Allah", …);
  * a match needs both enough shared words and a high containment ratio.

Translations differ in wording, so containment of the *canonical* text is
compared against the shorter side — a short hadith quoted inside a long
chunk still matches.
"""

from __future__ import annotations

import gzip
import json
import re
from pathlib import Path

DATA_FILE = Path(__file__).with_name("data") / "crossref.json.gz"
XREF_VERSION = "2"

STOPWORDS = {
    "the", "and", "for", "that", "this", "with", "from", "was", "were", "are",
    "his", "her", "him", "she", "they", "them", "their", "there", "then",
    "when", "what", "which", "who", "whom", "whose", "will", "would", "could",
    "should", "have", "has", "had", "been", "being", "into", "upon", "about",
    "also", "more", "than", "some", "such", "other", "only", "over", "after",
    "before", "these", "those", "where", "while", "because", "may", "might",
    "must", "not", "but", "you", "your", "our", "its", "it", "we", "us", "he",
    "she", "did", "does", "does", "do", "of", "to", "in", "on", "at", "by",
    "as", "is", "be", "or", "an", "a", "i", "said", "says", "say", "saying",
    "narrated", "narrates", "reported", "reports", "messenger", "prophet",
    "allah", "peace", "blessings", "mercy", "upon", "him", "hadith", "narration",
    "tradition", "companion", "father", "mother", "man", "people", "one",
}

_TOKEN = re.compile(r"[a-z][a-z'’\-]{2,}")


def _tokens(text: str) -> set[str]:
    return {t.strip("'-") for t in _TOKEN.findall((text or "").lower())} - STOPWORDS


def _clean_for_fts(text: str) -> str:
    return " ".join(sorted(_tokens(text)))


def _collection_from_title(title: str, collections: list[dict]) -> str | None:
    low = (title or "").lower()
    for c in collections:
        if c["name"].lower() in low or c["short"].lower() in low:
            return c["slug"]
    return None


def ensure_index(conn, notify=None) -> dict:
    """Create/populate the cross-reference tables. Idempotent and cheap to call."""
    conn.execute(
        "CREATE TABLE IF NOT EXISTS xref("
        " id INTEGER PRIMARY KEY, collection TEXT NOT NULL, number INTEGER NOT NULL,"
        " book TEXT, text TEXT NOT NULL, norm TEXT NOT NULL)"
    )
    conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS xref_fts USING fts5(norm)")
    conn.execute("CREATE TABLE IF NOT EXISTS xref_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    row = conn.execute("SELECT value FROM xref_meta WHERE key='version'").fetchone()
    if row and row["value"] == XREF_VERSION:
        n = conn.execute("SELECT COUNT(*) c FROM xref").fetchone()["c"]
        if n:
            return {"indexed": n, "fresh": False}

    if not DATA_FILE.is_file():
        return {"indexed": 0, "fresh": False, "missing": str(DATA_FILE)}

    with gzip.open(DATA_FILE, "rt", encoding="utf-8") as fh:
        payload = json.load(fh)
    collections = payload.get("collections", [])
    slugs = [c["slug"] for c in collections]
    rows = []
    for coll_idx, number, book, text in payload.get("hadiths", []):
        slug = slugs[coll_idx] if coll_idx < len(slugs) else "unknown"
        rows.append((slug, int(number), book or "", text, _clean_for_fts(text)))

    if notify:
        notify(op="crossref", stage="index", done=0, total=len(rows))
    with conn:
        # Rebuild from scratch so a version bump also repairs the schema.
        conn.execute("DROP TABLE IF EXISTS xref")
        conn.execute("DROP TABLE IF EXISTS xref_fts")
        conn.execute(
            "CREATE TABLE xref(id INTEGER PRIMARY KEY, collection TEXT NOT NULL,"
            " number INTEGER NOT NULL, book TEXT, text TEXT NOT NULL, norm TEXT NOT NULL)"
        )
        conn.execute("CREATE VIRTUAL TABLE xref_fts USING fts5(norm)")
        for i in range(0, len(rows), 2000):
            batch = rows[i:i + 2000]
            numbered = [(i + j + 1, *row) for j, row in enumerate(batch)]
            conn.executemany(
                "INSERT OR REPLACE INTO xref(id, collection, number, book, text, norm) VALUES (?,?,?,?,?,?)",
                numbered,
            )
            conn.executemany(
                "INSERT INTO xref_fts(rowid, norm) VALUES (?,?)",
                [(row[0], row[5]) for row in numbered],
            )
            if notify:
                notify(op="crossref", stage="index", done=min(i + 2000, len(rows)), total=len(rows))
        conn.execute("INSERT OR REPLACE INTO xref_meta(key, value) VALUES ('version', ?)", (XREF_VERSION,))
        conn.execute(
            "INSERT OR REPLACE INTO xref_meta(key, value) VALUES ('collections', ?)",
            (json.dumps(collections, ensure_ascii=False),),
        )
    return {"indexed": len(rows), "fresh": True}


def collections(conn) -> list[dict]:
    try:
        row = conn.execute("SELECT value FROM xref_meta WHERE key='collections'").fetchone()
    except Exception:
        return []  # index not built yet
    return json.loads(row["value"]) if row else []


import math

_DF_CACHE: dict[str, int] = {}


def narration(conn, collection: str, number: int) -> dict | None:
    """One canonical narration in full, so a cross-reference can be read through."""
    ensure_index(conn)
    row = conn.execute(
        "SELECT collection, number, book, text FROM xref"
        " WHERE collection = ? AND number = ? LIMIT 1",
        (str(collection), int(number)),
    ).fetchone()
    if row is None:
        return None
    meta = {c["slug"]: c for c in collections(conn)}.get(row["collection"], {})
    return {
        "collection": row["collection"],
        "collection_name": meta.get("name", row["collection"]),
        "short": meta.get("short", row["collection"]),
        "number": int(row["number"]),
        "book": row["book"] or "",
        "text": row["text"],
    }


def locate(conn, collection: str, number: int) -> dict:
    """Where this narration sits in the reader's own copy of the collection.

    The shipped corpus is text only, so a "same narration" line cannot open a
    page by itself. If the collection was imported as a book, the closest chunk
    of that book is found by full-text search and the page is returned with it.
    Books are matched by title, the same way citations are attributed.
    """
    ensure_index(conn)
    meta = {c["slug"]: c for c in collections(conn)}
    info = meta.get(collection, {})
    name = info.get("name", collection)
    short = info.get("short", collection)
    label = f"{short} {int(number)}"
    missing = {
        "found": False,
        "collection": collection,
        "collection_name": name,
        "short": short,
        "number": int(number),
        "label": label,
    }
    if not info:
        return missing
    row = conn.execute(
        "SELECT text FROM xref WHERE collection = ? AND number = ? LIMIT 1",
        (collection, int(number)),
    ).fetchone()
    if row is None:
        return missing
    books = conn.execute("SELECT id, title FROM books ORDER BY id").fetchall()
    book = next(
        (b for b in books if _collection_from_title(b["title"], [info]) == collection),
        None,
    )
    if book is None:
        return missing
    wanted = _tokens(row["text"])
    if not wanted:
        return missing
    query = " OR ".join(f'"{t}"' for t in sorted(wanted)[:40])
    try:
        chunks = conn.execute(
            "SELECT c.id, c.page_start, c.page_end, c.spans_json, c.text FROM chunks c"
            " JOIN chunks_fts ON chunks_fts.rowid = c.id"
            " WHERE chunks_fts MATCH ? AND c.book_id = ? LIMIT 25",
            (query, book["id"]),
        ).fetchall()
    except Exception:
        return missing
    best, best_score = None, 0.0
    for chunk in chunks:
        overlap = len(wanted & _tokens(chunk["text"])) / len(wanted)
        if overlap > best_score:
            best, best_score = chunk, overlap
    if best is None or best_score < 0.2:
        return missing
    try:
        spans = json.loads(best["spans_json"])
    except Exception:
        spans = {}
    return {
        **missing,
        "found": True,
        "book_id": int(book["id"]),
        "book_title": book["title"],
        "page_start": int(best["page_start"]),
        "page_end": int(best["page_end"]),
        "spans": spans,
        "score": round(best_score, 3),
    }


def _doc_count(conn) -> int:
    row = conn.execute("SELECT COUNT(*) c FROM xref").fetchone()
    return int(row["c"]) if row else 0


def _df(conn, token: str) -> int:
    cached = _DF_CACHE.get(token)
    if cached is not None:
        return cached
    try:
        row = conn.execute(
            "SELECT COUNT(*) c FROM xref_fts WHERE xref_fts MATCH ?", (f'"{token}"',)
        ).fetchone()
        count = int(row["c"])
    except Exception:
        count = 0
    _DF_CACHE[token] = count
    return count


def _weights(conn, text: str, n_docs: int) -> dict[str, float]:
    """Token → IDF weight. Ubiquitous words weigh almost nothing."""
    out: dict[str, float] = {}
    for token in _tokens(text):
        out[token] = math.log(1.0 + n_docs / (1.0 + _df(conn, token)))
    return out


def find_parallels(conn, text: str, *, limit: int = 3, min_weight: float = 8.0,
                   min_score: float = 0.45, exclude_collection: str | None = None) -> list[dict]:
    """Return canonical narrations carrying the same matn as `text`."""
    toks = list(_tokens(text))
    if len(toks) < 6:
        return []
    try:
        ensure_index(conn)
    except Exception:
        return []
    n_docs = _doc_count(conn)
    if n_docs == 0:
        return []
    weights = _weights(conn, text, n_docs)
    total_have = sum(weights.values()) or 1.0
    # Rarest (highest-IDF) words give FTS a compact, high-precision OR query.
    probe = sorted(weights, key=lambda t: -weights[t])[:24]
    if not probe:
        return []
    match = " OR ".join(f'"{t}"' for t in probe)
    try:
        rows = conn.execute(
            "SELECT x.collection, x.number, x.book, x.text, x.norm"
            " FROM xref_fts f JOIN xref x ON x.id = f.rowid"
            " WHERE xref_fts MATCH ? ORDER BY bm25(xref_fts) LIMIT 60",
            (match,),
        ).fetchall()
    except Exception:
        return []

    scored: list[dict] = []
    for row in rows:
        want = set(row["norm"].split()) if row["norm"] else set()
        if len(want) < 4:
            continue
        shared = [t for t in want if t in weights]
        if len(shared) < 4:
            continue
        weight = sum(weights[t] for t in shared)
        if weight < min_weight:
            continue
        total_want = sum(_weights(conn, row["text"], n_docs).values()) or 1.0
        score = weight / min(total_have, total_want)
        if score < min_score:
            continue
        scored.append({
            "collection": row["collection"],
            "number": int(row["number"]),
            "book": row["book"],
            "text": row["text"],
            "score": round(score, 3),
            "shared": len(shared),
            "same_collection": row["collection"] == exclude_collection,
        })

    # Prefer a genuine other-collection match; fall back to the same collection
    # (a parallel chain inside one collection is still useful) when nothing else
    # clears the bar.
    scored.sort(key=lambda m: (m["same_collection"], -m["score"], -m["shared"]))
    picked: list[dict] = []
    seen: set[tuple[str, int]] = set()
    for m in scored:
        key = (m["collection"], m["number"])
        if key in seen:
            continue
        seen.add(key)
        picked.append(m)
        if len(picked) >= limit:
            break
    return picked


def label_for(conn, match: dict) -> str:
    names = {c["slug"]: c["name"] for c in collections(conn)}
    return f"{names.get(match['collection'], match['collection'])} {match['number']}"


def daily(conn, day: str, *, min_chars: int = 150) -> dict | None:
    """The narration of the day: the same one all day, different tomorrow.

    The pick is a hash of the ISO date over narrations long enough to read on
    their own, so it is stable across launches without storing anything, and
    different on every calendar day.
    """
    import hashlib

    ensure_index(conn)
    rows = conn.execute(
        "SELECT id, collection, number, book, text FROM xref"
        " WHERE length(text) >= ? AND number > 0 ORDER BY id",
        (min_chars,),
    ).fetchall()
    if not rows:
        return None
    digest = hashlib.sha256(f"hadith-of-the-day:{day}".encode("utf-8")).hexdigest()
    index = int(digest[:12], 16) % len(rows)
    row = rows[index]
    meta = {c["slug"]: c for c in collections(conn)}
    info = meta.get(row["collection"], {})
    parallel_rows: list[dict] = []
    try:
        for match in find_parallels(conn, row["text"], limit=2,
                                    exclude_collection=row["collection"]):
            other = meta.get(match["collection"], {})
            parallel_rows.append({
                "collection": match["collection"],
                "collection_name": other.get("name", match["collection"]),
                "short": other.get("short", match["collection"]),
                "number": int(match["number"]),
                "book": match["book"],
                "snippet": " ".join(match["text"].split())[:240],
                "label": f"{other.get('short', match['collection'])} {match['number']}",
                "score": match["score"],
            })
    except Exception:
        parallel_rows = []
    return {
        "date": day,
        "collection": row["collection"],
        "collection_name": info.get("name", row["collection"]),
        "short": info.get("short", row["collection"]),
        "number": int(row["number"]),
        "book": row["book"] or "",
        "text": row["text"],
        "also_in": parallel_rows,
    }
