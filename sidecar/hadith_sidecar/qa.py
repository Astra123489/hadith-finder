"""Grounded question-answering over the loaded books.

Contract:
- Excerpts are all the model may use; it must cite them as [S#].
- If the excerpts don't answer the question, the model replies exactly
  NOT_FOUND, and we return zero citations (no invented hadiths, ever).
- Every returned citation carries page numbers + highlight spans + the
  book's own hadith reference, so the UI can point at the exact passage.
- Citation numbers in the answer keep their excerpt number (`source_index`),
  so [S4] in the text always points at the card labelled S4.
- If the model answers without citing anything we say so (`unverified`)
  instead of pinning its answer to an arbitrary excerpt.
- Each citation is cross-referenced against the canonical collections
  ("also recorded in Sahih al-Bukhari 1234") — see crossref.py.
"""

from __future__ import annotations

import json
import re

from . import crossref, db
from .retrieval import hybrid_search

NOT_FOUND = "I could not find this in the loaded books."

SYSTEM_PROMPT = f"""You are Hadith Finder, a meticulous research assistant over hadith and related classical works.

Rules:
1. Answer ONLY from the excerpts below. Never use outside knowledge. Never invent a hadith, narrator, page, or reference.
2. Cite the excerpts you use inline as [S1], [S2], etc., matching the excerpt numbers exactly. Every factual sentence needs a citation.
3. If the excerpts do not contain the answer, reply exactly: "{NOT_FOUND}"
4. Be precise: prefer short quotations in the language of the source (Arabic if Arabic), followed by explanation in the user's language.
5. Authenticity: if the excerpts state a grading (sahih/hasan/da'if/...), report it. If not, say the book does not grade this narration.
6. If asked to explain or summarize, reason step by step but keep the final answer compact.
"""

_MAX_EXCERPT_CHARS = 2600
_MAX_CITATION_EXCERPT = 700
_TAG_GROUP = re.compile(r"\[([^\]]{1,40})\]")
_TAG_INNER = re.compile(r"S(\d+)")


def _book_titles(conn, ids: list[int]) -> dict[int, str]:
    if not ids:
        return {}
    q = f"SELECT id, title FROM books WHERE id IN ({','.join('?' * len(ids))})"
    return {int(r["id"]): r["title"] for r in conn.execute(q, ids).fetchall()}


def _clip_to_query(text: str, query: str, limit: int) -> str:
    """Trim a long passage around the densest cluster of query words."""
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    terms = {t for t in re.findall(r"\w{3,}", query.lower())}
    if not terms:
        return text[:limit] + " …"
    best_start, best_hits = 0, -1
    step = max(1, limit // 4)
    windows = range(0, max(1, len(text) - limit + 1), step)
    for start in windows:
        window = text[start:start + limit].lower()
        hits = sum(1 for t in terms if t in window)
        if hits > best_hits:
            best_start, best_hits = start, hits
    if best_hits <= 0:
        return text[:limit] + " …"
    start = best_start
    end = min(len(text), start + limit)
    prefix = "… " if start > 0 else ""
    suffix = " …" if end < len(text) else ""
    return prefix + text[start:end].strip() + suffix


def _excerpt(titles, row, tag: str, query: str) -> str:
    pages = f"p. {row['page_start'] + 1}" if row["page_start"] == row["page_end"] else f"pp. {row['page_start'] + 1}-{row['page_end'] + 1}"
    ref = f" · hadith {row['hadith_ref']}" if row["hadith_ref"] else ""
    chapter = f" · {row['chapter']}" if row["chapter"] else ""
    body = _clip_to_query(row["text"], query, _MAX_EXCERPT_CHARS)
    return (
        f"[{tag}] {titles.get(row['book_id'], 'Unknown')}{chapter}{ref} · {pages}\n"
        f"{body}\n"
    )


def _tags_in(answer: str) -> list[int]:
    out: list[int] = []
    for group in _TAG_GROUP.findall(answer):
        for num in _TAG_INNER.findall(group):
            out.append(int(num))
    return out


def _is_not_found(answer: str) -> bool:
    text = " ".join(answer.strip().strip('"\u201c\u201d').split()).lower()
    return text.startswith(NOT_FOUND.lower()) or text == NOT_FOUND.lower()


def _parallels(conn, row, titles) -> list[dict]:
    try:
        # Build/verify the bundled corpus index before reading its metadata:
        # on a fresh library the tables do not exist yet.
        crossref.ensure_index(conn)
        exclude = crossref._collection_from_title(
            titles.get(row["book_id"], ""), crossref.collections(conn))
        found = crossref.find_parallels(conn, row["text"], exclude_collection=exclude)
    except Exception as exc:  # never fail an answer over a cross-reference
        import sys

        print(f"crossref: {exc}", file=sys.stderr)
        return []
    names = {c["slug"]: c for c in crossref.collections(conn)}
    for match in found:
        meta = names.get(match["collection"], {})
        match["collection_name"] = meta.get("name", match["collection"])
        match["label"] = f"{meta.get('short', match['collection'])} {match['number']}"
        match["snippet"] = " ".join(match["text"].split())[:240]
        match.pop("text", None)
    return found


def ask(conn, query: str, *, book_ids: list[int] | None, embedder, llm, reranker,
        top_k: int = 5, pool: int = 30, careful: bool = False,
        on_token=None, ask_id: str | None = None) -> dict:
    if book_ids is not None and len(book_ids) == 0:
        return {"answer": NOT_FOUND, "citations": [], "retrieved": 0,
                "unverified": False, "scoped_empty": True}

    hits = hybrid_search(conn, query, embedder, book_ids=book_ids, pool=pool)
    if not hits:
        return {"answer": NOT_FOUND, "citations": [], "retrieved": 0,
                "unverified": False, "scoped_empty": False}

    chunks = db.get_chunks(conn, [h.chunk_id for h in hits])
    by_id = {r["id"]: r for r in chunks}

    # Keep hit/score alignment even if a chunk row vanished between the two
    # queries: rerank only rows we actually have, then order those same rows.
    present = [h for h in hits if h.chunk_id in by_id]
    scores = reranker.rerank(query, [by_id[h.chunk_id]["text"] for h in present])
    ranked = sorted(zip(present, scores), key=lambda pair: pair[1], reverse=True)
    chosen = [h for h, _ in ranked[:top_k]]

    titles = _book_titles(conn, list({by_id[h.chunk_id]["book_id"] for h in chosen}))
    excerpts = []
    for i, h in enumerate(chosen):
        excerpts.append(_excerpt(titles, by_id[h.chunk_id], f"S{i + 1}", query))
    user = "Excerpts:\n\n" + "\n".join(excerpts) + f"\nQuestion: {query}"

    answer = llm.generate(SYSTEM_PROMPT, user, careful=careful, on_token=on_token)

    if _is_not_found(answer):
        return {"answer": NOT_FOUND, "citations": [], "retrieved": len(hits),
                "unverified": False, "scoped_empty": False}

    tag_numbers = _tags_in(answer)
    seen: list[int] = []
    for num in tag_numbers:
        if 1 <= num <= len(chosen) and num not in seen:
            seen.append(num)

    citations = []
    for num in seen:
        row = by_id[chosen[num - 1].chunk_id]
        gradings = list(dict.fromkeys(r["grading"] for r in db.gradings_for(conn, row["id"])))
        citations.append({
            "chunk_id": row["id"],
            "book_id": row["book_id"],
            "book_title": titles.get(row["book_id"], "Unknown"),
            "chapter": row["chapter"],
            "hadith_ref": row["hadith_ref"],
            "page_start": row["page_start"],
            "page_end": row["page_end"],
            "spans": json.loads(row["spans_json"] or "{}"),
            "excerpt": " ".join(row["text"].split())[:_MAX_CITATION_EXCERPT],
            "gradings": gradings,
            "source_index": num,
            "also_in": _parallels(conn, row, titles),
        })

    return {
        "answer": answer,
        "citations": citations,
        "retrieved": len(hits),
        "unverified": not citations,
        "scoped_empty": False,
        "ask_id": ask_id,
    }
