"""Grounded question-answering over the loaded books.

Contract:
- Excerpts are all the model may use; it must cite them as [S#].
- If the excerpts don't answer the question, the model replies exactly
  NOT_FOUND, and we return zero citations (no invented hadiths, ever).
- Every returned citation carries page numbers + highlight spans + the
  book's own hadith reference, so the UI can point at the exact passage.
"""

from __future__ import annotations

import json
import re

from . import db
from .retrieval import hybrid_search

NOT_FOUND = "I could not find this in the loaded books."

SYSTEM_PROMPT = f"""You are Hadith Finder, a meticulous research assistant over hadith and related classical works.

Rules:
1. Answer ONLY from the excerpts below. Never use outside knowledge. Never invent a hadith, narrator, page, or reference.
2. Cite excerpts inline as [S1], [S2], etc. matching the excerpt numbers.
3. If the excerpts do not contain the answer, reply exactly: "{NOT_FOUND}"
4. Be precise: prefer short quotations in the language of the source (Arabic if Arabic), followed by explanation in the user's language.
5. Authenticity: if the excerpts state a grading (sahih/hasan/da'if/...), report it. If not, say the book does not grade this narration.
6. If asked to explain or summarize, reason step by step but keep the final answer compact.
"""

_MAX_EXCERPT_CHARS = 2400
_TAG_RE = re.compile(r"\[(S\d+)\]")


def _book_titles(conn, ids: list[int]) -> dict[int, str]:
    if not ids:
        return {}
    q = f"SELECT id, title FROM books WHERE id IN ({','.join('?' * len(ids))})"
    return {int(r["id"]): r["title"] for r in conn.execute(q, ids).fetchall()}


def _excerpt(conn, titles, row, tag: str) -> str:
    pages = f"p. {row['page_start'] + 1}" if row["page_start"] == row["page_end"] else f"pp. {row['page_start'] + 1}-{row['page_end'] + 1}"
    ref = f" · hadith {row['hadith_ref']}" if row["hadith_ref"] else ""
    chapter = f" · {row['chapter']}" if row["chapter"] else ""
    body = row["text"][:_MAX_EXCERPT_CHARS]
    return (
        f"[{tag}] {titles.get(row['book_id'], 'Unknown')}{chapter}{ref} · {pages}\n"
        f"{body}\n"
    )


def ask(conn, query: str, *, book_ids: list[int] | None, embedder, llm, reranker,
        top_k: int = 5, pool: int = 30, careful: bool = False) -> dict:
    hits = hybrid_search(conn, query, embedder, book_ids=book_ids, pool=pool)
    if not hits:
        return {"answer": NOT_FOUND, "citations": [], "retrieved": 0}

    chunks = db.get_chunks(conn, [h.chunk_id for h in hits])
    by_id = {r["id"]: r for r in chunks}

    texts = [by_id[h.chunk_id]["text"] for h in hits if h.chunk_id in by_id]
    scores = reranker.rerank(query, texts)
    order = sorted(range(len(hits)), key=lambda i: scores[i] if i < len(scores) else 0.0, reverse=True)
    chosen = [hits[i] for i in order[:top_k] if hits[i].chunk_id in by_id]

    titles = _book_titles(conn, list({by_id[h.chunk_id]["book_id"] for h in chosen}))
    excerpts = []
    for i, h in enumerate(chosen):
        excerpts.append(_excerpt(conn, titles, by_id[h.chunk_id], f"S{i + 1}"))
    user = "Excerpts:\n\n" + "\n".join(excerpts) + f"\nQuestion: {query}"

    answer = llm.generate(SYSTEM_PROMPT, user, careful=careful)

    if NOT_FOUND.lower() in answer.lower():
        return {"answer": NOT_FOUND, "citations": [], "retrieved": len(hits)}

    tagged = _TAG_RE.findall(answer)
    seen: list[int] = []
    for tag in dict.fromkeys(tagged):
        idx = int(tag[1:]) - 1
        if 0 <= idx < len(chosen):
            seen.append(chosen[idx].chunk_id)
    if not seen and chosen:
        seen = [chosen[0].chunk_id]  # model answered but forgot to cite — safest anchor

    citations = []
    for cid in seen:
        row = by_id[cid]
        gradings = list(dict.fromkeys(r["grading"] for r in db.gradings_for(conn, cid)))
        citations.append({
            "chunk_id": cid,
            "book_id": row["book_id"],
            "book_title": titles.get(row["book_id"], "Unknown"),
            "chapter": row["chapter"],
            "hadith_ref": row["hadith_ref"],
            "page_start": row["page_start"],
            "page_end": row["page_end"],
            "spans": json.loads(row["spans_json"] or "{}"),
            "excerpt": row["text"][:500],
            "gradings": gradings,
        })
    return {"answer": answer, "citations": citations, "retrieved": len(hits)}
