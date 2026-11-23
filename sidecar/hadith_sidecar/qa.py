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

Retrieval is shaped by the question (intent.py): a request for ten
narrations fetches ten distinct passages instead of five near-duplicates, a
chapter summary reads the chapter in order instead of its best-matching
lines, and a question about "this chapter" is scoped to the chapter the
reader is actually looking at.
"""

from __future__ import annotations

import json
import re

from . import crossref, db
from .intent import Intent, classify, match_chapter
from .llm import count_tokens
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
7. Earlier turns are background only. Answer the new question from this turn's excerpts, and cite only this turn's [S#] numbers — never a number from an earlier answer.
"""

MAX_EXCERPT_CHARS = 2600
_MAX_CITATION_EXCERPT = 700
_TAG_GROUP = re.compile(r"\[([^\]]{1,40})\]")
_TAG_INNER = re.compile(r"S(\d+)")
_WORD = re.compile(r"[0-9A-Za-z\u0600-\u06FF]{3,}")


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


def _excerpt(titles, row, tag: str, query: str, limit: int = MAX_EXCERPT_CHARS) -> str:
    pages = f"p. {row['page_start'] + 1}" if row["page_start"] == row["page_end"] else f"pp. {row['page_start'] + 1}-{row['page_end'] + 1}"
    ref = f" · hadith {row['hadith_ref']}" if row["hadith_ref"] else ""
    chapter = f" · {row['chapter']}" if row["chapter"] else ""
    body = _clip_to_query(row["text"], query, limit)
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


# --- how the question changes the answer ------------------------------------

def _intent_rules(intent: Intent) -> str:
    """One extra numbered rule, only when the question asks for that shape."""
    if intent.kind == "list":
        n = intent.count or intent.top_k
        text = (
            f"8. The reader asked for {n} narrations. Give up to {n} DISTINCT narrations, "
            "numbered on separate lines, each with its [S#]. Never split one narration into two "
            "entries, never repeat a narration in other words, and never invent one to reach the "
            f"number: if the excerpts hold fewer, give what is there and say how many you found."
        )
    elif intent.kind == "summarize":
        text = (
            "8. The reader asked for a summary. Open with one sentence naming what is being "
            "summarised, then give its main themes in the order the excerpts present them, citing "
            "each point. Say plainly which parts the excerpts do not cover."
        )
    elif intent.kind == "locate":
        text = (
            "8. The reader asked where something is. Start with the exact location from the excerpt "
            "headers — book, chapter, page and hadith number — then the quotation."
        )
    else:
        text = ""
    if intent.novel:
        text = (text + " " if text else "") + (
            "The reader wants narrations they are unlikely to have seen: prefer the least familiar "
            "of the excerpts, and never call one rare or uncommon unless the excerpt says so."
        )
    return text


def _system_prompt(intent: Intent) -> str:
    rules = _intent_rules(intent)
    return SYSTEM_PROMPT + ("\n" + rules if rules else "")


def _answer_budget(intent: Intent) -> int:
    if intent.kind == "list":
        return min(1200, 320 + 90 * intent.top_k)
    if intent.kind == "summarize":
        return 700
    return 512


def _words(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def _near_duplicate(a: set[str], b: set[str], threshold: float = 0.7) -> bool:
    if not a or not b or not (a & b):
        return False
    return len(a & b) / len(a | b) >= threshold


def _diverse(scored: list[tuple], limit: int) -> list:
    """Take the best-ranked passages, skipping repeats of what is already in.

    A ten-narration request would otherwise spend half its slots on two
    different chunks of the same hadith, or on the same narration printed in
    two collections.  Same book + same hadith number, or near-identical text,
    counts as a repeat.
    """
    chosen: list = []
    seen_refs: set[tuple[int, str]] = set()
    seen_words: list[set[str]] = []
    for row, _score in scored:
        ref = row["hadith_ref"]
        key = (int(row["book_id"]), str(ref)) if ref else None
        if key is not None and key in seen_refs:
            continue
        words = _words(row["text"])
        if any(_near_duplicate(words, other) for other in seen_words):
            continue
        chosen.append(row)
        if key is not None:
            seen_refs.add(key)
        seen_words.append(words)
        if len(chosen) >= limit:
            break
    return chosen


def _sample_evenly(rows: list, count: int) -> list:
    """A chapter's arc, not its best-matching lines: beginning, middle, end."""
    if len(rows) <= count:
        return list(rows)
    step = len(rows) / count
    return [rows[min(len(rows) - 1, int(i * step))] for i in range(count)]


def ask(conn, query: str, *, book_ids: list[int] | None, embedder, llm, reranker,
        top_k: int | None = None, pool: int = 30,
        on_token=None, ask_id: str | None = None,
        context: str = "", excerpt_chars: int = MAX_EXCERPT_CHARS,
        history_chapter: str | None = None,
        history_chunk_id: int | None = None,
        intent: Intent | None = None) -> dict:
    base = {
        "answer": NOT_FOUND,
        "citations": [],
        "retrieved": 0,
        "unverified": False,
        "scoped_empty": False,
        "prompt_tokens": 0,
        "excerpt_tokens": 0,
        "chapter_scope": None,
    }
    if book_ids is not None and len(book_ids) == 0:
        plan = intent or classify(query, history_chapter=history_chapter)
        return {**base, "scoped_empty": True, "intent": plan.as_dict()}

    plan = intent or classify(query, history_chapter=history_chapter)
    meta = {"intent": plan.as_dict()}

    chapter = match_chapter(db.list_chapters(conn, book_ids), plan.chapter) if plan.chapter else None
    meta["chapter_scope"] = chapter
    want = max(1, int(top_k if top_k is not None else plan.top_k))
    search_text = plan.search_text or query

    # "this narration" / "that hadith": the reader is pointing at the passage
    # already on screen, so it is pinned in rather than searched for again.
    pinned: list = []
    if plan.previous and history_chunk_id and not chapter:
        pinned = db.get_chunks(conn, [int(history_chunk_id)])

    hits = []
    chapter_rows: list = []
    if plan.kind == "summarize" and chapter:
        chapter_rows = db.chapter_chunks(conn, chapter, book_ids)
        chosen_rows = _sample_evenly(chapter_rows, want)
        if not chosen_rows:
            return {**base, **meta}
    elif plan.kind == "summarize" and pinned:
        chosen_rows = pinned
    else:
        hits = hybrid_search(
            conn, search_text, embedder,
            book_ids=book_ids, pool=pool, chapter=chapter, queries=plan.queries,
        )
        if not hits and not pinned:
            return {**base, **meta}
        chosen_rows = list(pinned)
        if hits:
            chunks = db.get_chunks(conn, [h.chunk_id for h in hits])
            by_id = {r["id"]: r for r in chunks}
            # Keep hit/score alignment even if a chunk row vanished between the
            # two queries: rerank only rows we actually have, then order those
            # same rows.
            present = [h for h in hits if h.chunk_id in by_id]
            scores = reranker.rerank(search_text, [by_id[h.chunk_id]["text"] for h in present])
            ranked = sorted(zip(present, scores), key=lambda pair: pair[1], reverse=True)
            picked = _diverse(
                [(by_id[h.chunk_id], float(score)) for h, score in ranked], want
            )
            seen_ids = {row["id"] for row in chosen_rows}
            chosen_rows = (
                chosen_rows + [row for row in picked if row["id"] not in seen_ids]
            )[:want]
        if not chosen_rows:
            return {**base, **meta}

    titles = _book_titles(conn, list({row["book_id"] for row in chosen_rows}))
    excerpts = [
        _excerpt(titles, row, f"S{i + 1}", search_text, excerpt_chars)
        for i, row in enumerate(chosen_rows)
    ]
    excerpt_text = "\n".join(excerpts)
    user = "Excerpts:\n\n" + excerpt_text
    if context:
        user += "\nEarlier in this conversation:\n" + context + "\n"
    user += f"\nQuestion: {query}"
    system = _system_prompt(plan)
    prompt_tokens = count_tokens(llm, system + user)
    excerpt_tokens = count_tokens(llm, excerpt_text)

    answer = llm.generate(
        system, user, max_tokens=_answer_budget(plan), on_token=on_token,
    )

    if _is_not_found(answer):
        return {
            **base, **meta,
            "retrieved": len(hits) or len(chapter_rows) or len(chosen_rows),
            "prompt_tokens": prompt_tokens,
            "excerpt_tokens": excerpt_tokens,
        }

    tag_numbers = _tags_in(answer)
    seen: list[int] = []
    for num in tag_numbers:
        if 1 <= num <= len(chosen_rows) and num not in seen:
            seen.append(num)

    citations = []
    for num in seen:
        row = chosen_rows[num - 1]
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
        "retrieved": len(hits) or len(chapter_rows) or len(chosen_rows),
        "unverified": not citations,
        "scoped_empty": False,
        "ask_id": ask_id,
        "prompt_tokens": prompt_tokens,
        "excerpt_tokens": excerpt_tokens,
        **meta,
    }
