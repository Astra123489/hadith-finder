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
from .intent import COLLECTION_ALIASES, Intent, classify, match_chapter
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

# A halal/haram question may only be answered by quoting a ruling the loaded
# books print. When none does, the app says so instead of reasoning towards a
# verdict — and every ruling answer carries the same notice.
RULING_NOT_FOUND = "The loaded books do not state a ruling on this."
COMPARE_NEEDS_TWO = (
    "There are not two passages on screen to compare. Ask about a narration first, "
    "or name the two references (for example \"Bukhari 1 and Muslim 1907\")."
)
NOT_A_FATWA = (
    "This is a reading aid, not a fatwa: it reports only what the loaded books state. "
    "Ask a qualified scholar about your own situation."
)
# Words that mark an explicit ruling in a passage. Deliberately narrow: a
# passage that merely mentions the topic must not be dressed up as a verdict.
_RULING_TERMS = re.compile(
    r"\b(?:halal|haraam|haram|permissible|impermissible|permitted|forbidden|"
    r"prohibited|unlawful|makruh|makrooh|wajib|obligatory|fard|mustahabb|sinful)\b"
    r"|(?:حلال|حرام|مباح|مكروه|محرم|يجوز|لا\s+يجوز|واجب|فرض|مستحب|حكم)",
    re.I,
)
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


def _is_ruling_not_found(answer: str) -> bool:
    text = " ".join(answer.strip().strip('"\u201c\u201d').split()).lower()
    return text.startswith(RULING_NOT_FOUND.lower())


def _states_ruling(text: str) -> bool:
    """Whether a passage itself states a ruling, rather than just discussing it."""
    return bool(_RULING_TERMS.search(text or ""))


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
    elif intent.kind == "grade":
        text = (
            "8. The reader asked how sound the narration is. Give the grading exactly as the excerpts "
            "state it (sahih, hasan, da'if, mawdu', or the collector's own words). If no excerpt "
            "grades it, say that this book does not grade it — never infer a grade from the wording, "
            "and never repeat a grade the excerpts do not contain."
        )
    elif intent.kind == "narrator":
        text = (
            "8. The reader asked about the chain of narration. Give the isnad exactly as the excerpt "
            "gives it, in order, and name the Companion who reported it when the excerpt does. Never "
            "supply a missing link or correct a name from outside knowledge."
        )
    elif intent.kind == "explain":
        text = (
            "8. The reader asked what the passage means. Quote the key phrase, gloss unfamiliar "
            "terms, and explain it in the reader's language — but keep every claim inside the "
            "excerpts and say when they are too short to explain a term."
        )
    elif intent.kind == "ruling":
        text = (
            "8. The reader asked for a ruling. Report ONLY a ruling that an excerpt states in its "
            "own words, quoted and cited: begin with \"The loaded books state…\". Never infer a "
            "ruling, never extend it past what the excerpt says, and never give a verdict in your "
            "own voice. If no excerpt states a ruling, reply exactly: \"" + RULING_NOT_FOUND + "\""
        )
    elif intent.kind == "compare":
        text = (
            "8. The reader asked you to compare these passages. Compare them point by point — "
            "wording, chain of narration, and what each says or leaves out — citing each claim. "
            "Say where they agree and where they differ, and report each passage's own grading "
            "when it prints one. Never declare one narration correct, stronger, or more authentic: "
            "if asked which is more authentic, report what each source says and stop there."
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
    if intent.kind in ("summarize", "explain", "ruling", "compare"):
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
        history_chunk_ids: list[int] | None = None,
        pinned_chunk_ids: list[int] | None = None,
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
    plan = intent or classify(query, history_chapter=history_chapter)
    meta = {"intent": plan.as_dict()}
    if plan.kind == "ruling":
        meta["notice"] = NOT_A_FATWA
    silence = RULING_NOT_FOUND if plan.kind == "ruling" else NOT_FOUND
    if book_ids is not None and len(book_ids) == 0:
        return {**base, "scoped_empty": True, **meta}

    chapter = match_chapter(db.list_chapters(conn, book_ids), plan.chapter) if plan.chapter else None
    meta["chapter_scope"] = chapter
    want = max(1, int(top_k if top_k is not None else plan.top_k))
    search_text = plan.search_text or query

    # A bare reference ("Tirmidhi 3521") is not a keyword: resolve the
    # collection against the book titles and the number against the printed
    # hadith reference. If the library does not hold it, say so instead of
    # letting keyword search wander into unrelated passages.
    reference_rows: list = []
    references = plan.references or ((plan.reference,) if plan.reference else ())
    for collection, number in references:
        rows = db.find_reference_chunks(
            conn, COLLECTION_ALIASES.get(collection, ()), number, book_ids
        )
        if not rows:
            return {**base, **meta, "answer": silence}
        seen_refs = {row["id"] for row in reference_rows}
        reference_rows += [row for row in rows if row["id"] not in seen_refs]

    # The passages the last answer cited, in citation order: "this narration"
    # needs the first, a comparison needs the first two.
    history_ids: list[int] = []
    for value in list(history_chunk_ids or []) + (
            [history_chunk_id] if history_chunk_id is not None else []):
        if value is not None and int(value) not in history_ids:
            history_ids.append(int(value))

    # A source the reader clicked in an earlier answer: pin exactly that
    # passage, whatever the question sounds like, so "its chain" or "explain
    # this" is about the hadith they pointed at.
    requested_rows: list = []
    if pinned_chunk_ids:
        requested_rows = db.get_chunks(conn, [int(i) for i in pinned_chunk_ids])
        if not requested_rows:
            return {**base, **meta, "answer": silence}

    pinned: list = list(requested_rows)
    seen = {row["id"] for row in pinned}
    pinned += [row for row in reference_rows if row["id"] not in seen]
    if plan.kind == "compare" and history_ids:
        extra = db.get_chunks(conn, history_ids[:2])
        seen = {row["id"] for row in pinned}
        pinned += [row for row in extra if row["id"] not in seen]
    elif plan.previous and history_ids and not chapter:
        extra = db.get_chunks(conn, [history_ids[0]])
        seen = {row["id"] for row in pinned}
        pinned += [row for row in extra if row["id"] not in seen]

    # "Do these two agree" with only one passage on screen has no honest
    # answer: searching for two unrelated passages would invent the
    # comparison the reader thinks they are looking at.
    if plan.kind == "compare" and plan.previous and len(pinned) < 2:
        return {**base, **meta, "answer": COMPARE_NEEDS_TWO}

    hits = []
    chapter_rows: list = []
    if plan.kind == "summarize" and chapter:
        chapter_rows = db.chapter_chunks(conn, chapter, book_ids)
        chosen_rows = _sample_evenly(chapter_rows, want)
        if not chosen_rows:
            return {**base, **meta, "answer": silence}
    elif plan.kind == "summarize" and pinned:
        chosen_rows = pinned
    else:
        hits = hybrid_search(
            conn, search_text, embedder,
            book_ids=book_ids, pool=pool, chapter=chapter, queries=plan.queries,
        )
        if not hits and not pinned:
            return {**base, **meta, "answer": silence}
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
            return {**base, **meta, "answer": silence}

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

    # A halal/haram answer must not appear token by token before the gate has
    # seen the citations: the reader would read the verdict even when the
    # passage behind it turns out not to state one. Those answers arrive whole.
    sensitive = plan.kind == "ruling" or _states_ruling(query)
    answer = llm.generate(
        system, user, max_tokens=_answer_budget(plan),
        on_token=None if sensitive else on_token,
    )

    if plan.kind == "ruling" and (
            not (answer or "").strip()
            or _is_not_found(answer) or _is_ruling_not_found(answer)):
        return {
            **base, **meta,
            "answer": RULING_NOT_FOUND,
            "retrieved": len(hits) or len(chapter_rows) or len(chosen_rows),
            "prompt_tokens": prompt_tokens,
            "excerpt_tokens": excerpt_tokens,
        }

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

    # The gate: a ruling answer stands only if an excerpt it cites states a
    # ruling itself. A confident sentence with no such passage behind it is
    # replaced by silence, however plausible it reads. This also catches a
    # halal/haram verdict that slipped through as an ordinary lookup: if the
    # answer pronounces one and no cited passage does, it is not shown.
    stated = any(
        _states_ruling(chosen_rows[cite["source_index"] - 1]["text"]) for cite in citations
    )
    if not stated and (plan.kind == "ruling" or _states_ruling(answer)):
        meta["notice"] = NOT_A_FATWA
        return {
            **base, **meta,
            "answer": RULING_NOT_FOUND,
            "retrieved": len(hits) or len(chapter_rows) or len(chosen_rows),
            "prompt_tokens": prompt_tokens,
            "excerpt_tokens": excerpt_tokens,
        }

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
