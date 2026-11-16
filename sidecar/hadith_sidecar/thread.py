"""Multi-turn conversation over the loaded books.

The sidecar owns the transcript: ``ask`` returns it whole, so the window shows
the same conversation the prompt was built from.

A follow-up prompt looks like this::

    Excerpts:
    [S1] ...

    Earlier in this conversation:
    User: ...
    Assistant: ...            (citation tags stripped — numbering restarts)
    Summary of earlier turns: ...

    Question: ...

Compaction keeps the most recent turns verbatim (a follow-up almost always
refers to them) and folds everything older into a summary. It runs on its own
once the next prompt would pass 70% of the context window, and on demand from
the context panel in the Ask view.

Suggested follow-ups are the gaps an answer left open: a detail it skipped,
whether the narration is authentic, a related ruling, or another collection
that records it. The model phrases them; when it is unavailable or unhelpful
the gap list is built from the citations themselves.
"""

from __future__ import annotations

import datetime
import re

from .llm import count_tokens
from .qa import SYSTEM_PROMPT

# Fraction of the context window at which older turns get folded away.
COMPACT_AT = 0.7
# Turns kept verbatim through a compaction.
KEEP_TURNS = 2
MAX_SUMMARY_CHARS = 2400
# How much of a past answer the prompt keeps. The whole text lives in the
# transcript; the model only needs the gist of earlier turns.
MAX_ANSWER_CHARS = 900
# The current question's excerpts never shrink below this, or the answer loses
# the grounding that makes it worth having a conversation about.
MIN_EXCERPT_TOKENS = 600

SUMMARY_SYSTEM = """You compress a research conversation so it can continue in a smaller window.
Rewrite the exchange as a factual summary of at most 150 words: what was asked, what the sources
said, which books, chapters and hadith numbers were cited, and what stayed uncertain. Keep names
and numbers exact. No preamble, no bullet points, no new claims."""

SUGGEST_SYSTEM = """You help a reader keep researching hadith. Given one question and the answer it
received, write {n} short follow-up questions that dig into THE ANSWER ITSELF: a term or claim it
left unexplained, how its points relate, a condition or exception it skipped, or what it means in
practice. Quote or echo the answer's own wording. Do not ask whether the narration is authentic,
which collection records it, or what other chapters say — those are about the sources, not the
answer. One per line, numbered like "1.". No answers, no commentary, at most 14 words each."""

_CITE_TAG = re.compile(r"\[S\d+\]")
_NUMBERING = re.compile(r"^(?:[-*\u2022]|\d+[.)])\s*")
# Suggestions about where a narration is recorded or how it is graded come from
# the source cards, not from the answer — the reader asked for the answer.
_SOURCE_ONLY = re.compile(
    r"(narrate[sd]? this|grade[sd]? this|this narration is|is this narration|"
    r"which collection|where else .* recorded|do these passages agree|"
    r"what else (is|appears) in the (chapter|book)|other collections)",
    re.I,
)
_WORDS = re.compile(r"[A-Za-z\u0600-\u06FF']{3,}")
_STOP = {
    "the", "and", "for", "was", "were", "what", "which", "who", "whom", "does",
    "did", "are", "has", "have", "had", "this", "that", "these", "those", "with",
    "from", "about", "into", "when", "where", "why", "how", "his", "her", "their",
    "there", "they", "them", "but", "not", "you", "your", "can", "could", "would",
    "should", "tell", "explain", "find", "show", "give", "summarize", "summarise",
    "prophet", "allah", "hadith", "hadiths", "narrated", "say", "says", "said",
}


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def new_conversation(chat_id: str = "", title: str = "") -> dict:
    now = _now()
    return {
        "id": chat_id,
        "title": title,
        "turns": [],
        "summary": "",
        "compactions": 0,
        "next_id": 1,
        "created_at": now,
        "updated_at": now,
    }


def title_for(question: str, limit: int = 58) -> str:
    """A chat is named after the question that started it."""
    text = " ".join((question or "").split()).rstrip("?.!")
    if not text:
        return "New chat"
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return (cut or text[:limit]).rstrip() + "…"


def _plain(answer: str) -> str:
    """An answer without citation tags: numbering restarts every turn."""
    return " ".join(_CITE_TAG.sub("", answer or "").split())


def history_block(conv: dict, *, max_answer: int = MAX_ANSWER_CHARS) -> str:
    parts: list[str] = []
    if conv["summary"]:
        parts.append("Summary of earlier turns:\n" + conv["summary"])
    for turn in conv["turns"]:
        answer = _plain(turn["answer"])
        if len(answer) > max_answer:
            answer = answer[:max_answer].rstrip() + " …"
        parts.append(f"User: {turn['question']}\nAssistant: {answer}")
    return "\n\n".join(parts)


def add_turn(conv: dict, question: str, result: dict) -> dict:
    turn = {
        "id": conv["next_id"],
        "question": question,
        "answer": result.get("answer", ""),
        "citations": result.get("citations", []),
        "unverified": bool(result.get("unverified")),
        "created_at": _now(),
    }
    conv["next_id"] += 1
    conv["turns"].append(turn)
    conv["updated_at"] = turn["created_at"]
    if not conv.get("title"):
        conv["title"] = title_for(question)
    return turn


def rewind(conv: dict, turn_id) -> dict:
    """Drop one turn and everything asked after it, so it can be retried or edited.

    The reader is not deleting the conversation, only returning to the moment
    before a question: the transcript, the title and the id stay put and the
    tail is cut away.
    """
    try:
        wanted = int(turn_id)
    except (TypeError, ValueError):
        raise ValueError("params.turn_id must be a whole number") from None
    index = next((i for i, t in enumerate(conv["turns"]) if t["id"] == wanted), None)
    if index is None:
        raise ValueError(f"no such turn: {turn_id}")
    dropped = conv["turns"][index:]
    conv["turns"] = conv["turns"][:index]
    # A summary describes turns that are no longer in the transcript; with
    # nothing left it would be a memory of a conversation just rolled back.
    if not conv["turns"]:
        conv["summary"] = ""
    conv["updated_at"] = _now()
    return {"dropped": len(dropped), "question": dropped[0]["question"]}


def usage(llm, conv: dict, ctx: int, *, excerpt_tokens: int = 0) -> dict:
    """Size of the prompt the next question would send, against the window."""
    base = count_tokens(llm, SYSTEM_PROMPT)
    history = count_tokens(llm, history_block(conv))
    used = min(ctx, base + history + excerpt_tokens)
    return {
        "used": used,
        "ctx": ctx,
        "compact_at": int(ctx * COMPACT_AT),
        "pct": round(used / ctx, 4) if ctx else 0.0,
        "history_tokens": history,
        "excerpt_tokens": excerpt_tokens,
        "turns": len(conv["turns"]),
        "compactions": conv["compactions"],
    }


def projected(llm, conv: dict, ctx: int) -> int:
    """Tokens the next prompt is expected to need, before retrieval runs."""
    return (
        count_tokens(llm, SYSTEM_PROMPT)
        + count_tokens(llm, history_block(conv))
        + MIN_EXCERPT_TOKENS
    )


def needs_compaction(llm, conv: dict, ctx: int) -> bool:
    return bool(conv["turns"]) and projected(llm, conv, ctx) > int(ctx * COMPACT_AT)


def excerpt_chars(llm, conv: dict, ctx: int, limit: int, *, top_k: int = 5) -> int:
    """How long each excerpt may be once the conversation has taken its share.

    Retrieval clips every passage to this, so a long conversation makes the
    reminders shorter instead of pushing the prompt past the window.
    """
    base = count_tokens(llm, SYSTEM_PROMPT)
    history = count_tokens(llm, history_block(conv))
    available = max(MIN_EXCERPT_TOKENS, ctx - base - history - 320)
    per_excerpt = int(available * 2.6 / max(1, top_k))
    return max(600, min(limit, per_excerpt))


def _fallback_summary(previous: str, turns: list[dict]) -> str:
    asked = "; ".join(
        " ".join(t["question"].split()).rstrip("?.!")[:120] for t in turns[-8:]
    )
    books: list[str] = []
    for turn in turns:
        for cite in turn.get("citations") or []:
            name = cite.get("book_title")
            if name and name not in books:
                books.append(name)
    text = f"Earlier questions: {asked}."
    if books:
        text += f" Sources cited: {', '.join(books[:6])}."
    if previous:
        text = previous[:600] + " " + text
    return " ".join(text.split())[:MAX_SUMMARY_CHARS]


def _summarise(llm, previous: str, turns: list[dict]) -> str:
    transcript = "\n\n".join(
        f"User: {t['question']}\nAssistant: {_plain(t['answer'])[:600]}" for t in turns
    )
    if previous:
        transcript = f"Earlier summary:\n{previous}\n\nNew exchange:\n{transcript}"
    try:
        text = llm.generate(SUMMARY_SYSTEM, transcript, max_tokens=400, temperature=0.2)
    except Exception:
        text = ""
    text = " ".join((text or "").split())
    # A model that had nothing to work from echoes its refusal; that is not a
    # summary, so fall back to the questions that were actually asked.
    if len(text) >= 40 and "could not find this" not in text.lower():
        return text[:MAX_SUMMARY_CHARS]
    return _fallback_summary(previous, turns)


def compact(conv: dict, llm, *, keep: int = KEEP_TURNS, notify=None,
            force: bool = False) -> dict:
    """Fold the oldest turns into a summary. Returns what actually happened."""
    turns = conv["turns"]
    dropped = turns[:-keep] if keep > 0 else list(turns)
    if not dropped and force and len(turns) > 1:
        dropped = turns[:-1]
    if not dropped:
        return {"compacted": False, "dropped": 0, "summary": conv["summary"]}
    if notify:
        notify(stage="compacting", dropped=len(dropped))
    summary = _summarise(llm, conv["summary"], dropped)
    conv["summary"] = summary
    conv["turns"] = turns[len(dropped):]
    conv["compactions"] += 1
    if notify:
        notify(
            stage="compacted",
            dropped=len(dropped),
            compactions=conv["compactions"],
            summary=conv["summary"],
        )
    return {"compacted": True, "dropped": len(dropped), "summary": conv["summary"]}


def _topic(question: str) -> str:
    words = [w for w in _WORDS.findall(question) if w.lower() not in _STOP]
    return " ".join(words[:5]).lower() or "this topic"


# Words that must not end a quoted claim ("…intention so the reward").
_TRAILING = {
    "so", "and", "but", "or", "that", "which", "because", "if", "when", "then",
    "with", "for", "to", "of", "in", "as", "by", "from", "the", "a", "an",
    "is", "are", "was", "were", "it", "this", "these", "those", "they", "he",
    "she", "his", "her", "their",
}


def _answer_phrase(answer: str) -> str:
    """The answer's own opening claim, short enough to quote in a question."""
    text = _plain(answer)
    if not text:
        return ""
    sentence = re.split(r"(?<=[.!?])\s+", text.strip())[0]
    clause = re.split(r"[,;:\u2014]|\s-\s", sentence)[0].strip()
    if len(clause.split()) < 3:
        clause = sentence.strip()
    words = clause.split()
    if len(words) > 9:
        words = words[:9]
        while words and words[-1].lower().strip(",;:.") in _TRAILING:
            words.pop()
    return " ".join(words).strip(" ,.;:!?\u201c\u201d")[:90]


def _gaps(turn: dict) -> list[str]:
    """Questions the answer itself invites, used when the model says nothing useful.

    These follow the answer's content, not the source cards: an unexplained
    claim, a missing condition, what it looks like in practice.
    """
    if not turn.get("citations"):
        topic = _topic(turn["question"])
        return [
            f"Which of my books discusses {topic}?",
            f"What did the Prophet ﷺ teach about {topic}?",
        ]
    gaps: list[str] = []
    phrase = _answer_phrase(turn["answer"])
    if phrase:
        gaps.append(f"What does “{phrase}” mean in practice?")
    gaps.append("Are there conditions or exceptions to this?")
    gaps.append("How would that look in an ordinary day?")
    gaps.append("Which of these points matters most, and why?")
    return gaps


def _parse_suggestions(text: str, question: str) -> list[str]:
    seen = {" ".join(question.lower().split())}
    out: list[str] = []
    for raw in (text or "").splitlines():
        line = _NUMBERING.sub("", raw.strip()).strip().strip('"').strip()
        line = " ".join(line.split())
        if not (8 <= len(line) <= 160) or "?" not in line:
            continue
        if _SOURCE_ONLY.search(line):
            continue
        key = line.lower().rstrip("?")
        if key in seen:
            continue
        seen.add(key)
        out.append(line if line.endswith("?") else line + "?")
    return out


def _suggest_prompt(turn: dict) -> str:
    citations = turn.get("citations") or []
    labels: list[str] = []
    for c in citations[:5]:
        bits = [c.get("book_title") or "", c.get("chapter") or ""]
        if c.get("hadith_ref"):
            bits.append(f"hadith {c['hadith_ref']}")
        label = " · ".join(b for b in bits if b)
        if label:
            labels.append(label)
    lines = [
        f"Question: {turn['question']}",
        f"Answer: {_plain(turn['answer'])[:900]}",
    ]
    if labels:
        lines.append("Sources: " + "; ".join(labels))
    return "\n".join(lines)


def suggest(conv: dict, llm, count: int = 3) -> list[str]:
    count = max(1, min(5, int(count)))
    if not conv["turns"]:
        return []
    turn = conv["turns"][-1]
    try:
        text = llm.generate(
            SUGGEST_SYSTEM.format(n=count),
            _suggest_prompt(turn),
            max_tokens=220,
            temperature=0.5,
        )
        out = _parse_suggestions(text, turn["question"])
    except Exception:
        out = []
    seen = {" ".join(s.lower().split()) for s in out}
    for gap in _gaps(turn):
        if len(out) >= count:
            break
        if gap.lower() in seen:
            continue
        out.append(gap)
        seen.add(gap.lower())
    return out[:count]


def snapshot(conv: dict, usage_info: dict) -> dict:
    return {
        "id": conv.get("id") or "",
        "title": conv.get("title") or "",
        "turns": conv["turns"],
        "summary": conv["summary"],
        "compactions": conv["compactions"],
        "usage": usage_info,
    }
