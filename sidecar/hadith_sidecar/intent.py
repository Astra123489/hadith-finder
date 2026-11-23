"""What the reader is asking for, before any model runs.

Readers ask a handful of shapes of question over hadith: a lookup ("who
narrated the hadith of intentions"), a list ("bring me ten on patience"), a
summary ("summarise this chapter"), or a location ("which page is it on").
Retrieval treats those very differently — a list needs ten *distinct*
passages, a summary needs the chapter's arc rather than its best-matching
lines, a location needs the page metadata that is already in every excerpt.

This module makes that decision.  It is deterministic and free: no model
call, no I/O, no latency.  The model still writes the answer; this only
chooses the net that is cast before it does.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .normalize import fts_tokens

MAX_LIST = 20
DEFAULT_LIST = 10
DEFAULT_SUMMARY = 6
DEFAULT_K = 5

_AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")

_WORD_NUMBERS = {
    "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
}
_NUMBER_WORDS = "|".join(_WORD_NUMBERS)

_LIST = re.compile(
    r"\b(?:give|bring|show|list|find|get|fetch|pull|hand)\s+(?:me|us)\b"
    r"|\bwhat\s+are\s+some\b"
    r"|\b(?:hadiths|narrations|traditions)\s+(?:about|on|regarding|concerning|related\s+to)\b"
    r"|\b(?:list|enumerate)\s+(?:the\s+)?(?:hadiths?|narrations?)\b"
    r"|(?:أعطني|اعطني|هات|اذكر|أريد|اريد)\b",
    re.I,
)
_COUNT_AFTER_HADITH = re.compile(
    r"\b(\d{1,3}|" + _NUMBER_WORDS + r")\s+(?:more\s+)?"
    r"(?:hadiths?|narrations?|traditions?|examples?|ones?)\b",
    re.I,
)
_COUNT_AFTER_ME = re.compile(
    r"\b(?:give|bring|show|list|get|fetch|hand)\s+(?:me|us)\s+(?:up\s+)?"
    r"(\d{1,3}|" + _NUMBER_WORDS + r")\b",
    re.I,
)
_COUNT_ARABIC = re.compile(r"(\d{1,3})\s*(?:حديث|أحاديث|احاديث)")
_MANY = re.compile(
    r"\b(?:hadiths|narrations|traditions|examples|ones)\b"
    r"|\b(?:list|enumerate)\b|(?:أحاديث|احاديث)",
    re.I,
)

_SUMMARIZE = re.compile(
    r"\b(?:summar(?:y|ise|ize|ised|ized)|sum\s+up|overview|gist|tl;?dr|outline|condense)\b"
    r"|\bin\s+brief\b|(?:لخّ?ص|تلخيص|ملخص)",
    re.I,
)
_LOCATE = re.compile(
    r"\b(?:which|what)\s+(?:page|book|chapter|volume)\b"
    r"|\bwhere\s+(?:is|are|can\s+i\s+find|do\s+i\s+find|does)\b"
    r"|\bpage\s+number\b|\bwhich\s+hadith\b"
    r"|(?:في\s+أي\s+صفحة|في\s+اي\s+صفحة|أين\s+يوجد|اين\s+يوجد)",
    re.I,
)
_NOVEL = re.compile(
    r"\b(?:not|less)\s+(?:so\s+)?common(?:ly)?\b|\buncommon\b|\brare\b"
    r"|\blesser[- ]known\b|\bunfamiliar\b|\bobscure\b|\bunusual\b|\bunheard\b"
    r"|\bnew\s+(?:to\s+me|ones?|hadiths?)\b|\b(?:don'?t|do\s+not)\s+know\b"
    r"|(?:غير\s+مشهور|غير\s+معروف|نادرة)",
    re.I,
)
# The questions readers actually bring to hadith collections, in the order
# they show up on Islam Stack Exchange and in fatwa answers: "is this hadith
# authentic?", "who narrated it?", "what does it mean?".
_GRADE = re.compile(
    r"\bauthenticity\b"
    r"|\bgrad(?:e|ing)\b"
    r"|\bis\s+(?:this|it|that|the following|the narration|the hadith)\b[^?.!]{0,80}?"
    r"\b(?:authentic|sahih|hasan|da'?if|dha'?if|weak|fabricated|mawdu|reliable|valid|sound)\b"
    r"|\bsahih\s+or\s+(?:fake|fabricated|weak)\b"
    r"|\b(?:check|verify)\s+(?:this|the)\s+(?:hadith|narration)\b"
    r"|(?:ما\s+صحة|صحة\s+الحديث|درجة\s+الحديث|هل\s+هذا\s+الحديث|هل\s+الحديث"
    r"|حديث\s+صحيح|حديث\s+ضعيف|موضوع)",
    re.I,
)
_NARRATOR = re.compile(
    r"\bwho\s+(?:narrated|reported|transmitted|related|said)\s+(?:this|it|that|the)\b"
    r"|\bwho\s+is\s+the\s+narrator\b|\bnarrator(?:s)?\b"
    r"|\bisnad\b|\bsanad\b|\bchain\s+of\s+(?:narration|transmission|narrators)\b"
    r"|\bfrom\s+whom\b|\bwho\s+(?:did|has)\s+(?:narrate|report)"
    r"|(?:من\s+روى|من\s+حدث|رواة|الإسناد|الاسناد|السند|سند\s+الحديث)",
    re.I,
)
_EXPLAIN = re.compile(
    r"\bwhat\s+does\s+(?:this|it|that|the\s+(?:hadith|narration|word|phrase))\b[^?.!]{0,40}?\bmean\b"
    r"|\bmeaning\s+of\b|\bexplain\b|\bexplanation\b|\bcommentary\s+on\b"
    r"|(?:ما\s+معنى|معنى\s+الحديث|اشرح|شرح\s+الحديث|فسر|تفسير\s+الحديث)",
    re.I,
)
_CHAPTER_PHRASE = re.compile(
    r"\b(?:chapter|book|section|kitab|bab)\s+"
    r"(?:on|about|of|titled|called|named|entitled)\s+"
    r"([^?.!;:\n]{3,80})",
    re.I,
)
_THIS_CHAPTER = re.compile(
    r"\bthis\s+(?:chapter|book|section)\b"
    r"|\bthe\s+same\s+(?:chapter|book)\b"
    r"|(?:هذا\s+الباب)",
    re.I,
)
# "that hadith", "this narration", "the passage above": the reader is
# pointing at something already on screen, not naming a new subject.
_THIS_REF = re.compile(
    r"\bthis\s+(?:chapter|book|section|narration|hadith|passage|answer)\b"
    r"|\b(?:that|the above|the same)\s+(?:narration|hadith|passage)\b"
    r"|\b(?:who|what)\s+(?:said|narrated|reported|related)\s+this\b"
    r"|\bwhere\s+(?:is|was)\s+it\b"
    r"|(?:هذا\s+الحديث|هذا\s+النص)",
    re.I,
)

# Words that say nothing about the topic on their own: a query made only of
# these ("summarise this chapter") must stay as written, or retrieval gets
# "chapter" and nothing else.
_GENERIC_WORDS = {
    "this", "that", "it", "them", "these", "those", "one", "ones", "same",
    "chapter", "book", "section", "part", "narration", "narrations",
    "hadith", "hadiths", "passage", "passages", "excerpt", "excerpts",
    "more", "above", "previous", "answer", "text", "please",
    "the", "a", "an", "of", "on", "in", "and", "about", "regarding",
    "concerning", "related", "to",
}

_SCAFFOLD = [
    (re.compile(r"^\s*(?:please\s+)?(?:can|could|would)\s+you\s+", re.I), ""),
    (re.compile(
        r"^\s*(?:give|bring|show|list|find|get|fetch|pull|hand)\s+(?:me|us)\s+(?:up\s+)?",
        re.I), ""),
    (re.compile(
        r"\b(?:\d{1,3}|" + _NUMBER_WORDS + r")\s+(?:more\s+)?"
        r"(?:hadiths?|narrations?|traditions?|examples?|ones?)\b", re.I), " "),
    (re.compile(r"\b(?:that|which)\s+(?:i|we)\s+(?:do\s+not|don'?t|did\s+not)\s+know\b", re.I), " "),
    (re.compile(r"\b(?:not|less)\s+(?:so\s+)?common(?:ly)?\b", re.I), " "),
    (re.compile(r"\b(?:rare|uncommon|lesser[- ]known|unfamiliar|obscure|unusual|unheard)\b", re.I), " "),
    (re.compile(
        r"^\s*(?:summar(?:y|ise|ize)|sum\s+up|overview(?:\s+of)?|outline(?:\s+of)?|tl;?dr|condense)\b",
        re.I), " "),
    (re.compile(r"\b(?:the\s+)?(?:chapter|book|section)\s+(?:on|about|of|titled|called|named|entitled)\b",
                re.I), " "),
    (re.compile(r"\b(?:about|on|regarding|concerning|related\s+to)\b", re.I), " "),
]


@dataclass(frozen=True)
class Intent:
    """How to search and how to answer, derived from one question."""

    kind: str = "lookup"           # lookup | list | summarize | locate | grade | narrator | explain
    count: int | None = None       # narrations the reader asked for
    chapter: str | None = None     # chapter hint, resolved by the caller
    novel: bool = False            # "not common", "I don't know these"
    search_text: str = ""          # the topical core of the question
    queries: tuple[str, ...] = ()  # extra phrasings, fused in retrieval
    top_k: int = DEFAULT_K
    previous: bool = False         # "this narration", "that hadith"

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "count": self.count,
            "chapter": self.chapter,
            "novel": self.novel,
            "top_k": self.top_k,
            "previous": self.previous,
        }


def _to_int(raw: str | None) -> int | None:
    if not raw:
        return None
    raw = raw.translate(_AR_DIGITS).strip().lower()
    if raw.isdigit():
        value = int(raw)
        return value if 1 <= value <= 99 else None
    return _WORD_NUMBERS.get(raw)


def _count(text: str) -> int | None:
    for pattern in (_COUNT_AFTER_HADITH, _COUNT_AFTER_ME, _COUNT_ARABIC):
        match = pattern.search(text)
        if match:
            value = _to_int(match.group(1))
            if value:
                return value
    return None


def _chapter_hint(text: str, history_chapter: str | None) -> str | None:
    if _THIS_CHAPTER.search(text):
        return history_chapter
    match = _CHAPTER_PHRASE.search(text)
    if match:
        hint = match.group(1).strip(" \"'“”….,;")
        return hint or None
    return None


def _core(text: str) -> str:
    core = text
    for pattern, replacement in _SCAFFOLD:
        core = pattern.sub(replacement, core)
    core = re.sub(r"^\s*(?:the|a|an)\s+", "", core, flags=re.I)
    return " ".join(core.split()).strip(" ,;:-–—?.!")


def _topicless(text: str) -> bool:
    tokens = re.findall(r"[0-9A-Za-z\u0600-\u06FF]{2,}", text.lower())
    return not tokens or all(token in _GENERIC_WORDS for token in tokens)


def classify(query: str, *, history_chapter: str | None = None) -> Intent:
    """Turn a question into a retrieval/answer plan.

    ``history_chapter`` is the chapter of the previous answer's first
    citation, so "summarise this chapter" can mean the chapter the reader is
    looking at rather than a chapter named in the question.
    """
    text = " ".join((query or "").split())
    if not text:
        return Intent(search_text="", queries=())

    raw_count = _count(text)
    asks_for_many = raw_count is not None or bool(_MANY.search(text))

    if _SUMMARIZE.search(text):
        kind = "summarize"
    elif asks_for_many and (
            _LIST.search(text) or _COUNT_AFTER_HADITH.search(text) or _COUNT_AFTER_ME.search(text)):
        kind = "list"
    elif _GRADE.search(text):
        kind = "grade"
    elif _NARRATOR.search(text):
        kind = "narrator"
    elif _LOCATE.search(text):
        kind = "locate"
    elif _EXPLAIN.search(text):
        kind = "explain"
    else:
        kind = "lookup"

    count = raw_count if kind in ("list", "summarize") else None

    if kind == "list":
        top_k = max(3, min(MAX_LIST, count or DEFAULT_LIST))
    elif kind == "summarize":
        top_k = min(10, max(4, count or DEFAULT_SUMMARY))
    else:
        top_k = DEFAULT_K

    core = _core(text)
    search_text = text if _topicless(core) else core
    queries = (text,) if search_text != text and kind != "list" else ()

    return Intent(
        kind=kind,
        count=count,
        chapter=_chapter_hint(text, history_chapter),
        novel=bool(_NOVEL.search(text)),
        search_text=search_text,
        queries=queries,
        top_k=top_k,
        previous=bool(_THIS_REF.search(text)),
    )


def match_chapter(chapters: list[str], hint: str, *, minimum: float = 0.6) -> str | None:
    """The chapter the hint most likely means, or None when it is guesswork.

    Matching is by topic words only — "chapter on intentions" should reach
    "Book of Intentions" without the word "chapter" having to appear in the
    heading.  Ties go to the shorter heading, which is the more specific one.
    """
    tokens = {token for token in fts_tokens(hint) if token not in _GENERIC_WORDS}
    if not tokens:
        return None
    best: str | None = None
    best_score = 0.0
    for name in chapters:
        name_tokens = set(fts_tokens(name))
        if not name_tokens:
            continue
        overlap = len(tokens & name_tokens) / len(tokens)
        if overlap < minimum:
            continue
        score = overlap - 0.001 * len(name.split())
        if score > best_score:
            best, best_score = name, score
    return best
