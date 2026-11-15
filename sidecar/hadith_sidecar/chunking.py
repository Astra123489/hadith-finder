"""Hadith-aware chunking.

PDFs of hadith collections are usually either
  (a) numbered ("[123]", "123.", "(123)" — ASCII or Arabic-Indic digits), or
  (b) delimited by حَدَّثَنَا at the start of a line/paragraph.

We split on those boundaries instead of blind fixed windows. Every chunk
carries page range, hadith number (when detectable) and the enclosing
chapter heading (كتاب / باب / Book / Chapter). Word bounding boxes are
kept per line and merged into highlight spans per page.

Number detection is stricter than it looks necessary, because printed
hadith volumes are full of numbers that are *not* hadith numbers: Qur'anic
footnotes ("[1] Al-Baqarah 2:238"), page numbers, and TOC entries. A line
only starts a new hadith when it opens with a hadith-shaped marker (a
bracketed number followed by the customary ordinal/parenthetical tail) or
a plain "123." numbering that continues the sequence, and the number must
move forward through the book. Footnote markers therefore never split a
narration in half.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .normalize import normalize_arabic, normalize_for_index

MIN_CHUNK_CHARS = 120
MAX_CHUNK_CHARS = 6000  # ~1500 tokens; longer hadiths split at sentence breaks
MAX_WORDS_PER_FALLBACK = 400
MAX_FORWARD_JUMP = 400  # a marker further than this from the last one is suspect

_AR_INDIC = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")

# OCR confusions seen inside bracketed hadith numbers (digit-position only).
_MARKER_FIX = str.maketrans({
    "O": "0", "o": "0", "C": "0", "c": "0",
    "I": "1", "l": "1", "i": "1", "|": "1", "^": "1", ".": "1", "·": "1",
    ";": "", " ": "", "\u00a0": "",
})

_MARKER_BRACKET = re.compile(r"^\s*[\[\(\{]\s*([0-9OoIl|^cC.;·\s]{2,9})\s*[\]\)\}jJ]?")
_MARKER_PLAIN = re.compile(r"^\s*(\d{1,4})\s*[.\-:)]\s+(?=\S)")
# OCR sometimes drops the brackets but keeps "1161 1 - (520)".
_MARKER_UNBRACKETED = re.compile(
    r"^\s*(\d{3,6})\s+\d{1,3}\s*(?:[.\-–—·…]+\s*)?\(\s*(?:\d{1,3}|\.{1,4}|…)?\s*\)"
)
# Ordinal / parenthetical tail: "57 - (553)", "50 (547)", "(549)", "1312 (...)"
_MARKER_TAIL = re.compile(r"^[\s,.\-–—·…0-9]{0,14}\(\s*(?:\d{1,3}|\.{1,4}|…)?\s*\)")
# "[1] Al-Baqarah 2:238", "[21] A1-Baqarah 2:115" — Qur'anic references, not hadiths.
_QURAN_REF = re.compile(r"^[\s,.\-–—·…]*(?:[A-Za-zĀ-ž'’\-]{2,}\s+)?\d{1,3}\s*:\s*\d{1,3}")

# OCR/ligature forms of "حدثنا" after normalization (line starts)
_AR_NARR_BOUNDARY = re.compile(r"^\s*حدثنا\b")

# Chapter / book headings (checked on normalized, casefolded line)
_CHAPTER = re.compile(r"^\s*(كتاب|باب|book|chapter)\b", re.IGNORECASE)
_SENT_SPLIT = re.compile(r"(?<=[.!?؟؛])\s+")

_TOC_LINE = re.compile(r"(?:\.{5,}|…{4,})")
_PAGE_NOISE = re.compile(r"^[\s\-–—]*\d{1,4}[\s\-–—]*$")


def _arabic_digits_to_ascii(s: str) -> str:
    return s.translate(_AR_INDIC)


@dataclass
class Word:
    text: str
    rect: tuple[float, float, float, float]


@dataclass
class Line:
    page: int  # 0-based page index
    text: str
    words: list[Word]
    font_size: float = 0.0
    bold: bool = False


@dataclass
class Chunk:
    page_start: int
    page_end: int
    hadith_ref: str | None
    chapter: str | None
    lines: list[Line] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(l.text for l in self.lines).strip()

    def spans(self) -> dict[int, list[list[float]]]:
        """Merged highlight rects per page (line-granular)."""
        per_page: dict[int, list[list[float]]] = {}
        for line in self.lines:
            if not line.words:
                continue
            x0 = min(w.rect[0] for w in line.words)
            y0 = min(w.rect[1] for w in line.words)
            x1 = max(w.rect[2] for w in line.words)
            y1 = max(w.rect[3] for w in line.words)
            per_page.setdefault(line.page, []).append([round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)])
        # Merge lines that share a visual row (wrapped lines and two-column
        # rows whose boxes overlap vertically but not exactly).
        merged: dict[int, list[list[float]]] = {}
        for page, rects in per_page.items():
            rects.sort(key=lambda r: (r[1], r[0]))
            out: list[list[float]] = []
            for r in rects:
                if out and _same_row(out[-1], r) and _gap(out[-1], r) < 40:
                    p = out[-1]
                    p[0], p[2] = min(p[0], r[0]), max(p[2], r[2])
                    p[1], p[3] = min(p[1], r[1]), max(p[3], r[3])
                else:
                    out.append(list(r))
            merged[page] = out
        return merged


def _same_row(a: list[float], b: list[float]) -> bool:
    overlap = min(a[3], b[3]) - max(a[1], b[1])
    height = min(a[3] - a[1], b[3] - b[1])
    return height > 0 and overlap > 0.5 * height


def _gap(a: list[float], b: list[float]) -> float:
    return abs(a[0] - b[2]) if b[2] >= a[0] else abs(b[0] - a[2])


def _candidates(token: str) -> list[int]:
    digits = token.translate(_MARKER_FIX)
    if not digits.isdigit():
        return []
    if len(digits) <= 4:
        return [int(digits)]
    # OCR ran digits together ("[12351"): every 4-digit window is a candidate.
    return sorted({int(digits[i:i + 4]) for i in range(len(digits) - 3)})


def _pick_candidate(cands: list[int], prev: int | None) -> int | None:
    if not cands:
        return None
    if prev is None:
        plausible = [c for c in cands if 1 <= c <= 3000]
        if not plausible:
            return None
        # OCR duplicates a digit at an edge ("111611" for 1161): the true
        # number is the middle reading, not the smallest or largest window.
        return plausible[len(plausible) // 2]
    valid = [c for c in cands if prev < c <= prev + MAX_FORWARD_JUMP]
    return min(valid) if valid else None


def _hadith_marker(line_text: str, prev: int | None, mode: str | None) -> int | None:
    """Return the hadith number when this line starts a new narration.

    mode "bracket": the book uses [1234] markers — only bracketed numbers may
    split, so TOC page numbers and list numbering can never poison the
    sequence. mode "plain": the book uses "123." numbering, accepted only at
    the expected next number and only on a line that reads like a narration.
    """
    if mode is None:
        return None
    text = _arabic_digits_to_ascii(line_text)
    if mode == "bracket":
        m = _MARKER_BRACKET.match(text)
        if not m:
            # Bracketless marker left behind by OCR: same structural tail test.
            m = _MARKER_UNBRACKETED.match(text)
            if not m:
                return None
            rest = text[m.end():]
            if _QURAN_REF.match(rest):
                return None
            return _pick_candidate(_candidates(m.group(1)), prev)
        token = m.group(1)
        rest = text[m.end():]
        structural = bool(_MARKER_TAIL.match(rest))
        if _QURAN_REF.match(rest):
            return None
        if not structural:
            # Unambiguous sequence continuation only; a bare bracket can be a
            # footnote number, a Qur'an verse note or a page reference.
            if prev is None:
                return None
            cands = _candidates(token)
            if prev + 1 not in cands:
                return None
            return prev + 1
        pick = _pick_candidate(_candidates(token), prev)
        return pick

    m = _MARKER_PLAIN.match(text)
    if not m:
        return None
    token = m.group(1)
    rest = text[m.end():]
    if len(rest.split()) < 6:
        return None  # TOC entry or list item, not a narration
    if _TOC_LINE.search(rest) or _QURAN_REF.match(rest):
        return None
    if re.search(r"\s\d{1,4}\s*$", rest) and rest[:1].isupper():
        return None  # "Chapter title 357" — contents page
    n = int(token)
    if prev is None:
        return n if n <= 3 else None
    return n if n == prev + 1 else None


def _is_ar_narr_boundary(norm_line: str) -> bool:
    return bool(_AR_NARR_BOUNDARY.match(norm_line))


def _is_chapter(line: Line) -> bool:
    t = line.text.strip()
    if not (3 <= len(t) <= 90):
        return False
    if _TOC_LINE.search(t):
        return False
    norm = normalize_arabic(t).lower()
    if _CHAPTER.match(norm):
        return True
    if t[-1] in ".,;:!?،؛…-–—'\"”“":
        return False
    if t[0] in "[({\"'»«":
        return False
    if re.search(r"\d\s*$", t):
        return False
    if any(ch.isdigit() for ch in t) or "\ufffd" in t:
        return False
    words = t.split()
    if not 2 <= len(words) <= 12:
        return False
    letters = sum(ch.isalpha() or ch.isspace() for ch in t)
    if letters / len(t) < 0.9:
        return False
    # Prose lines inside a narration are routinely set in a bold-ish face, so
    # "bold and short" is not enough: a heading has to read like a title.
    lowercase_words = {"the", "of", "and", "in", "to", "a", "an", "that", "was",
                       "is", "said", "says", "saw", "stood", "for", "with", "on",
                       "at", "by", "from", "as", "his", "her", "he", "she", "they",
                       "we", "you", "it", "this", "these", "those", "who", "which"}
    if any(w in lowercase_words for w in words):
        return False
    capitalised = sum(1 for w in words if w[:1].isupper())
    if line.font_size >= 12.5 and capitalised >= 0.6 * len(words):
        return True
    return len(words) >= 3 and capitalised == len(words)


def _is_heading_continuation(text: str) -> bool:
    t = text.strip()
    if not (3 <= len(t) <= 60) or len(t.split()) > 8:
        return False
    if t[-1] in ".,;:!?،؛…-–—'\"”“":
        return False
    if any(ch.isdigit() for ch in t) or "\ufffd" in t:
        return False
    first = t.split()[0]
    return first[:1].islower() or first in {"Of", "And", "The", "In", "To", "For", "With"}


def _is_toc_line(text: str) -> bool:
    return bool(_TOC_LINE.search(text))


def _is_page_noise(text: str) -> bool:
    return bool(_PAGE_NOISE.match(text))


def _flush(lines: list[Line], hadith_ref: str | None, chapter: str | None) -> Chunk | None:
    lines = [l for l in lines if l.text.strip()]
    if not lines:
        return None
    text = "\n".join(l.text for l in lines).strip()
    if len(text) < 30:
        return None
    return Chunk(
        page_start=lines[0].page,
        page_end=lines[-1].page,
        hadith_ref=hadith_ref,
        chapter=chapter,
        lines=lines,
    )


def _marker_mode(all_lines: list[Line]) -> str | None:
    """Decide which numbering style this book uses (bracketed wins)."""
    bracketed = plain = 0
    for line in all_lines:
        text = _arabic_digits_to_ascii(line.text)
        m = _MARKER_BRACKET.match(text)
        if m:
            rest = text[m.end():]
            if _MARKER_TAIL.match(rest) and not _QURAN_REF.match(rest):
                bracketed += 1
    if bracketed >= 1:
        return "bracket"
    for line in all_lines:
        text = _arabic_digits_to_ascii(line.text)
        if _MARKER_PLAIN.match(text):
            rest = text[_MARKER_PLAIN.match(text).end():]
            if len(rest.split()) >= 6 and not _TOC_LINE.search(rest):
                plain += 1
    return "plain" if plain >= 3 else None


def chunk_lines(all_lines: list[Line]) -> list[Chunk]:
    mode = _marker_mode(all_lines)
    chunks: list[Chunk] = []
    cur: list[Line] = []
    cur_ref: str | None = None
    cur_chapter: str | None = None
    current_chapter: str | None = None
    prev_ref: int | None = None
    heading_counts: dict[str, int] = {}
    just_saw_heading = False

    def flush():
        nonlocal cur, cur_ref, cur_chapter
        c = _flush(cur, cur_ref, cur_chapter)
        if c is not None:
            chunks.append(c)
        cur, cur_ref, cur_chapter = [], None, None

    for line in all_lines:
        raw = line.text.strip()
        if not raw:
            continue

        marker = _hadith_marker(raw, prev_ref, mode)
        if marker is not None:
            flush()
            prev_ref = marker
            cur_ref = str(marker)
            cur_chapter = current_chapter
            cur.append(line)
            continue

        if _is_ar_narr_boundary(normalize_arabic(raw).lower().strip()):
            flush()
            cur_ref = None
            cur_chapter = current_chapter
            cur.append(line)
            continue

        if _is_chapter(line):
            text = " ".join(raw.split())
            heading_counts[text] = heading_counts.get(text, 0) + 1
            if heading_counts[text] <= 2:
                current_chapter = text
            just_saw_heading = True
            continue

        # A wrapped heading leaves one lowercase continuation line immediately
        # after it; fold that into the chapter instead of the narration. The
        # flag keeps ordinary lowercase body lines from being swallowed.
        if just_saw_heading and current_chapter and _is_heading_continuation(raw):
            current_chapter = f"{current_chapter} {raw.strip()}"
            continue
        just_saw_heading = False

        # Running-heads, TOC dot leaders and bare page numbers are not content.
        if _is_toc_line(raw) or _is_page_noise(raw):
            continue

        if not cur:
            cur_chapter = current_chapter
        cur.append(line)

    flush()

    # Front matter/TOC: with a numbered book, anything before the first numbered
    # narration is cover, copyright and contents pages — drop it.
    first_ref = next((i for i, c in enumerate(chunks) if c.hadith_ref), None)
    if first_ref:
        chunks = chunks[first_ref:]

    # Merge tiny reference-less fragments into the previous chunk (same chapter).
    merged: list[Chunk] = []
    for c in chunks:
        tiny_refless = c.hadith_ref is None and len(c.text) < MIN_CHUNK_CHARS
        if merged and merged[-1].chapter == c.chapter and tiny_refless:
            prev = merged[-1]
            prev.lines.extend(c.lines)
            prev.page_end = c.page_end
        else:
            merged.append(c)
    chunks = merged

    # Hard-split oversized chunks at line boundaries (never over the cap).
    final: list[Chunk] = []
    for c in chunks:
        if len(c.text) <= MAX_CHUNK_CHARS:
            final.append(c)
            continue
        final.extend(_split_oversized(c))
    return [c for c in final if len(c.text.strip()) >= 30]


def _fallback_split(chunks: list[Chunk]) -> list[Chunk]:
    lines = [l for c in chunks for l in c.lines]
    out: list[Chunk] = []
    cur: list[Line] = []
    count = 0
    for l in lines:
        cur.append(l)
        count += len(l.words)
        if count >= MAX_WORDS_PER_FALLBACK:
            flushed = _flush(cur, None, chunks[0].chapter if chunks else None)
            if flushed is not None:
                out.append(flushed)
            cur, count = [], 0
    if cur:
        flushed = _flush(cur, None, chunks[0].chapter if chunks else None)
        if flushed is not None:
            out.append(flushed)
    return out


def _split_oversized(c: Chunk) -> list[Chunk]:
    parts: list[Chunk] = []
    cur_lines: list[Line] = []
    cur_len = 0
    for l in c.lines:
        added = len(l.text) + 1
        if cur_lines and cur_len + added > MAX_CHUNK_CHARS:
            out = _flush(cur_lines, c.hadith_ref, c.chapter)
            if out is not None:
                parts.append(out)
            cur_lines, cur_len = [], 0
        cur_lines.append(l)
        cur_len += added
    if cur_lines:
        out = _flush(cur_lines, c.hadith_ref, c.chapter)
        if out is not None:
            parts.append(out)
    return [p for p in parts if p]


def chunk_text_for_index(text: str) -> str:
    return normalize_for_index(text)
