"""Hadith-aware chunking.

PDFs of hadith collections are usually either
  (a) numbered ("[123]", "123.", "(123)" — ASCII or Arabic-Indic digits), or
  (b) delimited by حَدَّثَنَا at the start of a line/paragraph.

We split on those boundaries instead of blind fixed windows. Every chunk
carries page range, hadith number (when detectable) and the enclosing
chapter heading (كتاب / باب / Book / Chapter). Word bounding boxes are
kept per line and merged into highlight spans per page.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .normalize import normalize_arabic, normalize_for_index

MIN_CHUNK_CHARS = 120
MAX_CHUNK_CHARS = 6000  # ~1500 tokens; longer hadiths split at sentence breaks
MAX_WORDS_PER_FALLBACK = 400

_AR_INDIC = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")

# Numbered hadith at line start: 123. / [123] / (123) / 123) / 123:
_NUM_BOUNDARY = re.compile(
    r"^\s*(?:\[\s*(\d{1,5})\s*\]|\(\s*(\d{1,5})\s*\)|(\d{1,5})\s*[.:)\]])\s*\S"
)

# Arabic-Indic digits line-start variant
_AR_NUM_BOUNDARY = re.compile(
    r"^\s*[\[\(]?([٠-٩۰-۹]{1,5})[\]\)]?[.:)\]]?\s+\S"
)

# OCR/ligature forms of "حدثنا" after normalization (line starts)
_AR_NARR_BOUNDARY = re.compile(r"^\s*حدثنا\b")

# Chapter / book headings (checked on normalized, casefolded line)
_CHAPTER = re.compile(r"^\s*(كتاب|باب|book|chapter)\b", re.IGNORECASE)
_SENT_SPLIT = re.compile(r"(?<=[.!?؟؛])\s+")


def _arabic_digits_to_ascii(s: str) -> str:
    return s.translate(_AR_INDIC)


def _boundary_ref(line_text: str) -> str | None:
    m = _NUM_BOUNDARY.match(line_text)
    if m:
        return next(g for g in m.groups() if g)
    m = _AR_NUM_BOUNDARY.match(line_text)
    if m:
        digits = _arabic_digits_to_ascii(m.group(1))
        return digits
    return None


def _is_ar_narr_boundary(norm_line: str) -> bool:
    return bool(_AR_NARR_BOUNDARY.match(norm_line))


def _is_chapter(norm_line: str) -> bool:
    return bool(_CHAPTER.match(norm_line))


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
        # Merge vertically adjacent same-ish lines (e.g. wrapped lines) into bands
        merged: dict[int, list[list[float]]] = {}
        for page, rects in per_page.items():
            rects.sort(key=lambda r: (r[1], r[0]))
            out: list[list[float]] = []
            for r in rects:
                if out and abs(out[-1][1] - r[1]) < 2 and _gap(out[-1], r) < 20:
                    p = out[-1]
                    p[0], p[2] = min(p[0], r[0]), max(p[2], r[2])
                    p[3] = max(p[3], r[3])
                else:
                    out.append(list(r))
            merged[page] = out
        return merged


def _gap(a: list[float], b: list[float]) -> float:
    return abs(a[0] - b[2]) if b[2] >= a[0] else abs(b[0] - a[2])


def _is_heading_candidate(line: Line) -> bool:
    norm = normalize_arabic(line.text).lower().strip()
    if not norm:
        return False
    if _is_chapter(norm):
        return True
    short = len(norm) <= 70
    return short and (line.bold or line.font_size >= 14)


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


def chunk_lines(all_lines: list[Line]) -> list[Chunk]:
    chunks: list[Chunk] = []
    cur: list[Line] = []
    cur_ref: str | None = None
    cur_chapter: str | None = None
    current_chapter: str | None = None
    saw_numbered_boundary = False

    def flush():
        nonlocal cur, cur_ref, cur_chapter
        c = _flush(cur, cur_ref, cur_chapter)
        if c is not None:
            chunks.append(c)
        cur, cur_ref, cur_chapter = [], None, None

    for line in all_lines:
        raw = line.text
        norm = normalize_arabic(raw).lower().strip()
        num_ref = _boundary_ref(_arabic_digits_to_ascii(raw))
        is_boundary = bool(num_ref) or _is_ar_narr_boundary(norm)

        if _is_heading_candidate(line):
            # Headings never emit a chunk of their own; they set context.
            text = raw.strip()
            if text and len(text) <= 120:
                current_chapter = text
            continue

        if is_boundary:
            flush()
            saw_numbered_boundary = saw_numbered_boundary or bool(num_ref)
            cur_ref = num_ref
            cur_chapter = current_chapter
            cur.append(line)
        else:
            if not cur:
                cur_chapter = current_chapter
            cur.append(line)

    flush()

    if not saw_numbered_boundary:
        # Pure حَدَّثَنَا-delimited or unnumbered text: our boundary split above
        # still applies; if nothing split anything, fall back to size windows.
        if len(chunks) <= 1 and sum(len(c.text) for c in chunks) > MAX_CHUNK_CHARS:
            chunks = _fallback_split(chunks)

    # Merge fragments into the previous chunk — but never swallow a chunk
    # that carries its own hadith number (numbered hadiths are atomic).
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

    # Hard-split oversized chunks at sentence boundaries
    final: list[Chunk] = []
    for c in chunks:
        if len(c.text) <= MAX_CHUNK_CHARS:
            final.append(c)
            continue
        final.extend(_split_oversized(c))
    return final


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
        cur_lines.append(l)
        cur_len += len(l.text) + 1
        if cur_len >= MAX_CHUNK_CHARS:
            parts.append(_flush_with_ref(cur_lines, c))
            cur_lines, cur_len = [], 0
    if cur_lines:
        parts.append(_flush_with_ref(cur_lines, c))
    return [p for p in parts if p]


def _flush_with_ref(lines: list[Line], src: Chunk) -> Chunk | None:
    out = _flush(lines, src.hadith_ref, src.chapter)
    return out


def chunk_text_for_index(text: str) -> str:
    return normalize_for_index(text)
