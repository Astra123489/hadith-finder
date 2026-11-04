"""Extraction + ingestion: PDF → hadith chunks → SQLite.

Uses PyMuPDF (fitz): the only open tool that reliably handles
glyph-ordered Arabic text PDFs *and* returns per-word bounding
boxes, which the viewer needs for highlighting.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pymupdf

from . import db
from .chunking import Line, Word, chunk_lines
from .gradings import extract_gradings
from .normalize import has_arabic, normalize_for_index

_BOLD_FLAG = 16


def sha256_of(path: str | Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk_size), b""):
            h.update(block)
    return h.hexdigest()


def _line_font_info(page) -> dict[tuple[int, int], tuple[float, bool]]:
    """Map (block_no, line_no) → (max font size, any-bold?) from the dict view."""
    info: dict[tuple[int, int], tuple[float, bool]] = {}
    d = page.get_text("dict")
    for bi, block in enumerate(d.get("blocks", [])):
        for li, line in enumerate(block.get("lines", [])):
            size, bold = 0.0, False
            for span in line.get("spans", []):
                size = max(size, float(span.get("size", 0.0)))
                bold = bold or bool(int(span.get("flags", 0)) & _BOLD_FLAG)
            info[(bi, li)] = (size, bold)
    return info


def _merge_same_rows(lines: list[Line]) -> list[Line]:
    """Merge lines that occupy the same visual row.

    Mixed-direction rows (e.g. "5. " + Arabic sentence) are emitted by
    MuPDF as separate blocks with identical y; chunking needs them as one
    line. Emission order is already reading order for our corpus.
    """
    out: list[Line] = []
    for l in lines:
        if out:
            prev = out[-1]
            if prev.page == l.page:
                p_lo = min(w.rect[1] for w in prev.words)
                p_hi = max(w.rect[3] for w in prev.words)
                l_lo = min(w.rect[1] for w in l.words)
                l_hi = max(w.rect[3] for w in l.words)
                overlap = min(p_hi, l_hi) - max(p_lo, l_lo)
                hgt = min(p_hi - p_lo, l_hi - l_lo)
                if hgt > 0 and overlap > 0.5 * hgt:
                    prev.text = prev.text + " " + l.text
                    prev.words = prev.words + l.words
                    prev.font_size = max(prev.font_size, l.font_size)
                    prev.bold = prev.bold or l.bold
                    continue
        out.append(l)
    return out


def _count_ar_chars(text: str) -> int:
    n = 0
    for ch in text:
        cp = ord(ch)
        if (0x0600 <= cp <= 0x06FF or 0xFB50 <= cp <= 0xFDFF or 0xFE70 <= cp <= 0xFEFF):
            n += 1
    return n


def extract_lines(pdf_path: str | Path) -> tuple[list[Line], int, str]:
    """Return (lines, page_count, detected_language)."""
    lines: list[Line] = []
    ar_chars = latin_chars = 0

    with pymupdf.open(str(pdf_path)) as doc:
        page_count = doc.page_count
        for page_no, page in enumerate(doc):
            font_info = _line_font_info(page)
            grouped: dict[tuple[int, int], list[tuple]] = {}
            for x0, y0, x1, y1, word, bno, lno, wno in page.get_text("words"):
                grouped.setdefault((bno, lno), []).append((wno, x0, y0, x1, y1, word))
            for (bno, lno), words in sorted(grouped.items()):
                words.sort(key=lambda w: w[0])
                size, bold = font_info.get((bno, lno), (0.0, False))
                text = " ".join(w[5] for w in words).strip()
                if not text:
                    continue
                ar_chars += _count_ar_chars(text)
                latin_chars += sum(1 for ch in text if ch.isascii() and ch.isalpha())
                lines.append(Line(
                    page=page_no,
                    text=text,
                    words=[Word(w[5], (w[1], w[2], w[3], w[4])) for w in words],
                    font_size=size,
                    bold=bold,
                ))

    if ar_chars == 0 and latin_chars == 0:
        lang = "unknown"
    elif ar_chars > latin_chars * 2:
        lang = "ar"
    elif latin_chars > ar_chars * 10:
        lang = "en"
    else:
        lang = "mixed"
    return _merge_same_rows(lines), page_count, lang


def ingest_pdf(conn, pdf_path: str | Path, *, title: str | None = None,
               author: str | None = None) -> dict:
    pdf_path = Path(pdf_path).expanduser().resolve()
    if not pdf_path.is_file():
        raise FileNotFoundError(pdf_path)
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"not a PDF: {pdf_path}")

    digest = sha256_of(pdf_path)
    existing = db.find_book_by_sha(conn, digest)
    if existing:
        return {
            "book_id": existing["id"], "title": existing["title"],
            "page_count": existing["page_count"],
            "chunks": db.count_chunks(conn, existing["id"]),
            "already_exists": True,
        }

    lines, page_count, language = extract_lines(pdf_path)
    if not lines:
        raise ValueError(
            "No extractable text found. Is this a scanned PDF? "
            "Hadith Finder currently supports text-based PDFs only."
        )

    final_title = (title or os.path.splitext(pdf_path.name)[0]).strip()
    chunks = chunk_lines(lines)

    with conn:  # single transaction; FTS kept in sync by trigger
        book_id = db.insert_book(
            conn, title=final_title, author=author, language=language,
            file_path=str(pdf_path), sha256=digest, page_count=page_count,
        )
        for i, ch in enumerate(chunks):
            text = ch.text
            chunk_id = db.insert_chunk(
                conn, book_id=book_id, chunk_index=i,
                page_start=ch.page_start, page_end=ch.page_end,
                hadith_ref=ch.hadith_ref, chapter=ch.chapter,
                text=text, norm_text=normalize_for_index(text),
                spans=ch.spans(),
            )
            db.insert_gradings(conn, chunk_id, extract_gradings(ch.text))

    return {
        "book_id": book_id, "title": final_title,
        "page_count": page_count, "chunks": len(chunks),
        "language": language, "already_exists": False,
    }
