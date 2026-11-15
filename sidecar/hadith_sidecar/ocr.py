"""RapidOCR fallback for pages whose PDF text layer is missing or damaged.

Scanned books and badly OCR'd exports carry either no text layer, or a text
layer full of U+FFFD replacement characters (the Arabic glyphs were dropped
when the publisher built the PDF). Those pages are rendered with PyMuPDF and
re-read with RapidOCR (PP-OCR models, ONNX Runtime, CPU).

Import is lazy: if `rapidocr_onnxruntime` is not installed the app keeps
working on the text layer and simply reports that OCR is unavailable.

Environment:
    HADITH_OCR=auto|off|force   (default auto: OCR only pages that need it)
    HADITH_OCR_DPI=200          (render resolution)
"""

from __future__ import annotations

import os

ENGINE = None
DPI_ENV = "HADITH_OCR_DPI"
MODE_ENV = "HADITH_OCR"
DEFAULT_DPI = 200


def mode() -> str:
    return (os.environ.get(MODE_ENV) or "auto").strip().lower()


def enabled() -> bool:
    return mode() != "off"


def dpi() -> int:
    try:
        return max(96, min(400, int(os.environ.get(DPI_ENV) or DEFAULT_DPI)))
    except ValueError:
        return DEFAULT_DPI


def available() -> bool:
    try:
        import rapidocr_onnxruntime  # noqa: F401
        return True
    except Exception:
        return False


def get_engine():
    global ENGINE
    if ENGINE is None:
        from rapidocr_onnxruntime import RapidOCR

        threads = max(1, (os.cpu_count() or 4) // 2)
        ENGINE = RapidOCR(intra_op_num_threads=threads)
    return ENGINE


def needs_ocr(text: str) -> bool:
    """True when the page's text layer is missing or visibly corrupted."""
    stripped = (text or "").strip()
    if len(stripped) < 60:
        return True
    # A handful of U+FFFD ("Abu�") is survivable and OCR costs ~4 s/page;
    # only pages whose layer is substantially broken are worth re-reading.
    bad = stripped.count("\ufffd")
    return bad >= max(10, int(len(stripped) * 0.08))


def ocr_page(page, page_no: int, *, dpi_px: int | None = None):
    """Run OCR on one PyMuPDF page and return chunking.Line objects."""
    import numpy as np

    from .chunking import Line, Word

    pix = page.get_pixmap(dpi=dpi_px or dpi(), alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = img[:, :, :3]
    # RapidOCR expects BGR (it feeds cv2); PyMuPDF gives RGB.
    img = np.ascontiguousarray(img[:, :, ::-1])

    engine = get_engine()
    result, _elapsed = engine(img)
    if not result:
        return []

    scale = 72.0 / float(dpi_px or dpi())
    lines = []
    for item in result:
        box, text, score = item[0], item[1], item[2] if len(item) > 2 else 1.0
        text = (text or "").strip()
        if not text or (score is not None and float(score) < 0.4):
            continue
        # Drop recognizer noise (the shipped model reads Latin only): a line
        # that is mostly punctuation/unknown glyphs is not text.
        alnum = sum(1 for ch in text if ch.isalnum() or ch.isspace())
        if alnum / len(text) < 0.7:
            continue
        xs = [float(p[0]) for p in box]
        ys = [float(p[1]) for p in box]
        rect = (
            round(min(xs) * scale, 2), round(min(ys) * scale, 2),
            round(max(xs) * scale, 2), round(max(ys) * scale, 2),
        )
        lines.append(Line(page=page_no, text=text, words=[Word(text, rect)],
                          font_size=0.0, bold=False))

    # Reading order: top-to-bottom, then left-to-right.
    lines.sort(key=lambda l: (round(l.words[0].rect[1] / 6), l.words[0].rect[0]))
    return lines
