"""Authenticity-grading extraction.

Only reports what the printed book actually says. When a chunk contains no
grading language, nothing is inserted and the UI says the book doesn't
grade it. Never invents a grading.

Precision beats recall here: a wrong grading on a hadith is worse than a
missed one. So tags require hadith-ish context in the same sentence and
standalone (non-name) matches for Arabic words.
"""

from __future__ import annotations

import re

_CONTEXT = re.compile(
    r"(حديث|احاديث|hadith|حدثنا|اخبرنا|سمعت|narrat|riwayat|رواه)",
    re.IGNORECASE,
)

# Tag → pattern; Arabic patterns avoid matching inside "الحسن/الحسن" name forms.
_GRADING_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("sahih", re.compile(r"(?<![ال\w])صحيح(?![\w])|صححه|\bsahih\b|\bauthentic\b")),
    ("hasan", re.compile(r"(?<![ال\w])حسن(?![\w])|حسنه|\bhasan\b|\bhassan\b")),
    ("daif",  re.compile(r"(?<![ال\w])ضعيف(?![\w])|ضعفه|\bda.?if\b|\bdaif\b|\bweak\b")),
    ("mawdu", re.compile(r"(?<![ال\w])موضوع(?![\w])|\bmawdu\b|\bfabricated\b")),
    ("munkar", re.compile(r"(?<![ال\w])منكر(?![\w])|\bmunkar\b")),
    ("mursal", re.compile(r"(?<![ال\w])مرسل(?![\w])|\bmursal\b")),
    ("mawquf", re.compile(r"(?<![ال\w])موقوف(?![\w])|\bmawquf\b")),
]

_SENT_SPLIT = re.compile(r"(?<=[.!?؟؛\n])\s*")


def extract_gradings(text: str) -> list[tuple[str, str]]:
    """Return list of (tag, raw_context) — context is the sentence carrying it."""
    from .normalize import normalize_arabic

    found: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for sentence in _SENT_SPLIT.split(text or ""):
        if not sentence.strip():
            continue
        probe = normalize_arabic(sentence).lower()
        if not _CONTEXT.search(probe):
            continue
        for tag, pattern in _GRADING_PATTERNS:
            if pattern.search(probe):
                raw = sentence.strip()
                key = (tag, raw)
                if key not in seen:
                    seen.add(key)
                    found.append(key)
    return found
