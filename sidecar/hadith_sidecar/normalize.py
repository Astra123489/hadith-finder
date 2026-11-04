"""Arabic/English normalization for the search index and query pipeline.

Originals are always preserved for display; these functions produce the
canonical form used for FTS5 and embedding input. Normalization is a
recall booster, not a rewrite of the text.
"""

from __future__ import annotations

import re
import unicodedata

# Tashkeel / harakat / Quranic annotation marks + tatweel (kashida)
_AR_DIACRITICS = re.compile(
    "["
    "ؐ-ؚ"  # 0610-061A honorifics/small sign ranges
    "ً-ٟ"  # 064B-065F fatha..tamma, hamza above/below
    "ٰ"   # 0670 superscript alef
    "ۖ-ۜ"  # 06D6-06DC Quranic verse signs
    "۟-ۨ"  # 06DF-06E8 rub el hizb etc.
    "۪-ۭ"  # 06EA-06ED empty centre marks
    "ـ"   # 0640 tatweel / kashida
    "]"
)

# Bidi control characters dropped from the search index (display keeps originals)
_BIDI_SET = {
    chr(cp)
    for cp in ([0x200E, 0x200F, 0x061C]
               + list(range(0x202A, 0x202F))
               + list(range(0x2066, 0x206A)))
}

_ALEF_FORMS = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ٲ": "ا", "ٳ": "ا"})
_YA_FORMS = str.maketrans({"ى": "ي", "ې": "ي", "ے": "ي", "ئ": "ي"})
_TA_MARBUTA = str.maketrans({"ة": "ه"})
_WAW_HAMZA = str.maketrans({"ؤ": "و"})
_WS = re.compile(r"\s+")

_AR_RANGE = re.compile("[؀-ۿ\ufb50-\ufdff\ufe70-\ufeff]")


def strip_diacritics(text: str) -> str:
    return _AR_DIACRITICS.sub("", text)


def normalize_arabic(text: str) -> str:
    t = unicodedata.normalize("NFKC", text or "")
    t = strip_diacritics(t)
    t = t.translate(_ALEF_FORMS).translate(_YA_FORMS).translate(_WAW_HAMZA)
    t = t.translate(_TA_MARBUTA)
    t = "".join(ch for ch in t if ch not in _BIDI_SET)
    return _WS.sub(" ", t).strip()


def has_arabic(text: str) -> bool:
    return bool(_AR_RANGE.search(text or ""))


def normalize_for_index(text: str) -> str:
    """Canonical form stored in FTS and fed to embeddings."""
    return normalize_arabic(text).lower()


def normalize_query(query: str) -> str:
    return normalize_for_index(query)


def fts_tokens(query: str) -> list[str]:
    """Tokenise a query into safe FTS5 terms (alphanumeric only)."""
    norm = normalize_query(query)
    return [t for t in re.findall(r"[0-9A-Za-z؀-ۿ_]+", norm) if len(t) >= 2]
