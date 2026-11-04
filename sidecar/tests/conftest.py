import sys
from pathlib import Path

import os
import pymupdf
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

AR_FONT = r"C:\Windows\Fonts\arial.ttf"

LINES_PAGE_1 = [
    "Chapter 1 Book of Intentions",
    "1. Narrated Tester: The Prophet said, 'Actions are judged by intentions.'",
    "2. Narrated Tester: The Prophet said, 'A person may lose worldly goods but intention.'",
]
LINES_PAGE_2 = [
    "Chapter 2 Book of Mercy",
    "3. Narrated Tester: The Prophet said, 'Whoever lacks mercy will",
    "   not be shown mercy.' Grading: this hadith is sahih.",
    "4. Narrated Tester: His companion said: where books do not grade",
    "   this narration, the reader must know it.",
    "قال رسول الله: من لا يرحم لا يُرحم. حديث صحيح.",
    "5. This final narration carries no grading at all in this book.",
]


def _insert_line(page, y, text, is_ar: bool):
    if is_ar and os.path.isfile(AR_FONT):
        page.insert_text((72, y), text, fontsize=11, fontfile=AR_FONT)
    else:
        page.insert_text((72, y), text, fontsize=11, fontname="helv")


@pytest.fixture(scope="session")
def sample_pdf(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("pdfs") / "FixtureBook_of_Testing.pdf"
    doc = pymupdf.open()
    for lines in (LINES_PAGE_1, LINES_PAGE_2):
        page = doc.new_page(width=595, height=842)
        y = 72.0
        for line in lines:
            is_ar = any("؀" <= ch <= "ۿ" for ch in line)
            _insert_line(page, y, line, is_ar)
            y += 18
    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture()
def conn(tmp_path):
    from hadith_sidecar import db

    c = db.connect(tmp_path / "test.haddb")
    yield c
    c.close()
