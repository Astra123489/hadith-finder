"""Regressions for the failures found in the real Sahih Muslim volume.

The fixture PDF cannot reproduce a 500-page book, so these drive the chunker
with synthetic lines copied from the damaged text layer.
"""

import pytest

from hadith_sidecar import db, qa
from hadith_sidecar.chunking import Line, Word, chunk_lines
from hadith_sidecar.embeddings import HashEmbedder
from hadith_sidecar.ingest import ingest_pdf
from hadith_sidecar.llm import EchoProvider, LLMProvider
from hadith_sidecar.rerank import IdentityReranker


@pytest.fixture()
def indexed_conn(conn, sample_pdf):
    ingest_pdf(conn, sample_pdf, title="Fixture")
    embedder = HashEmbedder()
    for row in db.iter_book_chunks(conn, 1):
        db.set_embedding(conn, row["id"], embedder.embed([row["norm_text"]])[0])
    conn.commit()
    return conn


def L(page: int, text: str, *, size: float = 11.0, bold: bool = False) -> Line:
    return Line(page=page, text=text, words=[Word(text, (72, 72, 500, 90))],
                font_size=size, bold=bold)


# --- marker detection -------------------------------------------------------

def test_quran_footnote_does_not_split_a_hadith():
    lines = [
        L(0, "[1233] 57 - (553) It was narrated from Abu Dharr that the"),
        L(0, "Prophet said: \"The deeds of my Ummah were shown to me.\""),
        L(0, "[1] Al-Baqarah 2:238. and the Messenger of Allah performed"),
        L(0, "Wudu' and so did we. Then the Prophet wiped his head."),
        L(1, "[1234] 58 - (554) It was narrated from Ibn 'Umar that the"),
    ]
    chunks = chunk_lines(lines)
    refs = [c.hadith_ref for c in chunks]
    assert refs == ["1233", "1234"], refs
    # The footnote line stays inside hadith 1233 instead of starting "hadith 1".
    assert "Al-Baqarah" in chunks[0].text
    assert "Wudu'" in chunks[0].text


def test_ocr_mangled_first_marker_recovers_its_number():
    lines = [
        L(0, "111611 1 - (520) It was narrated that Abu Dharr said:"),
        L(0, "\"I saw the Prophet.\""),
        L(0, "[1162] 2 - (...) Ibrahim bin Yazid At-Taimi said:"),
        L(0, "I used to recite the Qur'an to my father."),
    ]
    chunks = chunk_lines(lines)
    assert chunks[0].hadith_ref == "1161", [c.hadith_ref for c in chunks]
    assert chunks[1].hadith_ref == "1162"


def test_toc_entries_are_not_hadiths_and_front_matter_is_dropped():
    lines = [
        L(0, "King Fahd National Library Catalog-in-Publication Data"),
        L(0, "5. Masajid And Places Of Prayer................ 146"),
        L(0, "Chapter 55. Making Up A Missed Prayer........... 357"),
        L(1, "[1161] 1 - (520) It was narrated that Abu Dharr said:"),
        L(1, "\"The deeds of my Ummah, good and bad, were shown to me.\""),
    ]
    chunks = chunk_lines(lines)
    assert len(chunks) == 1
    assert chunks[0].hadith_ref == "1161"
    assert "Masajid" not in chunks[0].text  # contents page not indexed


def test_short_bold_prose_is_not_a_chapter():
    lines = [
        L(0, "Chapter 12. The Book Of Prayer", size=14, bold=True),
        L(0, "[1161] 1 - (520) It was narrated from Abu Dharr:", size=12, bold=True),
        L(0, "the Messenger of Allah stood and said something.", size=12, bold=True),
    ]
    chunks = chunk_lines(lines)
    assert chunks[0].chapter == "Chapter 12. The Book Of Prayer"
    assert "stood" in chunks[0].text


def test_oversized_chunks_split_under_the_cap():
    lines = [L(0, "[1161] 1 - (520) It was narrated:")] + [
        L(0, "word " * 40) for _ in range(40)
    ]
    chunks = chunk_lines(lines)
    assert len(chunks) > 1
    assert max(len(c.text) for c in chunks) <= 6000


# --- answer contract --------------------------------------------------------

class SecondOnly(LLMProvider):
    """Cites only excerpt S2, to exercise the badge/card mapping."""

    name = "test-second-only"

    def generate(self, system, user, *, max_tokens=512, temperature=0.2,
                 careful=False, on_token=None):
        text = "The narration answers it [S2]."
        if on_token:
            on_token(text)
        return text


def test_citation_keeps_its_excerpt_number(indexed_conn):
    result = qa.ask(
        indexed_conn, "mercy", book_ids=None,
        embedder=HashEmbedder(), llm=SecondOnly(), reranker=IdentityReranker(), top_k=3,
    )
    assert result["citations"], "S2 must resolve to a citation"
    assert result["citations"][0]["source_index"] == 2


def test_uncited_answer_is_flagged_not_pinned_to_excerpt_one(indexed_conn):
    class NoTags(LLMProvider):
        name = "no-tags"

        def generate(self, system, user, **kw):
            return "Mercy is emphasised repeatedly in these books."

    result = qa.ask(
        indexed_conn, "mercy", book_ids=None,
        embedder=HashEmbedder(), llm=NoTags(), reranker=IdentityReranker(),
    )
    assert result["unverified"] is True
    assert result["citations"] == []


def test_empty_scope_searches_nothing(indexed_conn):
    result = qa.ask(
        indexed_conn, "mercy", book_ids=[],
        embedder=HashEmbedder(), llm=EchoProvider(), reranker=IdentityReranker(),
    )
    assert result["answer"] == qa.NOT_FOUND
    assert result["scoped_empty"] is True
    assert result["citations"] == []
