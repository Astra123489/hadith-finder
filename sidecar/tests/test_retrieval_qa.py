import pytest

from hadith_sidecar import db, qa
from hadith_sidecar.embeddings import HashEmbedder
from hadith_sidecar.ingest import ingest_pdf
from hadith_sidecar.retrieval import hybrid_search
from hadith_sidecar.rerank import IdentityReranker
from hadith_sidecar.llm import EchoProvider, LLMProvider
from hadith_sidecar.qa import NOT_FOUND, ask


@pytest.fixture()
def ruled_conn(conn):
    """A book with one passage that prints a ruling and one that does not."""
    book = db.insert_book(
        conn, title="Rulings", author=None, language="en",
        file_path="rulings.pdf", sha256="rulings-fixture", page_count=1,
    )
    db.insert_chunk(
        conn, book_id=book, chunk_index=0, page_start=0, page_end=0,
        hadith_ref="1", chapter="Book of Trade",
        text="Narrated Tester: The Prophet said, 'Trade in wine is haram.'",
        norm_text="narrated tester: the prophet said, 'trade in wine is haram.'",
        spans={},
    )
    db.insert_chunk(
        conn, book_id=book, chunk_index=1, page_start=0, page_end=0,
        hadith_ref="2", chapter="Book of Intentions",
        text="Narrated Tester: The Prophet said, 'Actions are judged by intentions.'",
        norm_text="narrated tester: the prophet said, 'actions are judged by intentions.'",
        spans={},
    )
    conn.commit()
    return conn


@pytest.fixture()
def referenced_conn(conn):
    """A collection whose printed hadith number only exists in metadata."""
    book = db.insert_book(
        conn, title="Jami at-Tirmidhi", author=None, language="en",
        file_path="tirmidhi.pdf", sha256="tirmidhi-fixture", page_count=1,
    )
    db.insert_chunk(
        conn, book_id=book, chunk_index=0, page_start=0, page_end=0,
        hadith_ref="3521", chapter="Book of Supplication",
        text=("Narrated Tester: The Prophet used to supplicate with many words. "
              "The collector said: this hadith is hasan gharib."),
        norm_text=("narrated tester: the prophet used to supplicate with many words. "
                   "the collector said: this hadith is hasan gharib."),
        spans={},
    )
    conn.commit()
    return conn


@pytest.fixture()
def indexed_conn(conn, sample_pdf):
    ingest_pdf(conn, sample_pdf, title="Fixture")
    embedder = HashEmbedder()
    for row in db.iter_book_chunks(conn, 1):
        db.set_embedding(conn, row["id"], embedder.embed([row["norm_text"]])[0])
    conn.commit()
    return conn


def test_keyword_search_finds_mercy(indexed_conn):
    hits = hybrid_search(indexed_conn, "mercy", HashEmbedder(), pool=10)
    assert hits
    top = db.get_chunks(indexed_conn, [hits[0].chunk_id])[0]
    assert "mercy" in top["text"].lower()


def test_ask_returns_cited_answer(indexed_conn):
    result = ask(
        indexed_conn,
        "mercy",
        book_ids=None,
        embedder=HashEmbedder(),
        llm=EchoProvider(),
        reranker=IdentityReranker(),
        top_k=3,
    )
    assert result["citations"], "answer must be anchored to excerpts"
    cite = result["citations"][0]
    assert cite["page_start"] in (0, 1)
    assert cite["book_title"] == "Fixture"
    assert "spans" in cite and cite["spans"]


def test_ask_not_found_on_empty_library(conn):
    result = ask(
        conn,
        "anything",
        book_ids=None,
        embedder=HashEmbedder(),
        llm=EchoProvider(),
        reranker=IdentityReranker(),
    )
    assert result["answer"] == NOT_FOUND
    assert result["citations"] == []


def test_scoped_to_book_ids(indexed_conn):
    hits = hybrid_search(indexed_conn, "intentions", HashEmbedder(), book_ids=[999], pool=10)
    assert hits == []


class QuotingRuling(LLMProvider):
    name = "quoting-ruling"

    def generate(self, system, user, **kw):
        return "The loaded books state that trade in wine is haram [S1]."


class ConfidentRuling(LLMProvider):
    name = "confident-ruling"

    def generate(self, system, user, **kw):
        return "Music is haram [S1]."


def test_ruling_question_quotes_a_stated_ruling(ruled_conn):
    result = ask(
        ruled_conn, "Is trade in wine halal or haram?", book_ids=None,
        embedder=HashEmbedder(), llm=QuotingRuling(), reranker=IdentityReranker(),
    )
    assert result["intent"]["kind"] == "ruling"
    assert result["answer"].startswith("The loaded books state")
    assert result["citations"]
    assert result["notice"] == qa.NOT_A_FATWA


def test_ruling_question_is_silent_when_no_excerpt_states_one(indexed_conn):
    result = ask(
        indexed_conn, "Is music haram?", book_ids=None,
        embedder=HashEmbedder(), llm=ConfidentRuling(), reranker=IdentityReranker(),
    )
    assert result["intent"]["kind"] == "ruling"
    assert result["answer"] == qa.RULING_NOT_FOUND
    assert result["citations"] == []
    assert result["notice"] == qa.NOT_A_FATWA


def test_verdict_in_a_lookup_answer_is_gated_too(indexed_conn):
    result = ask(
        indexed_conn, "What do the books say about music?", book_ids=None,
        embedder=HashEmbedder(), llm=ConfidentRuling(), reranker=IdentityReranker(),
    )
    assert result["intent"]["kind"] != "ruling"
    assert result["answer"] == qa.RULING_NOT_FOUND
    assert result["citations"] == []
    assert result["notice"] == qa.NOT_A_FATWA


def test_ruling_answers_are_not_streamed_before_the_gate(ruled_conn):
    seen: list[str] = []
    result = ask(
        ruled_conn, "Is trade in wine halal or haram?", book_ids=None,
        embedder=HashEmbedder(), llm=EchoProvider(), reranker=IdentityReranker(),
        on_token=lambda text, phase="answer": seen.append(text),
    )
    assert seen == []
    assert result["citations"]


def test_model_refusal_becomes_the_same_ruling_silence(indexed_conn):
    class Refuses(LLMProvider):
        name = "refuses"

        def generate(self, system, user, **kw):
            return qa.RULING_NOT_FOUND

    result = ask(
        indexed_conn, "Is music haram?", book_ids=None,
        embedder=HashEmbedder(), llm=Refuses(), reranker=IdentityReranker(),
    )
    assert result["answer"] == qa.RULING_NOT_FOUND
    assert result["citations"] == []
    assert result["notice"] == qa.NOT_A_FATWA


@pytest.fixture()
def two_reference_conn(conn):
    """Two collections, so a comparison can pin one passage from each."""
    bukhari = db.insert_book(
        conn, title="Sahih al-Bukhari", author=None, language="en",
        file_path="bukhari.pdf", sha256="bukhari-fixture", page_count=1,
    )
    db.insert_chunk(
        conn, book_id=bukhari, chunk_index=0, page_start=0, page_end=0,
        hadith_ref="1", chapter="Book of Revelation",
        text="Narrated Tester: mercy is shown to the merciful.",
        norm_text="narrated tester: mercy is shown to the merciful.", spans={},
    )
    muslim = db.insert_book(
        conn, title="Sahih Muslim", author=None, language="en",
        file_path="muslim.pdf", sha256="muslim-fixture", page_count=1,
    )
    db.insert_chunk(
        conn, book_id=muslim, chunk_index=0, page_start=0, page_end=0,
        hadith_ref="1907", chapter="Book of Faith",
        text="Narrated Tester: the merciful receive mercy.",
        norm_text="narrated tester: the merciful receive mercy.", spans={},
    )
    conn.commit()
    return conn


class Recording(LLMProvider):
    name = "recording"

    def __init__(self):
        self.prompts: list[str] = []

    def generate(self, system, user, **kw):
        self.prompts.append(user)
        return "Both passages promise mercy [S1]. They differ in wording [S2]."


class Comparing(LLMProvider):
    name = "comparing"

    def generate(self, system, user, **kw):
        return "Both passages mention mercy [S1]. They differ in wording [S2]."


def test_comparison_pins_both_cited_passages(indexed_conn):
    rows = list(db.iter_book_chunks(indexed_conn, 1))[:2]
    ids = [row["id"] for row in rows]
    result = ask(
        indexed_conn, "Do these two narrations agree?", book_ids=None,
        embedder=HashEmbedder(), llm=Comparing(), reranker=IdentityReranker(),
        history_chunk_ids=ids,
    )
    assert result["intent"]["kind"] == "compare"
    assert [cite["chunk_id"] for cite in result["citations"]] == ids


def test_two_named_references_pin_and_cite_each_passage(two_reference_conn):
    llm = Recording()
    result = ask(
        two_reference_conn, "Compare Bukhari 1 and Muslim 1907", book_ids=None,
        embedder=HashEmbedder(), llm=llm, reranker=IdentityReranker(),
    )
    assert result["intent"]["kind"] == "compare"
    assert result["intent"]["references"] == [["bukhari", "1"], ["muslim", "1907"]]
    assert [(cite["book_title"], cite["hadith_ref"]) for cite in result["citations"]] == [
        ("Sahih al-Bukhari", "1"), ("Sahih Muslim", "1907"),
    ]
    assert "mercy is shown to the merciful" in llm.prompts[0]
    assert "the merciful receive mercy" in llm.prompts[0]


def test_comparison_of_these_without_two_passages_says_so(indexed_conn):
    result = ask(
        indexed_conn, "Do these two narrations agree?", book_ids=None,
        embedder=HashEmbedder(), llm=EchoProvider(), reranker=IdentityReranker(),
    )
    assert result["answer"] == qa.COMPARE_NEEDS_TWO
    assert result["citations"] == []


def test_comparison_without_history_falls_back_to_retrieval(indexed_conn):
    result = ask(
        indexed_conn, "compare mercy and intentions", book_ids=None,
        embedder=HashEmbedder(), llm=EchoProvider(), reranker=IdentityReranker(),
    )
    assert result["intent"]["kind"] == "compare"
    assert result["citations"]


def test_comparison_naming_an_unheld_reference_is_not_guessed(referenced_conn):
    result = ask(
        referenced_conn, "Compare Tirmidhi 3521 and Bukhari 1", book_ids=None,
        embedder=HashEmbedder(), llm=Comparing(), reranker=IdentityReranker(),
    )
    assert result["answer"] == NOT_FOUND
    assert result["citations"] == []


def test_bare_collection_reference_resolves_to_its_passage(referenced_conn):
    result = ask(
        referenced_conn, "What is the authenticity of Tirmidhi 3521?", book_ids=None,
        embedder=HashEmbedder(), llm=EchoProvider(), reranker=IdentityReranker(),
    )
    assert result["intent"]["reference"] == ["tirmidhi", "3521"]
    assert result["citations"]
    assert result["citations"][0]["hadith_ref"] == "3521"
    assert result["citations"][0]["book_title"] == "Jami at-Tirmidhi"


def test_unheld_reference_is_reported_before_the_model_runs(indexed_conn):
    class Explodes(LLMProvider):
        name = "explodes"

        def generate(self, system, user, **kw):
            raise AssertionError("model must not run for an unresolved reference")

    result = ask(
        indexed_conn, "Is Tirmidhi 3521 authentic?", book_ids=None,
        embedder=HashEmbedder(), llm=Explodes(), reranker=IdentityReranker(),
    )
    assert result["answer"] == NOT_FOUND
    assert result["citations"] == []


def test_reference_suffix_is_accepted_but_not_a_longer_number(conn):
    book = db.insert_book(
        conn, title="Sunan Ibn Majah", author=None, language="en",
        file_path="majah.pdf", sha256="majah-fixture", page_count=1,
    )
    db.insert_chunk(
        conn, book_id=book, chunk_index=0, page_start=0, page_end=0,
        hadith_ref="45a", chapter="Book of Purification",
        text="Narrated Tester: purification is half of faith.",
        norm_text="narrated tester: purification is half of faith.", spans={},
    )
    conn.commit()
    assert [r["hadith_ref"] for r in db.find_reference_chunks(conn, ("ibn majah",), "45")] == ["45a"]
    assert db.find_reference_chunks(conn, ("ibn majah",), "4") == []


def test_chapter_scope_stays_inside_that_chapter(indexed_conn):
    chapters = db.list_chapters(indexed_conn)
    assert chapters
    hits = hybrid_search(
        indexed_conn, "mercy", HashEmbedder(), chapter=chapters[0], pool=10
    )
    rows = db.get_chunks(indexed_conn, [h.chunk_id for h in hits])
    assert rows
    assert all(r["chapter"] == chapters[0] for r in rows)


def test_extra_phrasings_widen_the_search(indexed_conn):
    hits = hybrid_search(
        indexed_conn, "mercy", HashEmbedder(), queries=["intentions"], pool=10
    )
    rows = db.get_chunks(indexed_conn, [h.chunk_id for h in hits])
    text = " ".join(r["text"] for r in rows).lower()
    assert "mercy" in text
    assert "intention" in text


def test_list_request_is_routed_to_ten_excerpts(indexed_conn):
    result = ask(
        indexed_conn,
        "bring me 5 hadiths about mercy",
        book_ids=None,
        embedder=HashEmbedder(),
        llm=EchoProvider(),
        reranker=IdentityReranker(),
    )
    assert result["intent"]["kind"] == "list"
    assert result["intent"]["count"] == 5
    assert result["intent"]["top_k"] == 5
    assert result["citations"]


def test_follow_up_uses_the_last_cited_passage(indexed_conn):
    row = next(db.iter_book_chunks(indexed_conn, 1))
    result = ask(
        indexed_conn,
        "what did that hadith say",
        book_ids=None,
        embedder=HashEmbedder(),
        llm=EchoProvider(),
        reranker=IdentityReranker(),
        history_chunk_id=row["id"],
    )
    assert result["intent"]["previous"] is True
    assert result["citations"]
    assert result["citations"][0]["chunk_id"] == row["id"]


def test_authenticity_follow_up_reads_the_pinned_passage(indexed_conn):
    row = next(db.iter_book_chunks(indexed_conn, 1))
    result = ask(
        indexed_conn,
        "Is this hadith authentic?",
        book_ids=None,
        embedder=HashEmbedder(),
        llm=EchoProvider(),
        reranker=IdentityReranker(),
        history_chunk_id=row["id"],
    )
    assert result["intent"]["kind"] == "grade"
    assert result["citations"]
    assert result["citations"][0]["chunk_id"] == row["id"]


def test_chapter_summary_is_scoped_to_the_named_chapter(indexed_conn):
    chapter = next(c for c in db.list_chapters(indexed_conn) if "Mercy" in c)
    result = ask(
        indexed_conn,
        "summarize the chapter on mercy",
        book_ids=None,
        embedder=HashEmbedder(),
        llm=EchoProvider(),
        reranker=IdentityReranker(),
    )
    assert result["intent"]["kind"] == "summarize"
    assert result["chapter_scope"] == chapter
    assert result["citations"]
    assert all(cite["chapter"] == chapter for cite in result["citations"])
