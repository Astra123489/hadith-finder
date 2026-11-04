import pytest

from hadith_sidecar import db
from hadith_sidecar.embeddings import HashEmbedder
from hadith_sidecar.ingest import ingest_pdf
from hadith_sidecar.retrieval import hybrid_search
from hadith_sidecar.rerank import IdentityReranker
from hadith_sidecar.llm import EchoProvider
from hadith_sidecar.qa import NOT_FOUND, ask


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
