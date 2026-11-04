import json

from hadith_sidecar import db
from hadith_sidecar.ingest import ingest_pdf, sha256_of


def test_ingest_sample_pdf(conn, sample_pdf):
    stats = ingest_pdf(conn, sample_pdf, title="Fixture Book of Testing", author="Test Author")
    assert stats["already_exists"] is False
    assert stats["page_count"] == 2
    assert stats["chunks"] >= 5  # heading lines excluded, hadiths split by number

    books = db.list_books(conn)
    assert len(books) == 1
    book = books[0]
    assert book["title"] == "Fixture Book of Testing"
    assert book["sha256"] == sha256_of(sample_pdf)

    chapters = set()
    refs = []
    with_page2 = 0
    for row in conn.execute("SELECT * FROM chunks ORDER BY chunk_index"):
        chapters.add(row["chapter"])
        refs.append(row["hadith_ref"])
        if row["page_start"] == 1:
            with_page2 += 1
        spans = json.loads(row["spans_json"])
        assert spans, "every chunk must carry highlight spans"
    assert "Chapter 2 Book of Mercy" in chapters
    assert "4" in refs
    assert with_page2 >= 2


def test_reingest_is_idempotent(conn, sample_pdf):
    first = ingest_pdf(conn, sample_pdf, title="Fixture")
    second = ingest_pdf(conn, sample_pdf, title="Fixture")
    assert second["already_exists"] is True
    assert second["book_id"] == first["book_id"]
    assert len(db.list_books(conn)) == 1


def test_gradings_attached_where_printed(conn, sample_pdf):
    ingest_pdf(conn, sample_pdf, title="Fixture")
    rows = list(conn.execute(
        "SELECT c.hadith_ref, g.grading FROM chunks c JOIN gradings g ON g.chunk_id = c.id"
    ))
    assert rows, "expected printed gradings to be extracted"
    by_ref = {r["hadith_ref"]: r["grading"] for r in rows}
    assert by_ref.get("3") == "sahih"  # English: "this hadith is sahih"


def test_no_gradings_invented(conn, sample_pdf):
    ingest_pdf(conn, sample_pdf, title="Fixture")
    rows = list(conn.execute(
        "SELECT c.hadith_ref FROM chunks c LEFT JOIN gradings g ON g.chunk_id = c.id "
        "WHERE g.id IS NULL"
    ))
    refs = {r["hadith_ref"] for r in rows}
    assert "5" in refs  # explicitly ungraded narration stays ungraded
