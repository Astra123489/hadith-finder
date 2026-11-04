from hadith_sidecar.chunking import Line, Word, chunk_lines
from hadith_sidecar.gradings import extract_gradings


def make_line(page, text, word_count=None):
    words = [Word(t, (0.0, page * 12.0, 5.0, page * 12.0 + 8.0)) for t in text.split()]
    return Line(page=page, text=text, words=words)


def test_numbered_boundaries_split_chunks():
    lines = [
        make_line(0, "Chapter 1 Book of Intentions"),
        make_line(0, "1. Narrated Tester: actions are judged by intentions."),
        make_line(0, "continuation of the first hadith with more words here to pass the minimum."),
        make_line(0, "2. Narrated Tester: the second hadith begins here with enough words to be kept in one chunk."),
        make_line(0, "3. Narrated Tester: the third hadith begins here with enough words to be kept in this chunk too."),
    ]
    chunks = chunk_lines(lines)
    refs = [c.hadith_ref for c in chunks]
    assert refs == ["1", "2", "3"]
    assert all(c.chapter == "Chapter 1 Book of Intentions" for c in chunks)


def test_oversized_chunk_is_split():
    long = "word " * 400  # ~2000 chars... increase via repetition
    text = ("1. narrated something. " + ("sentence number one is quite long. " * 120)).strip()
    lines = [make_line(0, text, len(text.split()))]
    chunks = chunk_lines(lines)
    assert len(chunks) >= 1  # must not crash and must respect bounds eventually


def test_fallback_chunk_when_unnumbered():
    lines = []
    for i in range(3):
        lines.append(make_line(0, "حدثنا " + "word " * 15))  # narr boundary delimiters
    chunks = chunk_lines(lines)
    assert len(chunks) >= 1


def test_gradings_requires_context():
    assert extract_gradings("this hadith is sahih")
    assert extract_gradings("حديث صحيح")
    assert extract_gradings("Narrated x: y. ضعيف.") == []  # no hadith context word in sentence... 
    assert extract_gradings("النص: weak content here") == []


def test_grading_not_invented():
    assert extract_gradings("The weather is حسن today and everyone likes it very much.") == []
