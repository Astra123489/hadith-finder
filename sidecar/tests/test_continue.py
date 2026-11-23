"""Finishing an answer the token limit cut off.

The continuation must reuse the excerpt numbers the first part printed, so the
cards already on screen still match [S#], and it must join the two parts into
one answer without repeating the seam.
"""

from hadith_sidecar import qa, thread
from hadith_sidecar.llm import LLMProvider

CITATIONS = [
    {
        "source_index": 1,
        "book_id": 1,
        "book_title": "Sahih al-Bukhari",
        "chapter": "Revelation",
        "hadith_ref": "1",
        "page_start": 0,
        "page_end": 0,
        "excerpt": "Actions are judged by intentions.",
    },
    {
        "source_index": 2,
        "book_id": 1,
        "book_title": "Sahih al-Bukhari",
        "chapter": "Revelation",
        "hadith_ref": "2",
        "page_start": 1,
        "page_end": 1,
        "excerpt": "Each person will have what he intended.",
    },
]


class Rest(LLMProvider):
    """Writes a given continuation and records what it was asked."""

    last_finish_reason = "stop"

    def __init__(self, text: str):
        self.text = text
        self.user = ""
        self.system = ""
        self.max_tokens = 0

    def generate(self, system, user, *, max_tokens=512, temperature=0.2, on_token=None):
        self.system = system
        self.user = user
        self.max_tokens = max_tokens
        return self.text


def test_continuation_keeps_the_original_excerpt_numbers():
    llm = Rest("the reward follows what he intended [S2].")
    result = qa.continue_answer(
        question="What is the hadith of intentions?",
        partial="Deeds are judged by intention [S1], and",
        citations=CITATIONS,
        llm=llm,
        intent={"kind": "summarize", "top_k": 8},
    )
    # Same labels, same passages — a card labelled S2 still means excerpt 2.
    assert "[S1] Sahih al-Bukhari" in llm.user
    assert "[S2] Sahih al-Bukhari" in llm.user
    assert "Actions are judged by intentions." in llm.user
    assert "[S2] still means excerpt S2" in llm.system
    # The continuation gets the same answer budget the question had.
    assert llm.max_tokens == 700
    assert result["answer"] == (
        "Deeds are judged by intention [S1], and the reward follows what he intended [S2]."
    )
    assert result["truncated"] is False
    assert result["finish_reason"] == "stop"


def test_number_the_reader_has_no_card_for_is_dropped():
    llm = Rest("and the reward is for what he intended [S9].")
    llm.last_finish_reason = "length"
    result = qa.continue_answer(
        question="What is the hadith of intentions?",
        partial="Deeds are judged by intention [S1], and",
        citations=CITATIONS,
        llm=llm,
        intent={"kind": "lookup", "top_k": 5},
    )
    assert "[S9]" not in result["answer"]
    assert "[S1]" in result["answer"]
    # Cut off again: the reader can continue once more.
    assert result["truncated"] is True
    assert result["finish_reason"] == "length"


def test_transcript_keeps_the_finished_answer():
    conv = thread.new_conversation()
    turn = thread.add_turn(
        conv, "What is the hadith of intentions?", {"answer": "Deeds are judged", "truncated": True}
    )
    thread.append_answer(
        conv, turn["id"], "Deeds are judged by intention [S1].", truncated=False
    )
    saved = thread.snapshot(conv, {})["turns"][0]
    assert saved["answer"] == "Deeds are judged by intention [S1]."
    assert saved["truncated"] is False
