from hadith_sidecar.intent import classify, match_chapter


def test_plain_question_is_a_lookup():
    intent = classify("Who narrated the hadith about intentions?")
    assert intent.kind == "lookup"
    assert intent.top_k == 5


def test_show_me_one_hadith_is_still_a_lookup():
    intent = classify("Show me the hadith about mercy")
    assert intent.kind == "lookup"
    assert intent.top_k == 5


def test_counted_request_is_a_list():
    intent = classify("Bring me 10 hadiths about patience")
    assert intent.kind == "list"
    assert intent.count == 10
    assert intent.top_k == 10
    assert intent.search_text == "patience"


def test_word_number_and_novelty():
    intent = classify("give me ten hadiths I don't know, not common")
    assert intent.kind == "list"
    assert intent.count == 10
    assert intent.novel is True
    assert intent.top_k == 10


def test_uncommon_phrasing_keeps_the_topic():
    intent = classify("show me five uncommon narrations on mercy")
    assert intent.kind == "list"
    assert intent.count == 5
    assert intent.novel is True


def test_list_is_clamped_to_a_sane_window():
    assert classify("give me 99 hadiths on patience").top_k == 20
    assert classify("give me 2 hadiths on patience").top_k == 3


def test_summary_of_this_chapter_uses_the_open_chapter():
    intent = classify("summarize this chapter", history_chapter="Book of Mercy")
    assert intent.kind == "summarize"
    assert intent.chapter == "Book of Mercy"


def test_summary_of_this_narration_does_not_take_the_whole_chapter():
    intent = classify("summarize this narration", history_chapter="Book of Mercy")
    assert intent.kind == "summarize"
    assert intent.chapter is None
    assert intent.previous is True


def test_location_question_points_at_the_previous_passage():
    intent = classify("where is that hadith located?")
    assert intent.kind == "locate"
    assert intent.previous is True


def test_who_said_this_points_at_the_previous_passage():
    assert classify("Who said this?").previous is True


def test_where_is_it_located_points_at_the_previous_passage():
    intent = classify("Where is it located?")
    assert intent.kind == "locate"
    assert intent.previous is True


def test_summary_names_its_chapter():
    intent = classify("summarize the chapter on intentions")
    assert intent.kind == "summarize"
    assert intent.chapter == "intentions"


def test_location_question():
    intent = classify("which page is the hadith of intentions on?")
    assert intent.kind == "locate"


def test_a_count_without_a_topic_keeps_the_question_as_the_query():
    intent = classify("bring me 10 hadiths")
    assert intent.kind == "list"
    assert intent.search_text == "bring me 10 hadiths"


def test_arabic_count():
    intent = classify("أعطني ١٠ أحاديث عن الصبر")
    assert intent.kind == "list"
    assert intent.count == 10


def test_chapter_matching_ignores_generic_words():
    chapters = ["Book of Intentions", "Book of Mercy", "Chapter 3 On Patience"]
    assert match_chapter(chapters, "intentions") == "Book of Intentions"
    assert match_chapter(chapters, "chapter on mercy") == "Book of Mercy"
    assert match_chapter(chapters, "jihad") is None
