from hadith_sidecar.normalize import (
    has_arabic,
    fts_tokens,
    normalize_arabic,
    normalize_for_index,
    strip_diacritics,
)


def test_strip_diacritics():
    assert strip_diacritics("حَدَّثَنَا") == "حدثنا"
    assert strip_diacritics("مُحَمَّدٌ") == "محمد"


def test_alef_unification():
    assert normalize_arabic("أإآا") == "اااا"
    assert normalize_arabic("آمن") == "امن"
    assert normalize_arabic("إيمان") == "ايمان"


def test_tatweel_removed():
    assert normalize_arabic("اللّــه") == "الله"


def test_ya_unification():
    assert normalize_arabic("علي") == "علي"
    assert normalize_arabic("على") == "علي"


def test_index_normalization_lowercases():
    assert normalize_for_index("Hadith SAHIH") == "hadith sahih"


def test_has_arabic():
    assert has_arabic("قال رسول الله")
    assert not has_arabic("plain english")


def test_fts_tokens():
    assert fts_tokens("What is sahih?") == ["what", "is", "sahih"]
    assert "حدثنا" in fts_tokens("حَدَّثَنَا رسول الله")
