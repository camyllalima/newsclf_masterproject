"""Tests for newsclf.text_utils.count_sentences."""

from newsclf.text_utils import count_sentences


class TestCountSentences:
    def test_single_sentence(self) -> None:
        assert count_sentences("The cat sat on the mat.") == 1

    def test_three_sentences(self) -> None:
        text = "First sentence. Second sentence. Third sentence."
        assert count_sentences(text) == 3

    def test_abbreviation_dr(self) -> None:
        text = "Dr. Smith went to the store. He bought milk."
        assert count_sentences(text) == 2

    def test_abbreviation_us(self) -> None:
        text = "The U.S. economy grew last quarter. Exports rose sharply."
        assert count_sentences(text) == 2

    def test_empty_string(self) -> None:
        assert count_sentences("") == 0

    def test_none_returns_zero(self) -> None:
        assert count_sentences(None) == 0  # type: ignore[arg-type]

    def test_decimal_number(self) -> None:
        text = "The rate was 3.5 percent. Markets reacted positively."
        assert count_sentences(text) == 2

    def test_exclamation_and_question(self) -> None:
        text = "What happened? It was amazing! The crowd cheered."
        assert count_sentences(text) == 3
