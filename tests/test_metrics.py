import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from metrics import consistency_f1, fertility, morph_edit_distance, word_to_boundary_positions


def test_fertility():
    assert fertility([["a", "b"], ["c"], ["d", "e", "f"]]) == 6 / 3


def test_word_to_boundary_positions():
    assert word_to_boundary_positions(["un", "happy"]) == {2}
    assert word_to_boundary_positions(["a", "b", "c"]) == {1, 2}
    assert word_to_boundary_positions(["whole"]) == set()


def test_consistency_f1_perfect_match():
    reference_lexicon = {"unhappy": ["un", "happy"]}
    result = consistency_f1(
        test_words=["unhappy"],
        tokenized_words=[["un", "happy"]],
        reference_lexicon=reference_lexicon,
    )
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1"] == 1.0
    assert result["n_scored_words"] == 1


def test_consistency_f1_total_mismatch():
    reference_lexicon = {"unhappy": ["un", "happy"]}
    result = consistency_f1(
        test_words=["unhappy"],
        tokenized_words=[["u", "nhappy"]],
        reference_lexicon=reference_lexicon,
    )
    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["f1"] == 0.0


def test_consistency_f1_skips_words_without_reference():
    reference_lexicon = {"unhappy": ["un", "happy"]}
    result = consistency_f1(
        test_words=["unhappy", "zzz"],
        tokenized_words=[["un", "happy"], ["z", "zz"]],
        reference_lexicon=reference_lexicon,
    )
    assert result["n_scored_words"] == 1
    assert result["n_test_words"] == 2


def test_morph_edit_distance_perfect_match():
    reference_lexicon = {"unhappy": ["un", "happy"]}
    result = morph_edit_distance(
        test_words=["unhappy"],
        tokenized_words=[["un", "happy"]],
        reference_lexicon=reference_lexicon,
    )
    assert result["mean_edit_distance"] == 0.0
    assert result["n_scored_words"] == 1


def test_morph_edit_distance_known_value():
    reference_lexicon = {"unhappy": ["un", "happy"]}
    result = morph_edit_distance(
        test_words=["unhappy"],
        tokenized_words=[["u", "n", "happy"]],
        reference_lexicon=reference_lexicon,
    )
    assert abs(result["mean_edit_distance"] - 2 / 3) < 1e-9


def test_morph_edit_distance_skips_words_without_reference():
    reference_lexicon = {"unhappy": ["un", "happy"]}
    result = morph_edit_distance(
        test_words=["unhappy", "zzz"],
        tokenized_words=[["un", "happy"], ["z", "zz"]],
        reference_lexicon=reference_lexicon,
    )
    assert result["n_scored_words"] == 1
    assert result["n_test_words"] == 2


if __name__ == "__main__":
    test_fertility()
    test_word_to_boundary_positions()
    test_consistency_f1_perfect_match()
    test_consistency_f1_total_mismatch()
    test_consistency_f1_skips_words_without_reference()
    test_morph_edit_distance_perfect_match()
    test_morph_edit_distance_known_value()
    test_morph_edit_distance_skips_words_without_reference()
    print("metrics.py tests passed")
