import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bpe import (
    apply_bpe,
    apply_merges,
    apply_merges_with_checkpoints,
    finalize_tokens,
    learn_bpe,
    word_to_symbols,
)


def test_merge_order_and_application():
    word_counts = {"aab": 10, "aac": 1}
    merges = learn_bpe(word_counts, num_merges=2)
    assert merges == [("a", "a"), ("aa", "b</w>")], merges

    assert apply_bpe("aab", merges) == ["aab"]
    assert apply_bpe("aac", merges) == ["aa", "c"]


def test_stops_when_no_pair_left():
    word_counts = {"x": 5}
    merges = learn_bpe(word_counts, num_merges=10)
    assert merges == []
    assert apply_bpe("x", merges) == ["x"]


def test_word_to_symbols_marks_end():
    assert word_to_symbols("cat") == ("c", "a", "t</w>")


def test_checkpoints_match_truncated_full_apply():
    word_counts = {"aab": 10, "aac": 1, "aad": 4}
    merges = learn_bpe(dict(word_counts), num_merges=5)
    word = "aab"
    checkpoints = apply_merges_with_checkpoints(word_to_symbols(word), merges, [0, 1, 2, 3, 4, 5])
    for k in (0, 1, 2, 3, 4, 5):
        expected = apply_merges(word_to_symbols(word), merges[:k])
        assert checkpoints[k] == expected, (k, checkpoints[k], expected)


def test_finalize_tokens_strips_end_marker_and_empties():
    assert finalize_tokens(["c", "a", "t</w>"]) == ["c", "a", "t"]
    assert finalize_tokens(["cat</w>"]) == ["cat"]


if __name__ == "__main__":
    test_merge_order_and_application()
    test_stops_when_no_pair_left()
    test_word_to_symbols_marks_end()
    test_checkpoints_match_truncated_full_apply()
    test_finalize_tokens_strips_end_marker_and_empties()
    print("bpe.py toy-corpus tests passed")
