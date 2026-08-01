import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bpe import learn_bpe
from morphbpe import (
    BOUNDARY,
    apply_morphbpe,
    learn_morphbpe,
    word_to_marked_symbols,
)


def test_word_to_marked_symbols():
    lexicon = {"unhappy": ["un", "happy"]}
    assert word_to_marked_symbols("unhappy", lexicon) == (
        "u", "n", BOUNDARY, "h", "a", "p", "p", "y</w>",
    )
    assert word_to_marked_symbols("zzz", lexicon) == ("z", "z", "z</w>")


def test_boundary_actually_blocks_the_merge_plain_bpe_would_make():
    word_counts = {"unhappy": 10}
    boundary_lexicon = {"unhappy": ["un", "happy"]}

    plain_merges = learn_bpe(dict(word_counts), num_merges=3)
    assert plain_merges == [("u", "n"), ("un", "h"), ("unh", "a")], plain_merges

    constrained_merges = learn_morphbpe(word_counts, num_merges=10, boundary_lexicon=boundary_lexicon)
    assert all(BOUNDARY not in pair for pair in constrained_merges)
    assert ("un", "h") not in constrained_merges
    assert ("n", "h") not in constrained_merges


def test_apply_morphbpe_never_spans_the_boundary():
    word_counts = {"unhappy": 10}
    boundary_lexicon = {"unhappy": ["un", "happy"]}
    merges = learn_morphbpe(word_counts, num_merges=10, boundary_lexicon=boundary_lexicon)

    tokens = apply_morphbpe("unhappy", merges)
    assert "".join(tokens) == "unhappy"
    assert not any(t.startswith("un") and len(t) > 2 for t in tokens)

    assert "".join(apply_morphbpe("zzz", merges)) == "zzz"


if __name__ == "__main__":
    test_word_to_marked_symbols()
    test_boundary_actually_blocks_the_merge_plain_bpe_would_make()
    test_apply_morphbpe_never_spans_the_boundary()
    print("morphbpe.py toy-corpus tests passed")
