import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from morphbpe import learn_morphbpe
from run_experiment import select_vocab_size


def test_select_vocab_size_picks_the_length_matching_the_reference():
    word_counts = {"unhappy": 20, "unhelpful": 10}
    boundary_lexicon = {"unhappy": ["un", "happy"], "unhelpful": ["un", "helpful"]}
    merges = learn_morphbpe(word_counts, num_merges=20, boundary_lexicon=boundary_lexicon)

    dev_words = ["unhappy", "unhelpful"]
    best_length, distances = select_vocab_size("cree", merges, dev_words, boundary_lexicon)

    assert best_length in distances
    assert distances[best_length] == min(distances.values())
    assert distances[best_length] < 0.5


if __name__ == "__main__":
    test_select_vocab_size_picks_the_length_matching_the_reference()
    print("run_experiment.py tests passed")
