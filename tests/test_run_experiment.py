import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from morphbpe import learn_morphbpe
from run_experiment import _select_by_significance, select_vocab_size


def test_select_vocab_size_returns_a_valid_length_and_its_mean():
    word_counts = {"unhappy": 20, "unhelpful": 10}
    boundary_lexicon = {"unhappy": ["un", "happy"], "unhelpful": ["un", "helpful"]}
    merges = learn_morphbpe(word_counts, num_merges=20, boundary_lexicon=boundary_lexicon)

    dev_words = ["unhappy", "unhelpful"]
    best_length, means = select_vocab_size("cree", merges, dev_words, boundary_lexicon)

    assert best_length in means
    assert 0.0 <= means[best_length] <= 1.0


def test_select_by_significance_stops_once_further_lengths_stop_improving():
    lengths = [1, 2, 3, 4]
    # length 1 -> 2 is a clear, consistent improvement; 2 -> 3 -> 4 barely move at all,
    # so a significance test should stop at 2 rather than picking the raw-argmin length 4.
    per_word = {
        1: [0.9, 0.85, 0.95, 0.88, 0.92, 0.91, 0.87, 0.93],
        2: [0.4, 0.35, 0.45, 0.38, 0.42, 0.41, 0.37, 0.43],
        3: [0.41, 0.34, 0.44, 0.37, 0.43, 0.40, 0.38, 0.42],
        4: [0.39, 0.36, 0.43, 0.39, 0.41, 0.39, 0.38, 0.41],
    }
    means = {k: sum(v) / len(v) for k, v in per_word.items()}

    selected = _select_by_significance(lengths, per_word, means)

    assert selected == 2


def test_select_by_significance_picks_the_max_when_improvement_never_stops():
    lengths = [1, 2, 3]
    # every step is a clear, significant improvement over the last -- mirrors the
    # source paper's English/Arabic case, where morphology distance kept improving
    # all the way to the largest swept vocabulary size.
    per_word = {
        1: [0.9, 0.88, 0.91, 0.89, 0.92, 0.87],
        2: [0.6, 0.58, 0.61, 0.59, 0.62, 0.57],
        3: [0.3, 0.28, 0.31, 0.29, 0.32, 0.27],
    }
    means = {k: sum(v) / len(v) for k, v in per_word.items()}

    selected = _select_by_significance(lengths, per_word, means)

    assert selected == 3


if __name__ == "__main__":
    test_select_vocab_size_returns_a_valid_length_and_its_mean()
    test_select_by_significance_stops_once_further_lengths_stop_improving()
    test_select_by_significance_picks_the_max_when_improvement_never_stops()
    print("run_experiment.py tests passed")
