# Implements Algorithm 1 ("Morph-Aware Byte Pair Encoding") from:
#   Asgari, El Kheir & Sadraei Javaheri (2025), "MorphBPE: A Morpho-Aware
#   Tokenizer Bridging Linguistic Complexity for Efficient LLM Training
#   Across Morphologies," arXiv:2502.00894.
# This module is our own implementation from the paper's pseudocode; the
# authors' code is not publicly available, so nothing is reused from it.
from pathlib import Path

from bpe import END_MARKER, apply_merges, finalize_tokens, learn_merges, word_to_symbols

BOUNDARY = "<m>"

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BOUNDARIES_DIR = Path(__file__).resolve().parent.parent / "boundaries"


def load_boundary_lexicon(path):
    lexicon = {}
    if not path.exists():
        return lexicon
    for line in path.read_text(encoding="utf-8").splitlines():
        word, _, morphs = line.partition("\t")
        if morphs:
            lexicon[word] = morphs.split(" ")
    return lexicon


def word_to_marked_symbols(word, boundary_lexicon):
    morphs = boundary_lexicon.get(word)
    if not morphs:
        return word_to_symbols(word)

    symbols = []
    for i, morph in enumerate(morphs):
        chars = list(morph)
        if i < len(morphs) - 1:
            symbols.extend(chars)
            symbols.append(BOUNDARY)
        else:
            chars[-1] = chars[-1] + END_MARKER
            symbols.extend(chars)
    return tuple(symbols)


def _pair_allowed(pair):
    return BOUNDARY not in pair


def learn_morphbpe(word_counts, num_merges, boundary_lexicon):
    word_freqs = {}
    for word, freq in word_counts.items():
        symbols = word_to_marked_symbols(word, boundary_lexicon)
        word_freqs[symbols] = word_freqs.get(symbols, 0) + freq
    return learn_merges(word_freqs, num_merges, pair_allowed=_pair_allowed, log_label="morphbpe")


def apply_morphbpe(word, merges):
    return finalize_tokens(apply_merges(word_to_symbols(word), merges))


def build_word_counts(lines, tokenize_line):
    word_counts = {}
    for line in lines:
        for word in tokenize_line(line):
            word_counts[word] = word_counts.get(word, 0) + 1
    return word_counts


MERGE_COUNTS = {"cree": 10000, "inuktitut": 8000, "guarani": 12000}

BOUNDARY_FILES = {
    "cree": ["cree_fst.segments", "cree_morfessor.segments"],
    "inuktitut": ["inuktitut_uqailaut.segments", "inuktitut_morfessor.segments"],
    "guarani": ["guarani_apertium.segments", "guarani_morfessor.segments"],
}


def load_combined_boundary_lexicon(lang):
    combined = {}
    for filename in reversed(BOUNDARY_FILES[lang]):
        combined.update(load_boundary_lexicon(BOUNDARIES_DIR / filename))
    return combined


def train_and_apply(lang):
    from tokenize_utils import tokenize_line

    train_lines = (DATA_DIR / lang / "train.txt").read_text(encoding="utf-8").splitlines()
    word_counts = build_word_counts(train_lines, tokenize_line)
    boundary_lexicon = load_combined_boundary_lexicon(lang)

    merges = learn_morphbpe(word_counts, MERGE_COUNTS[lang], boundary_lexicon)
    print(f"{lang}: learned {len(merges)} MorphBPE merges from {len(word_counts)} word types")

    test_lines = (DATA_DIR / lang / "test.txt").read_text(encoding="utf-8").splitlines()
    tokenized = []
    for line in test_lines:
        for word in tokenize_line(line):
            tokenized.append(apply_morphbpe(word, merges))

    n_words = len(tokenized)
    n_tokens = sum(len(t) for t in tokenized)
    print(f"{lang}: MorphBPE fertility on test set = {n_tokens / n_words:.3f} ({n_tokens} tokens / {n_words} words)")
    return merges


if __name__ == "__main__":
    train_and_apply("cree")
    train_and_apply("inuktitut")
