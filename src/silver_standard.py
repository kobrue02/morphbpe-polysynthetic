from pathlib import Path

from tokenize_utils import tokenize_line

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BOUNDARIES_DIR = Path(__file__).resolve().parent.parent / "boundaries"


def novel_test_word_types(lang):
    train_lines = (DATA_DIR / lang / "train.txt").read_text(encoding="utf-8").splitlines()
    test_lines = (DATA_DIR / lang / "test.txt").read_text(encoding="utf-8").splitlines()
    train_types = {w for line in train_lines for w in tokenize_line(line)}
    test_types = {w for line in test_lines for w in tokenize_line(line)}
    novel = sorted(test_types - train_types)
    return novel, train_types, test_types


def _write(lang, lexicon, candidate_words):
    out_path = BOUNDARIES_DIR / f"{lang}_silver_test.segments"
    BOUNDARIES_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for word, morphs in lexicon.items():
            f.write(f"{word}\t{' '.join(morphs)}\n")
    coverage = 100 * len(lexicon) / len(candidate_words) if candidate_words else 0.0
    print(
        f"{lang}: silver test set -- {len(lexicon)}/{len(candidate_words)} novel "
        f"(train-vocabulary-disjoint) test word types got a confident analyzer parse "
        f"({coverage:.1f}%) -> {out_path}"
    )
    return out_path


def build_silver_test_cree():
    from fst_boundaries_cree import _build_combined_transducer, word_to_morphs
    from tqdm import tqdm

    novel, train_types, test_types = novel_test_word_types("cree")
    combined = _build_combined_transducer()
    lexicon = {}
    for word in tqdm(novel, desc="cree silver-test FST lookup", unit="word"):
        morphs = word_to_morphs(combined, word)
        if morphs is not None:
            lexicon[word] = morphs
    return _write("cree", lexicon, novel)


def build_silver_test_guarani():
    from fst_boundaries_guarani import load_or_build, word_to_morphs
    from tqdm import tqdm

    novel, train_types, test_types = novel_test_word_types("guarani")
    seg_fst = load_or_build()
    lexicon = {}
    for word in tqdm(novel, desc="guarani silver-test apertium lookup", unit="word"):
        morphs = word_to_morphs(seg_fst, word)
        if morphs is not None:
            lexicon[word] = morphs
    return _write("guarani", lexicon, novel)


def build_silver_test_inuktitut(max_words=5000):
    from fst_boundaries_iku import decompose_words, decomposition_to_morphs

    novel, train_types, test_types = novel_test_word_types("inuktitut")
    words = novel[:max_words] if max_words is not None else novel
    decompositions = decompose_words(words)
    lexicon = {}
    for word, decomposition in zip(words, decompositions):
        morphs = decomposition_to_morphs(decomposition, word)
        if morphs is not None:
            lexicon[word] = morphs
    return _write("inuktitut", lexicon, words)


BUILDERS = {
    "cree": build_silver_test_cree,
    "inuktitut": build_silver_test_inuktitut,
    "guarani": build_silver_test_guarani,
}

if __name__ == "__main__":
    import sys

    for lang in sys.argv[1:] or list(BUILDERS):
        BUILDERS[lang]()
