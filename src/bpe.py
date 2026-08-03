from collections import Counter, defaultdict
from pathlib import Path

from tqdm import tqdm

import wandb_utils

END_MARKER = "</w>"

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def word_to_symbols(word):
    chars = list(word)
    chars[-1] = chars[-1] + END_MARKER
    return tuple(chars)


def _pairs_in(symbols):
    return zip(symbols, symbols[1:])


def _init_pair_index(word_freqs, pair_allowed):
    pair_counts = Counter()
    pair_to_words = defaultdict(set)
    for symbols, freq in word_freqs.items():
        for pair in _pairs_in(symbols):
            if not pair_allowed(pair):
                continue
            pair_counts[pair] += freq
            pair_to_words[pair].add(symbols)
    return pair_counts, pair_to_words


def _apply_merge_to_word(symbols, a, b):
    merged = a + b
    new_symbols = []
    i = 0
    while i < len(symbols):
        if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
            new_symbols.append(merged)
            i += 2
        else:
            new_symbols.append(symbols[i])
            i += 1
    return tuple(new_symbols)


def learn_merges(word_freqs, num_merges, pair_allowed=lambda pair: True, log_label=""):
    word_freqs = dict(word_freqs)
    pair_counts, pair_to_words = _init_pair_index(word_freqs, pair_allowed)

    merges = []
    progress = tqdm(range(num_merges), desc=log_label or "merges", unit="merge", disable=not log_label)
    for merge_i in progress:
        if not pair_counts:
            break
        best_pair, best_count = max(pair_counts.items(), key=lambda kv: (kv[1], kv[0]))
        if best_count < 2:
            break
        merges.append(best_pair)
        if log_label:
            progress.set_postfix(pair=best_pair, count=best_count, distinct_pairs=len(pair_counts))
            if merge_i % 100 == 0:
                wandb_utils.log(
                    {
                        f"{log_label}/best_pair_count": best_count,
                        f"{log_label}/distinct_pairs": len(pair_counts),
                        f"{log_label}/word_forms": len(word_freqs),
                        f"{log_label}/merge_i": merge_i,
                    }
                )
        a, b = best_pair

        affected = pair_to_words.pop(best_pair, set())
        new_words_touched = set()
        for old_symbols in affected:
            freq = word_freqs.pop(old_symbols)
            for pair in _pairs_in(old_symbols):
                if not pair_allowed(pair):
                    continue
                pair_counts[pair] -= freq
                if pair_counts[pair] <= 0:
                    del pair_counts[pair]
                pair_to_words[pair].discard(old_symbols)

            new_symbols = _apply_merge_to_word(old_symbols, a, b)
            word_freqs[new_symbols] = word_freqs.get(new_symbols, 0) + freq
            new_words_touched.add(new_symbols)

        for new_symbols in new_words_touched:
            freq = word_freqs[new_symbols]
            for pair in _pairs_in(new_symbols):
                if not pair_allowed(pair):
                    continue
                pair_counts[pair] += freq
                pair_to_words[pair].add(new_symbols)

    return merges


def apply_merges(symbols, merges):
    symbols = list(symbols)
    for a, b in merges:
        merged = a + b
        i = 0
        new_symbols = []
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                new_symbols.append(merged)
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        symbols = new_symbols
    return symbols


def apply_merges_with_checkpoints(symbols, merges, checkpoint_lengths):
    symbols = list(symbols)
    checkpoints = sorted(set(checkpoint_lengths))
    snapshots = {}
    ci = 0
    if ci < len(checkpoints) and checkpoints[ci] == 0:
        snapshots[0] = list(symbols)
        ci += 1

    for i, (a, b) in enumerate(merges, start=1):
        if ci >= len(checkpoints):
            break
        merged = a + b
        j = 0
        new_symbols = []
        while j < len(symbols):
            if j < len(symbols) - 1 and symbols[j] == a and symbols[j + 1] == b:
                new_symbols.append(merged)
                j += 2
            else:
                new_symbols.append(symbols[j])
                j += 1
        symbols = new_symbols
        while ci < len(checkpoints) and checkpoints[ci] == i:
            snapshots[i] = list(symbols)
            ci += 1

    while ci < len(checkpoints):
        snapshots[checkpoints[ci]] = list(symbols)
        ci += 1

    return snapshots


def finalize_tokens(symbols):
    symbols = list(symbols)
    if symbols and symbols[-1].endswith(END_MARKER):
        symbols[-1] = symbols[-1][: -len(END_MARKER)]
    return [s for s in symbols if s]


def learn_bpe(word_counts, num_merges):
    word_freqs = {}
    for word, freq in word_counts.items():
        symbols = word_to_symbols(word)
        word_freqs[symbols] = word_freqs.get(symbols, 0) + freq
    return learn_merges(word_freqs, num_merges, log_label="bpe")


def apply_bpe(word, merges):
    return finalize_tokens(apply_merges(word_to_symbols(word), merges))


def build_word_counts(lines, tokenize_line):
    word_counts = {}
    for line in lines:
        for word in tokenize_line(line):
            word_counts[word] = word_counts.get(word, 0) + 1
    return word_counts


MERGE_COUNTS = {"cree": 4000, "inuktitut": 8000, "guarani": 12000}


def train_and_apply(lang):
    from tokenize_utils import tokenize_line

    train_lines = (DATA_DIR / lang / "train.txt").read_text(encoding="utf-8").splitlines()
    word_counts = build_word_counts(train_lines, tokenize_line)

    merges = learn_bpe(word_counts, MERGE_COUNTS[lang])
    print(f"{lang}: learned {len(merges)} BPE merges from {len(word_counts)} word types")

    test_lines = (DATA_DIR / lang / "test.txt").read_text(encoding="utf-8").splitlines()
    tokenized = []
    for line in test_lines:
        for word in tokenize_line(line):
            tokenized.append(apply_bpe(word, merges))

    n_words = len(tokenized)
    n_tokens = sum(len(t) for t in tokenized)
    print(f"{lang}: BPE fertility on test set = {n_tokens / n_words:.3f} ({n_tokens} tokens / {n_words} words)")
    return merges


if __name__ == "__main__":
    train_and_apply("cree")
    train_and_apply("inuktitut")
