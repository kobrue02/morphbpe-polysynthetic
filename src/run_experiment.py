import json
import os
from multiprocessing import Pool
from pathlib import Path

from tqdm import tqdm

import wandb_utils
from bpe import MERGE_COUNTS as BPE_MERGE_COUNTS
from bpe import apply_bpe, apply_merges_with_checkpoints, finalize_tokens, learn_bpe, word_to_symbols
from bpe import build_word_counts as _build_word_counts
from lm_transformer import train_and_evaluate as train_and_evaluate_transformer_lm
from merge_cache import load_or_train_merges
from metrics import consistency_f1, fertility, morph_edit_distance
from morphbpe import BOUNDARIES_DIR, BOUNDARY_FILES
from morphbpe import MERGE_COUNTS as MORPHBPE_MERGE_COUNTS
from morphbpe import apply_morphbpe, learn_morphbpe, load_combined_boundary_lexicon
from tokenize_utils import tokenize_line

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

CHECKPOINT_STEP = {"cree": 100, "inuktitut": 500, "guarani": 250}

_MIN_ITEMS_FOR_PARALLEL = 2000

_worker_apply_fn = None
_worker_merges = None


def _init_apply_worker(apply_fn, merges):
    global _worker_apply_fn, _worker_merges
    _worker_apply_fn = apply_fn
    _worker_merges = merges


def _tokenize_one_line(line):
    return [_worker_apply_fn(word, _worker_merges) for word in tokenize_line(line)]


def _apply_one_word(word):
    return _worker_apply_fn(word, _worker_merges)


def _n_workers():
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        try:
            return max(1, int(slurm_cpus))
        except ValueError:
            pass
    return os.cpu_count() or 1


def _parallel_map(worker_fn, items, apply_fn, merges, desc=None, unit="item"):
    n_workers = _n_workers()
    if n_workers <= 1 or len(items) < _MIN_ITEMS_FOR_PARALLEL:
        iterable = tqdm(items, desc=desc, unit=unit) if desc else items
        _init_apply_worker(apply_fn, merges)
        return [worker_fn(item) for item in iterable]

    chunksize = max(1, len(items) // (n_workers * 8))
    with Pool(processes=n_workers, initializer=_init_apply_worker, initargs=(apply_fn, merges)) as pool:
        results_iter = pool.imap(worker_fn, items, chunksize=chunksize)
        if desc:
            results_iter = tqdm(results_iter, total=len(items), desc=desc, unit=unit)
        return list(results_iter)


def _checkpoint_lengths(lang, max_len):
    step = CHECKPOINT_STEP[lang]
    lengths = list(range(step, max_len + 1, step))
    if not lengths or lengths[-1] != max_len:
        lengths.append(max_len)
    return lengths


def select_vocab_size(lang, morphbpe_merges, dev_words, reference_lexicon):
    lengths = _checkpoint_lengths(lang, len(morphbpe_merges))

    tokenized_by_length = {k: [] for k in lengths}
    for word in tqdm(dev_words, desc=f"{lang} vocab-size sweep", unit="word"):
        snapshots = apply_merges_with_checkpoints(word_to_symbols(word), morphbpe_merges, lengths)
        for k in lengths:
            tokenized_by_length[k].append(finalize_tokens(snapshots[k]))

    distances = {
        k: morph_edit_distance(dev_words, tokenized_by_length[k], reference_lexicon)["mean_edit_distance"]
        for k in lengths
    }
    best_length = min(distances, key=distances.get)
    return best_length, distances


def tokenize_sentences(lines, apply_fn, merges, desc=None):
    return _parallel_map(_tokenize_one_line, lines, apply_fn, merges, desc=desc, unit="line")


def run(lang):
    wandb_utils.init_run(
        project="morphbpe-polysynthetic-languages",
        name=lang,
        config={
            "language": lang,
            "bpe_merge_ceiling": BPE_MERGE_COUNTS[lang],
            "morphbpe_merge_ceiling": MORPHBPE_MERGE_COUNTS[lang],
        },
    )

    train_lines = (DATA_DIR / lang / "train.txt").read_text(encoding="utf-8").splitlines()
    word_counts = _build_word_counts(train_lines, tokenize_line)
    reference_lexicon = load_combined_boundary_lexicon(lang)

    dev_lines = (DATA_DIR / lang / "dev.txt").read_text(encoding="utf-8").splitlines()
    dev_words = [w for line in dev_lines for w in tokenize_line(line)]

    test_lines = (DATA_DIR / lang / "test.txt").read_text(encoding="utf-8").splitlines()
    test_words = [w for line in test_lines for w in tokenize_line(line)]

    train_path = DATA_DIR / lang / "train.txt"
    boundary_paths = [BOUNDARIES_DIR / fname for fname in BOUNDARY_FILES[lang]]

    bpe_ceiling = BPE_MERGE_COUNTS[lang]
    bpe_merges_max = load_or_train_merges(
        f"{lang}_bpe", [train_path], {"ceiling": bpe_ceiling},
        lambda: learn_bpe(word_counts, bpe_ceiling),
    )

    morphbpe_ceiling = MORPHBPE_MERGE_COUNTS[lang]
    morphbpe_merges_max = load_or_train_merges(
        f"{lang}_morphbpe", [train_path, *boundary_paths], {"ceiling": morphbpe_ceiling},
        lambda: learn_morphbpe(word_counts, morphbpe_ceiling, reference_lexicon),
    )

    selected_vocab_size, dev_distance_by_length = select_vocab_size(
        lang, morphbpe_merges_max, dev_words, reference_lexicon
    )
    print(f"{lang}: selected vocab size (merge count) = {selected_vocab_size} "
          f"(max ceiling {len(morphbpe_merges_max)}), dev mean edit distance "
          f"{dev_distance_by_length[selected_vocab_size]:.4f}")
    wandb_utils.log({
        f"{lang}/vocab_size_selection/selected": selected_vocab_size,
        f"{lang}/vocab_size_selection/dev_edit_distance_at_selected": dev_distance_by_length[selected_vocab_size],
    })
    for length, distance in dev_distance_by_length.items():
        wandb_utils.log({
            f"{lang}/vocab_size_selection/dev_edit_distance": distance,
            f"{lang}/vocab_size_selection/vocab_length": length,
        })

    bpe_merges = bpe_merges_max[:selected_vocab_size]
    morphbpe_merges = morphbpe_merges_max[:selected_vocab_size]

    bpe_tokenized = _parallel_map(_apply_one_word, test_words, apply_bpe, bpe_merges, desc=f"{lang} bpe apply (test)", unit="word")
    morphbpe_tokenized = _parallel_map(
        _apply_one_word, test_words, apply_morphbpe, morphbpe_merges, desc=f"{lang} morphbpe apply (test)", unit="word"
    )

    train_lm = lambda tr, te, variant: train_and_evaluate_transformer_lm(tr, te, lang=lang, variant=variant)

    print(f"{lang}: training small LMs (bpe, morphbpe) at the selected vocab size...")
    bpe_train_sentences = tokenize_sentences(train_lines, apply_bpe, bpe_merges, desc=f"{lang} bpe tokenize (train)")
    bpe_test_sentences = tokenize_sentences(test_lines, apply_bpe, bpe_merges, desc=f"{lang} bpe tokenize (test)")
    bpe_lm_result = train_lm(bpe_train_sentences, bpe_test_sentences, "bpe")

    morphbpe_train_sentences = tokenize_sentences(
        train_lines, apply_morphbpe, morphbpe_merges, desc=f"{lang} morphbpe tokenize (train)"
    )
    morphbpe_test_sentences = tokenize_sentences(
        test_lines, apply_morphbpe, morphbpe_merges, desc=f"{lang} morphbpe tokenize (test)"
    )
    morphbpe_lm_result = train_lm(morphbpe_train_sentences, morphbpe_test_sentences, "morphbpe")

    results = {
        "language": lang,
        "n_train_word_types": len(word_counts),
        "n_dev_words": len(dev_words),
        "n_test_words": len(test_words),
        "n_reference_lexicon_entries": len(reference_lexicon),
        "vocab_size_selection": {
            "max_ceiling": len(morphbpe_merges_max),
            "selected": selected_vocab_size,
            "dev_mean_edit_distance_by_length": dev_distance_by_length,
        },
        "bpe": {
            "num_merges": len(bpe_merges),
            "fertility": fertility(bpe_tokenized),
            "consistency": consistency_f1(test_words, bpe_tokenized, reference_lexicon),
            "morph_edit_distance": morph_edit_distance(test_words, bpe_tokenized, reference_lexicon),
            "lm": bpe_lm_result,
        },
        "morphbpe": {
            "num_merges": len(morphbpe_merges),
            "fertility": fertility(morphbpe_tokenized),
            "consistency": consistency_f1(test_words, morphbpe_tokenized, reference_lexicon),
            "morph_edit_distance": morph_edit_distance(test_words, morphbpe_tokenized, reference_lexicon),
            "lm": morphbpe_lm_result,
        },
    }

    out_path = RESULTS_DIR / f"{lang}_results.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{lang}: wrote {out_path}")

    for variant in ("bpe", "morphbpe"):
        wandb_utils.log({
            f"{lang}/{variant}/final/fertility": results[variant]["fertility"],
            f"{lang}/{variant}/final/consistency_f1": results[variant]["consistency"]["f1"],
            f"{lang}/{variant}/final/morph_edit_distance": results[variant]["morph_edit_distance"]["mean_edit_distance"],
            f"{lang}/{variant}/final/lm_test_cross_entropy": results[variant]["lm"]["test_cross_entropy"],
        })
    wandb_utils.finish()

    return results


if __name__ == "__main__":
    import sys

    for lang in sys.argv[1:] or ["cree", "inuktitut", "guarani"]:
        run(lang)
