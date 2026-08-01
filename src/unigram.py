import json
import tempfile
from pathlib import Path

import sentencepiece as spm

import merge_cache

MODELS_DIR = Path(__file__).resolve().parent.parent / "cache" / "unigram"

_META_SYMBOL = "▁"


def train_unigram(word_counts, vocab_size, model_name):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_prefix = str(MODELS_DIR / model_name)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False, encoding="utf-8") as f:
        for word, count in word_counts.items():
            f.write(f"{word}\t{count}\n")
        corpus_path = f.name

    try:
        spm.SentencePieceTrainer.train(
            input=corpus_path,
            input_format="tsv",
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type="unigram",
            character_coverage=1.0,
            unk_id=0,
            bos_id=-1,
            eos_id=-1,
            pad_id=-1,
        )
    finally:
        Path(corpus_path).unlink(missing_ok=True)

    return spm.SentencePieceProcessor(model_file=f"{model_prefix}.model")


def apply_unigram(word, sp):
    pieces = sp.encode(word, out_type=str)
    tokens = [p[len(_META_SYMBOL):] if p.startswith(_META_SYMBOL) else p for p in pieces]
    return [t for t in tokens if t]


def load_or_train_unigram(model_name, file_paths, extra, word_counts, vocab_size):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"{model_name}.model"
    fingerprint_path = MODELS_DIR / f"{model_name}.fingerprint"
    fp = merge_cache.fingerprint(file_paths, extra)

    if model_path.exists() and fingerprint_path.exists():
        cached_fp = json.loads(fingerprint_path.read_text(encoding="utf-8")).get("fingerprint")
        if cached_fp == fp:
            print(f"  [cache hit] {model_name}: reusing cached unigram model")
            return spm.SentencePieceProcessor(model_file=str(model_path))
        print(f"  [cache miss] {model_name}: inputs changed, retraining")
    else:
        print(f"  [cache miss] {model_name}: no cache yet, training")

    sp = train_unigram(word_counts, vocab_size, model_name)
    fingerprint_path.write_text(json.dumps({"fingerprint": fp}), encoding="utf-8")
    return sp
