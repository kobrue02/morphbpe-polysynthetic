import random
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SPLIT_SEED = 0
TRAIN_FRAC = 0.90
DEV_FRAC = 0.05


def split_language(lang):
    raw_path = DATA_DIR / lang / "raw.txt"
    lines = raw_path.read_text(encoding="utf-8").splitlines()

    rng = random.Random(SPLIT_SEED)
    shuffled = lines[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * TRAIN_FRAC)
    n_dev = int(n * DEV_FRAC)

    splits = {
        "train": shuffled[:n_train],
        "dev": shuffled[n_train : n_train + n_dev],
        "test": shuffled[n_train + n_dev :],
    }
    for name, split_lines in splits.items():
        out_path = DATA_DIR / lang / f"{name}.txt"
        out_path.write_text("\n".join(split_lines) + "\n", encoding="utf-8")
        print(f"{lang} {name}: {len(split_lines)} sentences -> {out_path}")


if __name__ == "__main__":
    import sys

    for lang in sys.argv[1:] or ["cree", "inuktitut"]:
        split_language(lang)
