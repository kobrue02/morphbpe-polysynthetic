from pathlib import Path

import morfessor
from tqdm import tqdm

import wandb_utils
from tokenize_utils import tokenize_line

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BOUNDARIES_DIR = Path(__file__).resolve().parent.parent / "boundaries"


def train_morfessor(lang):
    wandb_utils.init_run(project="morphbpe-polysynthetic-languages", name=f"{lang}-morfessor")

    train_path = DATA_DIR / lang / "train.txt"
    lines = train_path.read_text(encoding="utf-8").splitlines()

    word_counts = {}
    for line in lines:
        for word in tokenize_line(line):
            word_counts[word] = word_counts.get(word, 0) + 1

    model = morfessor.BaselineModel()
    model.load_data((count, word) for word, count in word_counts.items())
    model.train_batch()

    out_path = BOUNDARIES_DIR / f"{lang}_morfessor.segments"
    BOUNDARIES_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for word in tqdm(word_counts, desc=f"{lang} morfessor segment+dump", unit="word"):
            morphs, _score = model.viterbi_segment(word)
            f.write(f"{word}\t{' '.join(morphs)}\n")

    print(f"{lang}: trained Morfessor on {len(word_counts)} word types -> {out_path}")
    wandb_utils.log({f"{lang}/morfessor/n_word_types": len(word_counts)})
    wandb_utils.finish()


if __name__ == "__main__":
    import sys

    for lang in sys.argv[1:] or ["cree", "inuktitut", "guarani"]:
        train_morfessor(lang)
