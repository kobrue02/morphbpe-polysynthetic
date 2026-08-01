import sys
from pathlib import Path

from tqdm import tqdm

import wandb_utils

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "apertium-grn"))
from build_seg_transducer import load_or_build  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BOUNDARIES_DIR = Path(__file__).resolve().parent.parent / "boundaries"


def word_to_morphs(seg_fst, word):
    results = seg_fst.lookup(word, max_number=5, time_cutoff=2.0)
    if not results:
        return None
    surface_with_marks, _weight = results[0]
    if surface_with_marks.replace(">", "") != word:
        return None
    morphs = [m for m in surface_with_marks.split(">") if m]
    return morphs if len(morphs) > 1 else None


def build_boundary_lexicon():
    wandb_utils.init_run(project="morphbpe-polysynthetic-languages", name="guarani-fst-boundaries")

    seg_fst = load_or_build()

    train_path = DATA_DIR / "guarani" / "train.txt"
    from tokenize_utils import tokenize_line

    lines = train_path.read_text(encoding="utf-8").splitlines()
    word_types = sorted({w for line in lines for w in tokenize_line(line)})

    out_path = BOUNDARIES_DIR / "guarani_apertium.segments"
    BOUNDARIES_DIR.mkdir(parents=True, exist_ok=True)
    n_covered = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for word in tqdm(word_types, desc="guarani apertium lookup", unit="word"):
            morphs = word_to_morphs(seg_fst, word)
            if morphs is not None:
                n_covered += 1
                f.write(f"{word}\t{' '.join(morphs)}\n")

    coverage_pct = 100 * n_covered / len(word_types)
    print(
        f"guarani apertium: {n_covered}/{len(word_types)} word types got a multi-morph "
        f"boundary split ({coverage_pct:.1f}%) -> {out_path}"
    )
    wandb_utils.log({
        "guarani/apertium/n_covered": n_covered,
        "guarani/apertium/n_total": len(word_types),
        "guarani/apertium/coverage_pct": coverage_pct,
    })
    wandb_utils.finish()


if __name__ == "__main__":
    build_boundary_lexicon()
