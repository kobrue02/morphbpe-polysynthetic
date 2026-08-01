import re
from pathlib import Path

import hfst
from tqdm import tqdm

import wandb_utils

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BOUNDARIES_DIR = Path(__file__).resolve().parent.parent / "boundaries"
TOOLS_DIR = Path(__file__).resolve().parent.parent / "tools" / "lang-crk"

_FLAG_OR_EPSILON = re.compile(r"@[^@]*@")


def _load_fst(path):
    istr = hfst.HfstInputStream(str(path))
    fst = istr.read()
    istr.close()
    return fst


def _build_combined_transducer():
    analyzer = _load_fst(TOOLS_DIR / "crk-strict-analyzer.hfstol")
    generator_b = _load_fst(TOOLS_DIR / "crk-strict-generator-with-morpheme-boundaries.hfstol")
    analyzer.remove_optimization()
    generator_b.remove_optimization()
    combined = hfst.HfstTransducer(analyzer)
    combined.compose(generator_b)
    combined.minimize()
    return combined


def word_to_morphs(combined, word):
    results = combined.lookup(word)
    if not results:
        return None
    marked, _weight = results[0]
    surface_with_marks = _FLAG_OR_EPSILON.sub("", marked)
    if surface_with_marks.replace("<", "").replace(">", "") != word:
        return None
    morphs = [m for m in re.split(r"[<>]", surface_with_marks) if m]
    return morphs if len(morphs) > 1 else None


def build_boundary_lexicon():
    wandb_utils.init_run(project="morphbpe-polysynthetic-languages", name="cree-fst-boundaries")

    combined = _build_combined_transducer()

    train_path = DATA_DIR / "cree" / "train.txt"
    from tokenize_utils import tokenize_line

    lines = train_path.read_text(encoding="utf-8").splitlines()
    word_types = sorted({w for line in lines for w in tokenize_line(line)})

    out_path = BOUNDARIES_DIR / "cree_fst.segments"
    BOUNDARIES_DIR.mkdir(parents=True, exist_ok=True)
    n_covered = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for word in tqdm(word_types, desc="cree FST lookup", unit="word"):
            morphs = word_to_morphs(combined, word)
            if morphs is not None:
                n_covered += 1
                f.write(f"{word}\t{' '.join(morphs)}\n")

    coverage_pct = 100 * n_covered / len(word_types)
    print(
        f"cree FST: {n_covered}/{len(word_types)} word types got a multi-morph "
        f"boundary split ({coverage_pct:.1f}%) -> {out_path}"
    )
    wandb_utils.log({"cree/fst/n_covered": n_covered, "cree/fst/n_total": len(word_types), "cree/fst/coverage_pct": coverage_pct})
    wandb_utils.finish()


if __name__ == "__main__":
    build_boundary_lexicon()
