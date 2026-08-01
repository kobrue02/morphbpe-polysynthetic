import queue
import random
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

from tqdm import tqdm

import wandb_utils

TOOLS_DIR = Path(__file__).resolve().parent.parent / "tools" / "uqailaut"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BOUNDARIES_DIR = Path(__file__).resolve().parent.parent / "boundaries"

PER_WORD_TIMEOUT_SECONDS = 8.0
RESULT_PREFIX = "RESULT"

MAX_TOTAL_SECONDS = 90 * 60

_DECOMP_PART = re.compile(r"\{([^:{}]*):[^{}]*\}")


JDK_DIR = Path(__file__).resolve().parent.parent / "tools" / "jdk"


def _java_binary():
    for candidate in (JDK_DIR / "bin" / "java", "/opt/homebrew/opt/openjdk/bin/java", "java"):
        if candidate == "java":
            found = shutil.which("java")
            if found:
                return found
        elif Path(candidate).exists():
            return str(candidate)
    raise RuntimeError(
        "No Java runtime found. Run jobs/setup_jdk.sh to unpack a portable JDK into "
        "tools/jdk/, or install one locally (e.g. `brew install openjdk`)."
    )


def _start_process():
    java = _java_binary()
    classpath = f"{TOOLS_DIR}:{TOOLS_DIR / 'Uqailaut.jar'}"
    return subprocess.Popen(
        [java, "-cp", classpath, "BatchDecompose"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        bufsize=1,
    )


def decompose_words(words):
    results = [None] * len(words)
    idx = 0
    n_restarts = 0
    start_time = time.monotonic()
    progress = tqdm(total=len(words), desc="inuktitut Uqailaut decompose", unit="word")

    while idx < len(words):
        if time.monotonic() - start_time > MAX_TOTAL_SECONDS:
            progress.write(
                f"  ...hit the {MAX_TOTAL_SECONDS}s overall deadline at word {idx}/{len(words)}; "
                f"treating the rest as unanalyzable"
            )
            break
        remaining = words[idx:]
        proc = _start_process()
        out_q = queue.Queue()

        def writer(stdin=proc.stdin, to_send=remaining):
            try:
                for w in to_send:
                    stdin.write(w + "\n")
                stdin.close()
            except (BrokenPipeError, ValueError):
                pass

        def reader(stdout=proc.stdout, q=out_q):
            try:
                for line in iter(stdout.readline, ""):
                    line = "".join(c for c in line if c.isprintable()).rstrip("\n")
                    if line.startswith(RESULT_PREFIX):
                        q.put(line[len(RESULT_PREFIX):])
            except ValueError:
                pass
            q.put(None)

        threading.Thread(target=writer, daemon=True).start()
        threading.Thread(target=reader, daemon=True).start()

        local_i = 0
        stalled = False
        while local_i < len(remaining):
            try:
                line = out_q.get(timeout=PER_WORD_TIMEOUT_SECONDS)
            except queue.Empty:
                stalled = True
                break
            if line is None:
                break
            results[idx + local_i] = line
            local_i += 1

        proc.kill()
        proc.wait()

        progress.update(local_i)
        progress.set_postfix(stalls=n_restarts)

        if stalled:
            n_restarts += 1
            progress.write(
                f"  ...stall on {words[idx + local_i]!r} (word {idx + local_i}/{len(words)}), "
                f"skipping and restarting"
            )
            results[idx + local_i] = ""
            idx = idx + local_i + 1
            progress.update(1)
            wandb_utils.log({"inuktitut/uqailaut/n_restarts": n_restarts, "inuktitut/uqailaut/word_index": idx})
        elif local_i < len(remaining):
            results[idx + local_i] = ""
            idx = idx + local_i + 1
            progress.update(1)
        else:
            idx = idx + local_i

    progress.close()
    return [r if r is not None else "" for r in results]


def decomposition_to_morphs(decomposition, word):
    if not decomposition:
        return None
    morphs = _DECOMP_PART.findall(decomposition)
    if not morphs or "".join(morphs) != word:
        return None
    return morphs if len(morphs) > 1 else None


_LIKELY_UNANALYZABLE = re.compile(r"\d|--|[^\w\-]|^[A-Z]")

MAX_WORD_TYPES = 15_000


def build_boundary_lexicon():
    from collections import Counter

    from tokenize_utils import tokenize_line

    wandb_utils.init_run(project="morphbpe-polysynthetic-languages", name="inuktitut-uqailaut-boundaries")

    train_path = DATA_DIR / "inuktitut" / "train.txt"
    lines = train_path.read_text(encoding="utf-8").splitlines()
    word_counts = Counter(w for line in lines for w in tokenize_line(line))

    candidates = [w for w in word_counts if not _LIKELY_UNANALYZABLE.search(w)]
    n_skipped_unanalyzable = len(word_counts) - len(candidates)

    candidates.sort(key=lambda w: word_counts[w], reverse=True)
    word_types = candidates[:MAX_WORD_TYPES]
    random.Random(0).shuffle(word_types)
    n_skipped_rare = len(candidates) - len(word_types)

    print(
        f"inuktitut Uqailaut: decomposing {len(word_types)} most-frequent word types "
        f"({n_skipped_unanalyzable} pre-skipped as likely-unanalyzable, "
        f"{n_skipped_rare} more skipped as too rare to be worth the cost)..."
    )
    decompositions = decompose_words(word_types)

    out_path = BOUNDARIES_DIR / "inuktitut_uqailaut.segments"
    BOUNDARIES_DIR.mkdir(parents=True, exist_ok=True)
    n_covered = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for word, decomposition in tqdm(
            zip(word_types, decompositions), total=len(word_types), desc="inuktitut Uqailaut dump", unit="word"
        ):
            morphs = decomposition_to_morphs(decomposition, word)
            if morphs is not None:
                n_covered += 1
                f.write(f"{word}\t{' '.join(morphs)}\n")

    coverage_pct = 100 * n_covered / len(word_types)
    print(
        f"inuktitut Uqailaut: {n_covered}/{len(word_types)} word types got a multi-morph "
        f"boundary split ({coverage_pct:.1f}%) -> {out_path}"
    )
    wandb_utils.log({
        "inuktitut/uqailaut/n_covered": n_covered,
        "inuktitut/uqailaut/n_total": len(word_types),
        "inuktitut/uqailaut/coverage_pct": coverage_pct,
    })
    wandb_utils.finish()


if __name__ == "__main__":
    build_boundary_lexicon()
