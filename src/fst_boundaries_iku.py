import os
import queue
import re
import shutil
import subprocess
import threading
from pathlib import Path

from tqdm import tqdm

import wandb_utils

TOOLS_DIR = Path(__file__).resolve().parent.parent / "tools" / "uqailaut"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BOUNDARIES_DIR = Path(__file__).resolve().parent.parent / "boundaries"

PER_WORD_TIMEOUT_SECONDS = 8.0
RESULT_PREFIX = "RESULT"

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


def _num_workers():
    env = os.environ.get("SLURM_CPUS_PER_TASK")
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    return max(1, os.cpu_count() or 1)


def _decompose_chunk(indexed_words, results, progress, progress_lock, stall_counter, stall_lock):
    idx = 0
    while idx < len(indexed_words):
        remaining = indexed_words[idx:]
        proc = _start_process()
        out_q = queue.Queue()

        def writer(stdin=proc.stdin, to_send=remaining):
            try:
                for _, w in to_send:
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
            results[remaining[local_i][0]] = line
            local_i += 1
            with progress_lock:
                progress.update(1)

        proc.kill()
        proc.wait()

        if stalled:
            with stall_lock:
                stall_counter[0] += 1
                n_stalls = stall_counter[0]
            results[remaining[local_i][0]] = ""
            idx = idx + local_i + 1
            with progress_lock:
                progress.update(1)
                progress.set_postfix(stalls=n_stalls)
        elif local_i < len(remaining):
            # reader hit EOF before delivering a result for this word (process died)
            results[remaining[local_i][0]] = ""
            idx = idx + local_i + 1
            with progress_lock:
                progress.update(1)
        else:
            idx = idx + local_i


def decompose_words(words, num_workers=None):
    if not words:
        return []
    num_workers = max(1, min(num_workers or _num_workers(), len(words)))

    results = [None] * len(words)
    progress = tqdm(total=len(words), desc="inuktitut Uqailaut decompose", unit="word")
    progress_lock = threading.Lock()
    stall_counter = [0]
    stall_lock = threading.Lock()

    # Round-robin (not contiguous) assignment: words are frequency-sorted, and
    # frequency tends to correlate with word length/complexity in polysynthetic
    # languages, so a contiguous split would give some workers all the fast
    # (frequent, short) words and others all the slow (rare, long) ones.
    indexed = list(enumerate(words))
    threads = [
        threading.Thread(
            target=_decompose_chunk,
            args=(indexed[worker::num_workers], results, progress, progress_lock, stall_counter, stall_lock),
        )
        for worker in range(num_workers)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

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


def build_boundary_lexicon():
    from collections import Counter

    from tokenize_utils import tokenize_line

    wandb_utils.init_run(project="morphbpe-polysynthetic-languages", name="inuktitut-uqailaut-boundaries")

    train_path = DATA_DIR / "inuktitut" / "train.txt"
    lines = train_path.read_text(encoding="utf-8").splitlines()
    word_counts = Counter(w for line in lines for w in tokenize_line(line))

    word_types = [w for w in word_counts if not _LIKELY_UNANALYZABLE.search(w)]
    n_skipped_unanalyzable = len(word_counts) - len(word_types)
    word_types.sort(key=lambda w: word_counts[w], reverse=True)

    print(
        f"inuktitut Uqailaut: decomposing all {len(word_types)} candidate word types "
        f"({n_skipped_unanalyzable} pre-skipped as likely-unanalyzable), "
        f"using {_num_workers()} parallel workers..."
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
