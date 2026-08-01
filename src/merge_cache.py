import hashlib
import json
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"


def _hash_files(*paths):
    h = hashlib.sha256()
    for path in paths:
        if path.exists():
            h.update(path.read_bytes())
        else:
            h.update(b"<missing>")
        h.update(b"\x00")
    return h.hexdigest()


def fingerprint(file_paths, extra):
    h = hashlib.sha256()
    h.update(_hash_files(*file_paths).encode("utf-8"))
    h.update(repr(extra).encode("utf-8"))
    return h.hexdigest()[:24]


def load_or_train_merges(cache_name, file_paths, extra, train_fn):
    CACHE_DIR.mkdir(exist_ok=True)
    cache_path = CACHE_DIR / f"{cache_name}.json"
    fp = fingerprint(file_paths, extra)

    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("fingerprint") == fp:
            print(f"  [cache hit] {cache_name}: reusing {len(cached['merges'])} cached merges")
            return [tuple(pair) for pair in cached["merges"]]
        print(f"  [cache miss] {cache_name}: inputs changed, retraining")
    else:
        print(f"  [cache miss] {cache_name}: no cache yet, training")

    merges = train_fn()
    cache_path.write_text(
        json.dumps({"fingerprint": fp, "merges": merges}, ensure_ascii=False),
        encoding="utf-8",
    )
    return merges
