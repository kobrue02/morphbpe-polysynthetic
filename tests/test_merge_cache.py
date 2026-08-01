import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import merge_cache


def _with_temp_cache_dir(fn):
    original = merge_cache.CACHE_DIR
    tmp = Path(tempfile.mkdtemp())
    merge_cache.CACHE_DIR = tmp
    try:
        fn(tmp)
    finally:
        merge_cache.CACHE_DIR = original
        shutil.rmtree(tmp, ignore_errors=True)


def test_cache_hit_avoids_retraining():
    def run(tmp):
        train_file = tmp / "train.txt"
        train_file.write_text("hello world", encoding="utf-8")

        calls = []

        def train_fn():
            calls.append(1)
            return [("a", "b"), ("c", "d")]

        merges1 = merge_cache.load_or_train_merges("test", [train_file], {"ceiling": 100}, train_fn)
        merges2 = merge_cache.load_or_train_merges("test", [train_file], {"ceiling": 100}, train_fn)

        assert merges1 == [("a", "b"), ("c", "d")]
        assert merges2 == merges1
        assert len(calls) == 1, "train_fn should only run once (cache hit on the second call)"

    _with_temp_cache_dir(run)


def test_cache_invalidated_by_changed_file_content():
    def run(tmp):
        train_file = tmp / "train.txt"
        train_file.write_text("version one", encoding="utf-8")

        calls = []

        def train_fn():
            calls.append(1)
            return [("x", "y")]

        merge_cache.load_or_train_merges("test", [train_file], {"ceiling": 100}, train_fn)
        train_file.write_text("version two -- different content", encoding="utf-8")
        merge_cache.load_or_train_merges("test", [train_file], {"ceiling": 100}, train_fn)

        assert len(calls) == 2, "changed file content must invalidate the cache"

    _with_temp_cache_dir(run)


def test_cache_invalidated_by_changed_extra():
    def run(tmp):
        train_file = tmp / "train.txt"
        train_file.write_text("same content", encoding="utf-8")

        calls = []

        def train_fn():
            calls.append(1)
            return [("x", "y")]

        merge_cache.load_or_train_merges("test", [train_file], {"ceiling": 100}, train_fn)
        merge_cache.load_or_train_merges("test", [train_file], {"ceiling": 200}, train_fn)

        assert len(calls) == 2, "a changed merge ceiling (or other extra state) must invalidate the cache"

    _with_temp_cache_dir(run)


if __name__ == "__main__":
    test_cache_hit_avoids_retraining()
    test_cache_invalidated_by_changed_file_content()
    test_cache_invalidated_by_changed_extra()
    print("merge_cache.py tests passed")
