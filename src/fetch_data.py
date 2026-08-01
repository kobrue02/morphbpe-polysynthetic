import csv
import io
import json
import unicodedata
from pathlib import Path

import requests
from huggingface_hub import hf_hub_download

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

INUKTITUT_SUBSAMPLE_SIZE = None
INUKTITUT_SUBSAMPLE_SEED = 0

CREE_DATASET_REVISION = "5c5644bb85c4c978a0f6f84803f3a53af713faf9"
INUKTITUT_DATASET_REVISION = "af2ceb9d88433ea0174df8b4097e6851c6959a15"
GUARANI_COMMIT = "d5511aeef0a2d6875422a232437d73560fbc35d8"

GUARANI_CSV_URL = (
    f"https://raw.githubusercontent.com/pln-fing-udelar/jojajovai/{GUARANI_COMMIT}/data/jojajovai_all.csv"
)


def normalize_lines(lines):
    seen = set()
    out = []
    for line in lines:
        line = unicodedata.normalize("NFC", line).strip()
        if not line or line in seen:
            continue
        seen.add(line)
        out.append(line)
    return out


def fetch_cree():
    import pyarrow.parquet as pq

    texts = []
    for split in ("gold", "silver"):
        path = hf_hub_download(
            repo_id="KonradBRG/plains-cree-figurative",
            filename=f"{split}/{split}-00000-of-00001.parquet",
            repo_type="dataset",
            revision=CREE_DATASET_REVISION,
        )
        table = pq.read_table(path, columns=["text_cree"])
        texts.extend(table.column("text_cree").to_pylist())
    lines = normalize_lines(texts)

    out_path = DATA_DIR / "cree" / "raw.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"cree: {len(lines)} unique sentences -> {out_path}")
    return lines


def fetch_inuktitut():
    jsonl_path = hf_hub_download(
        repo_id="EdinburghNLP/nunavut-hansard-plusplus",
        filename="NunavutHansard.iu.rom.jsonl",
        repo_type="dataset",
        revision=INUKTITUT_DATASET_REVISION,
    )
    with open(jsonl_path, encoding="utf-8") as f:
        texts = [json.loads(line)["text"] for line in f]
    all_lines = normalize_lines(texts)

    if INUKTITUT_SUBSAMPLE_SIZE is not None and len(all_lines) > INUKTITUT_SUBSAMPLE_SIZE:
        import random

        rng = random.Random(INUKTITUT_SUBSAMPLE_SEED)
        lines = rng.sample(all_lines, INUKTITUT_SUBSAMPLE_SIZE)
    else:
        lines = all_lines

    out_path = DATA_DIR / "inuktitut" / "raw.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"inuktitut: {len(lines)} of {len(all_lines)} unique sentences -> {out_path}")
    return lines


def fetch_guarani():
    response = requests.get(GUARANI_CSV_URL, timeout=60)
    response.raise_for_status()
    reader = csv.DictReader(io.StringIO(response.text))

    lines_by_split = {"train": [], "dev": [], "test": []}
    for row in reader:
        lines_by_split[row["split"]].append(row["gn"])

    out_dir = DATA_DIR / "guarani"
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, raw_lines in lines_by_split.items():
        lines = normalize_lines(raw_lines)
        out_path = out_dir / f"{split}.txt"
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"guarani/{split}: {len(lines)} unique sentences -> {out_path}")
    return lines_by_split


FETCHERS = {"cree": fetch_cree, "inuktitut": fetch_inuktitut, "guarani": fetch_guarani}

if __name__ == "__main__":
    import sys

    langs = sys.argv[1:] or list(FETCHERS)
    for lang in langs:
        FETCHERS[lang]()
