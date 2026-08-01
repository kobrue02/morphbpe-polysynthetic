#!/bin/bash
set -e

PROJECT_ROOT=/home/tu/tu_tu/tu_zxoqp65/work/morphbpe-polysynthetic

module load devel/python/3.13.3-llvm-19.1
cd "$PROJECT_ROOT"
[ -d .venv ] || uv venv .venv
source .venv/bin/activate
uv sync

python3 src/fetch_data.py inuktitut
python3 src/split_data.py inuktitut

echo "Inuktitut prep complete -- if you haven't yet, also run: bash jobs/setup_jdk.sh"
echo "Then submit: sbatch jobs/pipeline_inuktitut.sh"
