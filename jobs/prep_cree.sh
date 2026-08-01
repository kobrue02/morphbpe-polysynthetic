#!/bin/bash
set -e

PROJECT_ROOT=/home/tu/tu_tu/tu_zxoqp65/work/morphbpe-polysynthetic

module load devel/python/3.13.3-llvm-19.1
cd "$PROJECT_ROOT"
[ -d .venv ] || uv venv .venv
source .venv/bin/activate
uv sync

python3 src/fetch_data.py cree
python3 src/split_data.py cree
python3 src/morfessor_train.py cree

echo "Cree prep complete -- now run: sbatch jobs/pipeline_cree.sh"
