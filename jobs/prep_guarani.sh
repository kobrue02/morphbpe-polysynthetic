#!/bin/bash
set -e

PROJECT_ROOT=/home/tu/tu_tu/tu_zxoqp65/work/morphbpe-polysynthetic

module load devel/python/3.13.3-llvm-19.1
cd "$PROJECT_ROOT"
[ -d .venv ] || uv venv .venv
source .venv/bin/activate
uv sync

python3 src/fetch_data.py guarani
python3 src/morfessor_train.py guarani
python3 src/silver_standard.py guarani

echo "Guarani prep complete -- now run: sbatch jobs/pipeline_guarani.sh"
