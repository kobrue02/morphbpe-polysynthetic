#!/bin/bash
#SBATCH --job-name=MorphBPE_Cree
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=konrad-rudolf.brueggemann@student.uni-tuebingen.de

PROJECT_ROOT=/home/tu/tu_tu/tu_zxoqp65/work/morphbpe-polysynthetic

module load devel/python/3.13.3-llvm-19.1
echo "Python: $(which python3)"

export PYTHONUNBUFFERED=1
export HF_HOME=$PROJECT_ROOT/.cache/huggingface
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
mkdir -p "$HF_HOME"

source $PROJECT_ROOT/.venv/bin/activate
cd $PROJECT_ROOT
uv sync
mkdir -p logs

echo "Starting Cree pipeline"
python3 src/fst_boundaries_cree.py && \
python3 src/run_experiment.py cree

if [ $? -eq 0 ]; then
    echo "Cree pipeline complete."
else
    echo "Cree pipeline failed." && exit 1
fi
