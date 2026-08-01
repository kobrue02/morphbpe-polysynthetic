#!/bin/bash
#SBATCH --job-name=MorphBPE_Inuktitut
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=konrad-rudolf.brueggemann@student.uni-tuebingen.de

PROJECT_ROOT=/home/tu/tu_tu/tu_zxoqp65/work/morphbpe-polysynthetic

module load devel/python/3.13.3-llvm-19.1
module load devel/cuda/12.8
echo "Python: $(which python3)"

export PYTHONUNBUFFERED=1
export HF_HOME=$PROJECT_ROOT/.cache/huggingface
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PATH=$PROJECT_ROOT/tools/jdk/bin:$PATH
mkdir -p "$HF_HOME"

if [ ! -x "$PROJECT_ROOT/tools/jdk/bin/java" ]; then
    echo "No JDK at tools/jdk/ -- run 'bash jobs/setup_jdk.sh' on the login node first." >&2
    exit 1
fi
echo "Java: $(which java) ($(java -version 2>&1 | head -1))"

source $PROJECT_ROOT/.venv/bin/activate
cd $PROJECT_ROOT
uv sync
mkdir -p logs
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available(), '--', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU visible')"

javac -cp tools/uqailaut/Uqailaut.jar tools/uqailaut/BatchDecompose.java

echo "Starting Inuktitut pipeline"
python3 src/morfessor_train.py inuktitut && \
python3 src/fst_boundaries_iku.py && \
python3 src/silver_standard.py inuktitut && \
python3 src/run_experiment.py inuktitut

if [ $? -eq 0 ]; then
    echo "Inuktitut pipeline complete."
else
    echo "Inuktitut pipeline failed." && exit 1
fi
