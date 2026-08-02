#!/bin/bash
#SBATCH --job-name=MorphBPE_Inuktitut_Segment
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=konrad-rudolf.brueggemann@student.uni-tuebingen.de

PROJECT_ROOT=/home/tu/tu_tu/tu_zxoqp65/work/morphbpe-polysynthetic

module load devel/python/3.13.3-llvm-19.1
echo "Python: $(which python3)"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PATH=$PROJECT_ROOT/tools/jdk/bin:$PATH

if [ ! -x "$PROJECT_ROOT/tools/jdk/bin/java" ]; then
    echo "No JDK at tools/jdk/ -- run 'bash jobs/setup_jdk.sh' on the login node first." >&2
    exit 1
fi
echo "Java: $(which java) ($(java -version 2>&1 | head -1))"
if ! javac -version 2>&1 | head -1; then
    echo "javac at tools/jdk/ is broken -- re-run 'bash jobs/setup_jdk.sh' on the login node." >&2
    exit 1
fi

source $PROJECT_ROOT/.venv/bin/activate
cd $PROJECT_ROOT
uv sync
mkdir -p logs

javac -cp tools/uqailaut/Uqailaut.jar tools/uqailaut/BatchDecompose.java || {
    echo "Failed to compile BatchDecompose.java -- aborting rather than risk running against a stale .class file." >&2
    exit 1
}

echo "Starting Inuktitut segmentation (CPU-only: Morfessor + Uqailaut boundary lexicon + silver standard)"
python3 src/morfessor_train.py inuktitut && \
python3 src/fst_boundaries_iku.py && \
python3 src/silver_standard.py inuktitut

if [ $? -eq 0 ]; then
    echo "Inuktitut segmentation complete -- now run: sbatch jobs/pipeline_inuktitut.sh"
else
    echo "Inuktitut segmentation failed." && exit 1
fi
