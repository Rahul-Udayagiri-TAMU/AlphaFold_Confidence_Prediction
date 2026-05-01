#!/bin/bash
#SBATCH --job-name=fixchk_esm_ridge
#SBATCH --output=/scratch/user/%u/ECEN766_final/logs/%x_%j.out
#SBATCH --error=/scratch/user/%u/ECEN766_final/logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

set -euxo pipefail

module purge
module load GCC/12.3.0 OpenMPI/4.1.5 PyTorch/2.1.2-CUDA-12.1.1

cd $SCRATCH/ECEN766_final

source venv_ecen766/bin/activate
unset PYTHONPATH
unset PYTHONHOME
export PYTHONNOUSERSITE=1
export HF_HOME=$SCRATCH/ECEN766_final/hf_home

mkdir -p logs
mkdir -p outputs/fixed_check

echo "===== JOB INFO ====="
echo "Hostname: $(hostname)"
echo "Date    : $(date)"
echo "PWD     : $(pwd)"
echo "Python  : $(which python)"
python --version
echo "===================="

python code/train_fixed.py \
    --feature_dir outputs/esm_small_2000 \
    --feature_prefix esm \
    --model ridge \
    --out_dir outputs/fixed_check/esm_ridge \
    --alpha 100
