#!/bin/bash
#SBATCH --job-name=final10k_esm_xgb
#SBATCH --output=/scratch/user/%u/ECEN766_final/logs/%x_%j.out
#SBATCH --error=/scratch/user/%u/ECEN766_final/logs/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

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
mkdir -p outputs/final_10000

python code/train_fixed.py \
    --feature_dir outputs/esm_small_10000 \
    --feature_prefix esm \
    --model xgb \
    --out_dir outputs/final_10000/esm_xgb \
    --n_estimators 500 \
    --max_depth 5 \
    --learning_rate 0.05 \
    --subsample 0.8 \
    --colsample_bytree 0.8

