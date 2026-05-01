#!/bin/bash
#SBATCH --job-name=fixed10000_models
#SBATCH --output=/scratch/user/%u/ECEN766_final/logs/%x_%j.out
#SBATCH --error=/scratch/user/%u/ECEN766_final/logs/%x_%j.err
#SBATCH --time=12:00:00
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
    --model ridge \
    --out_dir outputs/final_10000/esm_ridge \
    --alpha 100

python code/train_fixed.py \
    --feature_dir outputs/esm_small_10000 \
    --feature_prefix esm \
    --model svr \
    --out_dir outputs/final_10000/esm_svr \
    --C 10 \
    --epsilon 0.01 \
    --gamma scale

python code/train_fixed.py \
    --feature_dir outputs/esm_small_10000 \
    --feature_prefix basic \
    --model rf \
    --out_dir outputs/final_10000/basic_rf \
    --n_estimators 500 \
    --max_depth None \
    --min_samples_split 2 \
    --min_samples_leaf 1
