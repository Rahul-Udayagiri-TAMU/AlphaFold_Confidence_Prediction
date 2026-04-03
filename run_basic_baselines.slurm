#!/bin/bash
#SBATCH --job-name=ecen766_basic_baselines
#SBATCH --output=/scratch/user/%u/ECEN766_final/logs/basic_baselines_%j.out
#SBATCH --error=/scratch/user/%u/ECEN766_final/logs/basic_baselines_%j.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

set -euo pipefail

cd $SCRATCH/ECEN766_final

source venv_ecen766/bin/activate
unset PYTHONPATH
unset PYTHONHOME
export PYTHONNOUSERSITE=1

mkdir -p logs
mkdir -p outputs/baselines

echo "Running basic feature baselines..."

python code/train_baselines.py \
    --feature_dir outputs/esm_small_2000 \
    --feature_prefix basic \
    --model svr \
    --out_dir outputs/baselines/basic_svr \
    > logs/basic_svr.log 2>&1

python code/train_baselines.py \
    --feature_dir outputs/esm_small_2000 \
    --feature_prefix basic \
    --model rf \
    --out_dir outputs/baselines/basic_rf \
    > logs/basic_rf.log 2>&1

python code/train_baselines.py \
    --feature_dir outputs/esm_small_2000 \
    --feature_prefix basic \
    --model xgb \
    --out_dir outputs/baselines/basic_xgb \
    > logs/basic_xgb.log 2>&1

python code/train_baselines.py \
    --feature_dir outputs/esm_small_2000 \
    --feature_prefix basic \
    --model mlp \
    --out_dir outputs/baselines/basic_mlp \
    > logs/basic_mlp.log 2>&1

echo "All basic baselines completed."
