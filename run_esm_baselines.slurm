#!/bin/bash
#SBATCH --job-name=ecen766_esm_baselines
#SBATCH --output=/scratch/user/%u/ECEN766_final/logs/esm_baselines_%j.out
#SBATCH --error=/scratch/user/%u/ECEN766_final/logs/esm_baselines_%j.err
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G

set -euxo pipefail

module purge
module load GCC/12.3.0 OpenMPI/4.1.5 PyTorch/2.1.2-CUDA-12.1.1

cd $SCRATCH/ECEN766_final

source venv_ecen766/bin/activate
unset PYTHONPATH
unset PYTHONHOME
export PYTHONNOUSERSITE=1

mkdir -p logs
mkdir -p outputs/baselines

echo "PWD=$(pwd)"
echo "Python=$(which python)"
python --version

python code/train_baselines.py --feature_dir outputs/esm_small_2000 --feature_prefix esm --model ridge --out_dir outputs/baselines/esm_ridge > logs/esm_ridge.log 2>&1
python code/train_baselines.py --feature_dir outputs/esm_small_2000 --feature_prefix esm --model svr   --out_dir outputs/baselines/esm_svr   > logs/esm_svr.log 2>&1
python code/train_baselines.py --feature_dir outputs/esm_small_2000 --feature_prefix esm --model rf    --out_dir outputs/baselines/esm_rf    > logs/esm_rf.log 2>&1
python code/train_baselines.py --feature_dir outputs/esm_small_2000 --feature_prefix esm --model xgb   --out_dir outputs/baselines/esm_xgb   > logs/esm_xgb.log 2>&1
python code/train_baselines.py --feature_dir outputs/esm_small_2000 --feature_prefix esm --model mlp   --out_dir outputs/baselines/esm_mlp   > logs/esm_mlp.log 2>&1

echo "All ESM-small baselines completed."
