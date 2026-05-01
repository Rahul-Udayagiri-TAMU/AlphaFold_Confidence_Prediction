#!/bin/bash
#SBATCH --job-name=data10000_basic
#SBATCH --output=/scratch/user/%u/ECEN766_final/logs/%x_%j.out
#SBATCH --error=/scratch/user/%u/ECEN766_final/logs/%x_%j.err
#SBATCH --time=10:00:00
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
export HF_HOME=$SCRATCH/ECEN766_final/hf_home

mkdir -p logs
mkdir -p outputs
mkdir -p cache

python code/prepare_protein_level_data_esm2.py \
    --id_file data/uniprot_ids_10000.txt \
    --out_dir outputs/data10000_basic \
    --cache_dir cache/data10000 \
    --max_proteins 10000 \
    --skip_embeddings \
    --refresh_cache
