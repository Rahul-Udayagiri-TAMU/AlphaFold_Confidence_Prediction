# Predicting AlphaFold Confidence from Protein Sequences

This repository contains the code, scripts, and experiment setup for our ECEN-766 course project on predicting **AlphaFold confidence** directly from **protein sequence**.

## Project Overview

In this project, we study protein-level AlphaFold confidence prediction using lightweight machine learning methods. The main goal is to predict **protein-level mean confidence** from sequence-derived representations, so that confidence can be screened quickly without requiring full AlphaFold inference.

The primary task is:

- **Protein-level mean pLDDT prediction**

We also consider a secondary exploratory extension:

- **Coarse window-level confidence prediction**

## Data

The dataset is constructed using publicly available protein and AlphaFold metadata.

### Data source workflow

1. Collect a list of UniProt protein accessions
2. Query AlphaFold DB metadata for each protein
3. Extract the protein-level confidence label using `globalMetricValue`
4. Build train / validation / test splits
5. Generate embeddings

## Methods

We compare multiple regression baselines for predicting protein-level confidence.

### Feature representations

- Basic features
- ESM embeddings

### Models used

The project includes baseline and tuned experiments with the following model families:

- Ridge Regression
- Support Vector Regression (SVR)
- Random Forest
- XGBoost
- MLP

### Evaluation metrics

We evaluate the models using:

- Pearson correlation
- Spearman correlation
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- \(R^2\)

## Repository Structure

```text
.
├── code/
│   ├── fetch_uniprot_ids.py
│   ├── prepare_protein_level_data_esm2.py
│   ├── train_baselines.py
│   └── train_tuned.py
├── data/
├── outputs/
├── logs/
├── run_basic_baselines.slurm
├── run_esm_baselines.slurm
├── slurm_template.sh
└── README.md
