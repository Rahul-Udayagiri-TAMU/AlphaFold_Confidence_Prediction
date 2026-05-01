# Predicting AlphaFold Confidence from Protein Sequences

This repository contains the code, scripts, and experiment setup for our ECEN-766 course project on predicting **AlphaFold confidence** directly from **protein sequence**.

## Project Overview

In this project, we study whether AlphaFold confidence can be estimated from sequence-derived representations using lightweight machine learning methods. The main goal is to predict **protein-level mean confidence** from sequence alone, so confidence can be screened quickly without running full AlphaFold inference.

### Primary task
- **Protein-level mean pLDDT prediction**

### Exploratory extension
- **Coarse window-level confidence prediction**

## Data

The dataset is constructed using publicly available protein and AlphaFold metadata.

### Data source workflow

1. Collect a list of UniProt protein accessions
2. Query AlphaFold DB metadata for each protein
3. Extract the protein-level confidence label using `globalMetricValue`
4. Build train / validation / test splits
5. Generate feature representations and embeddings

## Methods

We compare multiple regression baselines for predicting protein-level confidence.

### Feature representations

- **Basic features**
  - sequence length
  - amino acid composition
  - simple physicochemical fractions

- **ESM embeddings**
  - pretrained protein language model representations

- **Combined features**
  - basic features + ESM embeddings

### Models used

The project includes baseline and tuned experiments with the following model families:

- Ridge Regression
- Support Vector Regression (SVR)
- Random Forest
- XGBoost
- MLP

## Evaluation Metrics

We evaluate the models using:

- Pearson correlation
- Spearman correlation
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R²

## Repository Structure

```text
.
├── code/
│   ├── fetch_uniprot_ids.py
│   ├── prepare_protein_level_data_esm2.py
│   ├── train_baselines.py
│   ├── train_tuned.py
│   ├── train_fixed.py
│   ├── train_fixed_flexible.py
│   ├── merge_features.py
│   ├── aggregate_results.py
│   ├── analyze_error_bins.py
│   ├── make_analysis_plots.py
│   ├── build_window_labels.py
│   └── prepare_window_basic_dataset.py
├── data/
├── cache/
├── outputs/
├── logs/
├── slurm_files
├── activate_ecen766.sh
├── slurm_template.sh
└── README.md
