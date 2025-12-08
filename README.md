# Middle Persian Ezafe Analysis  
Quantitative analysis of Ezafe marking and head–modifier directionality in Middle Persian nominal phrases.

This repository contains all code, notebooks, and preprocessed data required to reproduce the results reported in:

**Musavi, Raha. (2025). Head Directionality and Dependency Marking in Middle Persian Nominal Phrases: Quantitative Evidence from Ezafe Constructions.**

The project uses the Zoroastrian Middle Persian Corpus and Dictionary (MPCD), together with machine-learning methods (Random Forests, Boruta feature selection, and SHAP interpretability), to identify structural and lexical correlates of Ezafe marking and head directionality.

---

## Table of Contents
- Project Overview
- Repository Structure
- Data
- Installation
- Usage
- Notebooks
- Artifacts
- Citation
- License

---

## Project Overview

The project investigates:

- Predictors of Ezafe presence
- Predictors of head-initial vs. head-final ordering
- Variation and stability patterns in the nominal phrase
- Interpretable machine learning for historical syntax

The analysis is fully reproducible given:

1. The preprocessed dataset shipped in this repository
2. The trained model artifacts produced by the notebooks

---

## Repository Structure

mp-nominal-phrases/
│
├── data/
│ └── preprocessed/
│ └── head_modifiers_pairs.csv # shipped with repository
│
├── notebooks/
│ ├── 01_exploratory_analysis.ipynb
│ ├── 02_model_training.ipynb
│ └── 03_shap_analysis.ipynb
│
├── artifacts/
│ ├── best_pipeline_combined.joblib
│ ├── selected_feature_names.joblib
│ └── feature_analysis/
│ └── … generated outputs …
│
├── src/
│ ├── preprocessing.py
│ ├── features.py
│ ├── model.py
│ └── utils.py
│
├── README.md
├── LICENSE
└── requirements.txt

yaml
Code kopieren

---

## Data

### Preprocessed Data (included)

The file:

data/preprocessed/head_modifiers_pairs.csv

yaml
Code kopieren

contains all head–modifier pairs and derived structural features required to reproduce the analysis.

Because the MPCD is under ongoing revision and licensing constraints, raw data are **not included**, but the preprocessed dataset is sufficient for replication.

---

## Installation

```bash
git clone https://github.com/RahaaMusavi/mp-nominal-phrases.git
cd mp-nominal-phrases
pip install -r requirements.txt
Usage
The analysis proceeds in three conceptual phases:

Exploration — descriptive statistics and visualizations

Model Training — training classifiers, evaluating performance, running Boruta

Interpretability — global and per-class SHAP analysis

Everything is orchestrated through Jupyter notebooks.

Notebooks
01_exploratory_analysis.ipynb
Loads the preprocessed dataset

Computes descriptive statistics

Produces visual summaries

Prepares data for modelling

02_model_training.ipynb
Produces:

Random Forest classifier

Boruta feature selection

Final trained pipeline (saved to artifacts/best_pipeline_combined.joblib)

Selected feature names (artifacts/selected_feature_names.joblib)

The pipeline is saved using joblib and can be reloaded by downstream notebooks.

03_shap_analysis.ipynb
Uses:

Trained classifier

Preprocessing pipeline

Boruta mask

Selected feature names

Produces:

Global feature importance (SHAP)

Per-class SHAP summaries

Impurity-based class importance

Table of directional effects

Publication-ready figures

SHAP handling in this notebook is version-robust and correctly handles:

Binary classifiers

Multi-class classifiers

Older and newer SHAP API return formats

All generated tables and figures are written into:

bash
Code kopieren
artifacts/feature_analysis/
Artifacts / Files Expected
The SHAP notebook expects:

Code kopieren
artifacts/
│
├── best_pipeline_combined.joblib
├── selected_feature_names.joblib
└── feature_analysis/   (created automatically)
The Model Training notebook is responsible for creating those files.

Reproducibility Notes
All randomness uses fixed seeds to ensure repeatability.

Class imbalance is addressed using oversampling.

No data leakage occurs across folds.

The analysis is deterministic given the provided data and versions.

Citation
If you use this repository in your work, please cite:

bash
Code kopieren
Musavi, Raha. (2025). Head Directionality and Dependency Marking in Middle Persian:
Ezafe Analysis (Version 1.0) [Computer software]. Zenodo.
https://doi.org/10.5281/zenodo.17722545
License
MIT License — see LICENSE.

Contact / Issues
Questions, bug reports, or requests for additional documentation are welcome via GitHub Issues.

yaml
Code kopieren
