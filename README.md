# Middle Persian Ezafe Analysis

Quantitative analysis of Ezafe linking and head–modifier order in Middle Persian nominal phrases, using the Zoroastrian Middle Persian Corpus and Dictionary (MPCD). This repository contains scripts, notebooks, and results to reproduce the analyses from our paper:

**Head Directionality and Dependency Marking in Middle Persian Nominal Phrases: Quantitative Evidence from Ezafe Constructions**
*Abstract available in the paper.*

---

## Table of Contents

* [Project Overview](#project-overview)
* [Installation](#installation)
* [Data](#data)
* [Usage](#usage)
* [Results](#results)
* [Citation](#citation)
* [License](#license)

---

## Project Overview

This project investigates the interaction between head directionality and Ezafe marking in Middle Persian nominal phrases. Using 8,019 annotated head–modifier pairs, we implement a reproducible machine learning pipeline (Random Forests, Boruta feature selection, SHAP interpretability) to identify structural and lexical predictors of Ezafe presence and head-final/head-initial order.

Key contributions:

* Classification of Ezafe presence and head directionality with 81.53% mean accuracy.
* Feature analyses highlighting NP depth, modifier complexity, and anchoring dependents.
* Identification of distinct domains of stability and heterogeneity in Middle Persian nominal phrases.
* A fully reproducible pipeline for historical syntax research, addressing common pitfalls like data leakage and class imbalance.

---

## Installation

Clone this repository and install dependencies:

```bash
git clone https://github.com/[your-username]/middle-persian-ezafe-analysis.git
cd middle-persian-ezafe-analysis
pip install -r requirements.txt
```

Dependencies are listed in `requirements.txt`.

---

## Data

Preprocessed data:

We provide the preprocessed head–modifier pairs used for model training and evaluation in data/processed/.

This dataset is fully reproducible for running the analyses in this repository.

Raw corpus:

The original Zoroastrian Middle Persian Corpus and Dictionary (MPCD) can be downloaded from MPCD website (subject to updates and ongoing additions).

Users must obtain the raw data themselves due to licensing and potential updates.

Folder structure:

data/
├── raw/          # Not included in repo; users should download from MPCD
├── processed/    # Preprocessed head-modifier pairs included

Note: The preprocessed data captures all necessary information to reproduce the analyses in the paper, even if the raw corpus evolves over time.
---

## Usage

### Notebooks

* `notebooks/exploration.ipynb`: Exploratory data analysis, visualizations.
* `notebooks/modeling.ipynb`: Model training, evaluation, and feature analysis.

### Scripts

Scripts in `src/` can be run individually or integrated into a pipeline:

* `preprocessing.py` – prepares the data.
* `features.py` – computes features for ML.
* `model.py` – trains and evaluates Random Forest models.
* `utils.py` – helper functions.

Example command:

```bash
python src/model.py --input data/processed/train.csv --output results/
```

---

## Results

Figures, tables, and SHAP analyses from the paper are stored in `results/`.
Key insights:

* Ezafe probability increases with NP depth and modifier complexity.
* Unmarked head-final phrases are found in shallow, low-complexity environments.
* Three domain patterns emerge: Ezafe & head-initial, no-Ezafe & head-final, and no-Ezafe & head-initial (heterogeneous).

---

## Citation

If you use this repository or data pipeline in your work, please cite:

> [Your Paper Full Citation Here]

---

## License

This repository is licensed under the MIT License. See [LICENSE](LICENSE) for details.
