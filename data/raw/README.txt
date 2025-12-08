# Data Folder for Middle Persian Ezafe Project

This folder contains the data used in the analyses presented in the paper *“Head Directionality and Dependency Marking in Middle Persian Nominal Phrases: Quantitative Evidence from Ezafe Constructions”*.

## Folder Structure

```
data/
├── preprocessed/
│   ├── head_modifier_pairs.csv
│   └── metadata.csv (optional)
└── external/
    └── MPCD_link.txt
```

### `preprocessed/`

This subfolder contains the processed datasets derived from the Zoroastrian Middle Persian Corpus and Dictionary (MPCD).

* **`head_modifier_pairs.csv`**: The main dataset of annotated head–modifier pairs used for model training and analysis. Columns include:

  * `head` – the nominal head
  * `modifier` – the associated modifier
  * `ezafe` – binary label indicating the presence of Ezafe
  * `head_position` – head-initial or head-final
  * `np_depth` – structural depth of the noun phrase
  * `modifier_complexity` – complexity score for the modifier
  * Additional columns for context or other features used in the models

* **`metadata.csv`** (optional): Provides supplementary information about the preprocessed data, e.g., extraction date, processing notes, or source references.

> ⚠️ Note: This repository **does not include the raw MPCD data**. Only the processed head–modifier pairs are included, which are sufficient to replicate the analyses in the paper.

### `external/`

This subfolder contains information and references for the original MPCD dataset:

* **`MPCD_link.txt`**: Contains the official link to the corpus ([MPCD](https://mpcorpus.org)) and suggested citation for the source. Users must download the MPCD data themselves if needed.

---

### Usage

All analysis scripts and notebooks reference the `preprocessed/head_modifier_pairs.csv` file. To reproduce the analyses:

1. Ensure you have the required Python packages (see `requirements.txt` in the root folder).
2. Use relative paths to access the preprocessed dataset:

```python
import pandas as pd
df = pd.read_csv("data/preprocessed/head_modifier_pairs.csv")
```

3. For additional context or raw corpus references, see `external/MPCD_link.txt`.

---

### Citation

If you use these preprocessed data in your work, please cite the Zenodo release for this repository:

```
[Repository DOI:    ]
```
