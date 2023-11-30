# Data Engineering 

## Overview

This project focuses on text encoding using the Sentence Transformer model and applying three dimension reduction methods: PCA, UMAP, and t-SNE. The goal is to evaluate the methods using Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI) metrics. Additionally, a final step involves further reducing dimensions to 2D for cluster visualization.
### Bonus
We added a fonctionnality to choose a model, and also visualisation of the results in repo experiments.

## Setup

### Requirements

- Python 3.9
- Pip
- Additional requirements specified in `requirements.txt`

# Usage (locally)

## Clone the repo
```bash
git clone https://github.com/EliasKesr/Groupe34_Projet.git
cd Groupe34_Projet
```
## Install Dependencies
```bash
pip install -r requirements.txt
```
## Run the main.py File
```bash
python main.py
```
This script performs text encoding, applies dimension reduction methods, evaluates metrics, and visualizes clusters.

# Usage (Docker)
## Docker pull:
```bash
docker pull mondhirch2/my-python-app:latest
```

## Docker Execution:
```bash
docker run -it mondhirch2/my-python-app:latest
```

