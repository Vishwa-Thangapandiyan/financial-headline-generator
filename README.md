![banner](assets/banner.png)

# financial-headline-generator

Character-level language model that generates financial headlines.

## About

A from-scratch implementation of progressively deeper neural language models, trained on 53,000 financial headlines sourced from CNBC, The Guardian, and Reuters. Built following Andrej Karpathy's *makemore* series, with each notebook adding a new architectural layer — from raw bigram counts to a WaveNet-style hierarchical model.

## Architecture Progression

| Notebook | Model | Key Concept |
|---|---|---|
| `bigrams.ipynb` | Bigram | Count-based & single-layer neural |
| `mlp.ipynb` | MLP | Learned embeddings, hidden layer |
| `batchnorm.ipynb` | MLP + BatchNorm | Batch normalization, Kaiming init |
| `backprop.ipynb` | Manual Backprop | Hand-rolled gradients, no autograd |
| `wavenet.ipynb` | WaveNet | `FlattenConsecutive` hierarchical architecture |

## Dataset

- **Sources:** CNBC, The Guardian, Reuters
- **Size:** ~53,000 headlines
- **Format:** Raw text, one headline per line

## Results

| Notebook | Final Val Loss |
|---|---|
| `bigrams.ipynb` | ~2.47 |
| `mlp.ipynb` | ~2.10 |
| `batchnorm.ipynb` | ~2.08 |
| `backprop.ipynb` | ~2.08 |
| `wavenet.ipynb` | ~1.99 |

## How to Run

```bash
git clone https://github.com/Vishwa-Thangapandiyan/financial-headline-generator
cd financial-headline-generator
```

Open and run notebooks in order: `bigrams` → `mlp` → `batchnorm` → `backprop` → `wavenet`. Each notebook is self-contained.

## Tech Stack

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-76B900?style=flat&logo=nvidia&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?style=flat&logo=pandas&logoColor=white)
