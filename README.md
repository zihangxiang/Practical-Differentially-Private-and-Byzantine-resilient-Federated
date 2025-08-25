# PDPBRFL: Practical Differentially Private and Byzantine-Resilient Federated Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Published at SIGMOD 2023**  
- [ACM Digital Library](https://dl.acm.org/doi/10.1145/3589264)  
- [Full version (arXiv)](https://arxiv.org/abs/2304.09762)

## Overview

PDPBRFL is an implementation of practical algorithms for federated learning that are both differentially private and resilient to Byzantine (malicious) participants. This repository provides code to reproduce the experiments and results from our SIGMOD 2023 paper.

## Features

- Differential privacy
- Robustness to various Byzantine attacks
- Support for multiple datasets (MNIST, Fashion-MNIST, USPS, Colorectal)

## Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Edit `datasets/dataset_setup.py` to set your dataset storage path in the `get_dataset_data_path` function.

## Usage

You can run experiments using the provided shell scripts or directly via Python.  
**Example: Run federated learning with local DP and a Gaussian attacker on MNIST:**

```bash
python main.py \
  --dataset mnist \
  --att_key gaussian \
  --epsilon 1 \
  --DP_mode localDP \
  --seed 1 \
  --mal_worker_portion 20 \
  --anti_byz 1 \
  --non_iid 0 \
  --start_att 0.0 \
  --base_lr 0.2
```

Or use the batch script for multiple runs:
```bash
bash run_general.sh
```

## Arguments

- `--dataset`: Dataset to use (`mnist`, `fashion`, `usps`, `colorectal`)
- `--att_key`: Type of attacker (`gaussian`, `lf`, `local`, `nobyz`)
- `--epsilon`: Privacy budget (e.g., `1`)
- `--DP_mode`: `localDP` or `centralDP`
- `--seed`: Random seed
- `--mal_worker_portion`: Percentage of malicious workers (e.g., `20`)
- `--anti_byz`: Use anti-Byzantine aggregation (`1` for True, `0` for False)
- `--non_iid`: Use non-IID data distribution (`1` for True, `0` for False)
- `--start_att`: When the attack starts (e.g., `0.0`)
- `--base_lr`: Base learning rate (e.g., `0.2`)

## Citing

If you use this code, please cite:

```tex
@article{DBLP:journals/pacmmod/Xiang0L023,
  author       = {Zihang Xiang and Tianhao Wang and Wanyu Lin and Di Wang},
  title        = {Practical Differentially Private and Byzantine-resilient Federated Learning},
  journal      = {Proc. {ACM} Manag. Data},
  volume       = {1},
  number       = {2},
  pages        = {119:1--119:26},
  year         = {2023},
  url          = {https://doi.org/10.1145/3589264}
}
```