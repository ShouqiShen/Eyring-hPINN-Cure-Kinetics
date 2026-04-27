# Structure-Formulation hPINN v2

Physics-informed neural network code for cure kinetics modelling with structure and formulation inputs.

## Overview

This repository contains the v2 hPINN implementation used for:

- Leave-One-Ratio-Out validation with `experiments/exp1_ratio_loro.py`
- Leave-One-Molecule-Out validation with `experiments/exp2_structure_lomo.py`

The v2 model uses an Eyring transition-state-theory backbone, DiBenedetto/WLF diffusion gating, dynamic synergy terms, and MC dropout uncertainty estimates.

## Project Structure

- `core/` - model, dataset, feature processing, training, and inference utilities
- `experiments/` - experiment scripts for ratio and structure generalisation studies
- `data/` - input datasets and dummy data generator
- `results/` - generated experiment outputs, ignored by git

## Setup

Create and activate a Python environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

Depending on your CUDA/PyTorch setup, you may need to install `torch` and `torch-geometric` using the platform-specific commands from their official installation guides.

## Running Experiments

From the repository root:

```bash
python experiments/exp1_ratio_loro.py
python experiments/exp2_structure_lomo.py
```

Outputs are written to timestamped folders under `results/`.
