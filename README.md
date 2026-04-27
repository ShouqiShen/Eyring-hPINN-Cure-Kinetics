# Eyring hPINN for Structure-Formulation Cure Kinetics

Physics-informed neural network framework for modelling polymer curing kinetics from molecular structure and formulation composition.

This repository contains the v2 implementation of a hybrid physics-informed neural network (hPINN) for cure kinetics prediction. The model combines learned structure/formulation representations with an Eyring transition-state-theory kinetic backbone, DiBenedetto/WLF diffusion gating, dynamic synergy terms, and MC-dropout uncertainty estimation.

## Key Features

- Structure-aware molecular encoding from SMILES descriptors.
- Formulation-aware prediction across mixture ratios.
- Eyring physics head for activation enthalpy and entropy terms.
- DiBenedetto and WLF-inspired diffusion control for vitrification effects.
- Dynamic synergy correction `syn(T, alpha)` for non-ideal mixture behaviour.
- MC-dropout uncertainty estimates for residual kinetic corrections.
- Leave-One-Ratio-Out and Leave-One-Molecule-Out validation workflows.

## Experiments

The repository includes two main validation studies:

- `experiments/exp1_ratio_loro.py` - Leave-One-Ratio-Out validation for formulation-ratio generalisation.
- `experiments/exp2_structure_lomo.py` - Leave-One-Molecule-Out validation for molecular-structure generalisation.

Both scripts write timestamped outputs to `results/`, including plots, prediction tables, fitted physical parameters, and model checkpoints. The `results/` directory is intentionally ignored by git.

## Repository Structure

```text
.
├── core/
│   ├── config.py          # Physical constants, bounds, and training settings
│   ├── dataset.py         # Data preprocessing and PyTorch loaders
│   ├── features.py        # SMILES and chemical feature processing
│   ├── inference.py       # Evaluation, simulation, and uncertainty tools
│   ├── model.py           # hPINN v2 model architecture
│   └── trainer.py         # Curriculum training and physics penalties
├── data/
│   ├── Unified_Kinetics_Dataset_v1.csv
│   ├── Unified_Kinetics_Dataset_MR_v1.csv
│   └── dummy_generator.py
├── experiments/
│   ├── exp1_ratio_loro.py
│   └── exp2_structure_lomo.py
├── requirements.txt
└── README.md
```

## Installation

Create and activate a Python environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

PyTorch and PyTorch Geometric installation can depend on your CUDA version. If the generic install fails, install `torch` and `torch-geometric` using the platform-specific commands from their official documentation, then rerun the remaining dependency installation.

## Quick Start

Run the ratio generalisation experiment:

```bash
python experiments/exp1_ratio_loro.py
```

Run the molecular-structure generalisation experiment:

```bash
python experiments/exp2_structure_lomo.py
```

The scripts expect their input CSV files under `data/` and create new output folders under `results/`.

## Data Availability

The current project structure supports two data modes:

- Public reproducibility mode: keep the input CSV files in `data/` if the datasets are cleared for public release.
- Code-only release mode: remove private or unpublished experimental datasets from `data/`, keep `dummy_ratio_dataset.csv`, and document how approved users can request access to the real data.

Generated experiment outputs, trained checkpoints, and large result folders should not be committed. They are excluded through `.gitignore`.

## Citation

If this code supports a publication, add the preferred citation here before release:

```bibtex
@article{Shen2026hPINN,
  title   = {Physics-Informed Neural Networks for Predicting Polymer Curing Kinetics},
  author  = {Shen, Shouqi and Meo, Michele},
  journal = {To be added},
  year    = {2026}
}
```

## License

Add a license before wider release. If the intention is an open academic software release, MIT is a common choice and matches the style of the related `Kinetics-Digital-Twin` repository.
