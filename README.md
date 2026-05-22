# Eyring-hPINN-Cure-Kinetics
## Why this exists

This repository is a public reference implementation accompanying the v2 hPINN
manuscript on polymer cure kinetics. It is deliberately structured so that the
**methodology** — physics backbone, validation protocol, and uncertainty
estimation — can be reproduced and benchmarked even when the underlying
experimental dataset is under industrial IP review and cannot yet be released.

The repository pairs two model versions with a strict benchmarking discipline:

- **v1 (frozen):** an Arrhenius / Kamal–Sourour hPINN that serves as the
  field-standard empirical baseline.
- **v2 (active):** an Eyring transition-state-theory hPINN with diffusion
  gating, dynamic synergy, and MC-Dropout uncertainty estimation.

Both versions are exercised under the same Leave-One-Ratio-Out (LORO) and
Leave-One-Molecule-Out (LOMO) splits, so any claimed improvement of v2 over
v1 must come from physics, not from data leakage or test-set drift.

Dummy data generators are bundled so the full pipeline — training, LORO/LOMO
splitting, and MC-Dropout inference — runs end-to-end without the private
dataset. This makes the repository useful as a methodology template for other
groups working under similar industrial-data constraints.

Hybrid physics-informed benchmark for polymer cure kinetics, comparing a frozen v1 Arrhenius hPINN with a v2 Eyring hPINN for structure-formulation generalisation.

This repository benchmarks two related hPINN implementations for polymer curing kinetics. The v1 model provides a stable Arrhenius/Kamal-Sourour reference, while the v2 model introduces an automated Eyring transition-state-theory backbone with diffusion gating, dynamic synergy, and MC-dropout uncertainty estimation.

Suggested GitHub description:

> Benchmarking Arrhenius and Eyring physics-informed neural networks for polymer cure kinetics with structure-formulation inputs, LORO/LOMO validation, and constraint-aware uncertainty estimation.

## Model Versions

- `benchmarks/v1_arrhenius_hpinn/` - the frozen v1 benchmark using an Arrhenius/Kamal-Sourour-style physics backbone.
- `hpinn_v2_eyring/` - the v2 Eyring hPINN with diffusion gating, dynamic synergy, and MC-dropout uncertainty estimation.

The repository is organised to make the scientific comparison explicit: v1 is the baseline reference, while v2 is the newer automated model whose performance depends on physically meaningful constraints and in-domain operating conditions.

## Key Features

- Side-by-side v1 and v2 model snapshots for reproducible benchmarking.
- Leave-One-Ratio-Out validation for formulation-ratio generalisation.
- Leave-One-Molecule-Out validation for molecular-structure generalisation.
- Tri-modal molecular representation using SMILES, molecular graphs, and ECFP fingerprints.
- Constraint-aware v2 physics head based on Eyring transition-state theory.
- Documentation for data policy, model constraints, and version differences.

## Repository Structure

```text
.
├── benchmarks/
│   └── v1_arrhenius_hpinn/     # Frozen v1 benchmark
├── hpinn_v2_eyring/            # Main v2 Eyring hPINN implementation
├── data/
│   └── dummy/                  # Public dummy/sample data only
├── docs/
│   ├── constraints.md
│   ├── data.md
│   ├── v1_benchmark.md
│   └── v2_eyring_model.md
├── requirements.txt
└── README.md
```

Generated experiment outputs, checkpoints, and private experimental datasets are intentionally excluded from git.

## Installation

Create and activate a Python environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

PyTorch and PyTorch Geometric installation depends on your CUDA version. If the generic install fails, install `torch` and `torch-geometric` using the platform-specific commands from their official documentation, then rerun the remaining dependency installation.

## Quick Start

Run the v1 Arrhenius benchmark:

```bash
cd benchmarks/v1_arrhenius_hpinn
python experiments/exp1_ratio_loro.py
python experiments/exp2_structure_lomo.py
```

Run the v2 Eyring model:

```bash
cd hpinn_v2_eyring
python experiments/exp1_ratio_loro.py
python experiments/exp2_structure_lomo.py
```

If the real CSV files are not present, the scripts generate synthetic dummy datasets using their local `data/dummy_generator.py` modules.

## Data Policy

The real cure-kinetics datasets are not included in the current public layout. They contain molecular structures, formulation ratios, thermal conditions, conversion, and rate values, so they should only be released if they are cleared for publication and sharing.

For public demonstration and CI-style testing, use the dummy data under `data/dummy/` or the version-local dummy generators. See `docs/data.md` for the expected schema.

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

This repository is released under the MIT License. See `LICENSE`.
