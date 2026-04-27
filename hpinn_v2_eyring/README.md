# v2 Eyring hPINN

This folder contains the v2 Eyring hPINN implementation for structure-formulation cure kinetics.

## Model Role

v2 is the main research implementation. It extends the v1 benchmark with:

- Eyring transition-state-theory parameterisation.
- Activation enthalpy and entropy heads.
- DiBenedetto/WLF-style diffusion gating.
- Dynamic synergy correction `syn(T, alpha)`.
- MC-dropout uncertainty estimates for residual kinetic corrections.

## Running

From this folder:

```bash
python experiments/exp1_ratio_loro.py
python experiments/exp2_structure_lomo.py
```

Place approved real datasets in `data/` using the expected filenames:

- `Unified_Kinetics_Dataset_MR_v1.csv`
- `Unified_Kinetics_Dataset_v1.csv`

If those files are absent, the scripts generate synthetic dummy datasets.

## Constraint Awareness

v2 is more automated than v1, but its strongest performance depends on physically meaningful training and inference conditions. See `../docs/constraints.md` for the recommended validity checks before interpreting predictions.
