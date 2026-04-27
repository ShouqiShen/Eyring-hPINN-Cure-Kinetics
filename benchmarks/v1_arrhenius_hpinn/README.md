# v1 Arrhenius hPINN Benchmark

This folder contains the frozen v1 benchmark implementation for structure-formulation cure kinetics.

## Model Role

v1 is kept as a baseline reference. It uses a tri-modal molecular encoder with a Kamal-Sourour/Arrhenius-style physics backbone and staged residual learning.

The benchmark is useful for comparing:

- Leave-One-Ratio-Out formulation generalisation.
- Leave-One-Molecule-Out structure generalisation.
- Extracted Arrhenius-style parameters such as `Ea` and `lnA`.
- Pure-physics reconstruction against residual-corrected predictions.

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

## Notes

Generated outputs are written to `results/` and are ignored by git. Do not edit v1 when developing v2 unless you are deliberately updating the benchmark definition.
