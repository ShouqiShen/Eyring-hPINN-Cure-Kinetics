# Data Policy and Schema

The public repository keeps real experimental datasets out of git by default.

## Why Real Data Is Excluded

The real CSV files include molecular structures, formulation ratios, processing conditions, conversion, and cure-rate values. These data should only be made public after confirming publication, collaborator, and project-sharing permissions.

## Expected Columns

The experiment scripts expect CSV files with columns equivalent to:

- `Sample_ID` or `Molecule`
- `SMILES_E1`
- `SMILES_X1`
- `SMILES_Y1`
- `Ratio_X1`
- `Ratio_Y1`
- `Condition_Type`
- `Condition_Val`
- `Temp_K`
- `Alpha`
- `Rate_1_min`

The v1 and v2 scripts look for:

- `Unified_Kinetics_Dataset_MR_v1.csv` for ratio validation.
- `Unified_Kinetics_Dataset_v1.csv` for structure validation.

Place approved real datasets inside the relevant version folder's `data/` directory.

## Public Dummy Data

Use `data/dummy/` and the version-local `data/dummy_generator.py` files for public demonstrations and testing without exposing real experimental data.
