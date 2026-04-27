# -*- coding: utf-8 -*-
"""
data/dummy_generator.py
=======================
Synthetic cure-kinetics datasets for offline testing (no real CSV required).

Two generators are provided, one per experiment type:

make_ratio_dataset()
--------------------
  Exp-1 target: Leave-One-Ratio-Out (LORO)
  Fixed molecules: DGEBA epoxy + adipic acid (X) + isophthalic acid (Y)
  Variable: Ratio_X1 ∈ {0.0, 0.2, 0.5, 0.7, 1.0}
  Conditions: Dynamic β ∈ {5, 10, 20} K/min  +  Isothermal T ∈ {393, 413, 433} K

make_structure_dataset()
------------------------
  Exp-2 target: Leave-One-Molecule-Out (LOMO) — homologous diacid series
  Fixed ratio: rX = 0.70, rY = 0.30
  Variable: diacid chain length C6 → C14 (Ea increases ~2 kJ/mol per CH2)
  Conditions: Dynamic β ∈ {5, 10, 20} K/min  +  Isothermal T ∈ {393, 413, 433} K

Both functions return DataFrames with the required columns:
    Sample_ID, SMILES_E1, SMILES_X1, SMILES_Y1,
    Ratio_X1, Ratio_Y1, Condition_Type, Condition_Val,
    Temp_K, Alpha, Rate_1_min
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

R_GAS: float = 8.314

# SMILES constants
_SMILES_DGEBA = "CC(C)(c1ccc(OCC2CO2)cc1)c1ccc(OCC2CO2)cc1"
_SMILES_IPA   = "OC(=O)c1cccc(C(=O)O)c1"          # isophthalic acid (Y)

_DIACID_SMILES = {
    "C6":  "OC(=O)CCCC(=O)O",
    "C8":  "OC(=O)CCCCCC(=O)O",
    "C10": "OC(=O)CCCCCCCC(=O)O",
    "C12": "OC(=O)CCCCCCCCCC(=O)O",
    "C14": "OC(=O)CCCCCCCCCCCC(=O)O",
}

_BETAS    = [5.0, 10.0, 20.0]       # dynamic heating rates [K/min]
_T_ISOS   = [393.0, 413.0, 433.0]   # isothermal temperatures [K]
_M, _N    = 0.20, 1.20              # reaction orders
_N_PTS    = 60                       # points per simulated curve
_NOISE    = 0.03                     # fractional noise σ


# ---------------------------------------------------------------------------
def _integrate_dynamic(
    rate_fn,
    beta:   float,
    T0:     float = 323.0,
    T1:     float = 523.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t_end = (T1 - T0) / beta
    sol   = solve_ivp(
        fun=lambda t, y: [rate_fn(y[0], T0 + beta * t)],
        t_span=(0, t_end), y0=[1e-3],
        t_eval=np.linspace(0, t_end, _N_PTS),
        method="RK45", max_step=0.2,
    )
    T_arr     = T0 + beta * sol.t
    alpha_arr = np.clip(sol.y[0], 0, 0.9999)
    rate_arr  = np.array([rate_fn(a, T) for a, T in zip(alpha_arr, T_arr)])
    return T_arr, alpha_arr, rate_arr


def _integrate_isothermal(
    rate_fn,
    T_iso: float,
    t_max: float = 120.0,
) -> tuple[np.ndarray, np.ndarray]:
    sol = solve_ivp(
        fun=lambda t, y: [rate_fn(y[0], T_iso)],
        t_span=(0, t_max), y0=[1e-3],
        t_eval=np.linspace(0, t_max, _N_PTS),
        method="RK45", max_step=0.5,
    )
    return np.clip(sol.y[0], 0, 0.9999), np.array(
        [rate_fn(a, T_iso) for a in sol.y[0]])


def _add_noise(arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return np.clip(arr * (1.0 + rng.normal(0, _NOISE, size=len(arr))), 1e-10, None)


# ---------------------------------------------------------------------------
def make_ratio_dataset(seed: int = 0) -> pd.DataFrame:
    """
    Synthetic dataset for Experiment 1 (LORO by mixture ratio).

    One epoxy (DGEBA) cured with X-hardener (adipic acid, C6 aliphatic)
    and Y-hardener (IPA, aromatic) at five compositions.

    Kinetic parameters are interpolated between pure-X and pure-Y values
    so the mixture effect is non-trivial.

    Required CSV columns are all present in the returned DataFrame.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    # Pure-component kinetics
    params = {
        "X": {"Ea": 62_000, "lnA": 17.2, "smiles": "OC(=O)CCCC(=O)O"},
        "Y": {"Ea": 70_000, "lnA": 18.8, "smiles": _SMILES_IPA},
    }
    ratios = [0.0, 0.2, 0.5, 0.7, 1.0]    # Ratio_X1 values

    for rX in ratios:
        rY      = round(1.0 - rX, 6)
        sid     = f"R{int(rX*100):03d}X{int(rY*100):03d}Y"
        Ea_mix  = rX * params["X"]["Ea"] + rY * params["Y"]["Ea"]
        lnA_mix = rX * params["X"]["lnA"] + rY * params["Y"]["lnA"]
        sX      = params["X"]["smiles"]
        sY      = params["Y"]["smiles"]

        def rate_fn(alpha: float, T: float,
                    Ea=Ea_mix, lnA=lnA_mix) -> float:
            k = np.exp(lnA - Ea / (R_GAS * T))
            return float(max(k * (alpha + 1e-8) ** _M
                               * (1.0 - alpha + 1e-8) ** _N, 1e-10))

        for beta in _BETAS:
            T_arr, alpha_arr, rate_arr = _integrate_dynamic(rate_fn, beta)
            rate_arr = _add_noise(rate_arr, rng)
            for T_v, a_v, r_v in zip(T_arr, alpha_arr, rate_arr):
                rows.append(dict(
                    Sample_ID=sid, SMILES_E1=_SMILES_DGEBA,
                    SMILES_X1=sX, SMILES_Y1=sY,
                    Ratio_X1=rX, Ratio_Y1=rY,
                    Condition_Type="Dyn", Condition_Val=beta,
                    Temp_K=float(T_v), Alpha=float(a_v), Rate_1_min=float(r_v),
                ))

        for T_iso in _T_ISOS:
            alpha_arr, rate_arr = _integrate_isothermal(rate_fn, T_iso)
            rate_arr = _add_noise(rate_arr, rng)
            for a_v, r_v in zip(alpha_arr, rate_arr):
                rows.append(dict(
                    Sample_ID=sid, SMILES_E1=_SMILES_DGEBA,
                    SMILES_X1=sX, SMILES_Y1=sY,
                    Ratio_X1=rX, Ratio_Y1=rY,
                    Condition_Type="Iso",
                    Condition_Val=float(T_iso - 273.15),
                    Temp_K=float(T_iso),
                    Alpha=float(a_v), Rate_1_min=float(r_v),
                ))

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def make_structure_dataset(seed: int = 0) -> pd.DataFrame:
    """
    Synthetic dataset for Experiment 2 (LOMO by molecular structure).

    Five homologous aliphatic diacids (C6 → C14) cured with DGEBA epoxy
    in a 70:30 mixture with isophthalic acid (IPA) as co-hardener.
    Ea increases ~2 kJ/mol per CH₂ unit following the KCE.
    """
    rng    = np.random.default_rng(seed)
    rows: list[dict] = []

    mol_params = {
        "C6":  {"Ea": 62_000, "lnA": 17.2},
        "C8":  {"Ea": 64_000, "lnA": 17.5},
        "C10": {"Ea": 66_000, "lnA": 17.8},
        "C12": {"Ea": 68_000, "lnA": 18.1},
        "C14": {"Ea": 70_000, "lnA": 18.4},
    }
    rX, rY = 0.70, 0.30

    for mol, p in mol_params.items():
        Ea, lnA  = p["Ea"], p["lnA"]
        sX       = _DIACID_SMILES[mol]

        def rate_fn(alpha: float, T: float,
                    Ea=Ea, lnA=lnA) -> float:
            k = np.exp(lnA - Ea / (R_GAS * T))
            return float(max(k * (alpha + 1e-8) ** _M
                               * (1.0 - alpha + 1e-8) ** _N, 1e-10))

        for beta in _BETAS:
            T_arr, alpha_arr, rate_arr = _integrate_dynamic(rate_fn, beta)
            rate_arr = _add_noise(rate_arr, rng)
            for T_v, a_v, r_v in zip(T_arr, alpha_arr, rate_arr):
                rows.append(dict(
                    Sample_ID=mol, SMILES_E1=_SMILES_DGEBA,
                    SMILES_X1=sX, SMILES_Y1=_SMILES_IPA,
                    Ratio_X1=rX, Ratio_Y1=rY,
                    Condition_Type="Dyn", Condition_Val=beta,
                    Temp_K=float(T_v), Alpha=float(a_v), Rate_1_min=float(r_v),
                ))

        for T_iso in _T_ISOS:
            alpha_arr, rate_arr = _integrate_isothermal(rate_fn, T_iso)
            rate_arr = _add_noise(rate_arr, rng)
            for a_v, r_v in zip(alpha_arr, rate_arr):
                rows.append(dict(
                    Sample_ID=mol, SMILES_E1=_SMILES_DGEBA,
                    SMILES_X1=sX, SMILES_Y1=_SMILES_IPA,
                    Ratio_X1=rX, Ratio_Y1=rY,
                    Condition_Type="Iso",
                    Condition_Val=float(T_iso - 273.15),
                    Temp_K=float(T_iso),
                    Alpha=float(a_v), Rate_1_min=float(r_v),
                ))

    return pd.DataFrame(rows)
