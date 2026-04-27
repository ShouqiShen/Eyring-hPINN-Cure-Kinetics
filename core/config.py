# -*- coding: utf-8 -*-
"""
core/config.py  —  V2
=====================
Central configuration for the Structure-Formulation hPINN v2.

Adds the physical constants needed by the **Eyring Transition State Theory**
formulation and a few new hyperparameters introduced by V2:

  * LN_KB_OVER_H_MIN : pre-computed ln(k_B / h) + ln(60)  [for k in min⁻¹]
  * MC_DROPOUT_P     : persistent dropout probability for Bayesian residual heads
  * MC_SAMPLES       : default # of MC Dropout samples at inference time
  * DG_TARGET_*      : soft bounds on ΔG‡(T_ref) for the thermodynamic penalty
"""
from __future__ import annotations
import math
import os
import warnings

import torch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore")
torch.set_num_threads(4)

# ---------------------------------------------------------------------------
# Physics constants  (Eyring TST)
# ---------------------------------------------------------------------------
T_REF: float = 433.15       # Reference temperature [K]  (160 °C)
R_GAS: float = 8.314        # Universal gas constant     [J mol⁻¹ K⁻¹]
K_B:   float = 1.380649e-23 # Boltzmann constant         [J K⁻¹]
H_PL:  float = 6.62607015e-34  # Planck constant          [J s]

# Eyring prefactor expressed for k in min⁻¹:
#   k[min⁻¹] = 60 · (k_B T / h) · exp(ΔS‡/R) · exp(−ΔH‡/RT)
#   ln k     = ln(k_B/h) + ln(60) + ln T + ΔS‡/R − ΔH‡/(RT)
LN_KB_OVER_H_MIN: float = math.log(K_B / H_PL) + math.log(60.0)  # ≈ 27.85

# ---------------------------------------------------------------------------
# Thermodynamic penalty targets (replace the v1 fixed-lnA anchor)
# ---------------------------------------------------------------------------
# Typical Gibbs free energy of activation at the reference temperature for
# condensed-phase epoxy–amine cure reactions is ~80–120 kJ/mol.  We enforce a
# hinge penalty outside [DG_MIN, DG_MAX]; ΔG‡ *inside* the window is free.
DG_MIN_KJ: float = 70.0
DG_MAX_KJ: float = 130.0

# ---------------------------------------------------------------------------
# Physical parameter bounds (hard, via sigmoid re-scaling in model.py)
# ---------------------------------------------------------------------------
DH_MIN_J:  float =   40_000.0   # ΔH‡ lower bound  [J/mol]
DH_MAX_J:  float =  160_000.0   # ΔH‡ upper bound  [J/mol]
DS_MIN_SI: float = -250.0       # ΔS‡ lower bound  [J/(mol·K)]
DS_MAX_SI: float =   50.0       # ΔS‡ upper bound  [J/(mol·K)]
TG0_MIN_K: float =  240.0       # Tg of fully uncured resin (lower bound) — tightened from 200 K
TG0_MAX_K: float =  320.0       # Tg of fully uncured resin (upper bound) — tightened from 350 K
DTG_MIN_K: float =   50.0       # (Tg∞ − Tg0) lower bound  [K]  — tightened from 30 K
DTG_MAX_K: float =  200.0       # (Tg∞ − Tg0) upper bound  [K]  — tightened from 250 K
LAM_MIN:   float =   0.1        # DiBenedetto λ lower bound — raised from 0.05
LAM_MAX:   float =   1.0        # DiBenedetto λ upper bound — raised from 0.95
KD_SCALE:  float =   0.1        # softplus scaling for WLF slope [K⁻¹]

# DiBenedetto soft-prior targets (used in trainer._thermodynamic_penalty)
TG0_PRIOR_K: float = 260.0      # soft prior mean for Tg0  [K]
LAM_PRIOR:   float =   0.4      # soft prior mean for λ (DiBenedetto)

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
STAGE_EPOCHS: dict[int, int]   = {1: 60, 2: 30, 3: 20}
BATCH_SIZE:   int              = 64
LR_STAGE:     dict[int, float] = {1: 1e-3, 2: 5e-4, 3: 1e-4}
WD_STAGE1:    float            = 1e-3

# ---------------------------------------------------------------------------
# Bayesian (MC Dropout) — persistent dropout in residual heads
# ---------------------------------------------------------------------------
MC_DROPOUT_P: float = 0.15      # probability applied every forward pass
MC_SAMPLES:   int   = 30        # # forward passes for uncertainty at inference

# ---------------------------------------------------------------------------
# Ensemble (random-seed independence across restarts)
# ---------------------------------------------------------------------------
N_SEEDS: int       = 5
SEEDS:   list[int] = [0, 7, 13, 42, 99]

# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
