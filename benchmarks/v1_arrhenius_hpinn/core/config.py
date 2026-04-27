# -*- coding: utf-8 -*-
"""
core/config.py
==============
Central configuration — all physics constants and training hyperparameters.
Import this module everywhere instead of defining magic numbers inline.
"""
from __future__ import annotations
import os
import warnings

import torch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore")
torch.set_num_threads(4)

# ---------------------------------------------------------------------------
# Physics constants
# ---------------------------------------------------------------------------
T_REF: float = 433.15   # Reference temperature [K]  (160 °C)
R_GAS: float = 8.314    # Universal gas constant [J mol⁻¹ K⁻¹]

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
STAGE_EPOCHS: dict[int, int]   = {1: 60, 2: 30, 3: 20}
BATCH_SIZE:   int               = 64
LR_STAGE:     dict[int, float]  = {1: 1e-3, 2: 5e-4, 3: 1e-4}
WD_STAGE1:    float             = 1e-3   # L2 weight-decay on Stage-1 Adam

# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------
N_SEEDS: int       = 5
SEEDS:   list[int] = [0, 7, 13, 42, 99]

# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
