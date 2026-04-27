# -*- coding: utf-8 -*-
"""
core — Structure-Formulation hPINN public API
=============================================
Import everything from here for clean experiment scripts.
"""
from .config    import (T_REF, R_GAS, STAGE_EPOCHS, BATCH_SIZE, LR_STAGE,
                         WD_STAGE1, N_SEEDS, SEEDS, DEVICE)
from .features  import ChemicalProcessor
from .dataset   import (preprocess_global_dataframe, UnifiedKineticsDataset,
                         custom_collate, make_loader)
from .model     import TriModalEncoder, Unified_hPINN
from .trainer   import (set_global_seed, curriculum_train, train_stage,
                         _to, _physics_constraints, _residual_l2, _vit_weights)
from .inference import (evaluate, extract_mixture_params, make_rate_fn,
                         run_isothermal_report, run_dynamic_simulation)

__all__ = [
    # config
    "T_REF", "R_GAS", "STAGE_EPOCHS", "BATCH_SIZE", "LR_STAGE",
    "WD_STAGE1", "N_SEEDS", "SEEDS", "DEVICE",
    # features
    "ChemicalProcessor",
    # dataset
    "preprocess_global_dataframe", "UnifiedKineticsDataset",
    "custom_collate", "make_loader",
    # model
    "TriModalEncoder", "Unified_hPINN",
    # trainer
    "set_global_seed", "curriculum_train", "train_stage",
    "_to", "_physics_constraints", "_residual_l2", "_vit_weights",
    # inference
    "evaluate", "extract_mixture_params", "make_rate_fn",
    "run_isothermal_report", "run_dynamic_simulation",
]
