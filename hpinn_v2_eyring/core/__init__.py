# -*- coding: utf-8 -*-
"""
core — Structure-Formulation hPINN v2  public API
=================================================
Import everything from here for clean experiment scripts.

V2 highlights
-------------
  • `Unified_hPINN_v2`            Eyring + DiBenedetto + dynamic synergy + MC Dropout
  • `MCDropout`                   persistent dropout (active in eval())
  • `_thermodynamic_penalty`      ΔG‡-bound penalty (replaces v1 `_physics_constraints`)
  • `predict_with_uncertainty`    MC-Dropout posterior over residual heads
  • `extract_mixture_params`      decodes 8-param physics head + captures
                                   the dynamic synergy closure syn(T, α)
"""
from .config    import (T_REF, R_GAS, K_B, H_PL, LN_KB_OVER_H_MIN,
                         DG_MIN_KJ, DG_MAX_KJ,
                         DH_MIN_J, DH_MAX_J, DS_MIN_SI, DS_MAX_SI,
                         TG0_MIN_K, TG0_MAX_K, DTG_MIN_K, DTG_MAX_K,
                         LAM_MIN, LAM_MAX, KD_SCALE,
                         STAGE_EPOCHS, BATCH_SIZE, LR_STAGE, WD_STAGE1,
                         MC_DROPOUT_P, MC_SAMPLES,
                         N_SEEDS, SEEDS, DEVICE)
from .features  import ChemicalProcessor
from .dataset   import (preprocess_global_dataframe, UnifiedKineticsDataset,
                         custom_collate, make_loader)
from .model     import (TriModalEncoder, Unified_hPINN_v2,
                         Unified_hPINN, MCDropout)
from .trainer   import (set_global_seed, curriculum_train, train_stage,
                         _to, _thermodynamic_penalty, _physics_constraints,
                         _residual_l2, _vit_weights)
from .inference import (evaluate, predict_with_uncertainty,
                         extract_mixture_params, make_rate_fn,
                         run_isothermal_report, run_dynamic_simulation)

__all__ = [
    # config
    "T_REF", "R_GAS", "K_B", "H_PL", "LN_KB_OVER_H_MIN",
    "DG_MIN_KJ", "DG_MAX_KJ",
    "DH_MIN_J", "DH_MAX_J", "DS_MIN_SI", "DS_MAX_SI",
    "TG0_MIN_K", "TG0_MAX_K", "DTG_MIN_K", "DTG_MAX_K",
    "LAM_MIN", "LAM_MAX", "KD_SCALE",
    "STAGE_EPOCHS", "BATCH_SIZE", "LR_STAGE", "WD_STAGE1",
    "MC_DROPOUT_P", "MC_SAMPLES",
    "N_SEEDS", "SEEDS", "DEVICE",
    # features
    "ChemicalProcessor",
    # dataset
    "preprocess_global_dataframe", "UnifiedKineticsDataset",
    "custom_collate", "make_loader",
    # model
    "TriModalEncoder", "Unified_hPINN_v2", "Unified_hPINN", "MCDropout",
    # trainer
    "set_global_seed", "curriculum_train", "train_stage",
    "_to", "_thermodynamic_penalty", "_physics_constraints",
    "_residual_l2", "_vit_weights",
    # inference
    "evaluate", "predict_with_uncertainty",
    "extract_mixture_params", "make_rate_fn",
    "run_isothermal_report", "run_dynamic_simulation",
]
