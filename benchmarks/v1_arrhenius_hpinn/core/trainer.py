# -*- coding: utf-8 -*-
"""
core/trainer.py
===============
All training logic: loss functions, KCE-killer penalties, curriculum trainer.

Curriculum stages
-----------------
Stage 1 (60 ep, WD=1e-3) : physics backbone only — Kamal–Sourour + KCE anchors
Stage 2 (30 ep)           : + Flory-Huggins synergy head
Stage 3 (20 ep)           : + three residual correction heads

KCE-Killer penalties (applied every step)
------------------------------------------
  • _vit_weights     : vitrification down-weighting near α > 0.92
  • _physics_constraints :
      - Ea bounded to [40, 120] kJ/mol
      - lnA anchor: lnK_ref ≈ −1.0 at T_REF
      - m ~ 0.2, n ~ 1.2 reaction-order priors
  • _residual_l2 : strong L2 stiffness on delta, iso_head, dyn_corr outputs
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim

from .config import DEVICE, STAGE_EPOCHS, LR_STAGE, WD_STAGE1


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def set_global_seed(seed: int) -> None:
    """Fix torch, numpy, and CUDA seeds for a fully reproducible run."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def _to(batch: dict, device: torch.device) -> dict:
    """Recursively move all tensors in a nested dict to *device*."""
    result: dict = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            result[k] = v.to(device)
        elif isinstance(v, dict):
            result[k] = _to(v, device)
        else:
            result[k] = v
    return result


# ---------------------------------------------------------------------------
# Loss components
# ---------------------------------------------------------------------------
def _vit_weights(alpha: torch.Tensor) -> torch.Tensor:
    """
    Vitrification down-weighting: reduces loss weight for nearly-cured samples
    (α > 0.92) where diffusion-control invalidates the Kamal–Sourour kinetics.
    """
    return 1.0 / (1.0 + torch.exp(25.0 * (alpha - 0.92))) + 0.15


def _physics_constraints(pX: torch.Tensor) -> torch.Tensor:
    """
    KCE-Killer physics-anchor penalties applied to the X-component parameters:

      Ea range  : soft penalty outside [40, 120] kJ/mol
      lnA anchor: lnK_ref ≈ −1.0  (i.e. ln A ≈ lnK_ref + Ea/(R·T_REF))
      m/n priors: m ~ 0.2, n ~ 1.2
    """
    ea_kj   = F.softplus(pX[:, 1]) * 50.0
    ea_reg  = torch.mean(
        torch.clamp(ea_kj - 120.0, min=0.0) +
        torch.clamp(40.0 - ea_kj,  min=0.0))
    lna_reg = torch.mean(((pX[:, 0] * 2.0 - 2.0) - (-2.5)) ** 2)
    mn_reg  = torch.mean(
        (torch.abs(pX[:, 2]) - 0.2) ** 2 +
        (torch.abs(pX[:, 3]) - 1.2) ** 2)
    return 0.05 * ea_reg + 0.02 * lna_reg + 0.05 * mn_reg


def _residual_l2(
    delta:   torch.Tensor,
    iso_off: torch.Tensor,
    dyn_off: torch.Tensor,
) -> torch.Tensor:
    """Strong L2 stiffness on all three residual heads to prevent hallucination."""
    return (
        1e-3 * torch.mean(delta   ** 2) +
        3e-3 * torch.mean(iso_off ** 2) +
        1e-3 * torch.mean(dyn_off ** 2)
    )


# ---------------------------------------------------------------------------
# Stage trainer
# ---------------------------------------------------------------------------
def train_stage(
    model,
    loader,
    optimizer,
    stage:   int,
    epochs:  int,
    label:   str,
    verbose: bool         = True,
    device:  torch.device = DEVICE,
) -> None:
    """Run *epochs* of training for a given curriculum *stage*."""
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in loader:
            batch = _to(batch, device)
            optimizer.zero_grad()
            ln_pred, _, pX, pY, delta, iso_off, dyn_off = model(
                batch["struct_E"], batch["struct_X"], batch["struct_Y"],
                batch["T"], batch["alpha"], batch["rX"], batch["rY"],
                batch["phys_norm"], batch["is_dyn"],
                batch["beta_scaled"], batch["tiso_scaled"],
                stage=stage,
            )
            ln_tgt = torch.log(batch["target"].clamp(min=1e-10))
            w      = _vit_weights(batch["alpha"])
            main   = torch.mean(w * (ln_pred - ln_tgt) ** 2)
            loss   = (main
                      + _physics_constraints(pX)
                      + _residual_l2(delta, iso_off, dyn_off))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        if verbose and (epoch + 1) % 10 == 0:
            print(f"     {label}  Ep {epoch+1:3d}/{epochs}  "
                  f"Loss={total_loss/len(loader):.5f}")


# ---------------------------------------------------------------------------
# Curriculum trainer
# ---------------------------------------------------------------------------
def curriculum_train(
    model,
    loader,
    verbose: bool         = True,
    device:  torch.device = DEVICE,
) -> pd.DataFrame:
    """
    Three-stage curriculum training.

    Returns
    -------
    pd.DataFrame with columns [stage, epoch, loss] — Stage-1 losses only
    (Stages 2 and 3 losses are stored as NaN because train_stage does not
    return per-epoch values; extend if needed).
    """
    def _set(modules, req: bool) -> None:
        for m in modules:
            for p in m.parameters():
                p.requires_grad = req

    def _active():
        return filter(lambda p: p.requires_grad, model.parameters())

    history: list[dict] = []

    # ── Stage 1: physics backbone ────────────────────────────────────────────
    _set([model.synergy_head, model.delta_head,
          model.iso_head, model.dyn_corr], False)
    opt1 = optim.Adam(_active(), lr=LR_STAGE[1], weight_decay=WD_STAGE1)
    if verbose:
        print(f"  -- Stage 1: Physics only (WD={WD_STAGE1:.0e}) --")
    for ep in range(STAGE_EPOCHS[1]):
        model.train()
        tl = 0.0
        for batch in loader:
            batch = _to(batch, device)
            opt1.zero_grad()
            ln_p, _, pX, _, d, io, do = model(
                batch["struct_E"], batch["struct_X"], batch["struct_Y"],
                batch["T"], batch["alpha"], batch["rX"], batch["rY"],
                batch["phys_norm"], batch["is_dyn"],
                batch["beta_scaled"], batch["tiso_scaled"], stage=1)
            ln_t = torch.log(batch["target"].clamp(min=1e-10))
            w    = _vit_weights(batch["alpha"])
            loss = torch.mean(w * (ln_p - ln_t) ** 2) + _physics_constraints(pX)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt1.step()
            tl += loss.item()
        avg = tl / len(loader)
        history.append({"stage": 1, "epoch": ep + 1, "loss": avg})
        if verbose and (ep + 1) % 10 == 0:
            print(f"     Ep {ep+1:3d}/{STAGE_EPOCHS[1]}  Loss={avg:.5f}")

    # ── Stage 2: + Flory-Huggins synergy ────────────────────────────────────
    _set([model.synergy_head], True)
    opt2 = optim.Adam(_active(), lr=LR_STAGE[2])
    if verbose:
        print("  -- Stage 2: +Synergy --")
    train_stage(model, loader, opt2, stage=2,
                epochs=STAGE_EPOCHS[2], label="S2",
                verbose=verbose, device=device)
    for ep in range(STAGE_EPOCHS[2]):
        history.append({"stage": 2, "epoch": ep + 1, "loss": float("nan")})

    # ── Stage 3: + all residual heads ───────────────────────────────────────
    _set([model.delta_head, model.iso_head, model.dyn_corr], True)
    opt3 = optim.Adam(_active(), lr=LR_STAGE[3])
    if verbose:
        print("  -- Stage 3: Full fine-tune --")
    train_stage(model, loader, opt3, stage=3,
                epochs=STAGE_EPOCHS[3], label="S3",
                verbose=verbose, device=device)
    for ep in range(STAGE_EPOCHS[3]):
        history.append({"stage": 3, "epoch": ep + 1, "loss": float("nan")})

    return pd.DataFrame(history)
