# -*- coding: utf-8 -*-
"""
core/trainer.py  —  V2
======================
Training pipeline for the V2 hPINN.

Key differences vs v1
---------------------
* **Thermodynamic penalty instead of fixed `lnA` anchor.**
  v1 penalised the model whenever the decoded `lnK_ref` drifted away from
  a hardcoded value of −2.5 — a *semi-empirical* anchor that has no
  physical justification for a different resin chemistry.

  v2 replaces this with a *physically meaningful* bound on the Gibbs
  free energy of activation at the reference temperature:

      ΔG‡(T_ref)  =  ΔH‡  −  T_ref · ΔS‡           [J/mol]

  A hinge penalty kicks in only when ΔG‡ falls outside
  [DG_MIN_KJ, DG_MAX_KJ] (defaults 70–130 kJ/mol, the physically
  plausible window for condensed-phase epoxy–amine cure at 160 °C).
  Inside this window the model is free — exactly as it should be.

* **All parameter decoding is delegated to `Unified_hPINN_v2.decode_params`.**
  The trainer never re-implements the physical bounds; it just receives
  a dict of already-bounded quantities.

* **MCDropout-aware.**  Stage-3 loss weighting is unchanged, but the
  residual heads contain persistent dropout.  Training proceeds normally
  because `F.dropout` with `training=True` is still a valid unbiased
  estimator during optimisation.

Curriculum
----------
    Stage 1 (60 ep, WD = 1e-3) : physics backbone (Eyring + DiBenedetto)
    Stage 2 (30 ep)            : + dynamic synergy head
    Stage 3 (20 ep)            : + Bayesian residual heads
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim

from .config import (
    DEVICE, STAGE_EPOCHS, LR_STAGE, WD_STAGE1,
    T_REF, DG_MIN_KJ, DG_MAX_KJ,
    TG0_PRIOR_K, LAM_PRIOR,
)
from .model import Unified_hPINN_v2


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
    """Recursively move every tensor in a (nested) dict to *device*."""
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
    Vitrification down-weighting — reduces the loss weight for nearly-cured
    samples (α > 0.92) where heterogeneous glassy-state relaxation dominates.

    The minimum floor is raised from 0.15 → 0.35 so that vitrified samples
    (the diffusion tail, α > 0.92) still contribute meaningful gradients for
    the DiBenedetto Tg0 and λ parameters.  Without this floor the combined
    effect of the WLF gate exponential AND the near-zero loss weight causes
    total gradient collapse for those two parameters.
    """
    return 1.0 / (1.0 + torch.exp(25.0 * (alpha - 0.92))) + 0.35


def _thermodynamic_penalty(pX: torch.Tensor) -> torch.Tensor:
    """
    **Replaces v1's fixed lnA anchor.**

    Five physically motivated soft constraints on the *X*-component:

      1. ΔG‡(T_ref) hinge :  penalise ΔG‡ outside [DG_MIN_KJ, DG_MAX_KJ].
         This is the thermodynamically correct analogue of the v1
         "lnK_ref ≈ −2.5" anchor: a plausible range for the Gibbs free
         energy of activation, not a single value.

      2. m/n Kamal–Sourour priors :  m ~ 0.2, n ~ 1.2  (literature means
         for epoxy–amine cure).  Kept at v1 strength — weak enough that
         the data dominates but strong enough to prevent degenerate
         solutions like m = 0, n = 0.

      3. Gentle entropy-centering :  ΔS‡ ~ −100 J/(mol·K) as a soft prior,
         otherwise the Eyring prefactor has a trivial degeneracy with
         ΔH‡ (the classic Kinetic Compensation Effect, KCE).

      4. DiBenedetto Tg0 prior :  Tg0 ~ 260 K.
         Guides the network out of flat loss basins early in Stage 1 where
         the WLF gate provides negligible gradient signal.  The prior is
         normalised by 30 K (≈ half the new [240, 320] K window) so its
         scale is comparable to the other terms.

      5. DiBenedetto λ prior :  λ ~ 0.4.
         Analogous soft pull toward the middle of the [0.1, 1.0] window,
         preventing the network from parking at the boundary where the
         sigmoid gradient also vanishes.
    """
    q = Unified_hPINN_v2.decode_params(pX)

    # 1. Gibbs free-energy hinge at T_ref
    dG_J    = q["dH_J"] - T_REF * q["dS_SI"]                    # [J/mol]
    dG_kJ   = dG_J / 1000.0
    dG_reg  = torch.mean(
        F.relu(dG_kJ - DG_MAX_KJ) + F.relu(DG_MIN_KJ - dG_kJ))

    # 2. Kamal–Sourour reaction-order priors
    mn_reg  = torch.mean(
        (q["m"] - 0.2) ** 2 + (q["n"] - 1.2) ** 2)

    # 3. Soft entropy centre (discourages KCE drift)
    dS_reg  = torch.mean(((q["dS_SI"] - (-100.0)) / 100.0) ** 2)

    # 4. DiBenedetto Tg0 prior — normalised by 30 K (half the [240, 320] window)
    tg0_reg = torch.mean(((q["Tg0_K"] - TG0_PRIOR_K) / 30.0) ** 2)

    # 5. DiBenedetto λ prior — normalised by 0.3 (half the [0.1, 1.0] window)
    lam_reg = torch.mean(((q["lam"] - LAM_PRIOR) / 0.3) ** 2)

    return (0.08 * dG_reg
            + 0.05 * mn_reg
            + 0.01 * dS_reg
            + 0.02 * tg0_reg
            + 0.02 * lam_reg)


def _residual_l2(
    delta:   torch.Tensor,
    iso_off: torch.Tensor,
    dyn_off: torch.Tensor,
) -> torch.Tensor:
    """
    L2 stiffness on the three Bayesian residual heads.

    The residuals should absorb *instrument noise*, not rebuild the
    physics.  A strong L2 prevents the MCDropout heads from hallucinating
    systematic corrections that compete with the (now more accurate) V2
    physics backbone.
    """
    return (
        1e-3 * torch.mean(delta   ** 2) +
        3e-3 * torch.mean(iso_off ** 2) +
        1e-3 * torch.mean(dyn_off ** 2)
    )


# ---------------------------------------------------------------------------
# Low-level stage trainer (shared by stages 2 and 3)
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
    """Train *epochs* of a given curriculum *stage*."""
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in loader:
            batch = _to(batch, device)
            optimizer.zero_grad()
            ln_pred, _, pX, _, delta, iso_off, dyn_off = model(
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
                      + _thermodynamic_penalty(pX)
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
    Three-stage curriculum (identical structure to v1, new loss under the hood).

    Returns
    -------
    pd.DataFrame with columns [stage, epoch, loss] — per-epoch Stage-1 losses.
    Stages 2 and 3 losses are recorded as NaN (train_stage does not return
    per-epoch values; extend if per-stage curves are needed).
    """
    def _set(modules, req: bool) -> None:
        for m in modules:
            for p in m.parameters():
                p.requires_grad = req

    def _active():
        return filter(lambda p: p.requires_grad, model.parameters())

    history: list[dict] = []

    # ── Stage 1: Eyring backbone ──────────────────────────────────────────
    _set([model.synergy_head, model.delta_head,
          model.iso_head,     model.dyn_corr], False)
    opt1 = optim.Adam(_active(), lr=LR_STAGE[1], weight_decay=WD_STAGE1)
    if verbose:
        print(f"  -- Stage 1: Eyring backbone  (WD={WD_STAGE1:.0e}) --")
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
            loss = (torch.mean(w * (ln_p - ln_t) ** 2)
                    + _thermodynamic_penalty(pX))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt1.step()
            tl += loss.item()
        avg = tl / len(loader)
        history.append({"stage": 1, "epoch": ep + 1, "loss": avg})
        if verbose and (ep + 1) % 10 == 0:
            print(f"     Ep {ep+1:3d}/{STAGE_EPOCHS[1]}  Loss={avg:.5f}")

    # ── Stage 2: + dynamic synergy head ──────────────────────────────────
    _set([model.synergy_head], True)
    opt2 = optim.Adam(_active(), lr=LR_STAGE[2])
    if verbose:
        print("  -- Stage 2: +Dynamic Synergy syn(T, α) --")
    train_stage(model, loader, opt2, stage=2,
                epochs=STAGE_EPOCHS[2], label="S2",
                verbose=verbose, device=device)
    for ep in range(STAGE_EPOCHS[2]):
        history.append({"stage": 2, "epoch": ep + 1, "loss": float("nan")})

    # ── Stage 3: + Bayesian residual heads (MCDropout active) ───────────
    _set([model.delta_head, model.iso_head, model.dyn_corr], True)
    opt3 = optim.Adam(_active(), lr=LR_STAGE[3])
    if verbose:
        print("  -- Stage 3: +Bayesian residuals (MC Dropout) --")
    train_stage(model, loader, opt3, stage=3,
                epochs=STAGE_EPOCHS[3], label="S3",
                verbose=verbose, device=device)
    for ep in range(STAGE_EPOCHS[3]):
        history.append({"stage": 3, "epoch": ep + 1, "loss": float("nan")})

    return pd.DataFrame(history)


# ---------------------------------------------------------------------------
# Back-compat alias for legacy callers
# ---------------------------------------------------------------------------
# Old experiments import `_physics_constraints` — keep the name alive so the
# v1 experiment scripts can be dropped in unchanged.
_physics_constraints = _thermodynamic_penalty
