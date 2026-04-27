# -*- coding: utf-8 -*-
"""
core/model.py  —  V2
====================
Structure-Formulation hPINN v2.0.

Four scientific upgrades over v1:

1. **Eyring Transition-State Theory**  (replaces Arrhenius `Ea / lnA`)
       k(T) = (κ k_B T / h) · exp(ΔS‡/R) · exp(−ΔH‡/RT)       [κ = 1]

   Physics head now outputs ΔH‡ and ΔS‡ directly.  The thermodynamically
   meaningful Gibbs free energy of activation ΔG‡ = ΔH‡ − T·ΔS‡ can be
   reconstructed for any T, and is used as a *bounded* anchor in the
   trainer (replacing the hardcoded −2.5 lnA target of v1).

2. **Dynamic Synergy Function**  syn = syn(T, α)
       Previously the synergy head emitted a single scalar shared across
       the entire reaction.  V2 conditions it on the current normalised
       temperature and degree of cure, allowing the network to model
       shifting steric hindrance and competitive reactions as the
       crosslink density increases.

3. **DiBenedetto + WLF Diffusion Gate**  (replaces Rabinowitch sigmoid)
       Tg(α) = Tg0 + (Tg∞ − Tg0) · λα / (1 − (1−λ)α)            DiBenedetto
       under  = max(0, Tg(α) − T)                                 [K below]
       gate   = exp(−K_D · under)                                 WLF-like

   This reproduces the *exponential* slowdown observed in the glassy
   tail (α > 0.85) that the v1 sigmoid-gate severely under-damped.

4. **Monte-Carlo Dropout — Bayesian Residuals**
       Persistent `MCDropout` layers are injected into the three residual
       heads (delta / iso / dyn_corr).  They stay **active in eval() mode**
       so repeated forward passes give a Bayesian posterior over the
       residual correction — i.e. an epistemic confidence band on the
       data-driven bias-correction of the physics prediction.
"""
from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

from .config import (
    T_REF, R_GAS, LN_KB_OVER_H_MIN,
    DH_MIN_J, DH_MAX_J, DS_MIN_SI, DS_MAX_SI,
    TG0_MIN_K, TG0_MAX_K, DTG_MIN_K, DTG_MAX_K,
    LAM_MIN,   LAM_MAX,   KD_SCALE,
    MC_DROPOUT_P, MC_SAMPLES,
)


# =============================================================================
# Monte-Carlo Dropout  (persistent — active in eval mode)
# =============================================================================
class MCDropout(nn.Dropout):
    """Dropout that **remains active during `.eval()`** so that repeated
    forward passes at inference time yield a Monte-Carlo posterior."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # `training=True` forces the mask to be resampled every call,
        # regardless of the module's current training flag.
        return F.dropout(x, self.p, training=True)


# =============================================================================
# Tri-modal molecular encoder  (unchanged from v1)
# =============================================================================
class TriModalEncoder(nn.Module):
    """Fuses SMILES-CNN, Mol-GCN, and ECFP4-MLP paths into a 128-D embedding."""

    OUT_DIM: int = 128

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.embed    = nn.Embedding(vocab_size, 32, padding_idx=0)
        self.cnn      = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.gcn      = nn.ModuleList([GCNConv(3, 64), GCNConv(64, 128)])
        self.ecfp_mlp = nn.Sequential(
            nn.Linear(2048, 256), nn.ReLU(), nn.Linear(256, 128),
        )
        self.fusion = nn.Sequential(
            nn.Linear(64 + 128 + 128, 256), nn.ReLU(), nn.Linear(256, 128),
        )

    def forward(self, struct: dict) -> torch.Tensor:
        # CNN path over padded SMILES tokens
        x_seq = self.embed(struct["smiles_seq"])
        if x_seq.dim() == 2:
            x_seq = x_seq.unsqueeze(0)
        x_cnn = torch.max(F.relu(self.cnn(x_seq.transpose(1, 2))), dim=2)[0]

        # GCN path over molecular graph
        ei  = struct["edge_index"]
        nf  = struct["node_feat"]
        bid = struct.get(
            "batch_idx",
            torch.zeros(nf.size(0), dtype=torch.long, device=nf.device))
        x_g = F.relu(self.gcn[0](nf, ei))
        x_g = F.relu(self.gcn[1](x_g, ei))
        x_g = global_mean_pool(x_g, bid)

        # ECFP4 path
        ecfp = struct["ecfp"]
        if ecfp.dim() == 1:
            ecfp = ecfp.unsqueeze(0)
        x_fp = self.ecfp_mlp(ecfp)

        return self.fusion(torch.cat([x_cnn, x_g, x_fp], dim=1))


# =============================================================================
# Unified_hPINN_v2  — Eyring + DiBenedetto + Dynamic Synergy + MC Dropout
# =============================================================================
class Unified_hPINN_v2(nn.Module):
    """
    V2 Hybrid Physics-Informed Neural Network for co-curing kinetics.

    Physics head output layout (8 raw scalars per (E, hardener) pair) — each
    is mapped to a *physically bounded* quantity via `_decode_params`:

        index  raw → physical                     bound / units
        ─────  ───────────────────────────        ──────────────────────
          0    sigmoid → ΔH‡                      [40, 160] kJ/mol
          1    sigmoid → ΔS‡                      [−250, 50] J/(mol·K)
          2    abs()   → m                        autocatalytic order
          3    abs()   → n                        nth-order term
          4    sigmoid → Tg0                      [200, 350] K
          5    sigmoid → (Tg∞ − Tg0)              [30, 250] K
          6    sigmoid → λ (DiBenedetto)          [0.05, 0.95]
          7    softplus → K_D (WLF slope)         [~0, 0.1+] K⁻¹

    Curriculum stages (driven by `trainer.curriculum_train`)
    --------------------------------------------------------
        Stage 1 : encoder + physics_head           (Eyring backbone)
        Stage 2 : + synergy_head (dynamic)         (composition-coupling)
        Stage 3 : + delta / iso / dyn residuals    (Bayesian fine-tune)
    """

    PARAM_NAMES = ["dH_J", "dS_SI", "m", "n",
                   "Tg0_K", "dTg_K", "lam", "K_D"]
    N_PHYS_OUT  = 8

    # ------------------------------------------------------------------
    def __init__(
        self,
        vocab_size:    int,
        mc_dropout_p:  float = MC_DROPOUT_P,
    ) -> None:
        super().__init__()
        self.encoder      = TriModalEncoder(vocab_size)

        # --- deterministic physics branch --------------------------------
        self.physics_head = nn.Sequential(
            nn.Linear(128 * 2, 128), nn.ReLU(),
            nn.Dropout(0.3),                      # standard dropout — inactive at eval
            nn.Linear(128, self.N_PHYS_OUT),
        )

        # --- dynamic synergy head: conditioned on (T_norm, α) -----------
        # Input = [fE | fX | fY | T_norm | α]  →  128·3 + 2 = 386
        self.synergy_head = nn.Sequential(
            nn.Linear(128 * 3 + 2, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

        # --- Bayesian residual heads (MCDropout persists in eval) -------
        self.delta_head = nn.Sequential(
            nn.Linear(128 * 3 + 2, 64), nn.ReLU(), MCDropout(mc_dropout_p),
            nn.Linear(64, 32),          nn.ReLU(), MCDropout(mc_dropout_p),
            nn.Linear(32, 1),
        )
        self.iso_head = nn.Sequential(
            nn.Linear(128 * 3 + 2, 16), nn.ReLU(), MCDropout(mc_dropout_p),
            nn.Linear(16, 8),           nn.ReLU(),
            nn.Linear(8, 1),
        )
        # dyn_corr is upgraded from a single Linear to a tiny MLP with MCDropout
        self.dyn_corr = nn.Sequential(
            nn.Linear(1, 8), nn.ReLU(), MCDropout(mc_dropout_p),
            nn.Linear(8, 1),
        )

    # ------------------------------------------------------------------
    # Parameter decoding
    # ------------------------------------------------------------------
    @staticmethod
    def _bounded(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
        """Smoothly constrain a raw scalar to the physical interval [lo, hi]."""
        return lo + (hi - lo) * torch.sigmoid(x)

    @classmethod
    def decode_params(cls, p: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Decode the raw physics-head output into a dict of bounded physical
        parameters.  **Used by both the forward pass and the trainer's
        thermodynamic penalty**, so the bounds are applied in exactly one
        place.

        Parameters
        ----------
        p : [B, 8]  raw physics-head output

        Returns
        -------
        dict{str: [B, 1] tensor}
        """
        return {
            "dH_J":  cls._bounded(p[:, 0:1], DH_MIN_J,  DH_MAX_J),
            "dS_SI": cls._bounded(p[:, 1:2], DS_MIN_SI, DS_MAX_SI),
            "m":     torch.abs   (p[:, 2:3]),
            "n":     torch.abs   (p[:, 3:4]),
            "Tg0_K": cls._bounded(p[:, 4:5], TG0_MIN_K, TG0_MAX_K),
            "dTg_K": cls._bounded(p[:, 5:6], DTG_MIN_K, DTG_MAX_K),
            "lam":   cls._bounded(p[:, 6:7], LAM_MIN,   LAM_MAX),
            "K_D":   F.softplus  (p[:, 7:8]) * KD_SCALE + 1e-4,
        }

    # ------------------------------------------------------------------
    # Core kinetic rate calculation (Eyring + Kamal–Sourour + DiBenedetto)
    # ------------------------------------------------------------------
    @classmethod
    def _eyring_rate(
        cls,
        p_raw: torch.Tensor,
        T:     torch.Tensor,
        alpha: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Compute `ln r` for a pure component (ln-space for numerical stability).

        Mathematics
        -----------
        Eyring          :   ln k = ln(k_B/h) + ln 60 + ln T + ΔS‡/R − ΔH‡/(RT)
        Autocatalytic   :   ln r_chem = ln k + m·ln α + n·ln(1 − α)
        DiBenedetto     :   Tg(α)    = Tg0 + ΔTg · λα / (1 − (1 − λ)α)
        Free-volume gate:   ln gate  = −K_D · max(0, Tg(α) − T)           [K⁻¹ · K]

        The gate → 1 when T >> Tg(α) and decays *exponentially* as T slips
        below Tg(α), reproducing vitrification slowdown without the
        saturation artefact of the v1 sigmoid.

        Returns
        -------
        ln_r : [B, 1]     ln of the component rate
        q    : dict       decoded physical parameters (for penalties / logging)
        """
        eps = 1e-8
        q   = cls.decode_params(p_raw)
        T   = T.clamp(min=1.0)                                   # safety
        a   = alpha.clamp(min=eps, max=1.0 - eps)

        ln_k   = (LN_KB_OVER_H_MIN
                  + torch.log(T)
                  + q["dS_SI"] / R_GAS
                  - q["dH_J"]  / (R_GAS * T))

        ln_chem = ln_k + q["m"] * torch.log(a) + q["n"] * torch.log(1.0 - a)

        # DiBenedetto glass transition — denominator clamped to prevent division
        # by zero when λ is small and α ≈ 1.  With LAM_MIN = 0.1 the worst case
        # is (1−0.1)×0.9999 ≈ 0.9, so denom > 0.1 in practice, but an explicit
        # clamp makes the graph safe during the first noisy gradient steps.
        denom   = (1.0 - (1.0 - q["lam"]) * a).clamp(min=1e-3)
        Tg_a    = q["Tg0_K"] + q["dTg_K"] * q["lam"] * a / denom

        # WLF-inspired exponential gate — clamp the exponent to [-50, 0] so
        # that: (a) the gate never underflows to exactly 0 (gradient dies), and
        # (b) exp(-50) ≈ 2e-22 is still physically "vitrified" but the backward
        # pass through clamp retains a non-zero gradient signal for Tg0/λ.
        gate_arg = (q["K_D"] * F.relu(Tg_a - T)).clamp(max=50.0)
        ln_gate  = -gate_arg

        return ln_chem + ln_gate, q

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        sE, sX, sY,
        T, alpha, rX, rY,
        phys_norm, is_dyn, beta_scaled, tiso_scaled,
        stage: int = 3,
    ) -> tuple[torch.Tensor, ...]:
        """
        Parameters
        ----------
        sE, sX, sY        struct dicts from ChemicalProcessor
        T, alpha          [B, 1] physical state  (K, dimensionless)
        rX, rY            [B, 1] mole fractions of the two hardeners
        phys_norm         [B, 2] globally-normalised (T_norm, α)
                                ─ fed to synergy_head, delta_head, iso_head
        is_dyn            [B, 1] 1.0 for dynamic scan, 0.0 for isothermal
        beta_scaled       [B, 1] globally-scaled log β   (dynamic only)
        tiso_scaled       [B, 1] globally-scaled 1/T_iso (isothermal only)
        stage             curriculum stage (1 / 2 / 3) → controls active heads

        Returns
        -------
        ln_r_final, ln_r_phys, pX, pY, delta, iso_offset_masked, dyn_offset_masked

        Identity (by construction):
            ln_r_final = ln_r_phys + delta − iso_offset_masked − dyn_offset_masked
        so the zero-shot *pure-physics* reconstruction is simply
            pure_phys_ln = ln_r_final − delta + iso_offset_masked + dyn_offset_masked
        """
        fE = self.encoder(sE)
        fX = self.encoder(sX)
        fY = self.encoder(sY)

        pX = self.physics_head(torch.cat([fE, fX], dim=1))
        pY = self.physics_head(torch.cat([fE, fY], dim=1))

        # Eyring + Kamal–Sourour + DiBenedetto rate for each component
        ln_rX, _ = self._eyring_rate(pX, T, alpha)
        ln_rY, _ = self._eyring_rate(pY, T, alpha)

        # Linear-space weighted mixture
        r_base = rX * torch.exp(ln_rX) + rY * torch.exp(ln_rY)

        # ── Dynamic synergy syn(T, α) ──────────────────────────────────
        if stage >= 2:
            syn_input = torch.cat([fE, fX, fY, phys_norm], dim=1)   # [B, 386]
            syn       = self.synergy_head(syn_input)                 # [B, 1]
            # Shape-factor weight: zero at pure composition / full cure,
            # maximum at rX = rY = α = 0.5
            weight    = 16.0 * rX * rY * alpha * (1.0 - alpha)
            r_total   = r_base * (1.0 + syn * weight)
        else:
            r_total   = r_base

        ln_r_phys = torch.log(r_total.clamp(min=1e-12))

        # ── Bayesian residual corrections (stage 3 only) ───────────────
        if stage >= 3:
            delta   = self.delta_head(
                torch.cat([fE, fX, fY, phys_norm], dim=1))
            iso_off = self.iso_head(
                torch.cat([fE, fX, fY, tiso_scaled, alpha], dim=1)
            ) * (1.0 - is_dyn)
            dyn_off = self.dyn_corr(beta_scaled) * is_dyn
            ln_r_final = ln_r_phys + delta - iso_off - dyn_off
        else:
            delta = iso_off = dyn_off = torch.zeros_like(ln_r_phys)
            ln_r_final = ln_r_phys

        return (ln_r_final, ln_r_phys, pX, pY, delta, iso_off, dyn_off)

    # ------------------------------------------------------------------
    # Monte-Carlo predictive inference
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_mc(
        self,
        *fwd_args,
        n_samples: int = MC_SAMPLES,
        **fwd_kwargs,
    ) -> dict[str, torch.Tensor]:
        """
        Run `n_samples` forward passes **with persistent MC Dropout active**
        and return mean / std over the predictive posterior.

        Accepts exactly the same positional / keyword arguments as `forward`.

        Returns
        -------
        dict with keys:
            ln_r_final_mean, ln_r_final_std     [B, 1]
            ln_r_phys_mean,  ln_r_phys_std      [B, 1]
                (ln_r_phys should be ~constant — it is not touched by the
                 residual heads, so its std will be ≈ 0 apart from FP noise.)
            samples_ln_r_final  [n_samples, B, 1]   raw predictions
        """
        # MCDropout layers are always active regardless of `eval()`, but the
        # *plain* nn.Dropout inside physics_head must be kept off so that the
        # physics branch remains deterministic (only the residuals are noisy).
        self.eval()

        final_samples: list[torch.Tensor] = []
        phys_samples:  list[torch.Tensor] = []
        for _ in range(n_samples):
            out = self(*fwd_args, **fwd_kwargs)
            final_samples.append(out[0])
            phys_samples.append(out[1])

        stacked_final = torch.stack(final_samples, dim=0)   # [N, B, 1]
        stacked_phys  = torch.stack(phys_samples,  dim=0)
        return {
            "ln_r_final_mean":    stacked_final.mean(dim=0),
            "ln_r_final_std":     stacked_final.std (dim=0, unbiased=False),
            "ln_r_phys_mean":     stacked_phys .mean(dim=0),
            "ln_r_phys_std":      stacked_phys .std (dim=0, unbiased=False),
            "samples_ln_r_final": stacked_final,
        }


# =============================================================================
# Backward-compat alias
# =============================================================================
# Experiments can be ported from v1 by simply swapping the import; the name
# `Unified_hPINN` keeps existing call-sites working with the V2 implementation.
Unified_hPINN = Unified_hPINN_v2
