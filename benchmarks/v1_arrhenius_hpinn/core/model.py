# -*- coding: utf-8 -*-
"""
core/model.py
=============
Neural network architecture for the Structure-Formulation hPINN.

TriModalEncoder
---------------
Fuses three molecular representations into a 128-dim embedding:
  CNN path  : 1-D CNN over padded SMILES tokens → global-max-pool → 64 dim
  GCN path  : 2-layer GCNConv over molecular graph → global-mean-pool → 128 dim
  ECFP path : 2-layer MLP over 2048-bit Morgan fingerprint → 128 dim
  Fusion    : concat (64+128+128=320) → Linear → 256 → ReLU → Linear → 128

Unified_hPINN  (v4 explicit residual decoupling)
-------------------------------------------------
Forward returns 7 tensors:
    ln_r_final, ln_r_phys, pX, pY, delta, iso_offset_masked, dyn_offset_masked

Key identity:
    ln_r_final = ln_r_phys + delta - iso_offset_masked - dyn_offset_masked

Zero-shot pure-physics reconstruction (used at test time):
    pure_phys_ln = ln_r_final - delta + iso_offset_masked + dyn_offset_masked
                 ≡ ln_r_phys
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

from .config import T_REF, R_GAS


# =============================================================================
class TriModalEncoder(nn.Module):
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
        # CNN-SMILES path
        x_seq = self.embed(struct["smiles_seq"])
        if x_seq.dim() == 2:              # single sample: [seq_len, 32]
            x_seq = x_seq.unsqueeze(0)    # → [1, seq_len, 32]
        x_cnn = torch.max(
            F.relu(self.cnn(x_seq.transpose(1, 2))), dim=2)[0]

        # GCN path
        ei  = struct["edge_index"]
        nf  = struct["node_feat"]
        bid = struct.get(
            "batch_idx",
            torch.zeros(nf.size(0), dtype=torch.long, device=nf.device))
        x_g = F.relu(self.gcn[0](nf, ei))
        x_g = F.relu(self.gcn[1](x_g, ei))
        x_g = global_mean_pool(x_g, bid)

        # ECFP path
        ecfp = struct["ecfp"]
        if ecfp.dim() == 1:
            ecfp = ecfp.unsqueeze(0)
        x_fp = self.ecfp_mlp(ecfp)

        return self.fusion(torch.cat([x_cnn, x_g, x_fp], dim=1))


# =============================================================================
class Unified_hPINN(nn.Module):
    """
    Unified Hybrid Physics-Informed Neural Network.

    Curriculum training stages
    --------------------------
    Stage 1 : encoder + physics_head only  (Kamal–Sourour backbone)
    Stage 2 : + synergy_head               (Flory-Huggins mixture term)
    Stage 3 : + delta_head, iso_head, dyn_corr  (residual fine-tuning)
    """

    PARAM_NAMES = ["lnK_ref", "Ea_kJ_mol", "m", "n", "c", "ac0", "acT_K"]

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.encoder      = TriModalEncoder(vocab_size)
        self.physics_head = nn.Sequential(
            nn.Linear(128 * 2, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 7),
        )
        self.synergy_head = nn.Sequential(
            nn.Linear(128 * 3, 64), nn.ReLU(), nn.Linear(64, 1),
        )
        self.delta_head = nn.Sequential(
            nn.Linear(128 * 3 + 2, 64), nn.ReLU(), nn.Linear(64, 1),
        )
        self.iso_head = nn.Sequential(
            nn.Linear(128 * 3 + 2, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 1),
        )
        # Pure linear correction for dynamic scanning rate (1 parameter)
        self.dyn_corr = nn.Linear(1, 1, bias=True)

    # ------------------------------------------------------------------
    @staticmethod
    def _calc_rate(
        p: torch.Tensor,
        T: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        """Kamal–Sourour model with diffusion gate (vitrification)."""
        lnK_ref = p[:, 0:1] * 2.0 - 2.0
        Ea      = F.softplus(p[:, 1:2]) * 50_000.0   # J/mol, bounded > 0
        m       = torch.abs(p[:, 2:3])
        n       = torch.abs(p[:, 3:4])
        c       = F.softplus(p[:, 4:5])
        ac0     = torch.sigmoid(p[:, 5:6])
        acT     = 3e-3 * torch.tanh(p[:, 6:7])

        lnK    = lnK_ref - (Ea / R_GAS) * (1.0 / T - 1.0 / T_REF)
        r_chem = (
            torch.exp(lnK)
            * (alpha + 1e-8) ** m
            * (1.0 - alpha + 1e-8) ** n
        )
        alpha_c = ac0 + acT * T
        gate    = 1.0 / (1.0 + torch.exp(c * (alpha - alpha_c)))
        return r_chem * gate

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
        sE, sX, sY      : struct dicts from ChemicalProcessor (epoxy, X-hardener, Y-hardener)
        T, alpha         : [B, 1] physical state
        rX, rY           : [B, 1] mixture mole fractions
        phys_norm        : [B, 2] globally-normalised (T_norm, alpha)
        is_dyn           : [B, 1] 1.0 for dynamic scan, 0.0 for isothermal
        beta_scaled      : [B, 1] globally-scaled log(β) (dynamic only)
        tiso_scaled      : [B, 1] globally-scaled 1/T_iso (isothermal only)
        stage            : int, controls which heads are active

        Returns
        -------
        ln_r_final, ln_r_phys, pX, pY, delta, iso_offset_masked, dyn_offset_masked
        """
        fE = self.encoder(sE)
        fX = self.encoder(sX)
        fY = self.encoder(sY)

        pX = self.physics_head(torch.cat([fE, fX], dim=1))
        pY = self.physics_head(torch.cat([fE, fY], dim=1))

        r_X    = self._calc_rate(pX, T, alpha)
        r_Y    = self._calc_rate(pY, T, alpha)
        r_base = rX * r_X + rY * r_Y

        if stage >= 2:
            syn     = self.synergy_head(torch.cat([fE, fX, fY], dim=1))
            weight  = 16.0 * rX * rY * alpha * (1.0 - alpha)
            r_total = r_base * (1.0 + syn * weight)
        else:
            r_total = r_base

        ln_r_phys = torch.log(r_total.clamp(min=1e-10))

        if stage >= 3:
            delta = self.delta_head(
                torch.cat([fE, fX, fY, phys_norm], dim=1))

            # Masked residuals — each correction only fires for its condition type
            iso_offset_masked = (
                self.iso_head(
                    torch.cat([fE, fX, fY, tiso_scaled, alpha], dim=1))
                * (1.0 - is_dyn)
            )
            dyn_offset_masked = self.dyn_corr(beta_scaled) * is_dyn

            ln_r_final = ln_r_phys + delta - iso_offset_masked - dyn_offset_masked
        else:
            delta = iso_offset_masked = dyn_offset_masked = torch.zeros_like(ln_r_phys)
            ln_r_final = ln_r_phys

        return (ln_r_final, ln_r_phys, pX, pY,
                delta, iso_offset_masked, dyn_offset_masked)
