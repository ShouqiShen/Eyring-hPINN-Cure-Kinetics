# -*- coding: utf-8 -*-
"""
core/inference.py  —  V2
========================
Evaluation, parameter extraction, post-hoc simulation, and Bayesian
uncertainty quantification for the V2 hPINN.

V2 changes vs v1
----------------
* `extract_mixture_params` decodes the **8-element** physics-head output
  (ΔH‡, ΔS‡, m, n, Tg0, ΔTg, λ, K_D) and additionally returns:
      - effective Arrhenius equivalents (Ea_eff, ln_A_eff) for KCE plots
      - a *captured-state* dynamic-synergy closure  syn(T, α)
* `make_rate_fn` implements the full V2 rate equation:
      Eyring  +  Kamal–Sourour  +  DiBenedetto/WLF  +  syn(T, α)
* New `predict_with_uncertainty` runs `MC_SAMPLES` Monte-Carlo dropout
  forward passes per batch and returns the predictive ln-rate mean / std
  / 95% CI — the **epistemic** confidence of the residual heads (the
  physics branch is deterministic by construction).
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from .config  import (DEVICE, T_REF, R_GAS, LN_KB_OVER_H_MIN, MC_SAMPLES)
from .model   import Unified_hPINN_v2
from .trainer import _to


# =============================================================================
# Standard evaluation
# =============================================================================
@torch.no_grad()
def evaluate(
    model,
    loader,
    use_pure_physics: bool         = False,
    device:           torch.device = DEVICE,
) -> dict:
    """
    Evaluate model on *loader*.

    Note
    ----
    To stay consistent with v1, MC Dropout is **not** averaged here — this
    is a single deterministic forward pass.  Use `predict_with_uncertainty`
    when you want the Bayesian posterior over the residual correction.

    Parameters
    ----------
    use_pure_physics : bool
        True  → zero-shot mode: cancel residuals, score only ln_r_phys.
        False → full-model mode: score ln_r_final.

    Returns
    -------
    dict with keys: r2, rmse, mae, preds, targets, params_X
        params_X is the raw [N, 8] physics-head output (use
        `Unified_hPINN_v2.decode_params` to convert to physical units).
    """
    # NB: keep MCDropout layers in their always-on state but set the rest
    # to eval (BatchNorm etc.).  The *physics* branch sees no MCDropout.
    model.eval()
    preds, tgts, px_all = [], [], []
    for batch in loader:
        batch = _to(batch, device)
        ln_r_final, ln_r_phys, pX, pY, delta, iso_masked, dyn_masked = model(
            batch["struct_E"], batch["struct_X"], batch["struct_Y"],
            batch["T"], batch["alpha"], batch["rX"], batch["rY"],
            batch["phys_norm"], batch["is_dyn"],
            batch["beta_scaled"], batch["tiso_scaled"], stage=3)

        if use_pure_physics:
            ln_p = ln_r_final - delta + iso_masked + dyn_masked
        else:
            ln_p = ln_r_final

        ln_t = torch.log(batch["target"].clamp(min=1e-10))
        preds.append(ln_p.cpu())
        tgts.append(ln_t.cpu())
        px_all.append(pX.cpu())

    pred   = torch.cat(preds).numpy().flatten()
    tgt    = torch.cat(tgts).numpy().flatten()
    pX_np  = torch.cat(px_all).numpy()
    ss_res = np.sum((tgt - pred) ** 2)
    ss_tot = np.sum((tgt - tgt.mean()) ** 2)
    return dict(
        r2=float(1 - ss_res / (ss_tot + 1e-10)),
        rmse=float(np.sqrt(np.mean((tgt - pred) ** 2))),
        mae=float(np.mean(np.abs(tgt - pred))),
        preds=pred, targets=tgt, params_X=pX_np,
    )


# =============================================================================
# Bayesian (MC Dropout) inference
# =============================================================================
@torch.no_grad()
def predict_with_uncertainty(
    model,
    loader,
    n_samples: int          = MC_SAMPLES,
    device:    torch.device = DEVICE,
) -> dict:
    """
    Run **N** Monte-Carlo Dropout forward passes per batch and aggregate
    a per-sample predictive posterior.

    Returns
    -------
    dict with keys:
        ln_mean  : [M]    posterior mean   of ln_r_final
        ln_std   : [M]    posterior std    of ln_r_final
        ln_phys  : [M]    deterministic ln_r_phys (single pass, since
                          MCDropout does not touch the physics branch)
        targets  : [M]    ln of experimental rate
        ci95_lo  : [M]    ln_mean − 1.96·ln_std
        ci95_hi  : [M]    ln_mean + 1.96·ln_std

    where M = total number of samples in *loader*.
    """
    model.eval()                                # MCDropout stays active
    means, stds, phys_means, targets = [], [], [], []
    for batch in loader:
        batch = _to(batch, device)
        out = model.predict_mc(
            batch["struct_E"], batch["struct_X"], batch["struct_Y"],
            batch["T"], batch["alpha"], batch["rX"], batch["rY"],
            batch["phys_norm"], batch["is_dyn"],
            batch["beta_scaled"], batch["tiso_scaled"],
            stage=3, n_samples=n_samples,
        )
        means     .append(out["ln_r_final_mean"].cpu())
        stds      .append(out["ln_r_final_std"] .cpu())
        phys_means.append(out["ln_r_phys_mean"] .cpu())
        targets   .append(torch.log(batch["target"].clamp(min=1e-10)).cpu())

    mean = torch.cat(means)     .numpy().flatten()
    std  = torch.cat(stds)      .numpy().flatten()
    phys = torch.cat(phys_means).numpy().flatten()
    tgt  = torch.cat(targets)   .numpy().flatten()
    return dict(
        ln_mean=mean, ln_std=std, ln_phys=phys, targets=tgt,
        ci95_lo=mean - 1.96 * std, ci95_hi=mean + 1.96 * std,
    )


# =============================================================================
# Parameter extraction  (8-output physics head, V2)
# =============================================================================
def extract_mixture_params(
    model,
    smiles_E:  str,
    smiles_X:  str,
    smiles_Y:  str,
    processor,
    T_min_K:   float,
    T_max_K:   float,
    device:    torch.device = DEVICE,
) -> dict:
    """
    Single forward pass through encoder + physics_head + (captured)
    synergy_head to decode every physical parameter as a Python float.

    Parameters
    ----------
    smiles_E / X / Y : SMILES strings of epoxy and the two hardeners
    processor        : a fitted ChemicalProcessor
    T_min_K, T_max_K : the global temperature normalisation bounds used
                        by `preprocess_global_dataframe` — needed so that
                        the captured `_syn_fn` reproduces the same
                        normalised temperature seen during training.

    Returns
    -------
    dict with keys:
        # X-component physical parameters
        dH_X_J, dS_X_SI, m_X, n_X,
        Tg0_X_K, dTg_X_K, lam_X, K_D_X,
        # Y-component physical parameters
        dH_Y_J, dS_Y_SI, m_Y, n_Y,
        Tg0_Y_K, dTg_Y_K, lam_Y, K_D_Y,
        # Effective Arrhenius equivalents (for backward-compat KCE plots)
        Ea_X_J, lnA_X,  Ea_Y_J, lnA_Y,
        # Dynamic synergy callable  (T_K: float, alpha: float) → float
        _syn_fn,
        # The normalisation constants used inside _syn_fn
        T_min_K, T_max_K,
    """
    def _s2dev(s: dict) -> dict:
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in s.items()}

    sE = _s2dev(processor.get(smiles_E))
    sX = _s2dev(processor.get(smiles_X))
    sY = _s2dev(processor.get(smiles_Y))

    model.eval()
    with torch.no_grad():
        fE  = model.encoder(sE)
        fX  = model.encoder(sX)
        fY  = model.encoder(sY)
        pX  = model.physics_head(torch.cat([fE, fX], dim=1))
        pY  = model.physics_head(torch.cat([fE, fY], dim=1))

    qX = Unified_hPINN_v2.decode_params(pX)
    qY = Unified_hPINN_v2.decode_params(pY)

    # ── Capture closure for the dynamic synergy head ─────────────────────
    # The closure keeps a strong reference to (model, fE, fX, fY) so the
    # caller may safely `del model` after this function returns.
    syn_head = model.synergy_head
    _T_min, _T_max = float(T_min_K), float(T_max_K)

    def _syn_fn(T_K: float, alpha: float) -> float:
        T_norm = (float(T_K) - _T_min) / (_T_max - _T_min + 1e-12)
        phys   = torch.tensor(
            [[T_norm, float(alpha)]], dtype=torch.float32, device=device)
        with torch.no_grad():
            s = syn_head(torch.cat([fE, fX, fY, phys], dim=1))
        return float(s[0, 0].item())

    # ── Effective Arrhenius mapping for KCE-style plots ──────────────────
    # Eyring k(T) = (k_B T / h) exp(ΔS‡/R) exp(−ΔH‡/RT)
    # Compared with Arrhenius k(T) = A exp(−Ea/RT):
    #     Ea = ΔH‡ + R T              (one-temperature Tolman correction)
    #     ln A = ln(k_B T / h) + 1 + ΔS‡/R       (at the same temperature)
    def _arr(dH_J, dS_SI):
        Ea_J = float(dH_J) + R_GAS * T_REF
        lnA  = (math.log(1.380649e-23 / 6.62607015e-34)   # ln(k_B/h)
                + math.log(60.0)                          # → per-minute
                + math.log(T_REF) + 1.0
                + float(dS_SI) / R_GAS)
        return Ea_J, lnA

    Ea_X_J, lnA_X = _arr(qX["dH_J"][0, 0].item(), qX["dS_SI"][0, 0].item())
    Ea_Y_J, lnA_Y = _arr(qY["dH_J"][0, 0].item(), qY["dS_SI"][0, 0].item())

    return {
        # X component
        "dH_X_J":   float(qX["dH_J"]  [0, 0].item()),
        "dS_X_SI":  float(qX["dS_SI"] [0, 0].item()),
        "m_X":      float(qX["m"]     [0, 0].item()),
        "n_X":      float(qX["n"]     [0, 0].item()),
        "Tg0_X_K":  float(qX["Tg0_K"] [0, 0].item()),
        "dTg_X_K":  float(qX["dTg_K"] [0, 0].item()),
        "lam_X":    float(qX["lam"]   [0, 0].item()),
        "K_D_X":    float(qX["K_D"]   [0, 0].item()),
        # Y component
        "dH_Y_J":   float(qY["dH_J"]  [0, 0].item()),
        "dS_Y_SI":  float(qY["dS_SI"] [0, 0].item()),
        "m_Y":      float(qY["m"]     [0, 0].item()),
        "n_Y":      float(qY["n"]     [0, 0].item()),
        "Tg0_Y_K":  float(qY["Tg0_K"] [0, 0].item()),
        "dTg_Y_K":  float(qY["dTg_K"] [0, 0].item()),
        "lam_Y":    float(qY["lam"]   [0, 0].item()),
        "K_D_Y":    float(qY["K_D"]   [0, 0].item()),
        # Arrhenius equivalents (for KCE plots)
        "Ea_X_J":   Ea_X_J,
        "lnA_X":    lnA_X,
        "Ea_Y_J":   Ea_Y_J,
        "lnA_Y":    lnA_Y,
        # Dynamic synergy callable + normalisation constants
        "_syn_fn":  _syn_fn,
        "T_min_K":  _T_min,
        "T_max_K":  _T_max,
    }


# =============================================================================
# Rate-function factory  (Eyring + DiBenedetto + dynamic synergy)
# =============================================================================
def make_rate_fn(params: dict, rX: float, rY: float) -> Callable:
    """
    Build a **closure** rate(alpha, T_K) → float that evaluates the full
    V2 mixture rate including:

        * Eyring rates for each component (X and Y)
        * Kamal–Sourour autocatalytic α^m (1−α)^n
        * DiBenedetto + WLF diffusion gate
        * Dynamic synergy syn(T, α)                                  V2 NEW

    Parameters
    ----------
    params : dict returned by `extract_mixture_params`
    rX, rY : mole fractions of hardener X and Y

    Returns
    -------
    rate_fn : callable(alpha: float, T_K: float) → float
              Always returns a strictly positive scalar.
    """
    syn_fn = params["_syn_fn"]

    def _eyring_k(dH_J: float, dS_SI: float, T: float) -> float:
        return math.exp(LN_KB_OVER_H_MIN
                        + math.log(T)
                        + dS_SI / R_GAS
                        - dH_J / (R_GAS * T))

    def _dibenedetto_gate(
        Tg0: float, dTg: float, lam: float, K_D: float,
        a:   float, T:   float,
    ) -> float:
        Tg_a = Tg0 + dTg * lam * a / (1.0 - (1.0 - lam) * a + 1e-8)
        under = max(0.0, Tg_a - T)
        return math.exp(-K_D * under)

    # Pre-extract floats (avoid dict-lookup overhead in the inner loop)
    dHX, dSX = params["dH_X_J"], params["dS_X_SI"]
    mX,  nX  = params["m_X"],    params["n_X"]
    Tg0X, dTgX, lamX, K_D_X = (params["Tg0_X_K"], params["dTg_X_K"],
                               params["lam_X"],   params["K_D_X"])
    dHY, dSY = params["dH_Y_J"], params["dS_Y_SI"]
    mY,  nY  = params["m_Y"],    params["n_Y"]
    Tg0Y, dTgY, lamY, K_D_Y = (params["Tg0_Y_K"], params["dTg_Y_K"],
                               params["lam_Y"],   params["K_D_Y"])

    def _rate(alpha: float, T_K: float) -> float:
        a   = float(np.clip(alpha, 0.0, 0.9999))
        T   = max(float(T_K), 1.0)

        # --- per-component Eyring chemistry × DiBenedetto/WLF gate ------
        kX  = _eyring_k(dHX, dSX, T)
        rX_ = (kX * (a + 1e-8) ** mX * (1.0 - a + 1e-8) ** nX
               * _dibenedetto_gate(Tg0X, dTgX, lamX, K_D_X, a, T))

        kY  = _eyring_k(dHY, dSY, T)
        rY_ = (kY * (a + 1e-8) ** mY * (1.0 - a + 1e-8) ** nY
               * _dibenedetto_gate(Tg0Y, dTgY, lamY, K_D_Y, a, T))

        base  = rX * rX_ + rY * rY_
        # --- dynamic synergy (V2 NEW) -----------------------------------
        syn   = syn_fn(T, a)
        shape = 16.0 * rX * rY * a * (1.0 - a)
        return float(max(base * (1.0 + syn * shape), 1e-12))

    return _rate


# =============================================================================
# Post-hoc simulations (RK45)  — unchanged from v1
# =============================================================================
def run_isothermal_report(
    df_test:  pd.DataFrame,
    rate_fn:  Callable,
) -> pd.DataFrame:
    """
    For each isothermal condition in *df_test*, simulate with `solve_ivp`
    (RK45, adaptive step) and find the predicted time to reach the
    experimentally observed maximum α.  Returns a DataFrame of
    (Temp_C, Target_Alpha, Exp_Time_min, Pred_Time_min, Error_min,
     Rel_Error_pct).
    """
    df_iso      = df_test[df_test["Condition_Type"] == "Iso"].copy()
    iso_temps_C = sorted(df_iso["Condition_Val"].unique())
    rows: list[dict] = []

    for T_C in iso_temps_C:
        T_K = T_C + 273.15
        grp = df_iso[df_iso["Condition_Val"] == T_C].sort_values("Alpha")
        if grp.empty:
            continue

        target_alpha = float(grp["Alpha"].max())

        if "Time_min" in grp.columns:
            grp_t = grp.sort_values("Time_min")
            a_exp = grp_t["Alpha"].values
            t_exp = grp_t["Time_min"].values
        else:
            rates = np.clip(
                grp["Rate_Smooth"].values if "Rate_Smooth" in grp.columns
                else grp["Rate_1_min"].values, 1e-10, None)
            a_exp  = grp["Alpha"].values
            dalpha = np.diff(a_exp)
            avg_r  = 0.5 * (rates[:-1] + rates[1:])
            t_exp  = np.concatenate(
                [[0.0], np.cumsum(dalpha / np.clip(avg_r, 1e-10, None))])

        if len(t_exp) < 2:
            continue
        exp_time = float(
            interp1d(a_exp, t_exp,
                     bounds_error=False, fill_value="extrapolate")(target_alpha))

        t_max   = max(exp_time * 4.0, 300.0)
        sol_iso = solve_ivp(
            fun=lambda t, y: [rate_fn(y[0], T_K)],
            t_span=(0.0, t_max), y0=[1e-3],
            method="RK45", dense_output=False, max_step=2.0,
        )
        a_arr = np.clip(sol_iso.y[0], 0.0, 0.9999)
        t_arr = sol_iso.t
        pred_time = (
            float(t_arr[-1]) if target_alpha > a_arr.max()
            else float(
                interp1d(a_arr, t_arr,
                         bounds_error=False, fill_value="extrapolate")(target_alpha))
        )

        err = pred_time - exp_time
        rel = abs(err) / (exp_time + 1e-10) * 100.0
        rows.append(dict(
            Temp_C=T_C, Target_Alpha=target_alpha,
            Exp_Time_min=exp_time, Pred_Time_min=pred_time,
            Error_min=err, Rel_Error_pct=rel,
        ))

    return pd.DataFrame(rows)


def run_dynamic_simulation(
    df_test:     pd.DataFrame,
    rate_fn:     Callable,
    target_beta: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Simulate a dynamic DSC scan with `solve_ivp` evaluated at the
    experimental temperature grid.  Falls back to the nearest available β
    if *target_beta* is not present in *df_test*.

    Returns
    -------
    sim_T_C, sim_alpha, sim_rate, used_beta
    """
    df_dyn = df_test[
        (df_test["Condition_Type"] == "Dyn") &
        (df_test["Condition_Val"]  == target_beta)
    ].sort_values("Temp_K")

    if df_dyn.empty:
        avail = sorted(
            df_test[df_test["Condition_Type"] == "Dyn"]["Condition_Val"].unique())
        if not avail:
            raise ValueError("No dynamic conditions found in df_test.")
        target_beta = float(avail[0])
        df_dyn = df_test[
            (df_test["Condition_Type"] == "Dyn") &
            (df_test["Condition_Val"]  == target_beta)
        ].sort_values("Temp_K")

    exp_T_K = df_dyn["Temp_K"].values
    T0_K    = float(exp_T_K[0])
    t_eval  = (exp_T_K - T0_K) / target_beta
    t_end   = float(t_eval[-1]) * 1.05

    sol = solve_ivp(
        fun=lambda t, y: [rate_fn(y[0], T0_K + target_beta * t)],
        t_span=(0.0, t_end), y0=[1e-3],
        method="RK45", t_eval=t_eval, max_step=1.0,
    )
    sim_T_C   = T0_K + target_beta * sol.t - 273.15
    sim_alpha = np.clip(sol.y[0], 0.0, 0.9999)
    sim_T_K   = T0_K + target_beta * sol.t
    sim_rate  = np.array([rate_fn(a, T) for a, T in zip(sim_alpha, sim_T_K)])

    return sim_T_C, sim_alpha, sim_rate, target_beta
