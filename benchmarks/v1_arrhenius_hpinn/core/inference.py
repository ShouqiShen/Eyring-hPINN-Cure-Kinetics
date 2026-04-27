# -*- coding: utf-8 -*-
"""
core/inference.py
=================
Evaluation, parameter extraction, and post-hoc simulation utilities.

evaluate()
----------
Standard evaluation on a DataLoader.  Set `use_pure_physics=True` for the
zero-shot LOMO test mode: residuals are cancelled so only the Kamal–Sourour
mixture prediction is scored.

    pure_phys_ln = ln_r_final - delta + iso_offset_masked + dyn_offset_masked
                 ≡ ln_r_phys  (explicit, unambiguous reconstruction)

extract_mixture_params()
------------------------
Single torch.no_grad() forward through encoder + physics_head + synergy_head
to decode all physical parameters (lnK_ref, Ea, m, n for both X and Y
components, plus the synergy scalar) as plain Python floats.

make_rate_fn()
--------------
Builds a closure rate(alpha, T_K) → float from the extracted parameters,
implementing the full X+Y mixture + Flory-Huggins synergy equation.

run_isothermal_report() / run_dynamic_simulation()
--------------------------------------------------
Post-hoc simulations using scipy solve_ivp (RK45) — numerically stable,
adaptive step-size, no Euler blow-up.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from .config  import DEVICE, T_REF, R_GAS
from .trainer import _to


# =============================================================================
# Evaluation
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

    Parameters
    ----------
    use_pure_physics : bool
        True  → zero-shot mode: cancel residuals, score only ln_r_phys.
        False → full-model mode: score ln_r_final (train / in-distribution).

    Returns
    -------
    dict with keys: r2, rmse, mae, preds, targets, params_X
    """
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
# Parameter extraction
# =============================================================================
def extract_mixture_params(
    model,
    smiles_E:  str,
    smiles_X:  str,
    smiles_Y:  str,
    processor,
    device: torch.device = DEVICE,
) -> dict:
    """
    Run a single forward pass through encoder + physics_head + synergy_head
    to decode ALL physical parameters as plain Python floats.

    Returns
    -------
    dict with keys:
        lnKr_X, Ea_X_J, m_X, n_X   (component X kinetics)
        lnKr_Y, Ea_Y_J, m_Y, n_Y   (component Y kinetics)
        syn_mult                     (Flory-Huggins synergy scalar)
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
        syn = model.synergy_head(torch.cat([fE, fX, fY], dim=1))

    return {
        "lnKr_X":   float(pX[0, 0].item() * 2.0 - 2.0),
        "Ea_X_J":   float(F.softplus(pX[0, 1]).item() * 50_000.0),
        "m_X":      float(abs(pX[0, 2].item())),
        "n_X":      float(abs(pX[0, 3].item())),
        "lnKr_Y":   float(pY[0, 0].item() * 2.0 - 2.0),
        "Ea_Y_J":   float(F.softplus(pY[0, 1]).item() * 50_000.0),
        "m_Y":      float(abs(pY[0, 2].item())),
        "n_Y":      float(abs(pY[0, 3].item())),
        "syn_mult": float(syn[0, 0].item()),
    }


# =============================================================================
# Rate function factory
# =============================================================================
def make_rate_fn(params: dict, rX: float, rY: float) -> Callable:
    """
    Build a closure that evaluates the full X+Y mixture rate + synergy:

        rate_X  = K_X(T) · α^m_X · (1−α)^n_X
        rate_Y  = K_Y(T) · α^m_Y · (1−α)^n_Y
        base    = rX·rate_X + rY·rate_Y
        shape   = 16·rX·rY·α·(1−α)
        total   = base · (1 + syn · shape)

    Parameters
    ----------
    params : dict returned by `extract_mixture_params`
    rX, rY : mole fractions of hardener X and Y

    Returns
    -------
    rate_fn(alpha, T_K) → float
    """
    lnKr_X = params["lnKr_X"];  Ea_X = params["Ea_X_J"]
    m_X    = params["m_X"];      n_X  = params["n_X"]
    lnKr_Y = params["lnKr_Y"];  Ea_Y = params["Ea_Y_J"]
    m_Y    = params["m_Y"];      n_Y  = params["n_Y"]
    syn    = params["syn_mult"]

    def _rate(alpha: float, T_K: float) -> float:
        a    = float(np.clip(alpha, 0.0, 0.9999))
        K_X  = np.exp(lnKr_X - (Ea_X / R_GAS) * (1.0 / T_K - 1.0 / T_REF))
        K_Y  = np.exp(lnKr_Y - (Ea_Y / R_GAS) * (1.0 / T_K - 1.0 / T_REF))
        r_X  = K_X * (a + 1e-8) ** m_X * (1.0 - a + 1e-8) ** n_X
        r_Y  = K_Y * (a + 1e-8) ** m_Y * (1.0 - a + 1e-8) ** n_Y
        base  = rX * r_X + rY * r_Y
        shape = 16.0 * rX * rY * a * (1.0 - a)
        return float(max(base * (1.0 + syn * shape), 1e-10))

    return _rate


# =============================================================================
# Post-hoc simulations (RK45)
# =============================================================================
def run_isothermal_report(
    df_test:  pd.DataFrame,
    rate_fn:  Callable,
) -> pd.DataFrame:
    """
    For each isothermal condition in *df_test*, simulate with solve_ivp (RK45)
    and find the predicted time to reach the experimentally observed maximum α.

    Returns
    -------
    DataFrame with columns:
        Temp_C, Target_Alpha, Exp_Time_min, Pred_Time_min, Error_min, Rel_Error_pct
    """
    df_iso      = df_test[df_test["Condition_Type"] == "Iso"].copy()
    iso_temps_C = sorted(df_iso["Condition_Val"].unique())
    rows: list[dict] = []

    for T_C in iso_temps_C:
        T_K  = T_C + 273.15
        grp  = df_iso[df_iso["Condition_Val"] == T_C].sort_values("Alpha")
        if grp.empty:
            continue

        target_alpha = float(grp["Alpha"].max())

        # Experimental time reconstruction
        if "Time_min" in grp.columns:
            grp_t = grp.sort_values("Time_min")
            a_exp = grp_t["Alpha"].values
            t_exp = grp_t["Time_min"].values
        else:
            rates  = np.clip(
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

        # RK45 simulation at fixed T
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
    Simulate a dynamic DSC scan with solve_ivp evaluated at the experimental
    temperature grid.  Falls back to the nearest available β if *target_beta*
    is not present.

    Returns
    -------
    sim_T_C    : temperature array [°C]
    sim_alpha  : predicted degree of cure
    sim_rate   : predicted reaction rate
    used_beta  : actual heating rate used (may differ from target_beta)
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
