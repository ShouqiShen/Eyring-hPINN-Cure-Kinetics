# -*- coding: utf-8 -*-
"""
experiments/exp1_ratio_loro.py  —  V2
=====================================
Experiment 1 — Leave-One-Ratio-Out (LORO) Validation  [V2 hPINN]

V2 changes vs v1
----------------
* Physics backbone: Eyring TST (ΔH‡, ΔS‡)  in place of Arrhenius (Ea, lnA)
* Dynamic synergy   syn(T, α)
* DiBenedetto + WLF diffusion gate
* MC-Dropout Bayesian residuals → predict_with_uncertainty()

The ensemble seed-CI is still reported (cross-restart variance), and a
**second** uncertainty band is added: the within-model MC-Dropout posterior
on the residual heads.  These are physically distinct uncertainties:
ensemble = epistemic over weights, MC = epistemic over residual mask.

Output directory
----------------
results/exp1_loro_v2_YYYYMMDD_HHMMSS/
  plots/           — Figs 1–4
  parameters/      — CSV summary tables
  predictions/     — per-fold ensemble prediction CSV
  model_last_seed_<ratio>.pt  — analysis checkpoint for primary ratio
"""
#%% Imports & Setup
from __future__ import annotations

import gc
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.interpolate import interp1d

try:
    _THIS_FILE = Path(__file__).resolve()
except NameError:
    _THIS_FILE = Path().resolve() / "experiments" / "exp1_ratio_loro.py"

REPO_ROOT = _THIS_FILE.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core import (
    DEVICE, N_SEEDS, SEEDS, R_GAS, T_REF, MC_SAMPLES,
    ChemicalProcessor, preprocess_global_dataframe,
    UnifiedKineticsDataset, make_loader,
    Unified_hPINN_v2,
    set_global_seed, curriculum_train,
    evaluate, predict_with_uncertainty,
    extract_mixture_params, make_rate_fn,
    run_isothermal_report, run_dynamic_simulation,
)

# ---------------------------------------------------------------------------
# Paths & run configuration
# ---------------------------------------------------------------------------
DATA_DIR  = REPO_ROOT / "data"
CSV_PATH  = DATA_DIR / "Unified_Kinetics_Dataset_MR_v1.csv"

TIMESTAMP  = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR    = REPO_ROOT / "results" / f"exp1_loro_v2_{TIMESTAMP}"
PLOTS_DIR  = OUT_DIR / "plots"
PARAMS_DIR = OUT_DIR / "parameters"
PREDS_DIR  = OUT_DIR / "predictions"
for d in [OUT_DIR, PLOTS_DIR, PARAMS_DIR, PREDS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print(f"REPO_ROOT : {REPO_ROOT}")
print(f"CSV       : {CSV_PATH}  (exists={CSV_PATH.exists()})")
print(f"DEVICE    : {DEVICE}")

# Primary ratio to analyse in detail (change to match your dataset)
LORO_TARGET_RATIO: float = 0.5

# Dynamic heating rate used for physical-variance comparison
PHYS_VAR_BETA: float = 1.0   # K/min

# Neighbour map for physical-variance calculation  (ratio → adjacent ratios)
RATIO_NEIGHBOURS: dict[float, list[float]] = {
    0.0: [0.2],
    0.2: [0.0, 0.4],
    0.4: [0.2, 0.5],
    0.5: [0.4, 0.7],
    0.7: [0.5, 0.9],
    0.9: [0.7, 1.0],
    1.0: [0.9],
}

# ---------------------------------------------------------------------------
# Plot style helpers
# ---------------------------------------------------------------------------
for _style in ["seaborn-v0_8-paper", "seaborn-paper", "seaborn"]:
    try:
        plt.style.use(_style); break
    except OSError:
        continue
plt.rcParams.update({
    "text.usetex": False, "font.family": "serif",
    "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 12,
    "legend.fontsize": 9, "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 300,
})
_CB = dict(blue="#0072B2", orange="#E69F00", green="#009E73",
           red="#D55E00", purple="#CC79A7", cyan="#56B4E9",
           grey="#999999", black="#000000")


# ---------------------------------------------------------------------------
# V2 helper — decode raw physics-head row averages into physical units
# ---------------------------------------------------------------------------
def _summarise_pX(px_mean: np.ndarray) -> dict[str, float]:
    """
    Convert a single mean of the raw physics-head output (length-8 vector)
    into a dict of decoded physical quantities + effective Arrhenius
    equivalents (so KCE-style plots remain comparable with v1).
    """
    q = Unified_hPINN_v2.decode_params(
        torch.tensor(px_mean, dtype=torch.float32).unsqueeze(0))
    dH_J  = float(q["dH_J"]  [0, 0].item())
    dS_SI = float(q["dS_SI"] [0, 0].item())
    return {
        "dH_kJ":   dH_J / 1000.0,
        "dS_SI":   dS_SI,
        "Tg0_K":   float(q["Tg0_K"][0, 0].item()),
        "dTg_K":   float(q["dTg_K"][0, 0].item()),
        "lam":     float(q["lam"]  [0, 0].item()),
        "K_D":     float(q["K_D"]  [0, 0].item()),
        "m":       float(q["m"]    [0, 0].item()),
        "n":       float(q["n"]    [0, 0].item()),
        "Ea_eff_kJ": (dH_J + R_GAS * T_REF) / 1000.0,
        "lnA_eff":  (np.log(1.380649e-23 / 6.62607015e-34)
                     + np.log(60.0) + np.log(T_REF) + 1.0
                     + dS_SI / R_GAS),
    }


# =============================================================================
def main() -> None:
    # ── 1. Data loading ───────────────────────────────────────────────────────
    print("\n[EXP1-LORO V2] Loading dataset ...")
    if CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH).reset_index(drop=True)
        print(f"  Loaded real CSV: {CSV_PATH}")
    else:
        print(f"  CSV not found — generating synthetic ratio dataset ...")
        from data.dummy_generator import make_ratio_dataset
        df = make_ratio_dataset()
        df.to_csv(DATA_DIR / "dummy_ratio_dataset.csv", index=False)

    # ── 2. Global feature engineering ────────────────────────────────────────
    print("[EXP1-LORO V2] Global feature engineering ...")
    df = preprocess_global_dataframe(df).reset_index(drop=True)
    T_MIN_K = float(df["Temp_K"].min())
    T_MAX_K = float(df["Temp_K"].max())
    print(f"  Rows: {len(df)}  T_K [{T_MIN_K:.1f}, {T_MAX_K:.1f}]")

    # ── 3. ChemicalProcessor ─────────────────────────────────────────────────
    all_smiles = list(set(
        df["SMILES_E1"].tolist() + df["SMILES_X1"].tolist() +
        df["SMILES_Y1"].tolist()))
    processor  = ChemicalProcessor(all_smiles)
    print(f"  Vocab size: {processor.vocab_size}  "
          f"max SMILES len: {processor.max_len}")

    # ── 4. LORO loop ──────────────────────────────────────────────────────────
    unique_ratios = sorted(df["Ratio_X1"].unique())
    print(f"\n[EXP1-LORO V2] Starting LORO  "
          f"(N_SEEDS={N_SEEDS}, target={LORO_TARGET_RATIO})")

    ensemble_store: dict[float, dict] = {}
    loro_results:   list[dict]        = []

    with open(OUT_DIR / "config.json", "w") as f:
        json.dump(dict(experiment="exp1_ratio_loro_v2",
                       LORO_TARGET_RATIO=LORO_TARGET_RATIO,
                       unique_ratios=unique_ratios,
                       N_SEEDS=N_SEEDS, SEEDS=SEEDS,
                       MC_SAMPLES=MC_SAMPLES,
                       T_MIN_K=T_MIN_K, T_MAX_K=T_MAX_K,
                       TIMESTAMP=TIMESTAMP), f, indent=2)

    for ratio in unique_ratios:
        is_primary = (abs(ratio - LORO_TARGET_RATIO) < 1e-6)
        tag = " [*PRIMARY*]" if is_primary else ""
        print(f"\n{'='*60}")
        print(f"[LORO] Left-out ratio: {ratio:.3f}{tag}")

        df_train = df[df["Ratio_X1"] != ratio].reset_index(drop=True)
        df_test  = df[df["Ratio_X1"] == ratio].reset_index(drop=True)

        store: dict = {
            "ln_preds_phys": [], "ln_preds_full": [], "ln_tgts": None,
            "dH_kJ": [], "dS_SI": [], "Ea_eff_kJ": [], "lnA_eff": [],
            "Tg0_K": [], "dTg_K": [], "lam": [], "K_D": [],
            "m": [], "n": [], "df_test": df_test,
        }

        for seed_idx, seed in enumerate(SEEDS):
            print(f"  ── Seed {seed_idx+1}/{N_SEEDS}  (seed={seed}) ──")
            set_global_seed(seed)

            train_loader = make_loader(
                UnifiedKineticsDataset(df_train, processor),
                shuffle=True, seed=seed)
            test_loader  = make_loader(
                UnifiedKineticsDataset(df_test, processor),
                shuffle=False, seed=seed)

            model = Unified_hPINN_v2(processor.vocab_size).to(DEVICE)
            curriculum_train(model, train_loader,
                             verbose=(is_primary and seed_idx == 0))

            m_phys = evaluate(model, test_loader, use_pure_physics=True)
            m_full = evaluate(model, test_loader, use_pure_physics=False)

            store["ln_preds_phys"].append(m_phys["preds"])
            store["ln_preds_full"].append(m_full["preds"])
            if store["ln_tgts"] is None:
                store["ln_tgts"] = m_phys["targets"]

            # Decode bounded physical quantities (V2 8-element head)
            summary = _summarise_pX(m_phys["params_X"].mean(axis=0))
            for k, v in summary.items():
                store[k].append(v)

            print(f"     PurePhys R2={m_phys['r2']:.4f}  "
                  f"RMSE={m_phys['rmse']:.4f}  "
                  f"Full RMSE={m_full['rmse']:.4f}  "
                  f"ΔH={summary['dH_kJ']:.1f} kJ/mol  "
                  f"ΔS={summary['dS_SI']:.0f} J/(mol·K)  "
                  f"Tg0={summary['Tg0_K']:.0f}K")

            # Save checkpoint for primary ratio (last seed only)
            if is_primary and seed_idx == len(SEEDS) - 1:
                ckpt = OUT_DIR / f"model_last_seed_ratio{int(ratio*100)}.pt"
                torch.save(model.state_dict(), ckpt)
                print(f"  Saved analysis checkpoint: {ckpt.name}")

            del model
            gc.collect()

        # ── Ensemble statistics ───────────────────────────────────────────
        preds_phys = np.array(store["ln_preds_phys"])
        preds_full = np.array(store["ln_preds_full"])
        ln_tgts    = store["ln_tgts"]

        mean_ln_phys = preds_phys.mean(axis=0)
        ci95_ln_phys = 1.96 * preds_phys.std(axis=0, ddof=1)
        mean_ln_full = preds_full.mean(axis=0)
        ci95_ln_full = 1.96 * preds_full.std(axis=0, ddof=1)

        rmse_seeds = np.array([np.sqrt(np.mean((p - ln_tgts)**2))
                                for p in store["ln_preds_phys"]])
        mean_rmse  = float(rmse_seeds.mean())
        ci95_rmse  = float(1.96 * rmse_seeds.std(ddof=1))

        ss_res  = np.sum((ln_tgts - mean_ln_phys) ** 2)
        ss_tot  = np.sum((ln_tgts - ln_tgts.mean()) ** 2)
        mean_r2 = float(1.0 - ss_res / (ss_tot + 1e-10))

        # Aggregate decoded params with CI95
        def _agg(name):
            arr = np.array(store[name])
            return float(arr.mean()), float(1.96 * arr.std(ddof=1))

        m_dH,  c_dH  = _agg("dH_kJ")
        m_dS,  c_dS  = _agg("dS_SI")
        m_Ea,  c_Ea  = _agg("Ea_eff_kJ")
        m_lnA, c_lnA = _agg("lnA_eff")
        m_Tg0, c_Tg0 = _agg("Tg0_K")
        m_dTg, c_dTg = _agg("dTg_K")
        m_lam, c_lam = _agg("lam")
        m_KD,  c_KD  = _agg("K_D")

        # Save prediction CSV
        pred_df = df_test[["Sample_ID", "Condition_Type", "Condition_Val",
                            "Temp_K", "Alpha", "Rate_1_min"]].copy()
        pred_df["ln_Rate_Exp"]           = ln_tgts
        pred_df["ln_Rate_PurePhys_Mean"] = mean_ln_phys
        pred_df["ln_Rate_PurePhys_lo"]   = mean_ln_phys - ci95_ln_phys
        pred_df["ln_Rate_PurePhys_hi"]   = mean_ln_phys + ci95_ln_phys
        pred_df["Rate_PurePhys_Mean"]    = np.exp(mean_ln_phys)
        pred_df["Rate_PurePhys_CI95_lo"] = np.exp(mean_ln_phys - ci95_ln_phys)
        pred_df["Rate_PurePhys_CI95_hi"] = np.exp(mean_ln_phys + ci95_ln_phys)
        pred_df["Rate_Full_Mean"]        = np.exp(mean_ln_full)
        pred_df.to_csv(PREDS_DIR / f"LORO_ratio{int(ratio*100)}_ensemble.csv",
                       index=False)

        # Physical variance vs. neighbouring ratios
        cond_mask = ((df["Condition_Type"] == "Dyn") &
                     (df["Condition_Val"]  == PHYS_VAR_BETA))
        tgt_data  = df[(df["Ratio_X1"] == ratio) & cond_mask].sort_values("Temp_K")
        nb_diffs: list[float] = []
        for nb_r in RATIO_NEIGHBOURS.get(round(ratio, 6), []):
            nb_data = df[(df["Ratio_X1"] == nb_r) & cond_mask].sort_values("Temp_K")
            if tgt_data.empty or nb_data.empty:
                continue
            t_lo = max(tgt_data["Temp_K"].min(), nb_data["Temp_K"].min())
            t_hi = min(tgt_data["Temp_K"].max(), nb_data["Temp_K"].max())
            if t_lo >= t_hi:
                continue
            tg  = np.linspace(t_lo, t_hi, 100)
            f_t = interp1d(tgt_data["Temp_K"], tgt_data["Rate_1_min"],
                           bounds_error=False, fill_value="extrapolate")
            f_n = interp1d(nb_data["Temp_K"], nb_data["Rate_1_min"],
                           bounds_error=False, fill_value="extrapolate")
            nb_diffs.append(float(np.sqrt(np.mean(
                (np.log(f_t(tg) + 1e-10) - np.log(f_n(tg) + 1e-10)) ** 2))))
        phys_var = float(np.mean(nb_diffs)) if nb_diffs else float("nan")

        store.update(dict(
            pred_df=pred_df,
            mean_ln_phys=mean_ln_phys, ci95_ln_phys=ci95_ln_phys,
            mean_ln_full=mean_ln_full, ln_tgts=ln_tgts,
        ))
        ensemble_store[ratio] = store

        loro_results.append({
            "Ratio_X1":            ratio,
            "LORO_R2_mean":        mean_r2,
            "LORO_RMSE_mean":      mean_rmse,
            "LORO_RMSE_CI95":      ci95_rmse,
            "LORO_RMSE_worst":     mean_rmse + ci95_rmse,
            "dH_kJ_mean":          m_dH,   "dH_kJ_CI95":   c_dH,
            "dS_SI_mean":          m_dS,   "dS_SI_CI95":   c_dS,
            "Ea_eff_kJ_mean":      m_Ea,   "Ea_eff_kJ_CI95": c_Ea,
            "lnA_eff_mean":        m_lnA,  "lnA_eff_CI95": c_lnA,
            "Tg0_K_mean":          m_Tg0,  "Tg0_K_CI95":   c_Tg0,
            "dTg_K_mean":          m_dTg,  "dTg_K_CI95":   c_dTg,
            "lam_mean":            m_lam,  "lam_CI95":     c_lam,
            "K_D_mean":            m_KD,   "K_D_CI95":     c_KD,
            "Neighbour_Diff_ln":   phys_var,
        })

        print(f"\n  [rX={ratio:.2f}] Ensemble (pure physics):")
        print(f"    R²   = {mean_r2:.4f}")
        print(f"    RMSE = {mean_rmse:.4f} ± {ci95_rmse:.4f}  "
              f"(worst={mean_rmse+ci95_rmse:.4f})")
        print(f"    ΔH‡  = {m_dH:.1f} ± {c_dH:.1f} kJ/mol  "
              f"ΔS‡ = {m_dS:.0f} ± {c_dS:.0f} J/(mol·K)")
        print(f"    Tg0  = {m_Tg0:.0f} ± {c_Tg0:.0f} K   "
              f"λ = {m_lam:.3f} ± {c_lam:.3f}")
        print(f"    PhysVar = {phys_var:.4f}")

    results_df = (pd.DataFrame(loro_results)
                  .sort_values("Ratio_X1").reset_index(drop=True))
    results_df.to_csv(PARAMS_DIR / "LORO_Results_exp1_v2.csv", index=False)
    print("\n[EXP1-LORO V2] All LORO folds complete.")

    # ── 5. Publication plots ──────────────────────────────────────────────────
    print("\n[PLOTS] Generating publication figures ...")

    # Fig 1 — Pure-physics prediction curves + CI for primary ratio
    tgt_store = ensemble_store[LORO_TARGET_RATIO]
    pred_df   = tgt_store["pred_df"]
    dyn_conds = sorted(
        pred_df[pred_df["Condition_Type"] == "Dyn"]["Condition_Val"].unique())
    n_dyn     = max(len(dyn_conds), 1)
    pal       = [_CB["blue"], _CB["orange"], _CB["green"],
                 _CB["red"],  _CB["purple"], _CB["cyan"]]

    fig1, axes1 = plt.subplots(1, n_dyn, figsize=(5.5 * n_dyn, 4.8))
    if n_dyn == 1:
        axes1 = [axes1]
    for ax, beta_val in zip(axes1, dyn_conds):
        mask = ((pred_df["Condition_Type"] == "Dyn") &
                (pred_df["Condition_Val"]   == beta_val))
        grp  = pred_df[mask].sort_values("Temp_K")
        col  = pal[dyn_conds.index(beta_val) % len(pal)]
        ax.scatter(grp["Temp_K"], grp["Rate_1_min"],
                   color=col, alpha=0.55, s=22, zorder=3, label="Experimental")
        ax.plot(grp["Temp_K"], grp["Rate_PurePhys_Mean"],
                color=col, lw=2.0, zorder=4, label="Pure Physics mean (V2)")
        ax.fill_between(grp["Temp_K"],
                        grp["Rate_PurePhys_CI95_lo"],
                        grp["Rate_PurePhys_CI95_hi"],
                        color=col, alpha=0.20, zorder=2,
                        label="95% CI (seed ensemble)")
        ax.plot(grp["Temp_K"], grp["Rate_Full_Mean"],
                color=col, lw=1.2, ls="--", alpha=0.55, label="Full model mean")
        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel("d\u03b1/dt (min\u207b\u00b9)")
        ax.set_title(f"LORO rX={LORO_TARGET_RATIO}  |  \u03b2={beta_val:g} K/min",
                     fontweight="bold")
        ax.legend(loc="upper left", framealpha=0.9)
    fig1.suptitle(f"Exp-1 LORO (V2)  —  Pure-Physics Prediction  "
                  f"rX={LORO_TARGET_RATIO}  (N_seeds={N_SEEDS})",
                  fontsize=13, fontweight="bold", y=1.01)
    fig1.tight_layout()
    p = PLOTS_DIR / "Fig1_LORO_PurePhys_PredictionCurves.png"
    fig1.savefig(p, dpi=300, bbox_inches="tight"); plt.close(fig1)
    print(f"  Saved: {p.name}")

    # Fig 2 — ΔH‡ and ΔS‡ vs Ratio_X1   (V2 plot — was Ea / lnA in v1)
    r_vals    = results_df["Ratio_X1"].values.astype(float)
    dh_means  = results_df["dH_kJ_mean"].values
    dh_ci95   = results_df["dH_kJ_CI95"].values
    ds_means  = results_df["dS_SI_mean"].values
    ds_ci95   = results_df["dS_SI_CI95"].values
    is_tgt    = np.array([abs(r - LORO_TARGET_RATIO) < 1e-6 for r in r_vals])

    fig2, ax2s = plt.subplots(1, 2, figsize=(13, 5))
    for ax, y_mean, y_ci, ylabel, color, lbl in [
        (ax2s[0], dh_means, dh_ci95,
         "ΔH‡  (kJ/mol)",            _CB["red"],  "ΔH‡"),
        (ax2s[1], ds_means, ds_ci95,
         "ΔS‡  (J/(mol·K))",         _CB["blue"], "ΔS‡"),
    ]:
        ax.errorbar(r_vals, y_mean, yerr=y_ci, fmt="o", color=color,
                    elinewidth=1.8, capsize=5, ms=9, mfc="white", mew=2.2,
                    zorder=4, label=f"{lbl} ±95%CI")
        ax.scatter(r_vals[is_tgt], y_mean[is_tgt], color=color,
                   s=120, marker="*", zorder=5,
                   label=f"rX={LORO_TARGET_RATIO} (zero-shot)")
        sl, ic, rv, *_ = linregress(r_vals, y_mean)
        cx = np.linspace(r_vals.min() - 0.05, r_vals.max() + 0.05, 200)
        ax.plot(cx, sl * cx + ic, "--", color=color, alpha=0.55,
                label=f"R={rv:.3f}")
        ax.set_xlabel("Ratio X1")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{lbl} vs Composition", fontweight="bold")
        ax.legend(fontsize=8)
    fig2.suptitle("Eyring activation parameters across compositions (V2)",
                  fontsize=13, fontweight="bold", y=1.02)
    fig2.tight_layout()
    p = PLOTS_DIR / "Fig2_dH_dS_vs_Ratio.png"
    fig2.savefig(p, dpi=300, bbox_inches="tight"); plt.close(fig2)
    print(f"  Saved: {p.name}")

    # Fig 3 — Generalisation proof bar chart (unchanged structure)
    x_pos = np.arange(len(results_df)); w = 0.38
    fig3, ax3 = plt.subplots(figsize=(11, 5))
    ax3.bar(x_pos - w/2, results_df["LORO_RMSE_mean"], w,
            label="Pure-Physics RMSE (zero-shot, V2)",
            color=_CB["blue"], edgecolor="black", alpha=0.88, zorder=3)
    ax3.errorbar(x_pos - w/2, results_df["LORO_RMSE_mean"],
                 yerr=results_df["LORO_RMSE_CI95"],
                 fmt="none", ecolor="black", elinewidth=1.6,
                 capsize=5, capthick=1.6, zorder=4)
    ax3.bar(x_pos + w/2, results_df["Neighbour_Diff_ln"], w,
            label="Exp. neighbour variance",
            color=_CB["orange"], edgecolor="black", alpha=0.85, zorder=3)
    tgt_row = results_df[
        np.array([abs(r - LORO_TARGET_RATIO) < 1e-6
                  for r in results_df["Ratio_X1"]])].iloc[0]
    ax3.axhline(tgt_row["LORO_RMSE_worst"], color=_CB["red"], lw=1.5, ls="--",
                label=f"rX={LORO_TARGET_RATIO} worst={tgt_row['LORO_RMSE_worst']:.3f}")
    for xi, row in zip(x_pos, results_df.itertuples()):
        ax3.text(xi - w/2, row.LORO_RMSE_mean + row.LORO_RMSE_CI95 + 0.005,
                 f"R\u00b2={row.LORO_R2_mean:.3f}", ha="center",
                 fontsize=7.5, color=_CB["blue"], fontweight="bold")
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([f"rX={r:.2f}" for r in results_df["Ratio_X1"]],
                        fontsize=9, rotation=15, ha="right")
    ax3.set_ylabel("RMSE  (ln-rate space)")
    ax3.set_title("Exp-1 LORO (V2): Pure-Physics Generalisation vs Physical Variance",
                  fontweight="bold")
    ax3.legend(fontsize=8.5)
    fig3.tight_layout()
    p = PLOTS_DIR / "Fig3_LORO_Generalisation.png"
    fig3.savefig(p, dpi=300, bbox_inches="tight"); plt.close(fig3)
    print(f"  Saved: {p.name}")

    # ── 6. Post-hoc analysis for primary ratio (V2: + MC-Dropout band) ───────
    ckpt_path = OUT_DIR / f"model_last_seed_ratio{int(LORO_TARGET_RATIO*100)}.pt"
    if ckpt_path.exists():
        print(f"\n[EXP1] Post-hoc analysis for rX={LORO_TARGET_RATIO} ...")
        infer_model = Unified_hPINN_v2(processor.vocab_size).to(DEVICE)
        infer_model.load_state_dict(
            torch.load(ckpt_path, map_location=DEVICE, weights_only=True))

        # MC-Dropout uncertainty over the test loader
        test_loader = make_loader(
            UnifiedKineticsDataset(tgt_store["df_test"], processor),
            shuffle=False, seed=0)
        mc = predict_with_uncertainty(infer_model, test_loader,
                                       n_samples=MC_SAMPLES)
        print(f"  MC-Dropout band:  mean(σ_ln) = {mc['ln_std'].mean():.4f}  "
              f"(over {len(mc['ln_mean'])} samples, N={MC_SAMPLES})")

        # Append MC band to the prediction CSV
        mc_df = tgt_store["df_test"][[
            "Sample_ID", "Condition_Type", "Condition_Val",
            "Temp_K", "Alpha"]].copy()
        mc_df["ln_Rate_MC_Mean"] = mc["ln_mean"]
        mc_df["ln_Rate_MC_Std"]  = mc["ln_std"]
        mc_df["Rate_MC_CI95_lo"] = np.exp(mc["ci95_lo"])
        mc_df["Rate_MC_CI95_hi"] = np.exp(mc["ci95_hi"])
        mc_df.to_csv(PREDS_DIR
                     / f"LORO_ratio{int(LORO_TARGET_RATIO*100)}_MCdropout.csv",
                     index=False)

        # Decoded params for the post-hoc rate function
        rep_row = tgt_store["df_test"].iloc[0]
        params  = extract_mixture_params(
            infer_model,
            rep_row["SMILES_E1"], rep_row["SMILES_X1"], rep_row["SMILES_Y1"],
            processor, T_min_K=T_MIN_K, T_max_K=T_MAX_K)

        rX_val  = float(rep_row["Ratio_X1"])
        rY_val  = float(rep_row["Ratio_Y1"])
        rate_fn = make_rate_fn(params, rX_val, rY_val)

        print(f"  X-component:  ΔH‡={params['dH_X_J']/1000:.1f} kJ/mol  "
              f"ΔS‡={params['dS_X_SI']:.0f} J/(mol·K)  "
              f"Tg0={params['Tg0_X_K']:.0f}K  λ={params['lam_X']:.2f}")
        print(f"  Y-component:  ΔH‡={params['dH_Y_J']/1000:.1f} kJ/mol  "
              f"ΔS‡={params['dS_Y_SI']:.0f} J/(mol·K)  "
              f"Tg0={params['Tg0_Y_K']:.0f}K  λ={params['lam_Y']:.2f}")
        print(f"  Synergy syn(T_REF, α=0.5) = "
              f"{params['_syn_fn'](T_REF, 0.5):.4f}")

        # Isothermal report
        iso_df = run_isothermal_report(tgt_store["df_test"], rate_fn)
        W = 90
        print(f"\n  {'-'*W}")
        print(f"  {'Temp (°C)':<12}| {'Target Alpha':<16}| "
              f"{'Exp Time (min)':<16}| {'Pred Time (min)':<17}| "
              f"{'Error (min)':<14}| {'Rel Error (%)'}")
        print(f"  {'-'*W}")
        for _, r in iso_df.iterrows():
            print(f"  {r['Temp_C']:<12.1f}| {r['Target_Alpha']:<16.3f}| "
                  f"{r['Exp_Time_min']:<16.2f}| {r['Pred_Time_min']:<17.2f}| "
                  f"{r['Error_min']:<14.2f}| {r['Rel_Error_pct']:.2f}     %")
        print(f"  {'-'*W}")
        if not iso_df.empty:
            mape = float(iso_df["Rel_Error_pct"].mean())
            print(f"  Overall Model MAPE: {mape:.2f}%")
            iso_df.to_csv(
                PARAMS_DIR / f"Isothermal_Report_rX{int(LORO_TARGET_RATIO*100)}.csv",
                index=False)

        # Fig 4 — Dynamic validation (now with MC band overlay)
        try:
            sim_T_C, sim_alpha, sim_rate, used_beta = run_dynamic_simulation(
                tgt_store["df_test"], rate_fn, target_beta=PHYS_VAR_BETA)

            df_dyn_exp = tgt_store["df_test"][
                (tgt_store["df_test"]["Condition_Type"] == "Dyn") &
                (tgt_store["df_test"]["Condition_Val"]  == used_beta)
            ].sort_values("Temp_K")

            plt.close("all")
            fig4, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
            a1.plot(df_dyn_exp["Temp_K"] - 273.15, df_dyn_exp["Alpha"],
                    "o", color="#95a5a6", alpha=0.6, markersize=5,
                    label=f"Experimental ({used_beta}\u00b0C/min)")
            a1.plot(sim_T_C, sim_alpha, "-", color="#e74c3c", lw=3,
                    label="V2 Synergistic Model")
            a1.set_xlabel("Temperature (\u00b0C)", fontweight="bold")
            a1.set_ylabel("Degree of Cure (\u03b1)", fontweight="bold")
            a1.set_title("Dynamic Conversion Profile", fontweight="bold")
            a1.legend(); a1.grid(True, ls="--", alpha=0.6)

            # MC-Dropout posterior on the experimental Temp grid
            dyn_idx = df_dyn_exp.index.values
            # Build a flat-index → mc_array_index map (mc arrays are loader-ordered)
            test_df_full = tgt_store["df_test"].reset_index(drop=True)
            dyn_pos = test_df_full.index[
                (test_df_full["Condition_Type"] == "Dyn") &
                (test_df_full["Condition_Val"]  == used_beta)
            ].values
            order = np.argsort(test_df_full.loc[dyn_pos, "Temp_K"].values)
            dyn_pos = dyn_pos[order]

            a2.plot(df_dyn_exp["Temp_K"] - 273.15, df_dyn_exp["Rate_1_min"],
                    "o", color="#95a5a6", alpha=0.6, markersize=5,
                    label="Experimental Rate")
            a2.plot(sim_T_C, sim_rate, "-", color="#2ecc71", lw=3,
                    label="V2 Model Predicted Rate")
            a2.fill_between(df_dyn_exp["Temp_K"] - 273.15,
                            np.exp(mc["ci95_lo"][dyn_pos]),
                            np.exp(mc["ci95_hi"][dyn_pos]),
                            color="#2ecc71", alpha=0.20,
                            label="MC-Dropout 95% CI")
            a2.set_xlabel("Temperature (\u00b0C)", fontweight="bold")
            a2.set_ylabel("d\u03b1/dt", fontweight="bold")
            a2.set_title("Kinetics Phase Diagram + Bayesian residual band",
                         fontweight="bold")
            a2.legend(); a2.grid(True, ls="--", alpha=0.6)

            fig4.suptitle(
                f"Exp-1 LORO (V2)  —  rX={LORO_TARGET_RATIO}  "
                f"Dynamic Validation (\u03b2={used_beta}\u00b0C/min)",
                fontsize=13, fontweight="bold", y=1.01)
            fig4.tight_layout()
            p = PLOTS_DIR / f"Fig4_Dynamic_Validation_rX{int(LORO_TARGET_RATIO*100)}.png"
            fig4.savefig(p, dpi=300, bbox_inches="tight"); plt.close(fig4)
            print(f"\n  Saved: {p.name}")

            rmse_a = float(np.sqrt(np.mean(
                (df_dyn_exp["Alpha"].values - sim_alpha) ** 2)))
            print(f"  Dynamic RMSE(\u03b1) = {rmse_a:.4f}")
        except Exception as exc:
            print(f"  [WARN] Dynamic simulation failed: {exc}")

        # Closure (`params['_syn_fn']`) keeps a strong reference to
        # `infer_model`, so we delete it last to avoid dangling refs.
        del rate_fn, params, infer_model
        gc.collect()
    else:
        print(f"\n[EXP1] Checkpoint not found — skipping post-hoc analysis.")

    # ── 7. Summary ────────────────────────────────────────────────────────────
    print(f"""
{'='*64}
  Experiment 1 — LORO  V2  Complete
  Output: {OUT_DIR}
  Ratios tested : {unique_ratios}
  Primary ratio : {LORO_TARGET_RATIO}

  V2 features active:
    [1] Eyring TST physics head           (ΔH‡, ΔS‡)
    [2] Dynamic synergy syn(T, α)
    [3] DiBenedetto + WLF diffusion gate  (Tg0, ΔTg, λ, K_D)
    [4] MC-Dropout Bayesian residuals     (N={MC_SAMPLES})

  Figures:
    Fig1 — Pure-physics prediction CI
    Fig2 — ΔH‡ / ΔS‡ vs composition  (replaces v1 Ea/lnA)
    Fig3 — Generalisation vs physical variance
    Fig4 — Dynamic validation + MC-Dropout band
{'='*64}""")


#%% Run Experiment  (uncomment for interactive cell execution)
# main()

# =============================================================================
if __name__ == "__main__":
    main()
