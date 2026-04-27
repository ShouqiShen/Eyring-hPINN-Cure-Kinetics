# -*- coding: utf-8 -*-
"""
experiments/exp1_ratio_loro.py
==============================
Experiment 1 — Leave-One-Ratio-Out (LORO) Validation

Scientific question
-------------------
Can the hPINN predict cure kinetics for a **mixture composition it has never
seen** (e.g. rX = 0.50) by interpolating / extrapolating from neighbouring
ratio points?

Loop design
-----------
For every unique Ratio_X1 value in the dataset:
  • Leave out ALL rows where Ratio_X1 == target_ratio
  • Train on the remaining compositions (N_SEEDS ensembles)
  • Evaluate with PURE PHYSICS (zero-shot, no residuals) on the test set
  • Record RMSE, R², Ea, lnA

Post-hoc analysis for the primary target ratio
  • Isothermal quantitative report  (Table: Temp, target α, exp/pred time)
  • Dynamic validation plot          (α vs T and rate vs T with RK45)

Output directory
----------------
results/exp1_loro_YYYYMMDD_HHMMSS/
  plots/           — Figs 1–4
  parameters/      — CSV summary tables
  predictions/     — per-fold ensemble prediction CSV
  model_last_seed_<ratio>.pt  — analysis checkpoint for primary ratio
"""
#%% Imports & Setup
from __future__ import annotations

import gc
import io
import json
import sys
import contextlib
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.interpolate import interp1d

# ── Repo-root: works both as a script and as an interactive #%% cell ────────
try:
    _THIS_FILE = Path(__file__).resolve()
except NameError:
    _THIS_FILE = Path().resolve() / "experiments" / "exp1_ratio_loro.py"

REPO_ROOT = _THIS_FILE.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core import (
    DEVICE, N_SEEDS, SEEDS,
    ChemicalProcessor, preprocess_global_dataframe,
    UnifiedKineticsDataset, make_loader,
    Unified_hPINN,
    set_global_seed, curriculum_train,
    evaluate, extract_mixture_params, make_rate_fn,
    run_isothermal_report, run_dynamic_simulation,
)

# ---------------------------------------------------------------------------
# Paths & run configuration
# ---------------------------------------------------------------------------
DATA_DIR   = REPO_ROOT / "data"
CSV_PATH   = DATA_DIR / "Unified_Kinetics_Dataset_MR_v1.csv"

TIMESTAMP  = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR    = REPO_ROOT / "results" / f"exp1_loro_{TIMESTAMP}"
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

# Dynamic heating rate used for physical-variance comparison (must exist in data)
PHYS_VAR_BETA: float = 1.0   # K/min  (only dynamic rate in MR_v1 dataset)

# Neighbour map for physical-variance calculation  (ratio → adjacent ratios)
# Updated to cover all ratios present in the real dataset: 0.0, 0.2, 0.4, 0.5, 0.7, 0.9, 1.0
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


# =============================================================================
def main() -> None:
    # ── 1. Data loading ───────────────────────────────────────────────────────
    print("\n[EXP1-LORO] Loading dataset ...")
    if CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH).reset_index(drop=True)
        print(f"  Loaded real CSV: {CSV_PATH}")
    else:
        print(f"  CSV not found — generating synthetic ratio dataset ...")
        from data.dummy_generator import make_ratio_dataset
        df = make_ratio_dataset()
        df.to_csv(DATA_DIR / "dummy_ratio_dataset.csv", index=False)

    # ── 2. Global feature engineering (no per-fold leakage) ──────────────────
    print("[EXP1-LORO] Global feature engineering ...")
    df = preprocess_global_dataframe(df)
    print(f"  Rows: {len(df)}  "
          f"T_norm range: [{df['T_norm'].min():.3f}, {df['T_norm'].max():.3f}]")

    # ── 3. ChemicalProcessor ─────────────────────────────────────────────────
    all_smiles = list(set(
        df["SMILES_E1"].tolist() + df["SMILES_X1"].tolist() +
        df["SMILES_Y1"].tolist()))
    processor  = ChemicalProcessor(all_smiles)
    print(f"  Vocab size: {processor.vocab_size}  "
          f"max SMILES len: {processor.max_len}")

    # ── 4. LORO loop ──────────────────────────────────────────────────────────
    unique_ratios = sorted(df["Ratio_X1"].unique())
    print(f"\n[EXP1-LORO] Starting LORO  "
          f"(N_SEEDS={N_SEEDS}, target={LORO_TARGET_RATIO})")

    ensemble_store: dict[float, dict] = {}
    loro_results:   list[dict]        = []

    with open(OUT_DIR / "config.json", "w") as f:
        json.dump(dict(experiment="exp1_ratio_loro",
                       LORO_TARGET_RATIO=LORO_TARGET_RATIO,
                       unique_ratios=unique_ratios,
                       N_SEEDS=N_SEEDS, SEEDS=SEEDS,
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
            "Ea_kJ": [], "lnA": [], "lnK_ref": [], "m": [], "n": [],
            "df_test": df_test,
        }

        for seed_idx, seed in enumerate(SEEDS):
            print(f"  ── Seed {seed_idx+1}/{N_SEEDS}  (seed={seed}) ──")
            set_global_seed(seed)

            train_loader = make_loader(
                UnifiedKineticsDataset(df_train, processor),
                shuffle=True, seed=seed)
            test_loader  = make_loader(
                UnifiedKineticsDataset(df_test,  processor),
                shuffle=False, seed=seed)

            model = Unified_hPINN(processor.vocab_size).to(DEVICE)
            curriculum_train(model, train_loader,
                             verbose=(is_primary and seed_idx == 0))

            m_phys = evaluate(model, test_loader, use_pure_physics=True)
            m_full = evaluate(model, test_loader, use_pure_physics=False)

            store["ln_preds_phys"].append(m_phys["preds"])
            store["ln_preds_full"].append(m_full["preds"])
            if store["ln_tgts"] is None:
                store["ln_tgts"] = m_phys["targets"]

            px_mean = m_phys["params_X"].mean(axis=0)
            Ea_kJ   = float(F.softplus(torch.tensor(px_mean[1])).item() * 50.0)
            lnK_ref = float(px_mean[0] * 2.0 - 2.0)
            from core.config import R_GAS, T_REF
            lnA_val = float(lnK_ref + Ea_kJ * 1000.0 / (R_GAS * T_REF))
            store["Ea_kJ"].append(Ea_kJ)
            store["lnA"].append(lnA_val)
            store["lnK_ref"].append(lnK_ref)
            store["m"].append(float(abs(px_mean[2])))
            store["n"].append(float(abs(px_mean[3])))

            print(f"     PurePhys R2={m_phys['r2']:.4f}  "
                  f"RMSE={m_phys['rmse']:.4f}  "
                  f"Full RMSE={m_full['rmse']:.4f}  Ea={Ea_kJ:.1f} kJ/mol")

            # Save checkpoint for primary ratio
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

        mean_Ea  = float(np.mean(store["Ea_kJ"]))
        ci95_Ea  = float(1.96 * np.std(store["Ea_kJ"], ddof=1))
        mean_lnA = float(np.mean(store["lnA"]))
        ci95_lnA = float(1.96 * np.std(store["lnA"], ddof=1))

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

        # Physical variance vs. neighbouring ratio compositions
        cond_mask = (df["Condition_Type"] == "Dyn") & (df["Condition_Val"] == PHYS_VAR_BETA)
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
            f_n = interp1d(nb_data["Temp_K"],  nb_data["Rate_1_min"],
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
            "Ratio_X1":          ratio,
            "LORO_R2_mean":      mean_r2,
            "LORO_RMSE_mean":    mean_rmse,
            "LORO_RMSE_CI95":    ci95_rmse,
            "LORO_RMSE_worst":   mean_rmse + ci95_rmse,
            "Ea_kJ_mean":        mean_Ea,
            "Ea_kJ_CI95":        ci95_Ea,
            "lnA_mean":          mean_lnA,
            "lnA_CI95":          ci95_lnA,
            "Neighbour_Diff_ln": phys_var,
        })

        print(f"\n  [rX={ratio:.2f}] Ensemble (pure physics):")
        print(f"    R²   = {mean_r2:.4f}")
        print(f"    RMSE = {mean_rmse:.4f} ± {ci95_rmse:.4f}  "
              f"(worst={mean_rmse+ci95_rmse:.4f})")
        print(f"    Ea   = {mean_Ea:.1f} ± {ci95_Ea:.1f} kJ/mol")
        print(f"    PhysVar = {phys_var:.4f}")

    results_df = (pd.DataFrame(loro_results)
                  .sort_values("Ratio_X1").reset_index(drop=True))
    results_df.to_csv(PARAMS_DIR / "LORO_Results_exp1.csv", index=False)
    print("\n[EXP1-LORO] All LORO folds complete.")

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
                color=col, lw=2.0, zorder=4, label="Pure Physics mean")
        ax.fill_between(grp["Temp_K"],
                        grp["Rate_PurePhys_CI95_lo"],
                        grp["Rate_PurePhys_CI95_hi"],
                        color=col, alpha=0.20, zorder=2, label="95% CI")
        ax.plot(grp["Temp_K"], grp["Rate_Full_Mean"],
                color=col, lw=1.2, ls="--", alpha=0.55, label="Full model")
        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel("d\u03b1/dt (min\u207b\u00b9)")
        ax.set_title(f"LORO rX={LORO_TARGET_RATIO}  |  \u03b2={beta_val:g} K/min",
                     fontweight="bold")
        ax.legend(loc="upper left", framealpha=0.9)
    fig1.suptitle(f"Exp-1 LORO  —  Pure-Physics Prediction  "
                  f"rX={LORO_TARGET_RATIO}  (N_seeds={N_SEEDS})",
                  fontsize=13, fontweight="bold", y=1.01)
    fig1.tight_layout()
    p = PLOTS_DIR / "Fig1_LORO_PurePhys_PredictionCurves.png"
    fig1.savefig(p, dpi=300, bbox_inches="tight"); plt.close(fig1)
    print(f"  Saved: {p.name}")

    # Fig 2 — Ea and lnA vs Ratio_X1
    r_vals    = results_df["Ratio_X1"].values.astype(float)
    ea_means  = results_df["Ea_kJ_mean"].values
    ea_ci95   = results_df["Ea_kJ_CI95"].values
    lna_means = results_df["lnA_mean"].values
    lna_ci95  = results_df["lnA_CI95"].values
    is_tgt    = np.array([abs(r - LORO_TARGET_RATIO) < 1e-6 for r in r_vals])

    fig2, ax2s = plt.subplots(1, 2, figsize=(13, 5))
    for ax, y_mean, y_ci, ylabel, color, lbl in [
        (ax2s[0], ea_means,  ea_ci95,  "Ea (kJ/mol)",         _CB["red"],  "Ea"),
        (ax2s[1], lna_means, lna_ci95, "ln A (ln min\u207b\u00b9)", _CB["blue"], "lnA"),
    ]:
        ax.errorbar(r_vals, y_mean, yerr=y_ci, fmt="o", color=color,
                    elinewidth=1.8, capsize=5, ms=9, mfc="white", mew=2.2,
                    zorder=4, label=f"{lbl} \u00b195%CI")
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
    fig2.tight_layout()
    p = PLOTS_DIR / "Fig2_Ea_lnA_vs_Ratio.png"
    fig2.savefig(p, dpi=300, bbox_inches="tight"); plt.close(fig2)
    print(f"  Saved: {p.name}")

    # Fig 3 — Generalisation proof bar chart
    x_pos = np.arange(len(results_df)); w = 0.38
    fig3, ax3 = plt.subplots(figsize=(11, 5))
    ax3.bar(x_pos - w/2, results_df["LORO_RMSE_mean"], w,
            label="Pure-Physics RMSE (zero-shot)",
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
    ax3.set_title("Exp-1 LORO: Pure-Physics Generalisation vs Physical Variance",
                  fontweight="bold")
    ax3.legend(fontsize=8.5)
    fig3.tight_layout()
    p = PLOTS_DIR / "Fig3_LORO_Generalisation.png"
    fig3.savefig(p, dpi=300, bbox_inches="tight"); plt.close(fig3)
    print(f"  Saved: {p.name}")

    # ── 6. Post-hoc analysis for primary ratio ────────────────────────────────
    ckpt_path = OUT_DIR / f"model_last_seed_ratio{int(LORO_TARGET_RATIO*100)}.pt"
    if ckpt_path.exists():
        print(f"\n[EXP1] Post-hoc analysis for rX={LORO_TARGET_RATIO} ...")
        infer_model = Unified_hPINN(processor.vocab_size).to(DEVICE)
        infer_model.load_state_dict(
            torch.load(ckpt_path, map_location=DEVICE, weights_only=True))

        rep_row = tgt_store["df_test"].iloc[0]
        params  = extract_mixture_params(
            infer_model,
            rep_row["SMILES_E1"], rep_row["SMILES_X1"], rep_row["SMILES_Y1"],
            processor)
        del infer_model; gc.collect()

        rX_val  = float(rep_row["Ratio_X1"])
        rY_val  = float(rep_row["Ratio_Y1"])
        rate_fn = make_rate_fn(params, rX_val, rY_val)

        print(f"  lnKr_X={params['lnKr_X']:.3f}  Ea_X={params['Ea_X_J']/1000:.1f}"
              f"  m_X={params['m_X']:.3f}  n_X={params['n_X']:.3f}")
        print(f"  lnKr_Y={params['lnKr_Y']:.3f}  Ea_Y={params['Ea_Y_J']/1000:.1f}"
              f"  syn={params['syn_mult']:.4f}")

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

        # Dynamic validation + Fig 4
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
                    label="Synergistic Model")
            a1.set_xlabel("Temperature (\u00b0C)", fontweight="bold")
            a1.set_ylabel("Degree of Cure (\u03b1)", fontweight="bold")
            a1.set_title("Dynamic Conversion Profile", fontweight="bold")
            a1.legend(); a1.grid(True, ls="--", alpha=0.6)

            a2.plot(df_dyn_exp["Temp_K"] - 273.15, df_dyn_exp["Rate_1_min"],
                    "o", color="#95a5a6", alpha=0.6, markersize=5,
                    label="Experimental Rate")
            a2.plot(sim_T_C, sim_rate, "-", color="#2ecc71", lw=3,
                    label="Model Predicted Rate")
            a2.set_xlabel("Temperature (\u00b0C)", fontweight="bold")
            a2.set_ylabel("d\u03b1/dt", fontweight="bold")
            a2.set_title("Kinetics Phase Diagram", fontweight="bold")
            a2.legend(); a2.grid(True, ls="--", alpha=0.6)

            fig4.suptitle(
                f"Exp-1 LORO  —  rX={LORO_TARGET_RATIO}  "
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
    else:
        print(f"\n[EXP1] Checkpoint not found — skipping post-hoc analysis.")

    # ── 7. Summary ────────────────────────────────────────────────────────────
    print(f"""
{'='*64}
  Experiment 1 — LORO  Complete
  Output: {OUT_DIR}
  Ratios tested : {unique_ratios}
  Primary ratio : {LORO_TARGET_RATIO}
  Figures: Fig1 (prediction CI), Fig2 (Ea/lnA), Fig3 (generalisation),
           Fig4 (dynamic validation)
{'='*64}""")


#%% Run Experiment  (uncomment for interactive cell execution)
# main()

# =============================================================================
if __name__ == "__main__":
    main()
