# -*- coding: utf-8 -*-
"""
experiments/exp2_structure_lomo.py
===================================
Experiment 2 — Leave-One-Molecule-Out (LOMO) Validation  (Homologous Series)

Scientific question
-------------------
Can the hPINN predict cure kinetics for a **molecular structure it has never
seen** (e.g. the C10 diacid) by interpolating across the homologous series
(C6, C8, C12, C14)?

Loop design
-----------
For every unique Sample_ID in the dataset:
  • Leave out ALL rows where Sample_ID == target_mol
  • Train on the remaining structures (N_SEEDS ensembles)
  • Evaluate with PURE PHYSICS (zero-shot, no residuals) on the test set
  • Record RMSE, R², Ea, lnA per ensemble seed

Post-hoc analysis for the primary target molecule
  • Isothermal quantitative report  (Table: Temp, target α, exp/pred time)
  • Dynamic validation plot          (α vs T and rate vs T, RK45 integration)
  • Zero-shot proof report           (worst-case RMSE vs physical neighbour variance)
  • Ea / lnA vs chain-length plot    (KCE trend with uncertainty bars)

Output directory
----------------
results/exp2_lomo_YYYYMMDD_HHMMSS/
  plots/           — Figs 1–5
  parameters/      — CSV summary tables + proof report .txt
  predictions/     — per-fold ensemble prediction CSV
  model_last_seed_<mol>.pt  — analysis checkpoint for primary molecule
"""
#%% Imports & Setup
from __future__ import annotations

import contextlib
import gc
import io
import json
import sys
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
    _THIS_FILE = Path().resolve() / "experiments" / "exp2_structure_lomo.py"

REPO_ROOT = _THIS_FILE.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core import (
    DEVICE, N_SEEDS, SEEDS, R_GAS, T_REF,
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
CSV_PATH   = DATA_DIR / "Unified_Kinetics_Dataset_v1.csv"

print(f"REPO_ROOT : {REPO_ROOT}")
print(f"CSV       : {CSV_PATH}  (exists={CSV_PATH.exists()})")
print(f"DEVICE    : {DEVICE}")

TIMESTAMP  = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR    = REPO_ROOT / "results" / f"exp2_lomo_{TIMESTAMP}"
PLOTS_DIR  = OUT_DIR / "plots"
PARAMS_DIR = OUT_DIR / "parameters"
PREDS_DIR  = OUT_DIR / "predictions"
for d in [OUT_DIR, PLOTS_DIR, PARAMS_DIR, PREDS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Primary molecule to analyse in detail (leave this one out)
LOMO_TARGET: str = "C10"

# Neighbour map: which molecules are "adjacent" in the homologous series?
# Used to compute physical variance as a baseline for the zero-shot proof.
NEIGHBOURS: dict[str, list[str]] = {
    "C6":  ["C8"],
    "C8":  ["C6",  "C10"],
    "C10": ["C8",  "C12"],
    "C12": ["C10", "C14"],
    "C14": ["C12"],
}

# Dynamic condition used for physical-variance comparison (must exist in data)
PHYS_VAR_BETA: float = 10.0   # K/min

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
    print("\n[EXP2-LOMO] Loading dataset ...")
    if CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH).reset_index(drop=True)
        # Normalise column name: real data may use "Molecule" instead of "Sample_ID"
        if "Molecule" in df.columns and "Sample_ID" not in df.columns:
            df = df.rename(columns={"Molecule": "Sample_ID"})
        print(f"  Loaded real CSV: {CSV_PATH}")
    else:
        print(f"  CSV not found — generating synthetic structure dataset ...")
        from data.dummy_generator import make_structure_dataset
        df = make_structure_dataset()
        df.to_csv(DATA_DIR / "dummy_structure_dataset.csv", index=False)

    molecules = sorted(df["Sample_ID"].unique())
    print(f"  Rows: {len(df)},  Molecules: {molecules}")

    # ── 2. Global feature engineering (no per-fold leakage) ──────────────────
    print("[EXP2-LOMO] Global feature engineering ...")
    df = preprocess_global_dataframe(df)
    print(f"  T_norm [{df['T_norm'].min():.3f}, {df['T_norm'].max():.3f}]  "
          f"beta_scaled μ={df['beta_scaled'].mean():.3f}  "
          f"tiso_scaled μ={df['tiso_scaled'].mean():.3f}")

    # ── 3. ChemicalProcessor ─────────────────────────────────────────────────
    all_smiles = list(set(
        df["SMILES_E1"].tolist() + df["SMILES_X1"].tolist() +
        df["SMILES_Y1"].tolist()))
    processor  = ChemicalProcessor(all_smiles)
    print(f"  Vocab size: {processor.vocab_size}  "
          f"max SMILES len: {processor.max_len}")

    with open(OUT_DIR / "config.json", "w") as f:
        json.dump(dict(experiment="exp2_structure_lomo",
                       LOMO_TARGET=LOMO_TARGET,
                       molecules=molecules,
                       NEIGHBOURS=NEIGHBOURS,
                       N_SEEDS=N_SEEDS, SEEDS=SEEDS,
                       PHYS_VAR_BETA=PHYS_VAR_BETA,
                       TIMESTAMP=TIMESTAMP), f, indent=2)

    # ── 4. LOMO loop ──────────────────────────────────────────────────────────
    print(f"\n[EXP2-LOMO] Starting LOMO  "
          f"(N_SEEDS={N_SEEDS}, target={LOMO_TARGET})")
    print("  [v4] Test RMSE from PURE PHYSICS only (zero-shot mode)")

    ensemble_store: dict[str, dict] = {}
    lomo_results:   list[dict]      = []

    for mol in molecules:
        is_primary = (mol == LOMO_TARGET)
        tag = " [*PRIMARY*]" if is_primary else ""
        print(f"\n{'='*60}")
        print(f"[LOMO] Left-out: {mol}{tag}  "
              f"(train on {[m for m in molecules if m != mol]})")

        df_train = df[df["Sample_ID"] != mol].reset_index(drop=True)
        df_test  = df[df["Sample_ID"] == mol].reset_index(drop=True)

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
            if is_primary and seed_idx == 0:
                n_p = sum(p.numel() for p in model.parameters())
                print(f"  Model parameters: {n_p:,}")

            curriculum_train(model, train_loader,
                             verbose=(is_primary and seed_idx == 0))

            # Zero-shot: evaluate with pure physics (no residuals)
            m_phys = evaluate(model, test_loader, use_pure_physics=True)
            m_full = evaluate(model, test_loader, use_pure_physics=False)

            store["ln_preds_phys"].append(m_phys["preds"])
            store["ln_preds_full"].append(m_full["preds"])
            if store["ln_tgts"] is None:
                store["ln_tgts"] = m_phys["targets"]

            px_mean = m_phys["params_X"].mean(axis=0)
            Ea_kJ   = float(F.softplus(torch.tensor(px_mean[1])).item() * 50.0)
            lnK_ref = float(px_mean[0] * 2.0 - 2.0)
            lnA_val = float(lnK_ref + Ea_kJ * 1000.0 / (R_GAS * T_REF))
            store["Ea_kJ"].append(Ea_kJ)
            store["lnA"].append(lnA_val)
            store["lnK_ref"].append(lnK_ref)
            store["m"].append(float(abs(px_mean[2])))
            store["n"].append(float(abs(px_mean[3])))

            print(f"     PurePhys R2={m_phys['r2']:.4f}  "
                  f"RMSE={m_phys['rmse']:.4f}  "
                  f"Full RMSE={m_full['rmse']:.4f}  Ea={Ea_kJ:.1f} kJ/mol")

            # Save last-seed weights for primary molecule → post-hoc analysis
            if is_primary and seed_idx == len(SEEDS) - 1:
                ckpt = OUT_DIR / f"model_last_seed_{mol}.pt"
                torch.save(model.state_dict(), ckpt)
                print(f"  [v4] Saved analysis checkpoint: {ckpt.name}")

            del model
            gc.collect()

        # ── Ensemble statistics ───────────────────────────────────────────
        preds_phys = np.array(store["ln_preds_phys"])   # [N_SEEDS, N_test]
        preds_full = np.array(store["ln_preds_full"])
        ln_tgts    = store["ln_tgts"]

        mean_ln_phys = preds_phys.mean(axis=0)
        ci95_ln_phys = 1.96 * preds_phys.std(axis=0, ddof=1)
        mean_ln_full = preds_full.mean(axis=0)
        ci95_ln_full = 1.96 * preds_full.std(axis=0, ddof=1)

        rmse_seeds = np.array([np.sqrt(np.mean((p - ln_tgts) ** 2))
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

        # Save ensemble prediction CSV
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
        pred_df.to_csv(PREDS_DIR / f"LOMO_{mol}_ensemble.csv", index=False)

        store.update(dict(
            pred_df=pred_df,
            mean_ln_phys=mean_ln_phys, ci95_ln_phys=ci95_ln_phys,
            mean_ln_full=mean_ln_full, ln_tgts=ln_tgts,
        ))
        ensemble_store[mol] = store

        # Physical variance vs. neighbouring molecules in the series
        cond_mask = ((df["Condition_Type"] == "Dyn") &
                     (df["Condition_Val"]  == PHYS_VAR_BETA))
        tgt_data  = df[(df["Sample_ID"] == mol) & cond_mask].sort_values("Temp_K")
        nb_diffs: list[float] = []
        for nb in NEIGHBOURS.get(mol, []):
            nb_data = df[(df["Sample_ID"] == nb) & cond_mask].sort_values("Temp_K")
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

        lomo_results.append({
            "Sample_ID":         mol,
            "LOMO_R2_mean":      mean_r2,
            "LOMO_RMSE_mean":    mean_rmse,
            "LOMO_RMSE_CI95":    ci95_rmse,
            "LOMO_RMSE_worst":   mean_rmse + ci95_rmse,
            "Ea_kJ_mean":        mean_Ea,
            "Ea_kJ_CI95":        ci95_Ea,
            "lnA_mean":          mean_lnA,
            "lnA_CI95":          ci95_lnA,
            "Neighbour_Diff_ln": phys_var,
        })

        print(f"\n  [{mol}] Ensemble (pure physics):")
        print(f"    R²   = {mean_r2:.4f}")
        print(f"    RMSE = {mean_rmse:.4f} ± {ci95_rmse:.4f}  "
              f"(worst={mean_rmse+ci95_rmse:.4f})")
        print(f"    Ea   = {mean_Ea:.1f} ± {ci95_Ea:.1f} kJ/mol")
        print(f"    lnA  = {mean_lnA:.3f} ± {ci95_lnA:.3f}")
        print(f"    PhysVar = {phys_var:.4f}")

    results_df = pd.DataFrame(lomo_results).reset_index(drop=True)
    results_df.to_csv(PARAMS_DIR / "LOMO_Results_exp2.csv", index=False)
    print("\n[EXP2-LOMO] All LOMO folds complete.")

    # ── 5. Publication plots ──────────────────────────────────────────────────
    print("\n[PLOTS] Generating publication figures ...")

    # Fig 1 — Pure-physics prediction curves + 95% CI for primary molecule
    tgt_store = ensemble_store[LOMO_TARGET]
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
                color=col, lw=2.0, zorder=4, label="Pure Physics mean (v4)")
        ax.fill_between(grp["Temp_K"],
                        grp["Rate_PurePhys_CI95_lo"],
                        grp["Rate_PurePhys_CI95_hi"],
                        color=col, alpha=0.20, zorder=2, label="95% CI")
        ax.plot(grp["Temp_K"], grp["Rate_Full_Mean"],
                color=col, lw=1.2, ls="--", alpha=0.55, label="Full model mean")
        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel("d\u03b1/dt (min\u207b\u00b9)")
        ax.set_title(f"LOMO {LOMO_TARGET}  |  \u03b2={beta_val:g} K/min",
                     fontweight="bold")
        ax.legend(loc="upper left", framealpha=0.9)
    fig1.suptitle(
        f"Exp-2 LOMO  —  Pure-Physics Zero-Shot Prediction of {LOMO_TARGET}  "
        f"(N_seeds={N_SEEDS})", fontsize=13, fontweight="bold", y=1.01)
    fig1.tight_layout()
    p = PLOTS_DIR / "Fig1_PurePhys_PredictionCurves_CI.png"
    fig1.savefig(p, dpi=300, bbox_inches="tight"); plt.close(fig1)
    print(f"  Saved: {p.name}")

    # Fig 2 — Ea and lnA vs molecule (chain length) with CI error bars
    ea_means  = results_df["Ea_kJ_mean"].values
    ea_ci95   = results_df["Ea_kJ_CI95"].values
    lna_means = results_df["lnA_mean"].values
    lna_ci95  = results_df["lnA_CI95"].values
    x_labels  = results_df["Sample_ID"].values
    x_pos_p   = np.arange(len(results_df))
    is_tgt    = (x_labels == LOMO_TARGET)

    fig2, ax2s = plt.subplots(1, 2, figsize=(13, 5))
    for ax, y_mean, y_ci, ylabel, color, lbl in [
        (ax2s[0], ea_means,  ea_ci95,  "Ea (kJ/mol)",         _CB["red"],  "Ea"),
        (ax2s[1], lna_means, lna_ci95, "ln A (ln min\u207b\u00b9)", _CB["blue"], "lnA"),
    ]:
        ax.errorbar(x_pos_p, y_mean, yerr=y_ci, fmt="o", color=color,
                    elinewidth=1.8, capsize=5, ms=9, mfc="white", mew=2.2,
                    zorder=4, label=f"{lbl} \u00b195%CI")
        ax.scatter(x_pos_p[is_tgt], y_mean[is_tgt], color=color,
                   s=120, marker="*", zorder=5,
                   label=f"{LOMO_TARGET} (zero-shot)")
        sl, ic, rv, *_ = linregress(x_pos_p, y_mean)
        cx = np.linspace(-0.5, len(results_df) - 0.5, 200)
        ax.plot(cx, sl * cx + ic, "--", color=color, alpha=0.55,
                label=f"KCE trend  R={rv:.3f}")
        for xi, yi, ml in zip(x_pos_p, y_mean, x_labels):
            ax.annotate(ml, (xi, yi), xytext=(6, 5),
                        textcoords="offset points", fontsize=8)
        ax.set_xticks(x_pos_p)
        ax.set_xticklabels(x_labels, fontsize=9)
        ax.set_xlabel("Molecule (Chain Length)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{lbl} vs Chain Length  (ensemble UQ)",
                     fontweight="bold")
        ax.legend(fontsize=8)
    fig2.tight_layout()
    p = PLOTS_DIR / "Fig2_Ea_lnA_vs_ChainLength.png"
    fig2.savefig(p, dpi=300, bbox_inches="tight"); plt.close(fig2)
    print(f"  Saved: {p.name}")

    # Fig 3 — Generalisation proof bar chart
    x_pos = np.arange(len(results_df)); w = 0.38
    fig3, ax3 = plt.subplots(figsize=(11, 5))
    ax3.bar(x_pos - w/2, results_df["LOMO_RMSE_mean"], w,
            label="Pure-Physics RMSE (v4 zero-shot, ensemble mean)",
            color=_CB["blue"], edgecolor="black", alpha=0.88, zorder=3)
    ax3.errorbar(x_pos - w/2, results_df["LOMO_RMSE_mean"],
                 yerr=results_df["LOMO_RMSE_CI95"],
                 fmt="none", ecolor="black", elinewidth=1.6,
                 capsize=5, capthick=1.6, zorder=4)
    ax3.bar(x_pos + w/2, results_df["Neighbour_Diff_ln"], w,
            label=f"Exp. neighbour variance (\u03b2={PHYS_VAR_BETA}K/min)",
            color=_CB["orange"], edgecolor="black", alpha=0.85, zorder=3)
    tgt_row = results_df[results_df["Sample_ID"] == LOMO_TARGET].iloc[0]
    ax3.axhline(tgt_row["LOMO_RMSE_worst"], color=_CB["red"], lw=1.5, ls="--",
                label=f"{LOMO_TARGET} worst={tgt_row['LOMO_RMSE_worst']:.3f}")
    for xi, row in zip(x_pos, results_df.itertuples()):
        ax3.text(xi - w/2, row.LOMO_RMSE_mean + row.LOMO_RMSE_CI95 + 0.005,
                 f"R\u00b2={row.LOMO_R2_mean:.3f}", ha="center",
                 fontsize=7.5, color=_CB["blue"], fontweight="bold")
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(results_df["Sample_ID"], fontsize=9)
    ax3.set_ylabel("RMSE  (ln-rate space)")
    ax3.set_title(
        "Exp-2 LOMO: Pure-Physics Generalisation vs Physical Neighbour Variance",
        fontweight="bold")
    ax3.legend(fontsize=8.5)
    fig3.tight_layout()
    p = PLOTS_DIR / "Fig3_PurePhys_Generalisation.png"
    fig3.savefig(p, dpi=300, bbox_inches="tight"); plt.close(fig3)
    print(f"  Saved: {p.name}")

    # ── 6. Zero-shot proof statistical report ────────────────────────────────
    _SEP  = "=" * 72
    _SEP2 = "-" * 72
    delta_phys = tgt_row["Neighbour_Diff_ln"]
    worst_pred = tgt_row["LOMO_RMSE_worst"]

    print(f"\n{_SEP}")
    print(f"  ZERO-SHOT PROOF — Experiment 2 LOMO")
    print(f"  Left-out molecule: {LOMO_TARGET}  |  N_seeds: {N_SEEDS}")
    print(f"  Evaluation mode  : PURE PHYSICS (no residual heads)")
    print(_SEP2)
    print(f"\n  [A]  PHYSICAL VARIANCE OF HOMOLOGOUS SERIES  "
          f"(Dyn {PHYS_VAR_BETA} K/min)")
    print(f"  {'Molecule':<12} {'Neighbours':<24} {'Delta_phys (RMSE)'}")
    print(f"  {'-'*50}")
    for row in results_df.itertuples():
        nbs      = ", ".join(NEIGHBOURS.get(row.Sample_ID, []))
        pv_str   = f"{row.Neighbour_Diff_ln:.4f}" if not np.isnan(
            row.Neighbour_Diff_ln) else "N/A"
        flag     = "  <- TARGET" if row.Sample_ID == LOMO_TARGET else ""
        print(f"  {row.Sample_ID:<12} {nbs:<24} {pv_str}{flag}")
    print(f"\n  [B]  PREDICTION UNCERTAINTY")
    print(f"  Mean RMSE (pure physics)  = {tgt_row['LOMO_RMSE_mean']:.4f}")
    print(f"  95% CI                    ± {tgt_row['LOMO_RMSE_CI95']:.4f}")
    print(f"  Worst-Case RMSE           = {worst_pred:.4f}")
    print(f"\n  [C]  VERDICT")
    if not np.isnan(delta_phys):
        ratio  = worst_pred / delta_phys
        proof  = (ratio < 1.0)
        print(f"  PhysVar = {delta_phys:.4f},  Worst RMSE = {worst_pred:.4f}")
        print(f"  Ratio (worst / PhysVar) = {ratio:.3f}")
        if proof:
            print(f"  PROOF CONFIRMED — model error is "
                  f"{(1-ratio)*100:.1f}% smaller than intrinsic variance.")
        else:
            print(f"  Not yet confirmed — increase epochs or ensemble size.")
    print(f"\n{_SEP}")

    report_path = OUT_DIR / "ZeroShot_Proof_Report_exp2.txt"
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print(f"{_SEP}\n  ZERO-SHOT PROOF — Exp-2 LOMO  "
              f"({TIMESTAMP})\n{_SEP}")
        for row in results_df.itertuples():
            nbs  = ", ".join(NEIGHBOURS.get(row.Sample_ID, []))
            pv_s = (f"{row.Neighbour_Diff_ln:.4f}"
                    if not np.isnan(row.Neighbour_Diff_ln) else "NaN ")
            flag = "  [TARGET]" if row.Sample_ID == LOMO_TARGET else ""
            print(f"  {row.Sample_ID:<10}  RMSE={row.LOMO_RMSE_mean:.4f}"
                  f"±{row.LOMO_RMSE_CI95:.4f}  worst={row.LOMO_RMSE_worst:.4f}"
                  f"  PhysVar={pv_s}  Ea={row.Ea_kJ_mean:.1f}±{row.Ea_kJ_CI95:.1f}"
                  f"{flag}")
        print(f"{_SEP}")
        if not np.isnan(delta_phys):
            verdict = "CONFIRMED" if worst_pred < delta_phys else "NOT confirmed"
            print(f"  Ratio = {worst_pred/delta_phys:.3f}  ({verdict})\n{_SEP}")
    report_path.write_text(buf.getvalue())
    print(f"  Report saved: {report_path}")

    # ── 7. Post-hoc: isothermal + dynamic for primary molecule ───────────────
    ckpt_path = OUT_DIR / f"model_last_seed_{LOMO_TARGET}.pt"
    if not ckpt_path.exists():
        print(f"\n[EXP2] Checkpoint not found — skipping post-hoc analysis.")
        _print_summary(OUT_DIR, molecules, LOMO_TARGET)
        return

    print(f"\n[EXP2] Loading checkpoint for post-hoc analysis ({LOMO_TARGET}) ...")
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

    print(f"  X: lnKr={params['lnKr_X']:.3f}  "
          f"Ea={params['Ea_X_J']/1000:.1f}kJ/mol  "
          f"m={params['m_X']:.3f}  n={params['n_X']:.3f}")
    print(f"  Y: lnKr={params['lnKr_Y']:.3f}  "
          f"Ea={params['Ea_Y_J']/1000:.1f}kJ/mol  syn={params['syn_mult']:.4f}")

    # Isothermal quantitative report
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
            PARAMS_DIR / f"Isothermal_Report_{LOMO_TARGET}.csv", index=False)

    # Fig 4 — Dynamic validation
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
            f"Exp-2 LOMO  —  {LOMO_TARGET}  "
            f"Dynamic Validation  (\u03b2={used_beta}\u00b0C/min)",
            fontsize=13, fontweight="bold", y=1.01)
        fig4.tight_layout()
        p = PLOTS_DIR / f"Fig4_Dynamic_Validation_{LOMO_TARGET}.png"
        fig4.savefig(p, dpi=300, bbox_inches="tight"); plt.close(fig4)
        print(f"\n  Saved: {p.name}")

        dyn_rmse = float(np.sqrt(np.mean(
            (df_dyn_exp["Alpha"].values - sim_alpha) ** 2)))
        dyn_mae  = float(np.mean(np.abs(df_dyn_exp["Alpha"].values - sim_alpha)))
        print(f"  Dynamic RMSE(\u03b1) = {dyn_rmse:.4f}  "
              f"MAE(\u03b1) = {dyn_mae:.4f}")
    except Exception as exc:
        print(f"  [WARN] Dynamic simulation failed: {exc}")

    # Fig 5 — Ea vs chain-length parameter extraction scatter (all seeds)
    _chain_num = {mol: i for i, mol in enumerate(molecules)}
    fig5, ax5 = plt.subplots(figsize=(9, 5))
    for mol, st in ensemble_store.items():
        xi    = _chain_num[mol]
        ea_s  = st["Ea_kJ"]
        col   = _CB["red"] if mol == LOMO_TARGET else _CB["blue"]
        ax5.scatter([xi] * len(ea_s), ea_s, color=col, alpha=0.5, s=35, zorder=4)
        ax5.errorbar([xi], [float(np.mean(ea_s))],
                     yerr=[1.96 * float(np.std(ea_s, ddof=1))],
                     fmt="o", color=col, ms=9, capsize=5, mfc="white",
                     mew=2, elinewidth=1.8, zorder=5)
    ax5.scatter([], [], color=_CB["red"], label=f"{LOMO_TARGET} (zero-shot)")
    ax5.scatter([], [], color=_CB["blue"], label="Trained molecules")
    ax5.set_xticks(list(_chain_num.values()))
    ax5.set_xticklabels(list(_chain_num.keys()), fontsize=9)
    ax5.set_xlabel("Molecule")
    ax5.set_ylabel("Ea (kJ/mol)")
    ax5.set_title("Per-seed Ea across LOMO folds — KCE trend check",
                  fontweight="bold")
    ax5.legend(fontsize=9)
    fig5.tight_layout()
    p = PLOTS_DIR / "Fig5_Ea_PerSeed_AllMolecules.png"
    fig5.savefig(p, dpi=300, bbox_inches="tight"); plt.close(fig5)
    print(f"  Saved: {p.name}")

    _print_summary(OUT_DIR, molecules, LOMO_TARGET)


# ---------------------------------------------------------------------------
def _print_summary(out_dir: Path, molecules: list, lomo_target: str) -> None:
    print(f"""
{'='*64}
  Experiment 2 — Structure LOMO  Complete
  Output    : {out_dir}
  Molecules : {molecules}
  Target    : {lomo_target}

  v4 physics integrity:
    [1] Global pre-scaling — no per-fold StandardScaler leakage
    [2] Explicit residual decoupling: ln_r_final = ln_r_phys + delta
        - iso_offset_masked - dyn_offset_masked
    [3] Zero-shot evaluation: pure_phys_ln = ln_r_final - delta
        + iso_masked + dyn_masked  (≡ ln_r_phys)
    [4] KCE-Killer: vitrification weighting, lnA anchor,
        m/n priors, L2 stiffness, WD=1e-3 Stage-1 Adam

  Figures:
    Fig1 — Pure-physics curves + CI bands (target zero-shot)
    Fig2 — Ea & lnA vs chain length (KCE trend)
    Fig3 — Generalisation: Pure-Physics RMSE vs Physical Variance
    Fig4 — Dynamic validation (α and rate vs T, RK45)
    Fig5 — Per-seed Ea scatter across all LOMO folds
{'='*64}""")


#%% Run Experiment  (uncomment for interactive cell execution)
# main()

# =============================================================================
if __name__ == "__main__":
    main()
