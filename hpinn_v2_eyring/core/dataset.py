# -*- coding: utf-8 -*-
"""
core/dataset.py
===============
Global pre-scaling, Dataset class, collate function, and DataLoader factory.

v4 KEY DESIGN PRINCIPLE — no per-fold data leakage
----------------------------------------------------
`preprocess_global_dataframe(df)` must be called ONCE on the full dataframe
**before** any train/test split.  The StandardScaler is fitted on ALL rows so
every fold sees the same global feature distribution.  The Dataset class then
simply reads the pre-computed columns; it never fits any scaler of its own.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import Dataset, DataLoader

from .features import ChemicalProcessor
from .config   import BATCH_SIZE


# =============================================================================
# Global feature engineering
# =============================================================================
def preprocess_global_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add five engineered columns to *df* using **global** statistics.

    Added columns
    -------------
    Rate_Smooth  : Gaussian-smoothed (σ=1) reaction rate, per condition group
    T_norm       : (T_K - global_min) / (global_max - global_min)
    is_dyn       : 1.0 for dynamic DSC, 0.0 for isothermal
    beta_scaled  : StandardScaler-transformed log(β) for dynamic rows (0 elsewhere)
    tiso_scaled  : StandardScaler-transformed 1/T_iso for isothermal rows (0 elsewhere)

    Parameters
    ----------
    df : raw dataframe with columns
         Sample_ID, SMILES_E1, SMILES_X1, SMILES_Y1,
         Ratio_X1, Ratio_Y1, Condition_Type, Condition_Val,
         Temp_K, Alpha, Rate_1_min

    Returns
    -------
    df : new dataframe with all original columns plus the five above.
    """
    df = df.copy()

    # 1. Rate smoothing — per (molecule, condition-type, condition-value) group
    df["Rate_Smooth"] = np.nan
    for _, grp in df.groupby(
            ["Sample_ID", "Condition_Type", "Condition_Val"], sort=False):
        grp_s = grp.sort_values("Temp_K")
        sm    = gaussian_filter1d(grp_s["Rate_1_min"].to_numpy(float), sigma=1)
        df.loc[grp_s.index, "Rate_Smooth"] = sm

    # 2. Global temperature normalisation
    T_min = float(df["Temp_K"].min())
    T_max = float(df["Temp_K"].max())
    df["T_norm"] = (df["Temp_K"] - T_min) / (T_max - T_min)

    # 3. Condition-type flag
    df["is_dyn"] = (df["Condition_Type"] == "Dyn").astype(np.float32)

    # 4. Raw condition-value features
    proc_val   = df["Condition_Val"].values.astype(np.float32).reshape(-1, 1)
    is_dyn_arr = df["is_dyn"].values.astype(np.float32).reshape(-1, 1)
    beta_arr   = np.where(is_dyn_arr == 1.0, proc_val, 0.0)
    T_iso_arr  = np.where(is_dyn_arr == 0.0, proc_val + 273.15, 0.0)

    beta_feat = np.log(np.clip(beta_arr, 1e-6, None)) * is_dyn_arr
    tiso_feat = (1.0 / np.clip(T_iso_arr, 1.0, None)) * (1.0 - is_dyn_arr)

    # 5. Fit scalers on the FULL dataset — transform once
    df["beta_scaled"] = (
        StandardScaler().fit_transform(beta_feat).astype(np.float32))
    df["tiso_scaled"] = (
        StandardScaler().fit_transform(tiso_feat).astype(np.float32))

    return df


# =============================================================================
# PyTorch Dataset
# =============================================================================
class UnifiedKineticsDataset(Dataset):
    """
    Wraps a pre-processed dataframe slice (train or test) as a PyTorch Dataset.

    Expects the dataframe to already contain the five columns added by
    `preprocess_global_dataframe`: Rate_Smooth, T_norm, is_dyn,
    beta_scaled, tiso_scaled.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        processor: ChemicalProcessor,
    ) -> None:
        self.df   = dataframe.reset_index(drop=True)
        self.proc = processor

        self.phys_real   = torch.tensor(
            self.df[["Temp_K", "Alpha"]].values, dtype=torch.float32)
        self.phys_norm   = torch.tensor(
            self.df[["T_norm", "Alpha"]].values, dtype=torch.float32)
        self.ratios      = torch.tensor(
            self.df[["Ratio_X1", "Ratio_Y1"]].values, dtype=torch.float32)
        self.targets     = torch.tensor(
            np.clip(self.df["Rate_Smooth"].values, 1e-10, None),
            dtype=torch.float32)
        self.beta_scaled = torch.tensor(
            self.df["beta_scaled"].values.reshape(-1, 1), dtype=torch.float32)
        self.tiso_scaled = torch.tensor(
            self.df["tiso_scaled"].values.reshape(-1, 1), dtype=torch.float32)
        self.is_dynamic  = torch.tensor(
            self.df["is_dyn"].values.reshape(-1, 1), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        return {
            "struct_E":    self.proc.get(row["SMILES_E1"]),
            "struct_X":    self.proc.get(row["SMILES_X1"]),
            "struct_Y":    self.proc.get(row["SMILES_Y1"]),
            "T":           self.phys_real[idx, 0:1],
            "alpha":       self.phys_real[idx, 1:2],
            "rX":          self.ratios[idx, 0:1],
            "rY":          self.ratios[idx, 1:2],
            "phys_norm":   self.phys_norm[idx],
            "target":      self.targets[idx:idx+1],
            "is_dyn":      self.is_dynamic[idx],
            "beta_scaled": self.beta_scaled[idx],
            "tiso_scaled": self.tiso_scaled[idx],
        }


# =============================================================================
# Collate + DataLoader
# =============================================================================
def custom_collate(batch: list[dict]) -> dict:
    """
    Stack scalar tensors normally; merge variable-size molecular graphs by
    offsetting node indices and building a batch_idx vector for PyG pooling.
    """
    collated = {
        k: torch.stack([b[k] for b in batch])
        for k in ["T", "alpha", "rX", "rY", "phys_norm", "target",
                  "is_dyn", "beta_scaled", "tiso_scaled"]
    }
    for key in ["struct_E", "struct_X", "struct_Y"]:
        node_feats, edge_indices, ecfps, smiles_seqs, batch_idxs = \
            [], [], [], [], []
        offset = 0
        for i, b in enumerate(batch):
            s  = b[key]
            nf = s["node_feat"]
            n  = nf.size(0)
            node_feats.append(nf)
            ei = s["edge_index"]
            edge_indices.append(
                ei + offset if ei.numel() > 0
                else torch.empty((2, 0), dtype=torch.long))
            batch_idxs.append(torch.full((n,), i, dtype=torch.long))
            ecfps.append(s["ecfp"])
            smiles_seqs.append(s["smiles_seq"])
            offset += n
        collated[key] = {
            "node_feat":  torch.cat(node_feats,   dim=0),
            "edge_index": torch.cat(edge_indices, dim=1),
            "batch_idx":  torch.cat(batch_idxs,   dim=0),
            "ecfp":       torch.stack(ecfps),
            "smiles_seq": torch.stack(smiles_seqs),
        }
    return collated


def make_loader(
    ds: Dataset,
    shuffle: bool = True,
    seed: int = 42,
) -> DataLoader:
    """Return a seed-controlled DataLoader for reproducible shuffling."""
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        collate_fn=custom_collate,
        num_workers=0,
        generator=g if shuffle else None,
    )
