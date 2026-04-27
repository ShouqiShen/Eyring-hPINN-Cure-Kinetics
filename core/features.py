# -*- coding: utf-8 -*-
"""
core/features.py
================
ChemicalProcessor — builds a dynamic SMILES vocabulary from the dataset
and generates three molecular representations per SMILES string:

  1. ECFP4 fingerprint (Morgan radius=2, 2048 bits) via RDKit
  2. Molecular graph  (node_feat: atomic-num, degree, #H; edge_index: bidirectional)
     consumed by PyTorch Geometric GCNConv layers
  3. Padded SMILES token sequence consumed by a 1-D CNN

All representations are cached so repeated lookups are O(1).
"""
from __future__ import annotations

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, DataStructs


class ChemicalProcessor:
    """
    Build vocabulary and cache mol-level feature dicts for every unique SMILES.

    Parameters
    ----------
    smiles_list : list[str]
        All SMILES that will ever be queried (typically the union of all three
        SMILES columns across the full dataset).
    """

    def __init__(self, smiles_list: list[str]) -> None:
        self.fp_gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=2, fpSize=2048)
        unique_chars    = set("".join(smiles_list))
        self.char2idx   = {c: i + 1 for i, c in enumerate(sorted(unique_chars))}
        self.vocab_size = len(self.char2idx) + 1          # +1 for padding token 0
        self.max_len    = max(len(s) for s in smiles_list)
        self._cache: dict[str, dict] = {}
        for s in set(smiles_list):
            self._cache[s] = self._generate(s)

    # ------------------------------------------------------------------
    def _generate(self, smiles: str) -> dict:
        mol = Chem.MolFromSmiles(smiles)

        # ECFP4
        fp  = self.fp_gen.GetFingerprint(mol)
        arr = np.zeros(2048, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)

        # Graph
        nodes = [
            [a.GetAtomicNum(), a.GetDegree(), a.GetTotalNumHs()]
            for a in mol.GetAtoms()
        ]
        edges: list[list[int]] = []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edges += [[i, j], [j, i]]

        # SMILES token sequence (padded to max_len)
        seq = [self.char2idx.get(c, 0) for c in smiles]
        seq += [0] * (self.max_len - len(seq))

        return {
            "ecfp":       torch.tensor(arr, dtype=torch.float32),
            "node_feat":  torch.tensor(nodes, dtype=torch.float32),
            "edge_index": (
                torch.tensor(edges, dtype=torch.long).t().contiguous()
                if edges else torch.empty((2, 0), dtype=torch.long)
            ),
            "smiles_seq": torch.tensor(seq, dtype=torch.long),
        }

    # ------------------------------------------------------------------
    def get(self, smiles: str) -> dict:
        """Return the cached feature dict for *smiles* (O(1) lookup)."""
        return self._cache[smiles]
