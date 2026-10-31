# Migrated from 1002genertation.ipynb, cell 24. See README.md.
# ============================================================
# 1. Imports + cell-level KNN graph dataset
# ============================================================

import os, re, json, copy, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from PIL import Image
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm.auto import tqdm


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed); random.seed(worker_seed)


def safe_filename(name): return re.sub(r"[^\w\-.]+", "_", str(name))


def average_loss_dict(loss_dicts):
    out = defaultdict(float)
    for d in loss_dicts:
        for k, v in d.items(): out[k] += float(v)
    n = max(len(loss_dicts), 1)
    return {k: v / n for k, v in out.items()}


def compute_cell_knn_context(adata, celltypes, spatial_key="spatial", celltype_key="cell_type", k=20, distance_weighted=True, eps=1e-8):
    """Compute cell knn context."""
    celltypes = list(celltypes); K = len(celltypes); ct_to_idx = {str(ct): i for i, ct in enumerate(celltypes)}
    coords_all = np.asarray(adata.obsm[spatial_key])[:, :2].astype(np.float32)
    labels_str_all = adata.obs[celltype_key].astype(str).values

    known = np.array([x in ct_to_idx for x in labels_str_all], dtype=bool)
    finite = np.isfinite(coords_all).all(axis=1)
    valid = known & finite
    obs_indices = np.where(valid)[0].astype(np.int64)

    coords = coords_all[valid]
    labels_str = labels_str_all[valid]
    labels = np.array([ct_to_idx[x] for x in labels_str], dtype=np.int64)
    N = len(labels)

    if N == 0: raise ValueError(" slice has no valid cells .")

    self_x = np.eye(K, dtype=np.float32)[labels]

    if N == 1:
        neighbor_P = self_x.copy()
        return {"coords": coords, "labels": labels, "self_x": self_x, "neighbor_P": neighbor_P, "obs_indices": obs_indices}

    k_eff = min(int(k), N - 1)
    tree = cKDTree(coords)
    dists, inds = tree.query(coords, k=k_eff + 1, workers=-1)

    dists = np.asarray(dists[:, 1:], dtype=np.float32)
    inds = np.asarray(inds[:, 1:], dtype=np.int64)
    neigh_labels = labels[inds]

    if distance_weighted:
        local_sigma = np.maximum(0.5 * dists[:, -1:], eps)
        weights = np.exp(-0.5 * (dists / local_sigma) ** 2).astype(np.float32)
    else:
        weights = np.ones_like(dists, dtype=np.float32)

    neighbor_P = np.zeros((N, K), dtype=np.float32)
    rows = np.repeat(np.arange(N), k_eff)
    cols = neigh_labels.reshape(-1)
    vals = weights.reshape(-1)
    np.add.at(neighbor_P, (rows, cols), vals)

    neighbor_P /= np.maximum(neighbor_P.sum(axis=1, keepdims=True), eps)
    return {"coords": coords, "labels": labels, "self_x": self_x, "neighbor_P": neighbor_P, "obs_indices": obs_indices}


class ContextCellDataset(Dataset):
    def __init__(self, anndata_dict, celltypes, spatial_key="spatial", celltype_key="cell_type", k=20, distance_weighted=True, slice_names=None):
        self.slice_names = list(anndata_dict.keys()) if slice_names is None else list(slice_names)
        self.celltypes = list(celltypes); self.K = len(celltypes)

        xs, qs, labels_all, slice_ids = [], [], [], []

        for sid, name in enumerate(self.slice_names):
            d = compute_cell_knn_context(anndata_dict[name], self.celltypes, spatial_key, celltype_key, k, distance_weighted)
            xs.append(d["self_x"]); qs.append(d["neighbor_P"]); labels_all.append(d["labels"])
            slice_ids.append(np.full(len(d["labels"]), sid, dtype=np.int64))
            print(f"{name}: cells={len(d['labels'])}")

        self.self_x = np.concatenate(xs).astype(np.float32)
        self.neighbor_P = np.concatenate(qs).astype(np.float32)
        self.labels = np.concatenate(labels_all).astype(np.int64)
        self.slice_ids = np.concatenate(slice_ids).astype(np.int64)

        print(f"\n[ContextCellDataset] cells={len(self.labels)}, K={self.K}, slices={len(self.slice_names)}")

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        return {"self_x": torch.from_numpy(self.self_x[idx]), "neighbor_P": torch.from_numpy(self.neighbor_P[idx]), "label": int(self.labels[idx]), "slice_id": int(self.slice_ids[idx])}


def stratified_train_val_indices(labels, val_fraction=0.2, seed=1234):
    labels = np.asarray(labels, dtype=np.int64); rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []

    for k in np.unique(labels):
        idx = np.where(labels == k)[0]; rng.shuffle(idx)
        if val_fraction <= 0 or len(idx) == 1:
            train_idx.extend(idx.tolist()); continue
        n_val = max(1, min(int(round(len(idx) * val_fraction)), len(idx) - 1))
        val_idx.extend(idx[:n_val].tolist()); train_idx.extend(idx[n_val:].tolist())

    rng.shuffle(train_idx); rng.shuffle(val_idx)
    return np.asarray(train_idx, dtype=np.int64), np.asarray(val_idx, dtype=np.int64)