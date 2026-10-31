# Migrated from 1002genertation.ipynb, cell 61. See docs/MIGRATION.md.
import zlib
import numpy as np
import pandas as pd
import torch
import anndata as ad

from scipy import sparse
from scipy.spatial import cKDTree
from IPython.display import display


# ============================================================
# 1. Configuration
# ============================================================

expression_celltypes = [str(x) for x in optimization_celltypes]
same_omics = bool(config["same_omics"])
left_omics_name = str(config["left_omics"])
right_omics_name = str(config["right_omics"])


# ============================================================
# 2. Basic utilities
# ============================================================

def expr_assert_finite(X, name):
    arr = X.data if sparse.issparse(X) else np.asarray(X)
    if not np.isfinite(arr).all(): raise RuntimeError(f"{name}  contains  NaN/Inf")


def expr_prepare_flow_df(flow_df):
    required = [
        "model", "pair", "image_index", "site_id", "pixel_u", "pixel_v", "allocated_density",
        "left_real_pixel_u", "left_real_pixel_v", "right_real_pixel_u", "right_real_pixel_v",
        "left_mapping_valid", "right_mapping_valid",
    ]
    ctp_required_columns(flow_df, required, "flow_mapped_pixel_df")

    df = flow_df.copy()

    if "keep_after_flow_screen" in df.columns: df = df.loc[df["keep_after_flow_screen"].astype(bool)].copy()

    df["model"] = df["model"].astype(str); df["pair"] = df["pair"].astype(str)
    df["image_index"] = df["image_index"].astype(int); df["site_id"] = df["site_id"].astype(int)

    if "is_center_pixel" not in df.columns:
        ctp_required_columns(df, ["center_pixel_u", "center_pixel_v"], "flow_mapped_pixel_df")
        df["is_center_pixel"] = (df["pixel_u"].astype(int) == df["center_pixel_u"].astype(int)) & (df["pixel_v"].astype(int) == df["center_pixel_v"].astype(int))

    return df


# ============================================================
# 3. Real endpoint donor cache
# ============================================================

def expr_build_real_cache(adata, count_maps, feature_idx=None, feature_names=None, slice_name="unknown"):
    count_maps = np.rint(np.asarray(count_maps)).astype(np.int64)
    K, H, W = count_maps.shape

    if K != len(expression_celltypes): raise ValueError(f"{slice_name}: count_maps K  and  celltypes  inconsistent ")
    if celltype_col not in adata.obs.columns: raise KeyError(f"{slice_name}: obs missing  {celltype_col}")
    if "spatial" not in adata.obsm: raise KeyError(f"{slice_name}: obsm missing  spatial")

    labels = adata.obs[celltype_col].astype(str).to_numpy()
    unknown = sorted(set(labels) - set(expression_celltypes))
    if unknown: raise ValueError(f"{slice_name}:  unknown cell types: {unknown[:10]}")

    xy = np.asarray(adata.obsm["spatial"]).astype(np.float32)[:, :2]
    x_min, y_min, psx, psy = coord_grid_transform(grid_meta)

    u = np.floor((xy[:, 0] - x_min) / psx).astype(np.int64)
    v = np.floor((xy[:, 1] - y_min) / psy).astype(np.int64)
    valid = (u >= 0) & (u < W) & (v >= 0) & (v < H)

    ct_to_idx = {ct: k for k, ct in enumerate(expression_celltypes)}
    valid_idx = np.flatnonzero(valid)
    type_idx = np.asarray([ct_to_idx[labels[i]] for i in valid_idx], dtype=np.int64)

    reconstructed = np.zeros_like(count_maps)
    np.add.at(reconstructed, (type_idx, v[valid_idx], u[valid_idx]), 1)

    if not np.array_equal(reconstructed, count_maps):
        diff = np.abs(reconstructed - count_maps)
        raise RuntimeError(
            f"{slice_name}: REAL cell -> pixel  and  count_maps  inconsistent  | "
            f"reconstructed={int(reconstructed.sum())}, count_maps={int(count_maps.sum())}, "
            f"absolute difference={int(diff.sum())}"
        )

    X = adata.X if feature_idx is None else adata.X[:, np.asarray(feature_idx, dtype=np.int64)]
    X = X.tocsr().astype(np.float32) if sparse.issparse(X) else np.asarray(X, dtype=np.float32)
    expr_assert_finite(X, f"{slice_name}.X")

    feature_names = np.asarray(adata.var_names if feature_names is None else feature_names).astype(str)

    # Stable grouping by (pixel, cell type)
    code = ((v[valid_idx] * W + u[valid_idx]) * len(expression_celltypes) + type_idx).astype(np.int64)
    order = np.argsort(code, kind="stable")
    code, cells = code[order], valid_idx[order]

    starts = np.r_[0, 1 + np.flatnonzero(code[1:] != code[:-1])]
    ends = np.r_[starts[1:], len(code)]

    groups = {}

    for a, b in zip(starts, ends):
        c = int(code[a]); k = c % len(expression_celltypes); pixel = c // len(expression_celltypes)
        groups[(pixel % W, pixel // W, expression_celltypes[k])] = cells[a:b]

    return {
        "X": X,
        "feature_names": feature_names,
        "labels": labels,
        "coords": xy.astype(np.float32),
        "groups": groups,
        "type_indices": {ct: np.flatnonzero(labels == ct).astype(np.int64) for ct in expression_celltypes},
        "all_indices": np.arange(adata.n_obs, dtype=np.int64),
        "count_maps": count_maps,
        "celltype_to_idx": ct_to_idx,
        "mean_cache": {},
        "neighbor_features": None,
        "neighbor_norm": None,
        "slice_name": str(slice_name),
    }


def expr_same_type_pixel_mean(cache, u, v, cell_type):
    key = (int(u), int(v), str(cell_type))

    if key in cache["mean_cache"]: return cache["mean_cache"][key]

    idx = cache["groups"].get(key)

    if idx is None or not len(idx):
        cache["mean_cache"][key] = None
        return None

    mean = np.asarray(cache["X"][idx].mean(axis=0)).ravel().astype(np.float32)
    expr_assert_finite(mean, f"{cache['slice_name']} pixel mean")
    cache["mean_cache"][key] = mean

    return mean


# ============================================================
# 4. FLOW-local same-type propagation
# ============================================================

def expr_propagate_side(pred_adata, frame, cache, side):
    pixels = frame if use_neighborhood_pixels else frame.loc[frame["is_center_pixel"]]
    if not len(pixels): raise RuntimeError(" current frame no available FLOW pixels")

    flow_groups = {int(k): np.asarray(v, dtype=np.int64) for k, v in pixels.groupby("site_id", sort=False).indices.items()}

    pred_sites = pred_adata.obs["site_id"].to_numpy(dtype=np.int64)
    pred_types = pred_adata.obs[celltype_col].astype(str).to_numpy()

    pred_info = pd.DataFrame({"site": pred_sites, "type": pred_types})
    pred_groups = {(int(k[0]), str(k[1])): np.asarray(v, dtype=np.int64) for k, v in pred_info.groupby(["site", "type"], sort=False).indices.items()}

    real_u = pixels[f"{side}_real_pixel_u"].to_numpy(dtype=np.int64)
    real_v = pixels[f"{side}_real_pixel_v"].to_numpy(dtype=np.int64)
    valid = pixels[f"{side}_mapping_valid"].astype(bool).to_numpy()
    weight = pixels["allocated_density"].to_numpy(dtype=float)

    _, H, W = cache["count_maps"].shape
    N, F = pred_adata.n_obs, cache["X"].shape[1]

    X = np.full((N, F), np.nan, dtype=np.float32)
    available = np.zeros(N, dtype=bool)
    source = np.full(N, "missing", dtype=object)

    for (site, ct), pred_idx in pred_groups.items():
        rows = flow_groups.get(site)
        if rows is None: continue

        k = cache["celltype_to_idx"].get(ct)
        if k is None: raise ValueError(f" predicted type not in celltypes in : {ct}")

        expr_sum = np.zeros(F, dtype=np.float64)
        wsum = 0.0

        for pos in rows:
            if not valid[pos]: continue

            u, v = int(real_u[pos]), int(real_v[pos])
            w = float(weight[pos])

            if u < 0 or u >= W or v < 0 or v >= H or not np.isfinite(w) or w <= 0: continue

            expected_count = int(cache["count_maps"][k, v, u])
            if expected_count <= 0: continue

            donor_idx = cache["groups"].get((u, v, ct))

            if donor_idx is None or len(donor_idx) != expected_count:
                raise RuntimeError(
                    f"{side} | {cache['slice_name']} | pixel=({u},{v}) | type={ct}: "
                    f"count_maps={expected_count}, donor_lookup={0 if donor_idx is None else len(donor_idx)}"
                )

            mean = expr_same_type_pixel_mean(cache, u, v, ct)
            expr_sum += w * mean
            wsum += w

        if wsum <= 1e-12: continue

        X[pred_idx] = (expr_sum / wsum).astype(np.float32)
        available[pred_idx] = True
        source[pred_idx] = "flow_local_same_type"

    return {"X": X, "available": available, "source": source}


# ============================================================
# 5. Microenvironment features
# ============================================================

def expr_neighbor_features(coords, labels, k=50):
    coords = np.asarray(coords, dtype=np.float32)
    labels = np.asarray(labels).astype(str)

    ct_to_idx = {ct: i for i, ct in enumerate(expression_celltypes)}
    unknown = sorted(set(labels) - set(expression_celltypes))
    if unknown: raise ValueError(f" unknown types in neighborhood features : {unknown[:10]}")

    N, K = len(coords), len(expression_celltypes)
    F = np.zeros((N, K), dtype=np.float32)

    if N == 0: return F

    label_idx = np.asarray([ct_to_idx[x] for x in labels], dtype=np.int64)

    if N == 1:
        F[0, label_idx[0]] = 1
        return F

    k = min(int(k), N - 1)
    dists, idx = cKDTree(coords).query(coords, k=k + 1)

    if idx.ndim == 1: idx, dists = idx[:, None], dists[:, None]

    idx, dists = idx[:, 1:], dists[:, 1:]

    positive = dists[dists > 0]
    sigma = float(np.median(positive)) if positive.size else 1.0
    weights = np.exp(-dists / (sigma + 1e-8)).astype(np.float32)

    rows = np.repeat(np.arange(N), k)
    np.add.at(F, (rows, label_idx[idx].reshape(-1)), weights.reshape(-1))

    s = F.sum(axis=1, keepdims=True)
    good = s[:, 0] > 1e-8
    F[good] /= s[good]

    bad = np.flatnonzero(~good)
    if len(bad): F[bad, label_idx[bad]] = 1

    return F


def expr_normalize_features(F):
    F = np.asarray(F, dtype=np.float32)
    return F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-8)


def expr_similarity_probability(query_norm, donor_norm):
    sim = np.maximum(query_norm @ donor_norm.T, 0)
    sim[~np.isfinite(sim)] = 0

    s = sim.sum(axis=1, keepdims=True)
    prob = np.divide(sim, s, out=np.zeros_like(sim, dtype=np.float32), where=s > 1e-8)

    bad = s[:, 0] <= 1e-8
    if np.any(bad): prob[bad] = 1.0 / donor_norm.shape[0]

    return prob


# ============================================================
# 6. Feature-wise random donor sampling
# ============================================================

def expr_sample_features(probability, X_real, donor_idx, seed, feature_chunk_size=512):
    probability = np.asarray(probability, dtype=np.float32)
    donor_idx = np.asarray(donor_idx, dtype=np.int64)

    if not len(donor_idx): raise ValueError("donor_idx is empty ")

    X_donor = X_real[donor_idx]
    n_query, n_donor = probability.shape
    n_features = X_donor.shape[1]

    probs = torch.as_tensor(probability, dtype=torch.float32)
    probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-12)

    cdf = torch.cumsum(probs, dim=1)
    cdf[:, -1] = 1.0

    generator = torch.Generator()
    generator.manual_seed(int(seed))

    result = np.empty((n_query, n_features), dtype=np.float32)

    for start in range(0, n_features, feature_chunk_size):
        end = min(start + feature_chunk_size, n_features)

        X_chunk = X_donor[:, start:end]
        X_chunk = X_chunk.toarray().astype(np.float32) if sparse.issparse(X_chunk) else np.asarray(X_chunk, dtype=np.float32)
        X_chunk = torch.as_tensor(X_chunk.T, dtype=torch.float32)

        uniform = torch.rand((n_query, end - start), generator=generator)
        donor_pos = torch.searchsorted(cdf, uniform, right=False).clamp_max(n_donor - 1)

        result[:, start:end] = torch.gather(X_chunk, 1, donor_pos.T).T.numpy()

    expr_assert_finite(result, "microenvironment sampled expression")
    return result


# ============================================================
# 7. Microenvironment fallback
#
# same_omics:
#   same type only
#
# multi_omics:
#   same type -> all donor cells
# ============================================================

def expr_apply_fallback(result, pred_adata, cache, pred_norm, seed, hierarchical=False, fallback_mask=None):
    missing = ~result["available"]

    if fallback_mask is not None:
        fallback_mask = np.asarray(fallback_mask, dtype=bool)
        if fallback_mask.shape != missing.shape: raise ValueError("fallback_mask shape mismatch ")
        missing &= fallback_mask

    if not np.any(missing): return result

    pred_labels = pred_adata.obs[celltype_col].astype(str).to_numpy()

    if cache["neighbor_features"] is None:
        cache["neighbor_features"] = expr_neighbor_features(cache["coords"], cache["labels"])
        cache["neighbor_norm"] = expr_normalize_features(cache["neighbor_features"])

    X = result["X"].copy()
    available = result["available"].copy()
    source = result["source"].copy()

    for ct in pd.unique(pred_labels[missing]):
        query_idx = np.flatnonzero(missing & (pred_labels == ct))
        donor_idx = cache["type_indices"].get(str(ct), np.empty(0, dtype=np.int64))

        if not len(donor_idx):
            if not hierarchical:
                source[query_idx] = "missing_no_same_type"
                continue

            donor_idx = cache["all_indices"]
            source_name = "microenv_all_cells_missing_type"

        else:
            source_name = "microenv_same_type"

        donor_norm = cache["neighbor_norm"][donor_idx]

        for chunk_no, start in enumerate(range(0, len(query_idx), 1024)):
            idx = query_idx[start:start + 1024]
            probability = expr_similarity_probability(pred_norm[idx], donor_norm)
            current_seed = int(seed + zlib.crc32(str(ct).encode()) % 1_000_000 + chunk_no * 100003)

            X[idx] = expr_sample_features(probability, cache["X"], donor_idx, current_seed)
            available[idx] = True
            source[idx] = source_name

    result["X"], result["available"], result["source"] = X, available, source
    return result


# ============================================================
# 8. Pair donor cache
# ============================================================

def expr_common_features(left, right):
    left_names = np.asarray(left.var_names).astype(str)
    right_names = np.asarray(right.var_names).astype(str)

    right_map = {g: i for i, g in enumerate(right_names)}
    common = [g for g in left_names if g in right_map]

    if not common: raise ValueError(" observed endpoints have no common feature")

    left_map = {g: i for i, g in enumerate(left_names)}

    return (
        np.asarray(common, dtype=str),
        np.asarray([left_map[g] for g in common], dtype=np.int64),
        np.asarray([right_map[g] for g in common], dtype=np.int64),
    )


def expr_build_pair_cache(pair):
    left_token, right_token = pred_pair_to_real_slices[pair]
    left_key, right_key = adata_resolve_real_key(left_token), adata_resolve_real_key(right_token)

    left_adata, right_adata = anndata_dict[left_key], anndata_dict[right_key]

    left_maps = ctp_get_count_maps(outputs[ctp_resolve_output_key(outputs, left_token)], len(expression_celltypes))
    right_maps = ctp_get_count_maps(outputs[ctp_resolve_output_key(outputs, right_token)], len(expression_celltypes))

    if same_omics:
        features, left_idx, right_idx = expr_common_features(left_adata, right_adata)

        left_cache = expr_build_real_cache(left_adata, left_maps, left_idx, features, left_key)
        right_cache = expr_build_real_cache(right_adata, right_maps, right_idx, features, right_key)

    else:
        features = None
        left_cache = expr_build_real_cache(left_adata, left_maps, slice_name=left_key)
        right_cache = expr_build_real_cache(right_adata, right_maps, slice_name=right_key)

    return {
        "left": left_cache,
        "right": right_cache,
        "features": features,
        "left_key": left_key,
        "right_key": right_key,
    }


# ============================================================
# 9. Same-omics fusion weights
# ============================================================

def expr_fusion_weights(pred_adata, frame, model_name, pair, image_index, H, W, t):
    N = pred_adata.n_obs

    if not use_mask:
        return np.full(N, 1 - t, dtype=np.float32), np.full(N, t, dtype=np.float32), "time"

    model_root = ctp_resolve_model_result_root(save_dir, model_name)
    mask_df = ctp_compute_site_mask_weights(frame, model_root, pair, image_index, H, W, mask_mode)

    mask_df = mask_df.set_index("site_id")
    site_id = pred_adata.obs["site_id"].to_numpy(dtype=np.int64)

    left = mask_df["mask_left_weight"].reindex(site_id).to_numpy(dtype=np.float32)
    right = mask_df["mask_right_weight"].reindex(site_id).to_numpy(dtype=np.float32)

    if not np.isfinite(left).all() or not np.isfinite(right).all(): raise RuntimeError(" some predicted cells cannot match MASK site weights ")

    return left, right, f"mask:{mask_mode}"


def expr_fuse_same_omics(left, right, raw_left, raw_right):
    left_ok, right_ok = left["available"], right["available"]
    both, only_left, only_right = left_ok & right_ok, left_ok & ~right_ok, right_ok & ~left_ok
    neither = ~left_ok & ~right_ok

    if np.any(neither): raise RuntimeError(f"{int(neither.sum())}  predicted cells have no expression from either endpoint ")

    lw, rw = np.zeros(len(left_ok), dtype=np.float32), np.zeros(len(left_ok), dtype=np.float32)

    lw[both], rw[both] = raw_left[both], raw_right[both]
    lw[only_left], rw[only_right] = 1, 1

    s = lw + rw
    lw[s > 0] /= s[s > 0]
    rw[s > 0] /= s[s > 0]

    X = np.zeros_like(left["X"], dtype=np.float32)

    if np.any(both): X[both] = lw[both, None] * left["X"][both] + rw[both, None] * right["X"][both]
    if np.any(only_left): X[only_left] = left["X"][only_left]
    if np.any(only_right): X[only_right] = right["X"][only_right]

    expr_assert_finite(X, "same-omics final expression")

    return X, lw, rw, both, only_left, only_right


# ============================================================
# 10. Predict one reconstructed slice
# ============================================================
def expr_predict_middle(pred_adata, frame, cache, model_name, pair, image_name, image_index, seed):
    t = float(pred_adata.uns["interp_fraction"])

    left = expr_propagate_side(pred_adata, frame, cache["left"], "left")
    right = expr_propagate_side(pred_adata, frame, cache["right"], "right")

    # Microenvironment fallback only when required
    if same_omics:
        fallback_mask = (~left["available"]) & (~right["available"])
    else:
        fallback_mask = (~left["available"]) | (~right["available"])

    if np.any(fallback_mask):
        labels = pred_adata.obs[celltype_col].astype(str).to_numpy()
        pred_norm = expr_normalize_features(expr_neighbor_features(pred_adata.obsm["spatial"], labels))

        left = expr_apply_fallback(left, pred_adata, cache["left"], pred_norm, seed + 101, hierarchical=not same_omics, fallback_mask=fallback_mask)
        right = expr_apply_fallback(right, pred_adata, cache["right"], pred_norm, seed + 202, hierarchical=not same_omics, fallback_mask=fallback_mask)

    obs = pred_adata.obs.copy()
    obs["expr_left_available"] = left["available"]; obs["expr_right_available"] = right["available"]
    obs["expr_left_source"] = left["source"]; obs["expr_right_source"] = right["source"]

    if same_omics:
        H, W = cache["left"]["count_maps"].shape[1:]
        raw_left, raw_right, fusion_source = expr_fusion_weights(pred_adata, frame, model_name, pair, image_index, H, W, t)

        X, lw, rw, both, only_left, only_right = expr_fuse_same_omics(left, right, raw_left, raw_right)

        obs["expr_left_weight"] = lw
        obs["expr_right_weight"] = rw

        out = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=cache["features"]))
        out.layers["pred_from_left"] = np.nan_to_num(left["X"]).astype(np.float32)
        out.layers["pred_from_right"] = np.nan_to_num(right["X"]).astype(np.float32)

        n_both, n_only_left, n_only_right = int(both.sum()), int(only_left.sum()), int(only_right.sum())

    else:
        if np.any(~left["available"]): raise RuntimeError(f"LEFT still has  {int((~left['available']).sum())}  cells have missing expression ")
        if np.any(~right["available"]): raise RuntimeError(f"RIGHT still has  {int((~right['available']).sum())}  cells have missing expression ")

        expr_assert_finite(left["X"], "multi-omics left")
        expr_assert_finite(right["X"], "multi-omics right")

        out = ad.AnnData(obs=obs)

        out.obsm[f"pred_{left_omics_name}"] = left["X"]
        out.obsm[f"pred_{right_omics_name}"] = right["X"]

        out.uns["left_feature_names"] = cache["left"]["feature_names"].tolist()
        out.uns["right_feature_names"] = cache["right"]["feature_names"].tolist()

        fusion_source = "none_different_omics"
        n_both = n_only_left = n_only_right = 0

    for key in pred_adata.obsm:
        if key not in out.obsm: out.obsm[key] = pred_adata.obsm[key].copy()

    original_uns = dict(pred_adata.uns)
    original_uns.update(out.uns)
    out.uns = original_uns

    out.uns.update({
        "model": model_name,
        "pair_name": pair,
        "image_name": image_name,
        "image_index": int(image_index),
        "prediction_t": t,
        "same_omics": same_omics,
        "prediction_method": "flow_local_plus_microenvironment",
        "expression_fusion_source": fusion_source,
        "record_kind": "predicted_middle",
    })

    left_source, right_source = left["source"].astype(str), right["source"].astype(str)

    summary = {
        "model": model_name,
        "pair_name": pair,
        "image_name": image_name,
        "n_cells": pred_adata.n_obs,

        "left_flow_local": int(np.sum(left_source == "flow_local_same_type")),
        "right_flow_local": int(np.sum(right_source == "flow_local_same_type")),

        "left_microenv": int(np.sum(np.char.startswith(left_source, "microenv_"))),
        "right_microenv": int(np.sum(np.char.startswith(right_source, "microenv_"))),

        "left_available": int(left["available"].sum()),
        "right_available": int(right["available"].sum()),

        "both_sides": n_both,
        "only_left": n_only_left,
        "only_right": n_only_right,

        "fusion_source": fusion_source,
    }

    return out, summary


# ============================================================
# 11. Multi-omics: predict missing modality for real endpoints
# ============================================================

def expr_predict_real_missing(query_adata, donor_cache, missing_omics_name, seed):
    labels = query_adata.obs[celltype_col].astype(str).to_numpy()
    query_norm = expr_normalize_features(expr_neighbor_features(query_adata.obsm["spatial"], labels))

    if donor_cache["neighbor_features"] is None:
        donor_cache["neighbor_features"] = expr_neighbor_features(donor_cache["coords"], donor_cache["labels"])
        donor_cache["neighbor_norm"] = expr_normalize_features(donor_cache["neighbor_features"])

    X_pred = np.empty((query_adata.n_obs, donor_cache["X"].shape[1]), dtype=np.float32)
    source = np.empty(query_adata.n_obs, dtype=object)

    for ct in pd.unique(labels):
        query_idx = np.flatnonzero(labels == ct)
        donor_idx = donor_cache["type_indices"].get(str(ct), np.empty(0, dtype=np.int64))

        if len(donor_idx):
            source_name = "microenv_same_type"
        else:
            donor_idx = donor_cache["all_indices"]
            source_name = "microenv_all_cells_missing_type"

        donor_norm = donor_cache["neighbor_norm"][donor_idx]

        for chunk_no, start in enumerate(range(0, len(query_idx), 1024)):
            idx = query_idx[start:start + 1024]
            prob = expr_similarity_probability(query_norm[idx], donor_norm)
            current_seed = int(seed + zlib.crc32(str(ct).encode()) % 1_000_000 + chunk_no * 100003)

            X_pred[idx] = expr_sample_features(prob, donor_cache["X"], donor_idx, current_seed)
            source[idx] = source_name

    expr_assert_finite(X_pred, f"real endpoint missing {missing_omics_name}")

    out = query_adata.copy()
    out.obsm[f"pred_{missing_omics_name}"] = X_pred
    out.obs[f"{missing_omics_name}_prediction_source"] = source
    out.uns["missing_omics_name"] = missing_omics_name
    out.uns["missing_feature_names"] = donor_cache["feature_names"].tolist()
    out.uns["record_kind"] = "real_endpoint_missing_omics"

    return out


# ============================================================
# 12. Complete pipeline
# ============================================================

flow_df = expr_prepare_flow_df(flow_mapped_pixel_df)

flow_index = {
    (str(k[0]), str(k[1]), int(k[2])): np.asarray(v, dtype=np.int64)
    for k, v in flow_df.groupby(["model", "pair", "image_index"], sort=False, observed=True).indices.items()
}

pair_cache = {}
processed_endpoint_pairs = set()

pred_expr_adata_dict = {}
summary_rows = []


for key, pred_adata in pred_adata_dict.items():
    current_model = str(pred_adata.uns["model"])
    pair = str(pred_adata.uns["pair_name"])
    image_name = str(pred_adata.uns["image_name"])
    image_index = int(pred_adata.uns["image_index"])

    if pair not in pair_cache: pair_cache[pair] = expr_build_pair_cache(pair)
    cache = pair_cache[pair]

    frame_idx = flow_index.get((current_model, pair, image_index))
    if frame_idx is None: raise RuntimeError(f"flow_mapped_pixel_df not found in  {current_model}/{pair}/img{image_index}")

    frame = flow_df.iloc[frame_idx]

    pred_sites = set(pred_adata.obs["site_id"].astype(int))
    flow_sites = set(frame["site_id"].astype(int))

    if not pred_sites.issubset(flow_sites):
        missing = pred_sites - flow_sites
        raise RuntimeError(f"{current_model}/{pair}/img{image_index}: {len(missing)}  predicted site missing FLOW mapping")

    seed = int(config["seed"] + zlib.crc32(f"{current_model}/{pair}/{image_name}".encode()) % 1_000_000)

    pred_expr, summary = expr_predict_middle(
        pred_adata, frame, cache,
        current_model, pair, image_name, image_index, seed
    )

    pred_expr.uns["left_real_slice"] = cache["left_key"]
    pred_expr.uns["right_real_slice"] = cache["right_key"]

    pred_expr_adata_dict[key] = pred_expr
    summary["key"] = key
    summary_rows.append(summary)

    # Multi-omics: each pair predicts endpoint missing modality once
    if not same_omics and pair not in processed_endpoint_pairs:
        processed_endpoint_pairs.add(pair)

        left_real = anndata_dict[cache["left_key"]]
        right_real = anndata_dict[cache["right_key"]]

        endpoint_seed = int(config["seed"] + zlib.crc32(f"{pair}/real_endpoint".encode()) % 1_000_000)

        left_missing = expr_predict_real_missing(left_real, cache["right"], right_omics_name, endpoint_seed + 100001)
        right_missing = expr_predict_real_missing(right_real, cache["left"], left_omics_name, endpoint_seed + 200001)

        left_missing.uns.update({"model": "REAL_ENDPOINT", "pair_name": pair, "endpoint_side": "left"})
        right_missing.uns.update({"model": "REAL_ENDPOINT", "pair_name": pair, "endpoint_side": "right"})

        pred_expr_adata_dict[f"REAL__{pair}__{cache['left_key']}__missing_{right_omics_name}"] = left_missing
        pred_expr_adata_dict[f"REAL__{pair}__{cache['right_key']}__missing_{left_omics_name}"] = right_missing


# ============================================================
# 13. Output dictionaries + summary
# ============================================================

summary_df = pd.DataFrame(summary_rows)

pred_expr_adata_dict_by_model = {
    model_name: {
        key: obj for key, obj in pred_expr_adata_dict.items()
        if obj.uns.get("record_kind") == "predicted_middle"
    }
}

pred_expr_real_endpoint_dict = {
    key: obj for key, obj in pred_expr_adata_dict.items()
    if obj.uns.get("record_kind") == "real_endpoint_missing_omics"
}

if len(summary_df):
    model_expression_summary_df = summary_df.groupby("model", sort=False).agg(
        n_frames=("image_name", "size"),
        n_cells=("n_cells", "sum"),
        left_flow_local=("left_flow_local", "sum"),
        right_flow_local=("right_flow_local", "sum"),
        left_microenv=("left_microenv", "sum"),
        right_microenv=("right_microenv", "sum"),
        left_available=("left_available", "sum"),
        right_available=("right_available", "sum"),
    ).reset_index()
else:
    model_expression_summary_df = pd.DataFrame()


print(
    f"Expression propagation finished | model={model_name} | "
    f"predicted slices={len(pred_expr_adata_dict_by_model[model_name])} | "
    f"real endpoint predictions={len(pred_expr_real_endpoint_dict)} | "
    f"same_omics={same_omics}"
)

display(summary_df)

if len(model_expression_summary_df): display(model_expression_summary_df)