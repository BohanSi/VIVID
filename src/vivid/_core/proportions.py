# Migrated from 1002genertation.ipynb, cell 45. See README.md.
import re
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from IPython.display import display


SITE_KEYS = ["model", "pair", "image", "image_index", "site_id"]


# ============================================================
# Utilities
# ============================================================

def ctp_safe_name(x): return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(x))


def ctp_norm_slice_id(x):
    s = str(x).strip()
    m = re.search(r"slice[_\s-]*(\d+)", s, flags=re.I)
    if m: return int(m.group(1))
    return int(s) if re.fullmatch(r"\d+", s) else None


def ctp_parse_pair(pair_name):
    pair_name = str(pair_name)
    if "_to_" not in pair_name: raise ValueError(f" cannot parse  pair: {pair_name}")
    return tuple(pair_name.split("_to_", 1))


def ctp_resolve_output_key(outputs, slice_tag):
    if slice_tag in outputs: return slice_tag
    target_id = ctp_norm_slice_id(slice_tag)

    for key in outputs:
        if str(key) == str(slice_tag) or (target_id is not None and ctp_norm_slice_id(key) == target_id): return key

    raise KeyError(f"outputs  not found in  {slice_tag};  available  keys={list(outputs.keys())}")


def ctp_required_columns(df, cols, name):
    missing = [c for c in cols if c not in df.columns]
    if missing: raise KeyError(f"{name}  missing columns : {missing}")


def _ctp_const(value, n):
    out = np.empty(int(n), dtype=object)
    out[:] = value
    return out


def _ctp_clear_attrs(*dfs):
    for df in dfs:
        if isinstance(df, pd.DataFrame): df.attrs.clear()


def _ctp_offset_code(dx, dy):
    dx = np.asarray(dx, dtype=np.int64)
    dy = np.asarray(dy, dtype=np.int64)
    return (dx + 32768) * 65537 + (dy + 32768)


# ============================================================
# Neighborhood

# ============================================================

def ctp_make_neighborhood_offsets(radius_pixels=4):
    radius = int(radius_pixels)
    if radius < 0: raise ValueError("radius_pixels must not be less than 0")

    dy, dx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    dist = np.hypot(dx, dy)
    keep = dist <= radius + 1e-12

    dx = dx[keep].astype(np.int64)
    dy = dy[keep].astype(np.int64)
    dist = dist[keep].astype(np.float64)

    return dx, dy, dist, 1.0 / (1.0 + dist), (dx == 0) & (dy == 0)


# ============================================================
# Reuse previous FLOW mapping
# ============================================================

def _ctp_prepare_previous_mapping(previous_df, model_name):
    if not isinstance(previous_df, pd.DataFrame) or len(previous_df) == 0:
        raise RuntimeError(" cannot find previous-stage  flow_mapped_pixel_df,  run integer count estimation first /FLOW mapping.")

    required = [
        "model", "pair", "image_index", "center_pixel_u", "center_pixel_v",
        "pixel_u", "pixel_v", "offset_u", "offset_v",
        "left_real_x", "left_real_y", "right_real_x", "right_real_y",
        "left_real_pixel_u", "left_real_pixel_v",
        "right_real_pixel_u", "right_real_pixel_v",
        "left_mapping_valid", "right_mapping_valid",
    ]
    ctp_required_columns(previous_df, required, " previous-stage  flow_mapped_pixel_df")

    idx = np.flatnonzero(previous_df["model"].astype(str).to_numpy() == str(model_name))
    if len(idx) == 0: raise RuntimeError(f" previous-stage  flow_mapped_pixel_df  does not contain  model={model_name}")

    view = previous_df.iloc[idx]
    groups = {
        (str(k[0]), int(k[1])): idx[np.asarray(v, dtype=np.int64)]
        for k, v in view.groupby(["pair", "image_index"], sort=False, observed=True).indices.items()
    }

    return previous_df, groups


def _ctp_reuse_frame_mapping(
    src, idx, center_u, center_v, H, W, dx, dy, dist, spatial_w,
    is_center, pair_name, image_index
):
    n_sites = len(center_u)
    center_code = center_v.astype(np.int64) * W + center_u.astype(np.int64)

    src_cu = src["center_pixel_u"].to_numpy(dtype=np.int64, copy=False)[idx]
    src_cv = src["center_pixel_v"].to_numpy(dtype=np.int64, copy=False)[idx]
    src_center_code = src_cv * W + src_cu

    new_site = np.searchsorted(center_code, src_center_code)
    center_ok = (new_site < n_sites) & (center_code[np.minimum(new_site, n_sites - 1)] == src_center_code)

    src_dx = src["offset_u"].to_numpy(dtype=np.int64, copy=False)[idx]
    src_dy = src["offset_v"].to_numpy(dtype=np.int64, copy=False)[idx]

    req_code = _ctp_offset_code(dx, dy)
    order = np.argsort(req_code)
    sorted_code = req_code[order]

    src_code = _ctp_offset_code(src_dx, src_dy)
    pos = np.searchsorted(sorted_code, src_code)
    pos_safe = np.minimum(pos, len(sorted_code) - 1)

    keep = center_ok & (pos < len(sorted_code)) & (sorted_code[pos_safe] == src_code)

    if not np.any(keep):
        raise RuntimeError(f"{pair_name}/img{image_index}:  previous-stage mapping and current neighborhood parameters do not match ")

    idx = idx[keep]
    new_site = new_site[keep].astype(np.int64)
    rank = order[np.searchsorted(sorted_code, src_code[keep])]

    actual = np.bincount(new_site, minlength=n_sites)
    expected = np.zeros(n_sites, dtype=np.int64)

    for ox, oy in zip(dx, dy):
        expected += ((center_u + ox >= 0) & (center_u + ox < W) &
                     (center_v + oy >= 0) & (center_v + oy < H)).astype(np.int64)

    if not np.array_equal(actual, expected):
        raise RuntimeError(
            f"{pair_name}/img{image_index}:  current  neighborhood  exceeds previously saved mapping range ; "
            " keep support radius consistent ."
        )

    pu = src["pixel_u"].to_numpy(dtype=np.int64, copy=False)[idx]
    pv = src["pixel_v"].to_numpy(dtype=np.int64, copy=False)[idx]
    lx = src["left_real_x"].to_numpy(dtype=np.float64, copy=False)[idx]
    ly = src["left_real_y"].to_numpy(dtype=np.float64, copy=False)[idx]
    rx = src["right_real_x"].to_numpy(dtype=np.float64, copy=False)[idx]
    ry = src["right_real_y"].to_numpy(dtype=np.float64, copy=False)[idx]

    return {
        "site": new_site,
        "center_u": center_u[new_site], "center_v": center_v[new_site],
        "pixel_u": pu, "pixel_v": pv,
        "dx": dx[rank], "dy": dy[rank], "dist": dist[rank],
        "spatial_w": spatial_w[rank], "is_center": is_center[rank],
        "left_x": lx, "left_y": ly, "right_x": rx, "right_y": ry,
        "left_u": src["left_real_pixel_u"].to_numpy(dtype=np.int64, copy=False)[idx],
        "left_v": src["left_real_pixel_v"].to_numpy(dtype=np.int64, copy=False)[idx],
        "right_u": src["right_real_pixel_u"].to_numpy(dtype=np.int64, copy=False)[idx],
        "right_v": src["right_real_pixel_v"].to_numpy(dtype=np.int64, copy=False)[idx],
        "left_valid": src["left_mapping_valid"].to_numpy(dtype=bool, copy=False)[idx],
        "right_valid": src["right_mapping_valid"].to_numpy(dtype=bool, copy=False)[idx],
        "left_flow_dx": lx - pu, "left_flow_dy": ly - pv,
        "right_flow_dx": rx - pu, "right_flow_dy": ry - pv,
    }


# ============================================================
# Integer count map -> CENTER + neighborhood FLOW table
# ============================================================

def build_flow_tables_from_integer_count_maps(
    integer_count_results, global_flow_results, previous_flow_mapped_pixel_df,
    save_dir, neighborhood_radius_pixels=4, model_name="RIFE"
):
    model_name = str(model_name)
    dx, dy, dist, spatial_w, is_center = ctp_make_neighborhood_offsets(neighborhood_radius_pixels)
    src, groups = _ctp_prepare_previous_mapping(previous_flow_mapped_pixel_df, model_name)

    endpoints = {}

    for pair, info in global_flow_results.items():
        frames = info["frames"]
        if not frames: continue

        endpoints[str(pair)] = {
            "left_real_image": Path(info["left_image"]).name,
            "right_real_image": Path(info["right_image"]).name,
            "left_real_index": 0,
            "right_real_index": max(int(x["image_index"]) for x in frames) + 1,
        }

    site_frames, pixel_frames = [], []

    for pair_name, frames in integer_count_results.items():
        pair_name = str(pair_name)

        if pair_name not in endpoints:
            raise KeyError(f"{pair_name}: global_flow_results not found in ")

        for frame in frames:
            image_index = int(frame["image_index"])
            key = (pair_name, image_index)

            if key not in groups:
                raise KeyError(f" previous-stage flow_mapped_pixel_df cannot find  {key}")

            with np.load(frame["npz_path"]) as f:
                count_map = np.asarray(f["count_map"], dtype=np.int32)
                target = np.asarray(f["target_density"], dtype=np.float64) if "target_density" in f.files else np.zeros_like(count_map, dtype=np.float64)
                ratio = float(f["ratio"]) if "ratio" in f.files else float(frame["ratio"])

            H, W = count_map.shape

            if target.shape != (H, W):
                raise ValueError(f"target_density={target.shape}, count_map={(H, W)}")

            center_v, center_u = np.where(count_map > 0)
            n_sites = len(center_u)

            if n_sites == 0: continue

            cell_number = count_map[center_v, center_u].astype(np.int64)

            m = _ctp_reuse_frame_mapping(
                src, groups[key], center_u, center_v, H, W,
                dx, dy, dist, spatial_w, is_center, pair_name, image_index
            )

            site = m["site"]
            row_cells = cell_number[site]
            middle = target[m["pixel_v"], m["pixel_u"]]

            
            # inverse-distance × interpolated target density
            raw_w = m["spatial_w"] * np.maximum(middle, 0.0)
            wsum = np.bincount(site, weights=raw_w, minlength=n_sites)

            
            zero = wsum <= 1e-12

            if np.any(zero):
                raw_w[zero[site]] = m["spatial_w"][zero[site]]
                wsum = np.bincount(site, weights=raw_w, minlength=n_sites)

            allocated = row_cells.astype(np.float64) * raw_w / np.maximum(wsum[site], 1e-12)

            n_pix = np.bincount(site, minlength=n_sites).astype(np.int64)
            neigh_mass = np.bincount(site, weights=middle, minlength=n_sites)

            nlv = np.bincount(
                site, weights=m["left_valid"].astype(np.int8), minlength=n_sites
            ).astype(np.int64)

            nrv = np.bincount(
                site, weights=m["right_valid"].astype(np.int8), minlength=n_sites
            ).astype(np.int64)

            keep = (nlv > 0) | (nrv > 0)

            site_df = pd.DataFrame({
                "model": _ctp_const(model_name, n_sites),
                "pair": _ctp_const(pair_name, n_sites),
                "image": _ctp_const(f"img{image_index}.png", n_sites),
                "image_index": np.full(n_sites, image_index, dtype=np.int64),
                "site_id": np.arange(n_sites, dtype=np.int64),
                "center_pixel_u": center_u,
                "center_pixel_v": center_v,
                "cell_number": cell_number,
                "neighborhood_mass": neigh_mass,
                "n_neighborhood_pixels": n_pix,
                "n_left_valid_mapped_pixels": nlv,
                "n_right_valid_mapped_pixels": nrv,
                "interp_fraction": np.full(n_sites, ratio),
                "keep_after_flow_screen": keep,
            })

            n = len(site)
            ep = endpoints[pair_name]

            pixel_df = pd.DataFrame({
                "model": _ctp_const(model_name, n),
                "pair": _ctp_const(pair_name, n),
                "image": _ctp_const(f"img{image_index}.png", n),
                "image_index": np.full(n, image_index, dtype=np.int64),
                "site_id": site,
                "center_pixel_u": m["center_u"],
                "center_pixel_v": m["center_v"],
                "cell_number": row_cells,
                "pixel_u": m["pixel_u"],
                "pixel_v": m["pixel_v"],
                "offset_u": m["dx"],
                "offset_v": m["dy"],
                "neighborhood_distance": m["dist"],
                "is_center_pixel": m["is_center"],
                "middle_target_density": middle,
                "spatial_weight": m["spatial_w"],
                "allocated_density": allocated,
                "left_flow_dx": m["left_flow_dx"],
                "left_flow_dy": m["left_flow_dy"],
                "right_flow_dx": m["right_flow_dx"],
                "right_flow_dy": m["right_flow_dy"],
                "left_real_x": m["left_x"],
                "left_real_y": m["left_y"],
                "right_real_x": m["right_x"],
                "right_real_y": m["right_y"],
                "left_real_pixel_u": m["left_u"],
                "left_real_pixel_v": m["left_v"],
                "right_real_pixel_u": m["right_u"],
                "right_real_pixel_v": m["right_v"],
                "left_mapping_valid": m["left_valid"],
                "right_mapping_valid": m["right_valid"],
                "left_real_image": _ctp_const(ep["left_real_image"], n),
                "right_real_image": _ctp_const(ep["right_real_image"], n),
                "left_real_index": np.full(n, ep["left_real_index"], dtype=np.int64),
                "right_real_index": np.full(n, ep["right_real_index"], dtype=np.int64),
                "interp_fraction": np.full(n, ratio),
                "keep_after_flow_screen": keep[site],
            })

            site_frames.append(site_df)
            pixel_frames.append(pixel_df)

    if not site_frames:
        raise RuntimeError(" none constructed site.")

    all_sites = pd.concat(site_frames, ignore_index=True, copy=False)
    all_pixels = pd.concat(pixel_frames, ignore_index=True, copy=False)

    retained = all_sites.loc[
        all_sites["keep_after_flow_screen"].to_numpy(dtype=bool)
    ].copy().reset_index(drop=True)

    removed = all_sites.loc[
        ~all_sites["keep_after_flow_screen"].to_numpy(dtype=bool)
    ].copy().reset_index(drop=True)

    # allocated_density conservation
    pixel_mi = pd.MultiIndex.from_frame(all_pixels[SITE_KEYS])
    codes, unique = pd.factorize(pixel_mi, sort=False)

    alloc = np.bincount(
        codes,
        weights=all_pixels["allocated_density"].to_numpy(dtype=np.float64),
        minlength=len(unique),
    )

    loc = unique.get_indexer(pd.MultiIndex.from_frame(all_sites[SITE_KEYS]))

    if np.any(loc < 0):
        raise RuntimeError("allocated_density check found site key missing ")

    max_error = float(
        np.max(
            np.abs(
                alloc[loc]
                - all_sites["cell_number"].to_numpy(dtype=np.float64)
            )
        )
    )

    if max_error > 1e-6:
        raise RuntimeError(f"allocated_density total mass is not conserved , max error={max_error}")

    root = Path(save_dir).resolve() / "flow_count_pixel_mapping"
    root.mkdir(parents=True, exist_ok=True)

    all_sites.to_csv(root / "all_sites.csv", index=False, encoding="utf-8-sig")
    retained.to_csv(root / "retained_sites.csv", index=False, encoding="utf-8-sig")
    removed.to_csv(root / "removed_invalid_flow_sites.csv", index=False, encoding="utf-8-sig")
    all_pixels.to_csv(root / "all_neighborhood_flow_mapping.csv", index=False, encoding="utf-8-sig")

    _ctp_clear_attrs(all_sites, retained, removed, all_pixels)

    print(
        f"FLOW mapping finished | sites={len(retained):,}/{len(all_sites):,} | "
        f"cells={int(retained['cell_number'].sum()):,} | rows={len(all_pixels):,} | "
        f"allocation error={max_error:.2e}"
    )

    return retained, all_pixels, removed


# ============================================================
# Endpoint cell-type count maps
# ============================================================

def ctp_get_count_maps(result, K):
    if "count_maps" in result:
        cm = np.asarray(result["count_maps"], dtype=np.float64)

        if cm.ndim == 3 and cm.shape[0] == K: return cm
        if cm.ndim == 3 and cm.shape[-1] == K: return np.transpose(cm, (2, 0, 1))

        raise ValueError(f"count_maps shape invalid : {cm.shape}")

    if "P" not in result or "T_count" not in result:
        raise KeyError("outputs does not contain count_maps,  cannot recover from P×T_count recover .")

    P = np.asarray(result["P"], dtype=np.float64)
    T = np.asarray(result["T_count"], dtype=np.float64)

    H, W = T.shape

    if P.shape == (K, H, W): return P * T[None]
    if P.shape == (H, W, K): return np.transpose(P, (2, 0, 1)) * T[None]

    raise ValueError(f"P shape={P.shape}  cannot parse ")


def _ctp_get_maps(outputs, key, K, cache):
    if key not in cache: cache[key] = ctp_get_count_maps(outputs[key], K)
    return cache[key]


# ============================================================
# Prepare site codes
# ============================================================

def _ctp_prepare_site_codes(df):
    codes, site_ids = pd.factorize(
        df["site_id"].to_numpy(dtype=np.int64, copy=False),
        sort=False,
    )

    codes = codes.astype(np.int64)

    _, first = np.unique(codes, return_index=True)
    first = first[np.argsort(codes[first])]

    site_keys = df.iloc[first][SITE_KEYS].reset_index(drop=True).copy()
    weights = np.maximum(
        df["allocated_density"].to_numpy(dtype=np.float64, copy=False),
        0.0,
    )

    return codes, site_keys, weights, len(site_ids)


# ============================================================
# Bilinear endpoint cell-type sampling
# ============================================================

def _ctp_aggregate_side_bilinear(
    count_maps, df, codes, weights, n_sites, side, chunk_size=250000
):
    K, H, W = count_maps.shape

    valid_all = df[f"{side}_mapping_valid"].to_numpy(dtype=bool, copy=False)
    x_all = df[f"{side}_real_x"].to_numpy(dtype=np.float64, copy=False)
    y_all = df[f"{side}_real_y"].to_numpy(dtype=np.float64, copy=False)

    out = np.zeros((n_sites, K), dtype=np.float64)
    weight_sum = np.zeros(n_sites, dtype=np.float64)
    n_evidence = np.zeros(n_sites, dtype=np.int64)

    chunk_size = max(int(chunk_size), 1)

    for start in range(0, len(df), chunk_size):
        end = min(start + chunk_size, len(df))

        site_code = codes[start:end]
        pixel_weight = weights[start:end]
        x = x_all[start:end]
        y = y_all[start:end]

        inside = (
            valid_all[start:end]
            & np.isfinite(x)
            & np.isfinite(y)
            & (x >= 0)
            & (x <= W - 1)
            & (y >= 0)
            & (y <= H - 1)
        )

        idx = np.flatnonzero(inside)

        if len(idx) == 0: continue

        xv, yv = x[idx], y[idx]

        x0 = np.floor(xv).astype(np.int64)
        y0 = np.floor(yv).astype(np.int64)

        x1 = np.minimum(x0 + 1, W - 1)
        y1 = np.minimum(y0 + 1, H - 1)

        wx, wy = xv - x0, yv - y0

        # Bilinear sampling
        counts = (
            (1 - wx)[:, None] * (1 - wy)[:, None] * count_maps[:, y0, x0].T
            + wx[:, None] * (1 - wy)[:, None] * count_maps[:, y0, x1].T
            + (1 - wx)[:, None] * wy[:, None] * count_maps[:, y1, x0].T
            + wx[:, None] * wy[:, None] * count_maps[:, y1, x1].T
        )

        total = counts.sum(axis=1)
        has_evidence = total > 1e-12

        if not np.any(has_evidence): continue

        idx = idx[has_evidence]
        counts = counts[has_evidence]
        total = total[has_evidence]

        site = site_code[idx]
        w = pixel_weight[idx]

        n_evidence += np.bincount(site, minlength=n_sites).astype(np.int64)
        weight_sum += np.bincount(site, weights=w, minlength=n_sites)

        scale = w / total

        for k in range(K):
            out[:, k] += np.bincount(
                site,
                weights=counts[:, k] * scale,
                minlength=n_sites,
            )

    valid_site = weight_sum > 1e-12
    probs = np.zeros_like(out)

    if np.any(valid_site):
        probs[valid_site] = out[valid_site] / weight_sum[valid_site, None]
        probs[valid_site] /= np.maximum(
            probs[valid_site].sum(axis=1, keepdims=True),
            1e-12,
        )

    return probs, valid_site, n_evidence, weight_sum


# ============================================================
# RIFE MASK
# ============================================================

def ctp_resolve_model_result_root(save_dir, model_name):
    save_dir = Path(save_dir).resolve()

    candidates = [
        save_dir / f"{model_name}_result",
        save_dir / f"gaussian_{model_name}_result" / "global",
    ]

    for root in candidates:
        if root.exists(): return root

    raise FileNotFoundError(
        f" cannot find  {model_name}  result directory ;  check  save_dir={save_dir}"
    )


def ctp_load_frame_masks(model_root, pair_name, image_index, H, W):
    path = Path(model_root) / str(pair_name) / "flow" / f"img{int(image_index)}.npz"

    if not path.exists():
        raise FileNotFoundError(f"use_mask=True,  but cannot find  MASK  file : {path}")

    with np.load(path) as f:
        if "mask_to_left" not in f.files or "mask_to_right" not in f.files:
            raise KeyError(
                f"use_mask=True,  but  {path}  missing  mask_to_left / mask_to_right"
            )

        left = np.asarray(f["mask_to_left"], dtype=np.float64)
        right = np.asarray(f["mask_to_right"], dtype=np.float64)

    if left.shape != (H, W) or right.shape != (H, W):
        raise ValueError(
            f"MASK shape invalid : left={left.shape}, right={right.shape}, expected={(H, W)}"
        )

    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError(f"{path}: MASK contains NaN/Inf")

    left, right = np.clip(left, 0, 1), np.clip(right, 0, 1)

    total = left + right
    valid = total > 1e-12

    norm_left = np.full_like(left, 0.5)
    norm_right = np.full_like(right, 0.5)

    norm_left[valid] = left[valid] / total[valid]
    norm_right[valid] = right[valid] / total[valid]

    return norm_left, norm_right


def ctp_compute_site_mask_weights(
    frame_pixels, model_root, pair_name, image_index, H, W, mask_mode
):
    valid_modes = {
        "center",
        "neighborhood_mean",
        "center_neighborhood_mean",
    }

    if mask_mode not in valid_modes:
        raise ValueError(f"mask_mode must be  {sorted(valid_modes)}")

    mask_left, mask_right = ctp_load_frame_masks(
        model_root, pair_name, image_index, H, W
    )

    codes, site_ids = pd.factorize(
        frame_pixels["site_id"].to_numpy(dtype=np.int64, copy=False),
        sort=False,
    )

    codes = codes.astype(np.int64)
    n_sites = len(site_ids)

    u = frame_pixels["pixel_u"].to_numpy(dtype=np.int64, copy=False)
    v = frame_pixels["pixel_v"].to_numpy(dtype=np.int64, copy=False)
    w = np.maximum(
        frame_pixels["allocated_density"].to_numpy(dtype=np.float64, copy=False),
        0.0,
    )

    is_center = frame_pixels["is_center_pixel"].to_numpy(
        dtype=bool,
        copy=False,
    )

    if not np.all((u >= 0) & (u < W) & (v >= 0) & (v < H)):
        raise RuntimeError(
            f"{pair_name}/img{image_index}: neighborhood pixel  outside  MASK  range "
        )

    
    composite = (
        codes.astype(np.int64) * (np.int64(H) * W)
        + v.astype(np.int64) * W
        + u.astype(np.int64)
    )

    _, first = np.unique(composite, return_index=True)
    first = np.sort(first)

    c = codes[first]
    u = u[first]
    v = v[first]
    w = w[first]
    is_center = is_center[first]

    lpx = mask_left[v, u]
    rpx = mask_right[v, u]

    sw = np.bincount(c, weights=w, minlength=n_sites)

    neigh_left = np.divide(
        np.bincount(c, weights=lpx * w, minlength=n_sites),
        sw,
        out=np.full(n_sites, 0.5),
        where=sw > 1e-12,
    )

    neigh_right = np.divide(
        np.bincount(c, weights=rpx * w, minlength=n_sites),
        sw,
        out=np.full(n_sites, 0.5),
        where=sw > 1e-12,
    )

    center_left = np.full(n_sites, np.nan)
    center_right = np.full(n_sites, np.nan)

    if np.any(is_center):
        center_left[c[is_center]] = lpx[is_center]
        center_right[c[is_center]] = rpx[is_center]

    if mask_mode != "neighborhood_mean" and np.any(~np.isfinite(center_left)):
        raise RuntimeError(
            f"{pair_name}/img{image_index}:  some site missing center pixel, "
            f" cannot use  mask_mode={mask_mode}"
        )

    if mask_mode == "center":
        left, right = center_left, center_right

    elif mask_mode == "neighborhood_mean":
        left, right = neigh_left, neigh_right

    else:
        left = 0.5 * (center_left + neigh_left)
        right = 0.5 * (center_right + neigh_right)

    total = left + right

    left = np.divide(
        left,
        total,
        out=np.full(n_sites, 0.5),
        where=total > 1e-12,
    )

    right = np.divide(
        right,
        total,
        out=np.full(n_sites, 0.5),
        where=total > 1e-12,
    )

    return pd.DataFrame({
        "site_id": site_ids.astype(np.int64),
        "mask_left_weight": left,
        "mask_right_weight": right,
    })


# ============================================================
# One frame
# ============================================================

def ctp_process_one_frame(
    frame_pixels, outputs, celltypes, model_root=None,
    use_neighborhood_pixels=True,
    use_mask=False,
    mask_mode="center_neighborhood_mean",
    count_maps_cache=None,
    evidence_chunk_size=250000,
):
    cache = {} if count_maps_cache is None else count_maps_cache

    pair = str(frame_pixels["pair"].iloc[0])
    image_index = int(frame_pixels["image_index"].iloc[0])

    left_tag, right_tag = ctp_parse_pair(pair)
    K = len(celltypes)

    left_maps = _ctp_get_maps(
        outputs,
        ctp_resolve_output_key(outputs, left_tag),
        K,
        cache,
    )

    right_maps = _ctp_get_maps(
        outputs,
        ctp_resolve_output_key(outputs, right_tag),
        K,
        cache,
    )

    if left_maps.shape != right_maps.shape:
        raise ValueError(" left and right count_maps shape inconsistent ")

    _, H, W = left_maps.shape

    ratios = frame_pixels["interp_fraction"].dropna().unique()

    if len(ratios) != 1:
        raise RuntimeError(f"interp_fraction not unique : {ratios}")

    t = float(ratios[0])

    type_pixels = (
        frame_pixels
        if use_neighborhood_pixels
        else frame_pixels.loc[
            frame_pixels["is_center_pixel"].to_numpy(dtype=bool)
        ]
    )

    if len(type_pixels) == 0:
        raise RuntimeError(" missing type evidence pixels")

    codes, site_keys, weights, n_sites = _ctp_prepare_site_codes(type_pixels)

    left_probs, left_valid, left_n, left_wsum = _ctp_aggregate_side_bilinear(
        left_maps,
        type_pixels,
        codes,
        weights,
        n_sites,
        "left",
        evidence_chunk_size,
    )

    right_probs, right_valid, right_n, right_wsum = _ctp_aggregate_side_bilinear(
        right_maps,
        type_pixels,
        codes,
        weights,
        n_sites,
        "right",
        evidence_chunk_size,
    )

    # --------------------------------------------------------
    # Left / right fusion
    # --------------------------------------------------------

    if use_mask:
        if model_root is None:
            raise ValueError("use_mask=True  requires  model_root")

        mask_df = ctp_compute_site_mask_weights(
            frame_pixels,
            model_root,
            pair,
            image_index,
            H,
            W,
            mask_mode,
        )

        mask_index = pd.Index(
            mask_df["site_id"].to_numpy(dtype=np.int64)
        )

        loc = mask_index.get_indexer(
            site_keys["site_id"].to_numpy(dtype=np.int64)
        )

        if np.any(loc < 0):
            raise RuntimeError(
                f"{pair}/img{image_index}: MASK result is missing some site"
            )

        raw_left = mask_df["mask_left_weight"].to_numpy(dtype=np.float64)[loc]
        raw_right = mask_df["mask_right_weight"].to_numpy(dtype=np.float64)[loc]

        fusion_source = "mask"

    else:
        raw_left = np.full(n_sites, 1.0 - t, dtype=np.float64)
        raw_right = np.full(n_sites, t, dtype=np.float64)
        fusion_source = "time"

    left_weight = np.zeros(n_sites)
    right_weight = np.zeros(n_sites)

    both = left_valid & right_valid
    only_left = left_valid & ~right_valid
    only_right = right_valid & ~left_valid

    left_weight[both] = raw_left[both]
    right_weight[both] = raw_right[both]

    left_weight[only_left] = 1.0
    right_weight[only_right] = 1.0

    total = left_weight + right_weight
    valid_any = total > 1e-12

    left_weight[valid_any] /= total[valid_any]
    right_weight[valid_any] /= total[valid_any]

    fused_probs = (
        left_probs * left_weight[:, None]
        + right_probs * right_weight[:, None]
    )

    if np.any(valid_any):
        fused_probs[valid_any] /= np.maximum(
            fused_probs[valid_any].sum(axis=1, keepdims=True),
            1e-12,
        )

    prop_cols = [
        f"prop__{ctp_safe_name(ct)}"
        for ct in celltypes
    ]

    # --------------------------------------------------------
    # Left
    # --------------------------------------------------------

    left_meta = site_keys.copy()
    left_meta["interp_fraction"] = t
    left_meta["n_type_evidence_pixels"] = left_n
    left_meta["type_weight_sum"] = left_wsum
    left_meta["has_type_evidence"] = left_valid

    # --------------------------------------------------------
    # Right
    # --------------------------------------------------------

    right_meta = site_keys.copy()
    right_meta["interp_fraction"] = t
    right_meta["n_type_evidence_pixels"] = right_n
    right_meta["type_weight_sum"] = right_wsum
    right_meta["has_type_evidence"] = right_valid

    # --------------------------------------------------------
    # Fused
    # --------------------------------------------------------

    fused_meta = site_keys.copy()

    fused_meta["interp_fraction"] = t
    fused_meta["use_neighborhood_pixels"] = bool(use_neighborhood_pixels)
    fused_meta["real_type_sampling"] = "bilinear"

    fused_meta["use_mask_requested"] = bool(use_mask)
    fused_meta["mask_available"] = bool(use_mask)
    fused_meta["mask_mode"] = mask_mode if use_mask else "none"

    fused_meta["fusion_weight_source"] = fusion_source

    fused_meta["left_has_type_evidence"] = left_valid
    fused_meta["right_has_type_evidence"] = right_valid
    fused_meta["has_type_evidence"] = valid_any

    fused_meta["left_n_type_evidence_pixels"] = left_n
    fused_meta["right_n_type_evidence_pixels"] = right_n

    fused_meta["left_type_weight_sum"] = left_wsum
    fused_meta["right_type_weight_sum"] = right_wsum

    fused_meta["raw_left_fusion_weight"] = raw_left
    fused_meta["raw_right_fusion_weight"] = raw_right

    fused_meta["left_fusion_weight"] = left_weight
    fused_meta["right_fusion_weight"] = right_weight

    left_df = pd.concat(
        [
            left_meta.reset_index(drop=True),
            pd.DataFrame(left_probs, columns=prop_cols),
        ],
        axis=1,
    )

    right_df = pd.concat(
        [
            right_meta.reset_index(drop=True),
            pd.DataFrame(right_probs, columns=prop_cols),
        ],
        axis=1,
    )

    fused_df = pd.concat(
        [
            fused_meta.reset_index(drop=True),
            pd.DataFrame(fused_probs, columns=prop_cols),
        ],
        axis=1,
    )

    _ctp_clear_attrs(
        left_df,
        right_df,
        fused_df,
    )

    return left_df, right_df, fused_df


# ============================================================
# Complete pipeline
# ============================================================

def compute_flow_celltype_proportions(
    save_dir, interpolation_model_names,
    flow_screened_site_df, flow_mapped_pixel_df,
    outputs, celltypes,
    use_neighborhood_pixels=True,
    use_mask=False,
    mask_mode="center_neighborhood_mean",
    evidence_chunk_size=250000,
):
    site_df = flow_screened_site_df.copy()
    pixel_df = flow_mapped_pixel_df

    ctp_required_columns(
        site_df,
        SITE_KEYS + [
            "center_pixel_u",
            "center_pixel_v",
            "cell_number",
        ],
        "flow_screened_site_df",
    )

    ctp_required_columns(
        pixel_df,
        SITE_KEYS + [
            "pixel_u",
            "pixel_v",
            "is_center_pixel",
            "allocated_density",
            "left_real_x",
            "left_real_y",
            "right_real_x",
            "right_real_y",
            "left_mapping_valid",
            "right_mapping_valid",
            "interp_fraction",
        ],
        "flow_mapped_pixel_df",
    )

    if "keep_after_flow_screen" in pixel_df.columns:
        pixel_df = pixel_df.loc[
            pixel_df["keep_after_flow_screen"].to_numpy(dtype=bool)
        ].copy()

    else:
        pixel_df = pixel_df.merge(
            site_df[SITE_KEYS].drop_duplicates(),
            on=SITE_KEYS,
            how="inner",
            validate="many_to_one",
            sort=False,
        )

    out_root = Path(save_dir).resolve() / "flow_celltype_proportion"
    out_root.mkdir(parents=True, exist_ok=True)

    celltypes = [str(x) for x in celltypes]

    available_models = set(
        site_df["model"].astype(str).unique()
    )

    count_maps_cache = {}

    all_left = []
    all_right = []
    all_fused = []

    for model_name in map(str, interpolation_model_names):
        if model_name not in available_models:
            continue

        model_sites = site_df.loc[
            site_df["model"].astype(str).to_numpy()
            == model_name
        ].copy()

        model_pixels = pixel_df.loc[
            pixel_df["model"].astype(str).to_numpy()
            == model_name
        ]

        frame_groups = sorted(
            model_pixels.groupby(
                ["pair", "image", "image_index"],
                sort=False,
                observed=True,
            ).indices.items(),
            key=lambda x: (
                str(x[0][0]),
                int(x[0][2]),
            ),
        )

        model_root = (
            ctp_resolve_model_result_root(
                save_dir,
                model_name,
            )
            if use_mask
            else None
        )

        model_left = []
        model_right = []
        model_fused = []

        for _, idx in tqdm(
            frame_groups,
            total=len(frame_groups),
            desc=f"{model_name} cell-type proportions",
        ):
            frame_pixels = model_pixels.iloc[
                np.asarray(
                    idx,
                    dtype=np.int64,
                )
            ]

            left_df, right_df, fused_df = ctp_process_one_frame(
                frame_pixels,
                outputs,
                celltypes,
                model_root=model_root,
                use_neighborhood_pixels=use_neighborhood_pixels,
                use_mask=use_mask,
                mask_mode=mask_mode,
                count_maps_cache=count_maps_cache,
                evidence_chunk_size=evidence_chunk_size,
            )

            model_left.append(left_df)
            model_right.append(right_df)
            model_fused.append(fused_df)

        if not model_fused:
            continue

        left_df = pd.concat(
            model_left,
            ignore_index=True,
            copy=False,
        )

        right_df = pd.concat(
            model_right,
            ignore_index=True,
            copy=False,
        )

        fused_df = pd.concat(
            model_fused,
            ignore_index=True,
            copy=False,
        )

        def add_site_meta(result_df):
            existing = set(result_df.columns) - set(SITE_KEYS)

            meta_cols = SITE_KEYS + [
                c
                for c in model_sites.columns
                if c not in SITE_KEYS
                and c not in existing
            ]

            return (
                model_sites[
                    meta_cols
                ]
                .drop_duplicates(
                    subset=SITE_KEYS
                )
                .merge(
                    result_df,
                    on=SITE_KEYS,
                    how="inner",
                    validate="one_to_one",
                    sort=False,
                )
            )

        left_df = add_site_meta(left_df)
        right_df = add_site_meta(right_df)
        fused_df = add_site_meta(fused_df)

        model_out = out_root / model_name
        model_out.mkdir(parents=True, exist_ok=True)

        left_df.to_csv(
            model_out
            / "left_celltype_proportions.csv",
            index=False,
            encoding="utf-8-sig",
        )

        right_df.to_csv(
            model_out
            / "right_celltype_proportions.csv",
            index=False,
            encoding="utf-8-sig",
        )

        fused_df.to_csv(
            model_out
            / "fused_celltype_proportions.csv",
            index=False,
            encoding="utf-8-sig",
        )

        all_left.append(left_df)
        all_right.append(right_df)
        all_fused.append(fused_df)

    left_df = (
        pd.concat(
            all_left,
            ignore_index=True,
            copy=False,
        )
        if all_left
        else pd.DataFrame()
    )

    right_df = (
        pd.concat(
            all_right,
            ignore_index=True,
            copy=False,
        )
        if all_right
        else pd.DataFrame()
    )

    fused_df = (
        pd.concat(
            all_fused,
            ignore_index=True,
            copy=False,
        )
        if all_fused
        else pd.DataFrame()
    )

    left_df.to_csv(
        out_root
        / "all_models_left_celltype_proportions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    right_df.to_csv(
        out_root
        / "all_models_right_celltype_proportions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    fused_df.to_csv(
        out_root
        / "all_models_fused_celltype_proportions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    _ctp_clear_attrs(
        left_df,
        right_df,
        fused_df,
    )

    if len(fused_df):
        print(
            f"Cell-type proportions finished | centers={len(fused_df):,} | "
            f"cells={int(fused_df['cell_number'].sum()):,} | "
            f"evidence={int(fused_df['has_type_evidence'].sum()):,} | "
            f"sampling=bilinear | fusion={'mask' if use_mask else 'time'}"
        )

    return left_df, right_df, fused_df


# ============================================================
# Run
# ============================================================

neighborhood_radius_pixels = 4
evidence_chunk_size = 250000


previous_flow_mapped_pixel_df = flow_mapped_pixel_df


# Integer count map -> CENTER + neighborhood FLOW mapping
flow_screened_site_df, flow_mapped_pixel_df, flow_removed_site_df = build_flow_tables_from_integer_count_maps(
    integer_count_results,
    global_flow_results,
    previous_flow_mapped_pixel_df,
    save_dir,
    neighborhood_radius_pixels=neighborhood_radius_pixels,
    model_name=str(interpolation_model_names[0]),
)


# FLOW mapping -> cell-type proportions
left_celltype_prop_df, right_celltype_prop_df, fused_celltype_prop_df = compute_flow_celltype_proportions(
    save_dir,
    interpolation_model_names,
    flow_screened_site_df,
    flow_mapped_pixel_df,
    outputs,
    celltypes,
    use_neighborhood_pixels=use_neighborhood_pixels,
    use_mask=use_mask,
    mask_mode=mask_mode,
    evidence_chunk_size=evidence_chunk_size,
)


# ============================================================
# Dictionaries

# ============================================================

left_celltype_prop_dict = {
    m: g.reset_index(drop=True)
    for m, g in left_celltype_prop_df.groupby("model", sort=False)
} if len(left_celltype_prop_df) else {}

right_celltype_prop_dict = {
    m: g.reset_index(drop=True)
    for m, g in right_celltype_prop_df.groupby("model", sort=False)
} if len(right_celltype_prop_df) else {}

fused_celltype_prop_dict = {
    m: g.reset_index(drop=True)
    for m, g in fused_celltype_prop_df.groupby("model", sort=False)
} if len(fused_celltype_prop_df) else {}


