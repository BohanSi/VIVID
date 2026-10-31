# Migrated from 1002genertation.ipynb, cell 14. See docs/MIGRATION.md.
def compute_global_grid_fixed_shape(
    anndata_dict,
    spatial_key="spatial",
    target_H=1024,
    target_W=1024,
    margin_pixels=0,
):
    """Compute global grid fixed shape."""
    target_H, target_W, margin_pixels = int(target_H), int(target_W), int(margin_pixels)

    if target_H <= 0 or target_W <= 0: raise ValueError("target_H  and  target_W  must be greater than  0.")
    if margin_pixels < 0: raise ValueError("margin_pixels  must not be less than  0.")
    if 2 * margin_pixels >= target_H or 2 * margin_pixels >= target_W: raise ValueError(f"margin_pixels={margin_pixels}  too large ,  must satisfy  2*margin_pixels < target_H  and  target_W.")

    all_x, all_y = [], []

    for name, adata in anndata_dict.items():
        if spatial_key not in adata.obsm: raise KeyError(f"{name}  of  obsm  not found in  {spatial_key}.")

        coords = np.asarray(adata.obsm[spatial_key])
        if coords.ndim != 2 or coords.shape[1] < 2: raise ValueError(f"{name}  invalid spatial coordinate shape : {coords.shape}")

        valid = np.isfinite(coords[:, 0]) & np.isfinite(coords[:, 1])
        if np.any(valid):
            all_x.append(coords[valid, 0])
            all_y.append(coords[valid, 1])

    if not all_x: raise ValueError(" no valid spatial coordinates found .")

    all_x, all_y = np.concatenate(all_x), np.concatenate(all_y)

    raw_x_min, raw_x_max = float(np.min(all_x)), float(np.max(all_x))
    raw_y_min, raw_y_max = float(np.min(all_y)), float(np.max(all_y))
    raw_width, raw_height = raw_x_max - raw_x_min, raw_y_max - raw_y_min

    if raw_width <= 0 or raw_height <= 0: raise ValueError(" invalid spatial coordinate extent ,  check  spatial coordinates.")

    inner_W, inner_H = target_W - 2 * margin_pixels, target_H - 2 * margin_pixels
    pixel_size = max(raw_width / inner_W, raw_height / inner_H)

    canvas_width, canvas_height = target_W * pixel_size, target_H * pixel_size
    x_center, y_center = 0.5 * (raw_x_min + raw_x_max), 0.5 * (raw_y_min + raw_y_max)

    x_min, x_max = x_center - 0.5 * canvas_width, x_center + 0.5 * canvas_width
    y_min, y_max = y_center - 0.5 * canvas_height, y_center + 0.5 * canvas_height

    actual_margin_left = (raw_x_min - x_min) / pixel_size
    actual_margin_right = (x_max - raw_x_max) / pixel_size
    actual_margin_top = (raw_y_min - y_min) / pixel_size
    actual_margin_bottom = (y_max - raw_y_max) / pixel_size

    return {
        "H": target_H,
        "W": target_W,
        "x_min": float(x_min),
        "x_max": float(x_max),
        "y_min": float(y_min),
        "y_max": float(y_max),
        "raw_x_min": raw_x_min,
        "raw_x_max": raw_x_max,
        "raw_y_min": raw_y_min,
        "raw_y_max": raw_y_max,
        "pixel_size": float(pixel_size),
        "pixel_size_x": float(pixel_size),
        "pixel_size_y": float(pixel_size),
        "margin_pixels": margin_pixels,
        "inner_H": inner_H,
        "inner_W": inner_W,
        "actual_margin_left_pixels": float(actual_margin_left),
        "actual_margin_right_pixels": float(actual_margin_right),
        "actual_margin_top_pixels": float(actual_margin_top),
        "actual_margin_bottom_pixels": float(actual_margin_bottom),
    }


def get_global_celltypes(
    anndata_dict,
    celltype_key="cell_type",
    celltypes=None,
):
    """Get global celltypes."""

    if celltypes is not None:
        return list(celltypes)

    all_types = []

    for _, adata in anndata_dict.items():
        labels = adata.obs[celltype_key].astype(str).values
        all_types.extend(labels)

    return sorted(pd.unique(all_types).tolist())


def rasterize_one_slice_fixed_shape(
    adata,
    grid_meta,
    celltypes,
    spatial_key="spatial",
    celltype_key="cell_type",
    eps=1e-8,
    total_transform="normalized",
    return_extra=True,
):
    """Rasterize one slice fixed shape."""

    H = grid_meta["H"]
    W = grid_meta["W"]

    x_min = grid_meta["x_min"]
    y_min = grid_meta["y_min"]

    pixel_size_x = grid_meta["pixel_size_x"]
    pixel_size_y = grid_meta["pixel_size_y"]

    K = len(celltypes)
    celltype_to_idx = {ct: i for i, ct in enumerate(celltypes)}

    coords = np.asarray(adata.obsm[spatial_key]).astype(np.float32)
    labels = adata.obs[celltype_key].astype(str).values

    x = coords[:, 0]
    y = coords[:, 1]

    u = np.floor((x - x_min) / pixel_size_x).astype(np.int64)
    v = np.floor((y - y_min) / pixel_size_y).astype(np.int64)

    valid = (u >= 0) & (u < W) & (v >= 0) & (v < H)

    u = u[valid]
    v = v[valid]
    labels = labels[valid]

    count_maps = np.zeros((K, H, W), dtype=np.float32)

    for ct in celltypes:
        k = celltype_to_idx[ct]
        mask = labels == ct

        if np.any(mask):
            np.add.at(count_maps[k], (v[mask], u[mask]), 1.0)

    
    D_type = count_maps.copy()

    
    T_count = count_maps.sum(axis=0).astype(np.float32)

    
    total_count = float(T_count.sum())

    if total_count > eps:
        T_norm = (T_count / total_count).astype(np.float32)
    else:
        T_norm = np.zeros_like(T_count, dtype=np.float32)

    
    P = np.zeros_like(count_maps, dtype=np.float32)

    nonzero = T_count > eps
    P[:, nonzero] = count_maps[:, nonzero] / T_count[nonzero][None, :]

    if total_transform in ["normalized", "density", "prob"]:
        T_channel = T_norm
        total_channel_name = "total_density_normalized"
    elif total_transform == "raw":
        T_channel = T_count
        total_channel_name = "total_count_raw"
    elif total_transform == "log1p":
        T_channel = np.log1p(T_count).astype(np.float32)
        total_channel_name = "total_count_log1p"
    else:
        raise ValueError(
            "total_transform must be one of: "
            "'normalized', 'density', 'prob', 'raw', 'log1p'."
        )

    X = np.concatenate([T_channel[None, :, :], P, T_count[None, :, :]], axis=0)

    X = np.transpose(X, (1, 2, 0)).astype(np.float32)

    result = {
        "X": X,
        "meta": {
            **grid_meta,
            "celltypes": list(celltypes),
            "total_transform": total_transform,
            "smoothing": "none",
            "channels": (
                [total_channel_name]
                + [f"prop_{ct}" for ct in celltypes]
                + ["total_count_raw"]
            ),
            "n_valid_cells": int(total_count),
            "n_obs": int(adata.n_obs),
        },
    }

    if return_extra:
        result["T"] = T_norm
        result["T_norm"] = T_norm
        result["T_count"] = T_count
        result["P"] = P
        result["D_type"] = D_type
        result["count_maps"] = count_maps

    return result

def build_fixed_shape_density_tensors(
    anndata_dict,
    spatial_key="spatial",
    celltype_key="cell_type",
    target_H=1024,
    target_W=1024,
    margin_pixels=0,
    truncate=None,
    celltypes=None,
    total_transform="normalized",
    return_extra=True,
    #spatial_unit_mode="cell",
    gaussian_sigma_pixels=1,
    r_eff_pixels=5,
    print_per_slice=True,
):
    """Build fixed shape density tensors."""
    if config['resolution'] == 'singlecell': 
        spatial_unit_mode = 'cell'
    else:
        spatial_unit_mode = 'bin'

    if spatial_unit_mode not in {"cell", "bin"}:
        raise ValueError("spatial_unit_mode must be 'cell' or 'bin'.")

    if spatial_unit_mode == "cell":
        if gaussian_sigma_pixels is None:
            raise ValueError(
                "spatial_unit_mode='cell'  requires  gaussian_sigma_pixels."
            )
        if gaussian_sigma_pixels <= 0:
            raise ValueError("gaussian_sigma_pixels  must  > 0.")

    if spatial_unit_mode == "bin":
        if r_eff_pixels is None:
            raise ValueError(
                "spatial_unit_mode='bin'  requires  r_eff_pixels."
            )
        if r_eff_pixels <= 0:
            raise ValueError("r_eff_pixels  must  > 0.")

    # ============================================================
    # 1. Construct global raster grid
    # ============================================================

    grid_meta = compute_global_grid_fixed_shape(
        anndata_dict=anndata_dict,
        spatial_key=spatial_key,
        target_H=target_H,
        target_W=target_W,
        margin_pixels=margin_pixels,
    )

    celltypes = get_global_celltypes(
        anndata_dict=anndata_dict,
        celltype_key=celltype_key,
        celltypes=celltypes,
    )

    outputs = {}

    print("Global grid:")
    for key, value in grid_meta.items():
        print(f"  {key}: {value}")

    print(f"\nNumber of cell types: {len(celltypes)}")
    print(f"Output shape per slice: ({target_H}, {target_W}, {len(celltypes) + 2})")

    # ============================================================
    # 2. Per-cell / per-bin normalized NN spacing
    #
    #       s_H,n = d_NN / Delta_H
    # ============================================================

    delta_H = float(grid_meta["pixel_size"])

    all_s_H = []
    s_H_by_slice = {}

    for name, adata in anndata_dict.items():
        s_H_n = compute_nn_spacing_pixels(
            adata=adata,
            delta_H=delta_H,
            spatial_key=spatial_key,
        )

        s_H_by_slice[name] = s_H_n

        if len(s_H_n) > 0:
            all_s_H.append(s_H_n)

    if len(all_s_H) > 0:
        all_s_H = np.concatenate(all_s_H)
    else:
        all_s_H = np.array([], dtype=np.float64)

    # ============================================================
    # 3. Print spatial-resolution diagnostics
    # ============================================================

    print("\nSpatial resolution diagnostics:")
    print(f"  Raster size              : {target_H} × {target_W}")
    print(f"  Pixel size Δ_H           : {delta_H:.6g}")
    print(f"  Spatial-unit mode        : {spatial_unit_mode}")

    if spatial_unit_mode == "cell":
        threshold = 2.0 * np.sqrt(2.0 * np.log(2.0)) * float(gaussian_sigma_pixels)

        print(f"  Gaussian sigma           : {gaussian_sigma_pixels:.6g} pixels")
        print(f"  Gaussian FWHM            : {threshold:.6g} pixels")
        print("  V_H^cell definition      : fraction(s_H,n < Gaussian FWHM)")

    else:
        threshold = float(r_eff_pixels)

        print(f"  Effective bin radius     : {threshold:.6g} pixels")
        print("  V_H^bin definition       : fraction(s_H,n < r_eff)")

    if print_per_slice:
        print("\n  Per-slice s_H,n distribution:")

        for name, s_H_n in s_H_by_slice.items():
            if len(s_H_n) == 0:
                print(f"    {name}: no valid non-zero NN distances")
                continue

            q10, q25, q50 = np.percentile(s_H_n, [10, 25, 50])
            V = np.mean(s_H_n < threshold)

            if spatial_unit_mode == "cell":
                print(
                    f"    {name}: "
                    f"N={len(s_H_n)}, "
                    f"P10={q10:.4f}, "
                    f"P25={q25:.4f}, "
                    f"median={q50:.4f}, "
                    f"V_H^cell={V:.4%}"
                )
            else:
                print(
                    f"    {name}: "
                    f"N={len(s_H_n)}, "
                    f"P10={q10:.4f}, "
                    f"P25={q25:.4f}, "
                    f"median={q50:.4f}, "
                    f"V_H^bin={V:.4%}"
                )

    print("\n  Full s_H,n distribution:")

    if len(all_s_H) == 0:
        print("    No valid non-zero nearest-neighbor distances.")
    else:
        q10, q25, q50 = np.percentile(all_s_H, [10, 25, 50])
        V_H = float(np.mean(all_s_H < threshold))

        print(f"    N                      = {len(all_s_H)}")
        print(f"    10th percentile        = {q10:.4f} pixels")
        print(f"    25th percentile        = {q25:.4f} pixels")
        print(f"    median                 = {q50:.4f} pixels")

        if spatial_unit_mode == "cell":
            print(f"    V_H^cell               = {V_H:.4%}")
        else:
            print(f"    V_H^bin                = {V_H:.4%}")

    print()

    # ============================================================
    # 4. Original rasterization
    # ============================================================

    for name, adata in anndata_dict.items():
        print(f"Processing slice: {name}")

        result = rasterize_one_slice_fixed_shape(
            adata=adata,
            grid_meta=grid_meta,
            celltypes=celltypes,
            spatial_key=spatial_key,
            celltype_key=celltype_key,
            total_transform=total_transform,
            return_extra=return_extra,
        )

        outputs[name] = result

        T_norm = result["T_norm"] if return_extra else result["X"][:, :, 0]
        T_count = result["T_count"] if return_extra else result["X"][:, :, -1]

        print(
            f"  X shape = {result['X'].shape}, "
            f"sum T_norm = {T_norm.sum():.6f}, "
            f"sum T_count = {T_count.sum():.0f}, "
            f"n_obs = {adata.n_obs}"
        )

    return outputs, grid_meta, celltypes


def pixel_to_spatial(u, v, grid_meta):
    """Pixel to spatial."""
    x = grid_meta["x_min"] + (u + 0.5) * grid_meta["pixel_size_x"]
    y = grid_meta["y_min"] + (v + 0.5) * grid_meta["pixel_size_y"]

    return x, y


def spatial_to_pixel(x, y, grid_meta):
    """Spatial to pixel."""
    u = np.floor((x - grid_meta["x_min"]) / grid_meta["pixel_size_x"]).astype(np.int64)
    v = np.floor((y - grid_meta["y_min"]) / grid_meta["pixel_size_y"]).astype(np.int64)

    return u, v


def safe_filename(name):
    """Safe filename."""
    name = str(name)
    name = re.sub(r"[^\w\-.]+", "_", name)
    return name


def save_all_density_outputs(
    outputs,
    grid_meta,
    celltypes,
    save_dir="./density_results",
    save_extra=True,
):
    """Save all density outputs."""
    os.makedirs(save_dir, exist_ok=True)

    global_meta = {
        "grid_meta": grid_meta,
        "celltypes": list(celltypes),
        "note": (
            "No Gaussian smoothing is used. "
            "By default, X[:, :, 0] is normalized total cell distribution; "
            "X[:, :, 1:1+K] are cell-type proportions; "
            "X[:, :, 1+K] is raw total count map."
        ),
    }

    global_meta_path = os.path.join(save_dir, "global_meta.json")

    with open(global_meta_path, "w", encoding="utf-8") as f:
        json.dump(global_meta, f, ensure_ascii=False, indent=2)

    print(f"Saved global meta to: {global_meta_path}")

    for slice_name, result in outputs.items():
        safe_name = safe_filename(slice_name)
        save_path = os.path.join(save_dir, f"{safe_name}_density.npz")

        arrays = {
            "X": result["X"].astype(np.float32),
        }

        if "T_count" in result:
            arrays["T_count"] = result["T_count"].astype(np.float32)

        if save_extra:
            for key in ["T", "T_norm", "P", "D_type", "count_maps"]:
                if key in result:
                    arrays[key] = result[key].astype(np.float32)

        meta_json = json.dumps(result.get("meta", {}), ensure_ascii=False)

        np.savez_compressed(
            save_path,
            **arrays,
            meta_json=np.array(meta_json),
            slice_name=np.array(str(slice_name)),
        )

        print(f"Saved {slice_name} to: {save_path}")


def load_global_meta(save_dir="./density_results"):
    """Load global meta."""
    path = os.path.join(save_dir, "global_meta.json")

    with open(path, "r", encoding="utf-8") as f:
        global_meta = json.load(f)

    grid_meta = global_meta["grid_meta"]
    celltypes = global_meta["celltypes"]

    return grid_meta, celltypes


def load_one_density_npz(path):
    """Load one density npz."""

    data = np.load(path, allow_pickle=True)

    result = {}

    for key in data.files:
        if key == "meta_json":
            result["meta"] = json.loads(str(data[key]))
        elif key == "slice_name":
            result["slice_name"] = str(data[key])
        else:
            result[key] = data[key]

    return result


def load_all_density_outputs(save_dir="./density_results"):
    """Load all density outputs."""

    grid_meta, celltypes = load_global_meta(save_dir)
    outputs = {}

    for fname in os.listdir(save_dir):
        if not fname.endswith("_density.npz"):
            continue

        path = os.path.join(save_dir, fname)
        result = load_one_density_npz(path)

        slice_name = result.get("slice_name", fname.replace("_density.npz", ""))
        outputs[slice_name] = result

        print(f"Loaded {slice_name}: {result['X'].shape}")

    return outputs, grid_meta, celltypes