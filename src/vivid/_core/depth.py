# Migrated from 1002genertation.ipynb, cell 52. See README.md.
import time, zlib
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, map_coordinates
from tqdm.auto import tqdm
from IPython.display import display


# ============================================================
# 1. Continuous Z generation
# ============================================================

def z_density(df, H, W, x0, y0, sigma):
    img = np.zeros((H, W), dtype=np.float32)

    if len(df):
        x = np.clip(np.rint(df["x_pixel"].to_numpy(float) - x0).astype(int), 0, W - 1)
        y = np.clip(np.rint(df["y_pixel"].to_numpy(float) - y0).astype(int), 0, H - 1)
        np.add.at(img, (y, x), 1.0)

    return gaussian_filter(img, sigma, mode="constant") if sigma > 0 else img


def z_sample_density(img, x, y, x0, y0):
    return map_coordinates(img, [np.asarray(y, float) - y0, np.asarray(x, float) - x0], order=1, mode="constant", cval=0.0)


def generate_continuous_z(
    reconstructed_df, section_thickness=10.0, sigma_xy=2.0, sigma_z_ratio=0.5,
    neighbor_sections=1, use_celltype=True, seed=0
):
    df = reconstructed_df.copy().reset_index(drop=True)
    required = ["model", "pair", "image_index", "interp_fraction", "x_pixel", "y_pixel"]
    ctp_required_columns(df, required + (["cell_type"] if use_celltype else []), "final_reconstructed_cell_df")

    
    z_grid_size, chunk_size = 129, 5000
    thickness = float(section_thickness)
    sigma_z = max(float(sigma_z_ratio) * thickness, 1e-8)
    neighbor_sections = max(0, int(neighbor_sections))

    n = len(df)
    section_order = np.full(n, -1, dtype=np.int64)
    nominal = np.full(n, np.nan)
    slab_left = np.full(n, np.nan)
    slab_right = np.full(n, np.nan)
    z = np.full(n, np.nan)
    z_fraction = np.full(n, np.nan)
    source = np.empty(n, dtype=object); source[:] = ""

    groups = df.groupby(["model", "pair"], sort=False).indices

    for (model, pair), idx in tqdm(groups.items(), total=len(groups), desc="Generating continuous z"):
        idx = np.asarray(idx, dtype=np.int64)
        g = df.iloc[idx]

        frames = g[["image_index", "interp_fraction"]].drop_duplicates("image_index")
        frames = frames.sort_values(["interp_fraction", "image_index"] if frames["interp_fraction"].notna().all() else ["image_index"])

        frame_ids = frames["image_index"].to_numpy(dtype=np.int64)
        n_sections = len(frame_ids)
        centers = (np.arange(n_sections, dtype=float) + 0.5) * thickness
        total_thickness = n_sections * thickness

        frame_arr = g["image_index"].to_numpy(dtype=np.int64)
        x_arr = g["x_pixel"].to_numpy(dtype=float)
        y_arr = g["y_pixel"].to_numpy(dtype=float)
        ct_arr = g["cell_type"].astype(str).to_numpy() if use_celltype else None

        for fi, fid in enumerate(frame_ids):
            rows = idx[frame_arr == fid]
            section_order[rows] = fi
            nominal[rows] = centers[fi]
            slab_left[rows] = fi * thickness
            slab_right[rows] = (fi + 1) * thickness

        margin = max(2, int(np.ceil(4 * sigma_xy)))
        x0 = float(np.floor(x_arr.min())) - margin
        y0 = float(np.floor(y_arr.min())) - margin
        W = max(2, int(np.ceil(x_arr.max() - x0)) + margin + 1)
        H = max(2, int(np.ceil(y_arr.max() - y0)) + margin + 1)

        global_maps = {
            int(fid): z_density(g.iloc[np.flatnonzero(frame_arr == fid)], H, W, x0, y0, sigma_xy)
            for fid in frame_ids
        }

        celltypes = np.unique(ct_arr) if use_celltype else np.array([None], dtype=object)

        for ct in celltypes:
            ct_mask = (ct_arr == ct) if use_celltype else np.ones(len(g), dtype=bool)

            density_maps = (
                {
                    int(fid): z_density(g.iloc[np.flatnonzero(ct_mask & (frame_arr == fid))], H, W, x0, y0, sigma_xy)
                    for fid in frame_ids
                }
                if use_celltype else global_maps
            )

            for fi, fid in enumerate(frame_ids):
                local_target = np.flatnonzero(ct_mask & (frame_arr == fid))
                if not len(local_target): continue

                target_rows = idx[local_target]
                left, right = fi * thickness, (fi + 1) * thickness

                src_pos = np.arange(max(0, fi - neighbor_sections), min(n_sections, fi + neighbor_sections + 1))
                src_ids, src_z = frame_ids[src_pos], centers[src_pos]

                z_grid = np.linspace(left, right, z_grid_size)
                z_kernel = np.exp(-0.5 * ((z_grid[None, :] - src_z[:, None]) / sigma_z) ** 2)

                rng_seed = (int(seed) + zlib.crc32(f"{model}/{pair}/{fid}/{ct}".encode())) % (2 ** 32)
                rng = np.random.default_rng(rng_seed)

                for start in range(0, len(target_rows), chunk_size):
                    rows = target_rows[start:start + chunk_size]
                    x = df.loc[rows, "x_pixel"].to_numpy(float)
                    y = df.loc[rows, "y_pixel"].to_numpy(float)

                    weights = np.column_stack([z_sample_density(density_maps[int(sf)], x, y, x0, y0) for sf in src_ids])
                    current_source = np.full(len(rows), "celltype_density" if use_celltype else "global_density", dtype=object)

                    
                    bad = weights.sum(axis=1) <= 1e-20

                    if use_celltype and np.any(bad):
                        weights[bad] = np.column_stack([
                            z_sample_density(global_maps[int(sf)], x[bad], y[bad], x0, y0)
                            for sf in src_ids
                        ])
                        current_source[bad] = "global_density_fallback"

                    prob = weights @ z_kernel

                    
                    bad = prob.sum(axis=1) <= 1e-20

                    if np.any(bad):
                        prob[bad] = np.exp(-0.5 * ((z_grid - centers[fi]) / sigma_z) ** 2)
                        current_source[bad] = "nominal_gaussian_fallback"

                    # inverse-CDF sampling
                    cdf = np.cumsum(prob, axis=1)
                    u = rng.random(len(rows)) * cdf[:, -1]
                    k = np.minimum((cdf < u[:, None]).sum(axis=1), len(z_grid) - 1)

                    
                    dz = thickness / (len(z_grid) - 1)
                    z[rows] = np.clip(z_grid[k] + rng.uniform(-0.5, 0.5, len(rows)) * dz, left, right)
                    source[rows] = current_source

        if np.isnan(z[idx]).any(): raise RuntimeError(f"{model}/{pair}:  missing generated z objects ")
        z_fraction[idx] = z[idx] / total_thickness

    df["z_section_order"] = section_order
    df["z_nominal"] = nominal
    df["z"] = z
    df["z_slab_left"] = slab_left
    df["z_slab_right"] = slab_right
    df["z_fraction"] = z_fraction
    df["z_offset"] = z - nominal
    df["z_source"] = source

    return df


# ============================================================
# 2. Parameters
# ============================================================

z_section_thickness = float(config.get("z_section_thickness", 10.0))
z_sigma_xy = float(config.get("z_sigma_xy_pixel", 2.0))
z_sigma_z_ratio = float(config.get("z_sigma_z_ratio", 0.5))
z_neighbor_sections = int(config.get("z_neighbor_sections", 1))
z_use_celltype = bool(config.get("z_use_celltype", coordinate_resolution == "singlecell"))


# ============================================================
# 3. Run
# ============================================================

if config["generation_z"]:
    z_start = time.perf_counter()

    final_reconstructed_cell_df = generate_continuous_z(
        final_reconstructed_cell_df,
        section_thickness=z_section_thickness,
        sigma_xy=z_sigma_xy,
        sigma_z_ratio=z_sigma_z_ratio,
        neighbor_sections=z_neighbor_sections,
        use_celltype=z_use_celltype,
        seed=config["seed"],
    )

    final_reconstructed_cell_dict = {
        str(model): g.reset_index(drop=True)
        for model, g in final_reconstructed_cell_df.groupby("model", sort=False)
    }

    final_reconstructed_cell_df.to_csv(out_root / "all_models_reconstructed_cells_3d.csv", index=False, encoding="utf-8-sig")

    for model, g in final_reconstructed_cell_dict.items():
        model_out = out_root / model
        model_out.mkdir(parents=True, exist_ok=True)
        g.to_csv(model_out / "reconstructed_cells_3d.csv", index=False, encoding="utf-8-sig")

    print(
        f"Z generation finished | objects={len(final_reconstructed_cell_df):,} | "
        f"runtime={time.perf_counter() - z_start:.2f}s | "
        f"thickness={z_section_thickness:g} | sigma_xy={z_sigma_xy:g} | "
        f"sigma_z={z_sigma_z_ratio * z_section_thickness:g}"
    )

else: 
 print("no generation")


# ============================================================
# 4. Quick check
# ============================================================

if config["generation_z"]:
    display_cols = [
        "model", "pair", "image_index", "cell_id", "cell_type",
        "x", "y", "z_section_order", "z_nominal", "z",
        "z_slab_left", "z_slab_right", "z_offset", "z_source",
    ]

    display(final_reconstructed_cell_df[[c for c in display_cols if c in final_reconstructed_cell_df.columns]].head(30))

    z_summary = final_reconstructed_cell_df.groupby(
        ["model", "pair", "z_section_order", "image_index"], as_index=False
    ).agg(
        n=("z", "size"),
        nominal_z=("z_nominal", "first"),
        mean_z=("z", "mean"),
        std_z=("z", "std"),
        min_z=("z", "min"),
        max_z=("z", "max"),
    )

    display(z_summary)