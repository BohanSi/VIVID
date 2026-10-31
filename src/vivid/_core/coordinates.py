# Migrated from 1002genertation.ipynb, cell 50. See docs/MIGRATION.md.
import time, zlib
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from IPython.display import display
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching


# ============================================================
# 1. Common utilities
# ============================================================

def coord_grid_transform(grid_meta):
    if "x_min" not in grid_meta or "y_min" not in grid_meta: raise KeyError("grid_meta  missing  x_min / y_min")
    px = grid_meta.get("pixel_size_x", grid_meta.get("pixel_size")); py = grid_meta.get("pixel_size_y", grid_meta.get("pixel_size"))
    if px is None or py is None: raise KeyError("grid_meta  missing  pixel_size  or  pixel_size_x / pixel_size_y")
    return float(grid_meta["x_min"]), float(grid_meta["y_min"]), float(px), float(py)


def coord_sample_in_pixel(u, v, rng, inset=0.08):
    return float(u) + rng.uniform(inset, 1 - inset), float(v) + rng.uniform(inset, 1 - inset)


# ============================================================
# 2. SINGLECELL: neighborhood lookup
# ============================================================

def coord_build_neighborhood_lookup(flow_df, required_keys):
    required = ["model", "pair", "image_index", "site_id", "center_pixel_u", "center_pixel_v", "pixel_u", "pixel_v", "allocated_density"]
    ctp_required_columns(flow_df, required, "flow_mapped_pixel_df")

    df = flow_df.copy(); df["model"] = df["model"].astype(str); df["pair"] = df["pair"].astype(str)
    required_keys = {(str(m), str(p), int(im), int(s)) for m, p, im, s in required_keys}
    lookup = {}

    for key, idx in tqdm(df.groupby(["model", "pair", "image_index", "site_id"], sort=False, observed=True).indices.items(), desc="Preparing neighborhood lookup"):
        key = (str(key[0]), str(key[1]), int(key[2]), int(key[3]))
        if key not in required_keys: continue

        g = df.iloc[np.asarray(idx, dtype=np.int64)].drop_duplicates(["pixel_u", "pixel_v"])
        w = np.maximum(g["allocated_density"].to_numpy(dtype=float), 0); source = "allocated_density"

        if not len(w) or w.max() <= 1e-12: w = np.ones(len(g), dtype=float); source = "uniform_fallback"
        else: w /= w.max()

        lookup[key] = {
            "center_u": int(g["center_pixel_u"].iloc[0]), "center_v": int(g["center_pixel_v"].iloc[0]),
            "pixel_u": g["pixel_u"].to_numpy(dtype=np.int64), "pixel_v": g["pixel_v"].to_numpy(dtype=np.int64),
            "weights": w, "weight_source": source,
        }

    return lookup


# ============================================================
# 3. SINGLECELL: dispersed points
# ============================================================

def coord_sample_neighborhood_points(nb, n, rng, center_u, center_v, inset=0.08):
    pixel_u, pixel_v, weights = nb["pixel_u"], nb["pixel_v"], nb["weights"]

    cpp = max(4, int(np.ceil((n + 2) / max(len(pixel_u), 1))))
    candidate_u, candidate_v, candidate_w = np.repeat(pixel_u, cpp), np.repeat(pixel_v, cpp), np.repeat(weights, cpp)

    xy = np.column_stack([
        candidate_u + rng.uniform(inset, 1 - inset, len(candidate_u)),
        candidate_v + rng.uniform(inset, 1 - inset, len(candidate_v)),
    ])

    first = np.asarray(coord_sample_in_pixel(center_u, center_v, rng, inset), dtype=float)
    selected = [first]
    min_dist = np.sqrt(((xy - first) ** 2).sum(axis=1))
    available = np.ones(len(xy), dtype=bool)
    available[int(np.argmin(min_dist))] = False

    while len(selected) < n:
        idx = np.flatnonzero(available)

        if not len(idx):
            p = np.maximum(weights, 1e-12); p /= p.sum(); j = int(rng.choice(len(pixel_u), p=p))
            point = np.asarray(coord_sample_in_pixel(pixel_u[j], pixel_v[j], rng, inset), dtype=float)

        else:
            score = np.maximum(candidate_w[idx], 0.05) * np.maximum(min_dist[idx], 1e-6) ** 1.5
            j = int(idx[np.argmax(score)])
            point = xy[j].copy()
            available[j] = False

        selected.append(point)
        min_dist = np.minimum(min_dist, np.sqrt(((xy - point) ** 2).sum(axis=1)))

    return np.asarray(selected, dtype=float)


# ============================================================
# 4. SINGLECELL: generate coordinates
# ============================================================

def generate_singlecell_coordinates(site_df, flow_df, celltypes, grid_meta, seed=0):
    required = ["model", "pair", "image", "image_index", "site_id", "cell_number", "center_pixel_u", "center_pixel_v", "interp_fraction"]
    alloc_cols = [f"alloc__{ctp_safe_name(ct)}" for ct in celltypes]
    ctp_required_columns(site_df, required + alloc_cols, "final_site_type_df")

    df = site_df.copy().reset_index(drop=True)
    alloc = df[alloc_cols].to_numpy(dtype=np.int64)
    n = df["cell_number"].to_numpy(dtype=np.int64)

    if np.any(alloc < 0) or not np.array_equal(alloc.sum(axis=1), n): raise RuntimeError("allocation  and  cell_number  inconsistent ")

    need_neighborhood = n > 1
    required_keys = set(zip(
        df.loc[need_neighborhood, "model"].astype(str),
        df.loc[need_neighborhood, "pair"].astype(str),
        df.loc[need_neighborhood, "image_index"].astype(int),
        df.loc[need_neighborhood, "site_id"].astype(int),
    ))

    lookup = coord_build_neighborhood_lookup(flow_df, required_keys)
    x_min, y_min, pixel_size_x, pixel_size_y = coord_grid_transform(grid_meta)

    celltypes_array = np.asarray(celltypes, dtype=object)
    parts = []

    for i in tqdm(range(len(df)), desc="Generating single-cell coordinates"):
        row = df.iloc[i]

        model, pair, image = str(row["model"]), str(row["pair"]), str(row["image"])
        image_index, site_id, count = int(row["image_index"]), int(row["site_id"]), int(row["cell_number"])

        if count <= 0: continue

        center_u, center_v = int(row["center_pixel_u"]), int(row["center_pixel_v"])
        labels = np.repeat(celltypes_array, alloc[i])

        rng_seed = (int(seed) + zlib.crc32(f"{model}/{pair}/{image_index}/{site_id}".encode())) % (2 ** 32)
        rng = np.random.default_rng(rng_seed)
        rng.shuffle(labels)

        key = (model, pair, image_index, site_id)

        if count == 1:
            points = np.asarray([coord_sample_in_pixel(center_u, center_v, rng)], dtype=float)
            source, weight_source = "center_pixel", "none"

        else:
            nb = lookup.get(key)

            if nb is None:
                nb = {
                    "pixel_u": np.array([center_u]), "pixel_v": np.array([center_v]),
                    "weights": np.array([1.0]), "weight_source": "uniform_fallback",
                }
                source = "center_fallback"

            else:
                if nb["center_u"] != center_u or nb["center_v"] != center_v: raise RuntimeError(f"{key}: optimized center  and  FLOW neighborhood center  inconsistent ")
                source = "countmap_neighborhood"

            points = coord_sample_neighborhood_points(nb, count, rng, center_u, center_v)
            weight_source = nb["weight_source"]

        px, py = points[:, 0], points[:, 1]
        x, y = x_min + px * pixel_size_x, y_min + py * pixel_size_y
        image_stem = Path(image).stem

        parts.append(pd.DataFrame({
            "model": model,
            "pair": pair,
            "image": image,
            "image_index": image_index,
            "interp_fraction": float(row["interp_fraction"]),
            "site_id": site_id,
            "cell_in_site": np.arange(count),
            "cell_id": [f"{ctp_safe_name(model)}__{pair}__{image_stem}__site_{site_id:06d}__cell_{j:03d}" for j in range(count)],
            "cell_type": labels,
            "mid_x_pixel": center_u,
            "mid_y_pixel": center_v,
            "site_pixel_u": center_u,
            "site_pixel_v": center_v,
            "x_pixel": px,
            "y_pixel": py,
            "x": x,
            "y": y,
            "site_cell_number": count,
            "coordinate_source": source,
            "coordinate_weight_source": weight_source,
            "distance_from_site_center_pixel": np.sqrt((px - (center_u + 0.5)) ** 2 + (py - (center_v + 0.5)) ** 2),
            "type_prop_fallback_global": bool(row.get("type_prop_fallback_global", False)),
        }))

    result = pd.concat(parts, ignore_index=True, copy=False) if parts else pd.DataFrame()

    if len(result) != int(n.sum()): raise RuntimeError(f" generated cell count ={len(result)} != expected={int(n.sum())}")

    expected_types = df[alloc_cols].sum().to_numpy(dtype=np.int64)
    actual_types = result["cell_type"].astype(str).value_counts().reindex(celltypes, fill_value=0).to_numpy(dtype=np.int64)

    if not np.array_equal(expected_types, actual_types): raise RuntimeError(" final cell_type counts and optimized allocation inconsistent ")

    return result


# ============================================================
# 5. BIN: estimate spacing
# ============================================================

def coord_estimate_axis_spacing(values):
    x = np.asarray(values, dtype=float)
    x = np.unique(np.round(x[np.isfinite(x)], 8))

    if len(x) < 2: raise ValueError(" not enough valid spatial coordinates ,  cannot estimate bin spacing ")

    d = np.diff(np.sort(x))
    d = d[d > max(float(np.ptp(x)), 1.0) * 1e-10]

    if not len(d): raise ValueError(" no positive coordinate spacing ")

    base = np.sort(d)[:max(1, int(np.ceil(0.35 * len(d))))]
    med = float(np.median(base))
    keep = base[(base >= 0.5 * med) & (base <= 1.5 * med)]

    return float(np.median(keep)) if len(keep) else med


def coord_estimate_bin_spacing(anndata_dict, spatial_key="spatial", spacing_x=None, spacing_y=None):
    if spacing_x is not None and spacing_y is not None: return float(spacing_x), float(spacing_y)
    if anndata_dict is None: raise ValueError("bin mode requires  anndata_dict  or explicit  bin spacing")

    xs, ys = [], []

    for adata in anndata_dict.values():
        if spatial_key not in adata.obsm: continue

        xy = np.asarray(adata.obsm[spatial_key], dtype=float)
        xy = xy[np.all(np.isfinite(xy[:, :2]), axis=1)]

        if len(xy) < 2: continue

        try: xs.append(coord_estimate_axis_spacing(xy[:, 0]))
        except Exception: pass

        try: ys.append(coord_estimate_axis_spacing(xy[:, 1]))
        except Exception: pass

    if spacing_x is None:
        if not xs: raise ValueError(" cannot estimate  bin_spacing_x")
        spacing_x = float(np.median(xs))

    if spacing_y is None:
        if not ys: raise ValueError(" cannot estimate  bin_spacing_y")
        spacing_y = float(np.median(ys))

    return float(spacing_x), float(spacing_y)


# ============================================================
# 6. BIN: lattice phase
# ============================================================

def coord_optimize_phase(values, spacing):
    values, spacing = np.asarray(values, dtype=float), float(spacing)

    def loss(phase):
        r = ((values - phase + 0.5 * spacing) % spacing - 0.5 * spacing) / spacing
        a, delta = np.abs(r), 0.25
        return float(np.where(a <= delta, 0.5 * a * a, delta * (a - 0.5 * delta)).sum())

    phases = np.linspace(0, spacing, 64, endpoint=False)
    best = float(phases[np.argmin([loss(p) for p in phases])])
    half = spacing / 64

    for _ in range(3):
        candidates = np.mod(np.linspace(best - half, best + half, 33), spacing)
        best = float(candidates[np.argmin([loss(p) for p in candidates])])
        half *= 0.25

    return best


# ============================================================
# 7. BIN: unique lattice assignment
# ============================================================

def coord_assign_unique_bins(x, y, phase_x, phase_y, spacing_x, spacing_y, radius=2.0, drop_penalty=1e6, chunk_size=25000):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    N = len(x)

    base_i = np.rint((x - phase_x) / spacing_x).astype(np.int64)
    base_j = np.rint((y - phase_y) / spacing_y).astype(np.int64)

    R = int(np.ceil(radius + 1))
    d = np.arange(-R, R + 1, dtype=np.int64)
    offset_i, offset_j = np.repeat(d, len(d)), np.tile(d, len(d))
    M = len(offset_i)

    rows, grid_i_parts, grid_j_parts, costs = [], [], [], []

    for start in range(0, N, chunk_size):
        end = min(N, start + chunk_size)

        gi = base_i[start:end, None] + offset_i
        gj = base_j[start:end, None] + offset_j

        gx = phase_x + gi * spacing_x
        gy = phase_y + gj * spacing_y

        cost = ((x[start:end, None] - gx) / spacing_x) ** 2 + ((y[start:end, None] - gy) / spacing_y) ** 2
        valid = cost <= radius ** 2 + 1e-12

        if not np.any(valid): continue

        mask = valid.ravel()

        rows.append(np.repeat(np.arange(start, end), M)[mask])
        grid_i_parts.append(gi.ravel()[mask])
        grid_j_parts.append(gj.ravel()[mask])
        costs.append(cost.ravel()[mask])

    if not rows: raise RuntimeError(" no candidates constructed bin lattice sites ")

    edge_rows = np.concatenate(rows)
    edge_i = np.concatenate(grid_i_parts)
    edge_j = np.concatenate(grid_j_parts)
    edge_cost = np.concatenate(costs)

    nodes, edge_cols = np.unique(np.column_stack([edge_i, edge_j]), axis=0, return_inverse=True)
    n_nodes = len(nodes)

    matrix = coo_matrix(
        (
            np.concatenate([edge_cost + 1e-12, np.full(N, drop_penalty)]),
            (
                np.concatenate([edge_rows, np.arange(N)]),
                np.concatenate([edge_cols, n_nodes + np.arange(N)]),
            ),
        ),
        shape=(N, n_nodes + N),
    ).tocsr()

    row_ind, col_ind = min_weight_full_bipartite_matching(matrix)
    col_ind = col_ind[np.argsort(row_ind)]

    if len(col_ind) != N: raise RuntimeError("Sparse matching  does not cover all proposal")

    matched = col_ind < n_nodes

    grid_i = np.full(N, np.iinfo(np.int64).min, dtype=np.int64)
    grid_j = np.full(N, np.iinfo(np.int64).min, dtype=np.int64)

    if np.any(matched):
        grid_i[matched] = nodes[col_ind[matched], 0]
        grid_j[matched] = nodes[col_ind[matched], 1]

    final_x, final_y = np.full(N, np.nan), np.full(N, np.nan)

    final_x[matched] = phase_x + grid_i[matched] * spacing_x
    final_y[matched] = phase_y + grid_j[matched] * spacing_y

    distance = np.full(N, np.nan)

    distance[matched] = np.sqrt(
        ((x[matched] - final_x[matched]) / spacing_x) ** 2
        + ((y[matched] - final_y[matched]) / spacing_y) ** 2
    )

    return matched, grid_i, grid_j, final_x, final_y, distance


# ============================================================
# 8. BIN: generate coordinates
# ============================================================

def generate_bin_coordinates(site_df, celltypes, grid_meta, anndata_dict, spacing_x=None, spacing_y=None):
    required = ["model", "pair", "image", "image_index", "site_id", "cell_number", "center_pixel_u", "center_pixel_v", "interp_fraction"]
    alloc_cols = [f"alloc__{ctp_safe_name(ct)}" for ct in celltypes]

    ctp_required_columns(site_df, required + alloc_cols, "final_site_type_df")

    df = site_df.copy().reset_index(drop=True)
    x_min, y_min, pixel_size_x, pixel_size_y = coord_grid_transform(grid_meta)

    spacing_x, spacing_y = coord_estimate_bin_spacing(anndata_dict, spacing_x=spacing_x, spacing_y=spacing_y)

    if spacing_x / pixel_size_x < 1 - 1e-6 or spacing_y / pixel_size_y < 1 - 1e-6:
        raise ValueError(" current  raster  resolution is insufficient : bin spacing  is smaller in at least one direction than  1 pixel")

    parts, stats = [], []

    groups = df.groupby(["model", "pair", "image", "image_index"], sort=False, observed=True)

    for (model, pair, image, image_index), g in tqdm(groups, total=groups.ngroups, desc="Generating bin coordinates"):
        g = g.reset_index(drop=True)

        proposal_u = g["center_pixel_u"].to_numpy(dtype=float) + 0.5
        proposal_v = g["center_pixel_v"].to_numpy(dtype=float) + 0.5

        proposal_x = x_min + proposal_u * pixel_size_x
        proposal_y = y_min + proposal_v * pixel_size_y

        phase_x = coord_optimize_phase(proposal_x, spacing_x)
        phase_y = coord_optimize_phase(proposal_y, spacing_y)

        matched, grid_i, grid_j, final_x, final_y, distance = coord_assign_unique_bins(
            proposal_x, proposal_y, phase_x, phase_y, spacing_x, spacing_y
        )

        
        if np.any(matched):
            phase_x = float(np.mod(phase_x + np.median(proposal_x[matched] - final_x[matched]), spacing_x))
            phase_y = float(np.mod(phase_y + np.median(proposal_y[matched] - final_y[matched]), spacing_y))

            matched, grid_i, grid_j, final_x, final_y, distance = coord_assign_unique_bins(
                proposal_x, proposal_y, phase_x, phase_y, spacing_x, spacing_y
            )

        idx = np.flatnonzero(matched)

        alloc = g[alloc_cols].to_numpy(dtype=float)
        labels = np.asarray(celltypes, dtype=object)[np.argmax(alloc, axis=1)]

        if len(idx):
            final_u = (final_x[idx] - x_min) / pixel_size_x
            final_v = (final_y[idx] - y_min) / pixel_size_y

            site_ids = g["site_id"].to_numpy(dtype=np.int64)[idx]
            center_u = g["center_pixel_u"].to_numpy(dtype=np.int64)[idx]
            center_v = g["center_pixel_v"].to_numpy(dtype=np.int64)[idx]

            image_stem = Path(str(image)).stem
            bin_ids = [f"{ctp_safe_name(model)}__{pair}__{image_stem}__site_{int(s):06d}__bin" for s in site_ids]

            parts.append(pd.DataFrame({
                "model": str(model),
                "pair": str(pair),
                "image": str(image),
                "image_index": int(image_index),
                "interp_fraction": g["interp_fraction"].to_numpy(dtype=float)[idx],

                "site_id": site_ids,
                "cell_in_site": 0,
                "cell_id": bin_ids,
                "bin_id": bin_ids,
                "cell_type": labels[idx],

                "mid_x_pixel": center_u,
                "mid_y_pixel": center_v,
                "site_pixel_u": center_u,
                "site_pixel_v": center_v,

                "proposal_x_pixel": proposal_u[idx],
                "proposal_y_pixel": proposal_v[idx],
                "proposal_x": proposal_x[idx],
                "proposal_y": proposal_y[idx],

                "x_pixel": final_u,
                "y_pixel": final_v,
                "x": final_x[idx],
                "y": final_y[idx],

                "site_cell_number": 1,
                "estimated_cell_number_raw": g["cell_number"].to_numpy(dtype=np.int64)[idx],

                "coordinate_source": "bin_lattice_assignment",
                "coordinate_weight_source": "none",

                "distance_from_site_center_pixel": np.sqrt(
                    (final_u - proposal_u[idx]) ** 2
                    + (final_v - proposal_v[idx]) ** 2
                ),

                "bin_grid_i": grid_i[idx],
                "bin_grid_j": grid_j[idx],

                "bin_spacing_x": spacing_x,
                "bin_spacing_y": spacing_y,
                "bin_phase_x": phase_x,
                "bin_phase_y": phase_y,

                "bin_snap_distance_bins": distance[idx],

                "type_prop_fallback_global":
                    g["type_prop_fallback_global"].fillna(False).astype(bool).to_numpy()[idx]
                    if "type_prop_fallback_global" in g.columns else False,
            }))

        stats.append({
            "model": str(model),
            "pair": str(pair),
            "image": str(image),
            "image_index": int(image_index),
            "n_proposals": len(g),
            "n_bins": int(matched.sum()),
            "n_dropped": int((~matched).sum()),
            "phase_x": phase_x,
            "phase_y": phase_y,
            "mean_snap_distance_bins": float(np.nanmean(distance)) if np.any(matched) else np.nan,
        })

    result = pd.concat(parts, ignore_index=True, copy=False) if parts else pd.DataFrame()

    if len(result) and result.duplicated(["pair", "image_index", "bin_grid_i", "bin_grid_j"]).any():
        raise RuntimeError(" final bin duplicate lattice sites ")

    return result, pd.DataFrame(stats), spacing_x, spacing_y


# ============================================================
# 9. Run
# ============================================================

coordinate_resolution = str(config["resolution"]).lower()

if coordinate_resolution not in {"singlecell", "bin"}:
    raise ValueError("config['resolution']  must be  'singlecell'  or  'bin'")

coordinate_celltypes = [str(x) for x in optimization_celltypes]

available_models = set(final_site_type_df["model"].astype(str))
requested_models = [str(x) for x in interpolation_model_names if str(x) in available_models]

if not requested_models:
    raise ValueError(" no model available for coordinate reconstruction ")

site_df = final_site_type_df.loc[
    final_site_type_df["model"].astype(str).isin(requested_models)
].copy().reset_index(drop=True)

coordinate_output_folder = (
    "optimized_celltype_reconstruction_hardglobal"
    if bool(config["enforce_global_hard_constraint"])
    else "optimized_celltype_reconstruction_no_global_hard"
)

out_root = Path(save_dir).resolve() / coordinate_output_folder
out_root.mkdir(parents=True, exist_ok=True)

start = time.perf_counter()


# ------------------------------------------------------------
# SINGLECELL
# ------------------------------------------------------------

if coordinate_resolution == "singlecell":
    final_reconstructed_cell_df = generate_singlecell_coordinates(
        site_df, flow_mapped_pixel_df, coordinate_celltypes, grid_meta, seed=config["seed"]
    )

    coordinate_summary_df = site_df.groupby("model", sort=False).agg(
        n_sites=("site_id", "size"),
        expected_cells=("cell_number", "sum"),
    ).reset_index()

    generated_counts = final_reconstructed_cell_df.groupby("model", sort=False).size()
    coordinate_summary_df["generated_cells"] = coordinate_summary_df["model"].map(generated_counts).fillna(0).astype(int)


# ------------------------------------------------------------
# BIN
# ------------------------------------------------------------

else:
    config_bin_spacing = config.get("bin_spacing")

    final_reconstructed_cell_df, bin_frame_summary_df, bin_spacing_x, bin_spacing_y = generate_bin_coordinates(
        site_df,
        coordinate_celltypes,
        grid_meta,
        anndata_dict,
        spacing_x=config.get("bin_spacing_x", config_bin_spacing),
        spacing_y=config.get("bin_spacing_y", config_bin_spacing),
    )

    coordinate_summary_df = site_df.groupby("model", sort=False).size().rename("n_sites").reset_index()

    generated_counts = final_reconstructed_cell_df.groupby("model", sort=False).size()
    coordinate_summary_df["n_bins"] = coordinate_summary_df["model"].map(generated_counts).fillna(0).astype(int)

    bin_frame_summary_df.to_csv(out_root / "bin_coordinate_frame_summary.csv", index=False, encoding="utf-8-sig")
    final_reconstructed_cell_df.to_csv(out_root / "all_models_reconstructed_bins.csv", index=False, encoding="utf-8-sig")


# ============================================================
# 10. Save + QC
# ============================================================

if len(final_reconstructed_cell_df) and final_reconstructed_cell_df["cell_id"].duplicated().any():
    raise RuntimeError("cell_id contains duplicates ")

for model, g in final_reconstructed_cell_df.groupby("model", sort=False):
    model_out = out_root / str(model)
    model_out.mkdir(parents=True, exist_ok=True)
    g.to_csv(model_out / "reconstructed_cells.csv", index=False, encoding="utf-8-sig")

final_reconstructed_cell_df.to_csv(out_root / "all_models_reconstructed_cells.csv", index=False, encoding="utf-8-sig")
coordinate_summary_df.to_csv(out_root / "coordinate_summary.csv", index=False, encoding="utf-8-sig")

final_reconstructed_cell_dict = {
    str(model): g.reset_index(drop=True)
    for model, g in final_reconstructed_cell_df.groupby("model", sort=False)
} if len(final_reconstructed_cell_df) else {}

runtime = time.perf_counter() - start

print(
    f"Coordinate generation finished | resolution={coordinate_resolution} | "
    f"rows={len(final_reconstructed_cell_df):,} | runtime={runtime:.2f}s | output={out_root}"
)

display_cols = [
    "model", "pair", "image", "image_index", "site_id", "cell_in_site",
    "cell_id", "cell_type", "x_pixel", "y_pixel", "x", "y",
    "coordinate_source", "distance_from_site_center_pixel",
    "type_prop_fallback_global",
]

if coordinate_resolution == "bin":
    display_cols += [
        "bin_grid_i", "bin_grid_j",
        "bin_spacing_x", "bin_spacing_y",
        "bin_phase_x", "bin_phase_y",
        "bin_snap_distance_bins",
    ]

if len(final_reconstructed_cell_df):
    display(final_reconstructed_cell_df[[c for c in display_cols if c in final_reconstructed_cell_df.columns]].head(30))