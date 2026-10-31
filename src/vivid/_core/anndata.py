# Migrated from 1002genertation.ipynb, cell 57. See README.md.
import numpy as np
import pandas as pd
import anndata as ad
from IPython.display import display


# ============================================================
# 1. Configuration
# ============================================================

generation_z = bool(config.get("generation_z", False))
adata_celltypes = [str(x) for x in optimization_celltypes]

available_models = final_reconstructed_cell_df["model"].astype(str).unique().tolist()
if len(available_models) != 1: raise ValueError(f" this pipeline requires one model ,  but detected models : {available_models}")

model_name = available_models[0]


# ============================================================
# 2. Utilities
# ============================================================

def adata_resolve_real_key(slice_name):
    if slice_name in anndata_dict: return slice_name
    target_id = ctp_norm_slice_id(slice_name)
    matches = [k for k in anndata_dict if target_id is not None and ctp_norm_slice_id(k) == target_id]
    if len(matches) == 1: return matches[0]
    if not matches: raise KeyError(f"anndata_dict observed slice not found in  {slice_name}")
    raise ValueError(f"{slice_name}  matched multiple AnnData: {matches}")


def adata_spatial_to_pixel(xy):
    xy = np.asarray(xy, dtype=float)
    x_min, y_min, psx, psy = coord_grid_transform(grid_meta)
    return np.column_stack([(xy[:, 0] - x_min) / psx, (xy[:, 1] - y_min) / psy])


# ============================================================
# 3. Prepare reconstructed cells
# ============================================================

def prepare_cells_for_anndata(df):
    df = df.copy().reset_index(drop=True)
    required = ["model", "pair", "image", "image_index", "site_id", "cell_id", "cell_type", "x", "y"]
    ctp_required_columns(df, required, "final_reconstructed_cell_df")

    df = df.loc[df["model"].astype(str) == model_name].copy().reset_index(drop=True)
    df["model"] = df["model"].astype(str); df["pair"] = df["pair"].astype(str); df["image"] = df["image"].astype(str)
    df["cell_type"] = df["cell_type"].astype(str); df["cell_id"] = df["cell_id"].astype(str)

    df["pair_name"] = df["pair"]
    df["image_name"] = df["image"]
    df["slice_id"] = df["model"] + "__" + df["pair"] + "__" + df["image"]

    spatial = df[["x", "y"]].to_numpy(float)
    if not np.all(np.isfinite(spatial)): raise ValueError(" predicted cells  x/y  contains  NaN/Inf")

    if not {"x_pixel", "y_pixel"}.issubset(df.columns):
        pixel = adata_spatial_to_pixel(spatial)
        df["x_pixel"], df["y_pixel"] = pixel[:, 0], pixel[:, 1]

    if not np.all(np.isfinite(df[["x_pixel", "y_pixel"]].to_numpy(float))): raise ValueError(" predicted cells  x_pixel/y_pixel  contains  NaN/Inf")
    if df["cell_id"].duplicated().any(): raise ValueError(f"cell_id contains duplicates : {int(df['cell_id'].duplicated().sum())}")

    unknown = sorted(set(df["cell_type"]) - set(adata_celltypes))
    if unknown: raise ValueError(f" contains unknown cell types: {unknown}")

    if generation_z:
        if "z" not in df.columns: raise KeyError("generation_z=True,  but  final_reconstructed_cell_df  does not contain  z")
        if not np.all(np.isfinite(df["z"].to_numpy(float))): raise ValueError(" predicted cells  z  contains  NaN/Inf")

    return df


assigned_cells_for_adata = prepare_cells_for_anndata(final_reconstructed_cell_df)
pred_pair_to_real_slices = {pair: ctp_parse_pair(pair) for pair in assigned_cells_for_adata["pair"].unique()}


# ============================================================
# 4. generation_z=True: add z to real endpoint AnnData
# ============================================================

def add_z_to_real_endpoints(pred_df):
    modified_keys, summary = set(), []

    for pair, (left_slice, right_slice) in pred_pair_to_real_slices.items():
        p = pred_df.loc[pred_df["pair"] == pair].copy()
        if not len(p): continue

        frames = p[["image_index", "interp_fraction"]].drop_duplicates("image_index").sort_values(["interp_fraction", "image_index"])
        middle_frames = frames.loc[(frames["interp_fraction"] > 1e-8) & (frames["interp_fraction"] < 1 - 1e-8)].reset_index(drop=True)
        if not len(middle_frames): raise ValueError(f"{pair}:  no intermediate slices detected ")

        n_middle = len(middle_frames)
        middle_order = {int(fid): i + 1 for i, fid in enumerate(middle_frames["image_index"])}

        middle = p.loc[p["image_index"].astype(int).isin(middle_order)].copy()
        middle["image_index"] = middle["image_index"].astype(int).map(middle_order)
        middle["interp_fraction"] = middle["image_index"] / float(n_middle + 1)
        middle = middle[["model", "pair", "image_index", "interp_fraction", "x_pixel", "y_pixel", "cell_type"]]

        real_parts = []

        for side, slice_name, section_index, t in [
            ("left", left_slice, 0, 0.0),
            ("right", right_slice, n_middle + 1, 1.0),
        ]:
            key = adata_resolve_real_key(slice_name)
            if key in modified_keys: raise ValueError(f"{key}  used by multiple pair reused as an endpoint ,  single  obs['z']  cannot represent multiple 3D stack")

            real = anndata_dict[key]
            if celltype_col not in real.obs.columns: raise KeyError(f"{key}: obs missing  {celltype_col}")
            if "spatial" not in real.obsm: raise KeyError(f"{key}: obsm missing  spatial")

            xy = np.asarray(real.obsm["spatial"], dtype=float)[:, :2]
            pixel = adata_spatial_to_pixel(xy)

            real_parts.append(pd.DataFrame({
                "model": model_name,
                "pair": pair,
                "image_index": section_index,
                "interp_fraction": t,
                "x_pixel": pixel[:, 0],
                "y_pixel": pixel[:, 1],
                "cell_type": real.obs[celltype_col].astype(str).to_numpy(),
                "_real_key": str(key),
                "_real_pos": np.arange(real.n_obs),
            }))

        middle["_real_key"] = ""
        middle["_real_pos"] = -1

        temp = pd.concat([real_parts[0], middle, real_parts[1]], ignore_index=True)

        temp_z = generate_continuous_z(
            temp,
            section_thickness=float(config.get("z_section_thickness", 10.0)),
            sigma_xy=float(config.get("z_sigma_xy_pixel", 2.0)),
            sigma_z_ratio=float(config.get("z_sigma_z_ratio", 0.5)),
            neighbor_sections=int(config.get("z_neighbor_sections", 1)),
            use_celltype=bool(config.get("z_use_celltype", True)),
            seed=config["seed"],
        )

        for side, slice_name in [("left", left_slice), ("right", right_slice)]:
            key = adata_resolve_real_key(slice_name)
            real = anndata_dict[key]
            sub = temp_z.loc[temp_z["_real_key"] == str(key)]

            if len(sub) != real.n_obs: raise RuntimeError(f"{pair}/{key}: z count ={len(sub)} != real cells={real.n_obs}")

            z = np.full(real.n_obs, np.nan)
            z[sub["_real_pos"].to_numpy(int)] = sub["z"].to_numpy(float)

            if not np.all(np.isfinite(z)): raise RuntimeError(f"{pair}/{key}:  missing generated z observed cells ")

            real.obs["z"] = z
            real.uns.update({
                "generation_z": True,
                "z_unit": "um",
                "z_section_thickness_um": float(config.get("z_section_thickness", 10.0)),
                "z_pair": pair,
                "z_side": side,
            })

            modified_keys.add(key)
            summary.append({
                "pair": pair,
                "real_slice": str(key),
                "side": side,
                "n_cells": real.n_obs,
                "min_z": float(z.min()),
                "mean_z": float(z.mean()),
                "std_z": float(z.std()),
                "max_z": float(z.max()),
            })

    return pd.DataFrame(summary), sorted(modified_keys, key=str)


real_z_summary_df, real_z_modified_keys = (add_z_to_real_endpoints(assigned_cells_for_adata) if generation_z else (pd.DataFrame(), []))


# ============================================================
# 5. Reconstructed cells -> AnnData
# ============================================================

def cells_to_anndata_dict(df):
    obs_cols = [
        "image_index", "interp_fraction", "site_id", "cell_in_site",
        "mid_x_pixel", "mid_y_pixel", "site_pixel_u", "site_pixel_v",
        "site_cell_number", "coordinate_source", "coordinate_weight_source",
        "distance_from_site_center_pixel", "type_prop_fallback_global",
    ]

    z_cols = ["z_section_order", "z_nominal", "z", "z_slab_left", "z_slab_right", "z_fraction", "z_offset", "z_source"]
    result = {}

    for (model, pair, image), g in df.groupby(["model", "pair", "image"], sort=False):
        g = g.reset_index(drop=True)
        ids = g["cell_id"].astype(str).to_numpy()

        if len(np.unique(ids)) != len(ids): raise ValueError(f"{pair}/{image}: cell_id not unique ")

        obs = pd.DataFrame(index=pd.Index(ids, name="cell_id"))
        obs[celltype_col] = pd.Categorical(g["cell_type"].astype(str), categories=adata_celltypes)
        obs["model"] = str(model); obs["pair_name"] = str(pair); obs["image_name"] = str(image)
        obs["slice_id"] = f"{model}__{pair}__{image}"

        for col in obs_cols + (z_cols if generation_z else []):
            if col in g.columns: obs[col] = g[col].to_numpy()

        adata = ad.AnnData(X=np.empty((len(g), 0), dtype=np.float32), obs=obs)
        adata.obsm["spatial"] = g[["x", "y"]].to_numpy(float)
        adata.obsm["pixel"] = g[["x_pixel", "y_pixel"]].to_numpy(float)

        if {"site_pixel_u", "site_pixel_v"}.issubset(g.columns): adata.obsm["site_pixel"] = g[["site_pixel_u", "site_pixel_v"]].to_numpy(float)
        elif {"mid_x_pixel", "mid_y_pixel"}.issubset(g.columns): adata.obsm["site_pixel"] = g[["mid_x_pixel", "mid_y_pixel"]].to_numpy(float)

        left_slice, right_slice = pred_pair_to_real_slices[str(pair)]

        adata.uns.update({
            "source": "VIVID_reconstruction",
            "model": str(model),
            "pair_name": str(pair),
            "image_name": str(image),
            "slice_id": f"{model}__{pair}__{image}",
            "n_cells": len(g),
            "celltype_col": str(celltype_col),
            "celltypes": adata_celltypes,
            "left_slice": str(left_slice),
            "right_slice": str(right_slice),
            "generation_z": generation_z,
        })

        if generation_z:
            adata.uns["z_key"] = "z"
            adata.uns["z_unit"] = "um"
            adata.uns["z_section_thickness_um"] = float(config.get("z_section_thickness", 10.0))

        if g["image_index"].nunique() == 1: adata.uns["image_index"] = int(g["image_index"].iloc[0])
        if g["interp_fraction"].dropna().nunique() == 1: adata.uns["interp_fraction"] = float(g["interp_fraction"].dropna().iloc[0])

        key = f"{model}__{pair}__{image}"
        if key in result: raise ValueError(f"AnnData dict key duplicate : {key}")

        result[key] = adata

    return result


pred_adata_dict = cells_to_anndata_dict(assigned_cells_for_adata)
pred_adata_dict_by_model = {model_name: pred_adata_dict}


# ============================================================
# 6. Consistency checks
# ============================================================

if len(assigned_cells_for_adata) != sum(x.n_obs for x in pred_adata_dict.values()):
    raise RuntimeError("AnnData cell totals changed during construction ")

for key, adata in pred_adata_dict.items():
    spatial = np.asarray(adata.obsm["spatial"], dtype=float)
    pixel = np.asarray(adata.obsm["pixel"], dtype=float)

    if spatial.shape != (adata.n_obs, 2) or pixel.shape != (adata.n_obs, 2): raise RuntimeError(f"{key}: spatial/pixel shape invalid ")
    if not np.all(np.isfinite(spatial)) or not np.all(np.isfinite(pixel)): raise RuntimeError(f"{key}: spatial/pixel contains NaN/Inf")
    if np.max(np.abs(pixel - adata_spatial_to_pixel(spatial))) > 1e-6: raise RuntimeError(f"{key}: spatial and pixel inconsistent ")

    unknown = set(adata.obs[celltype_col].astype(str)) - set(adata_celltypes)
    if unknown: raise RuntimeError(f"{key}:  unknown celltypes: {sorted(unknown)}")

    if generation_z:
        z = adata.obs["z"].to_numpy(float)
        if not np.all(np.isfinite(z)): raise RuntimeError(f"{key}: z contains NaN/Inf")

        if {"z_slab_left", "z_slab_right"}.issubset(adata.obs.columns):
            left = adata.obs["z_slab_left"].to_numpy(float); right = adata.obs["z_slab_right"].to_numpy(float)
            if np.any(z < left - 1e-8) or np.any(z > right + 1e-8): raise RuntimeError(f"{key}: z outside its assigned section range ")

if generation_z:
    for key in real_z_modified_keys:
        if "z" not in anndata_dict[key].obs: raise RuntimeError(f"{key}:  observed AnnData missing z")
        if not np.all(np.isfinite(anndata_dict[key].obs["z"].to_numpy(float))): raise RuntimeError(f"{key}:  observed slice z contains NaN/Inf")


# ============================================================
# 7. Quick summary
# ============================================================

print(
    f"AnnData construction finished | model={model_name} | "
    f"slices={len(pred_adata_dict)} | cells={sum(x.n_obs for x in pred_adata_dict.values()):,} | "
    f"generation_z={generation_z}"
)

if generation_z and len(real_z_summary_df): display(real_z_summary_df)

if pred_adata_dict:
    example_key = next(iter(pred_adata_dict))
    print(example_key, pred_adata_dict[example_key])
    display(pred_adata_dict[example_key].obs.head())