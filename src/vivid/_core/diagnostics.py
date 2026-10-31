# Migrated from 1002genertation.ipynb, cell 13. See README.md.
from scipy.spatial import cKDTree

def compute_characteristic_spacing(adata, spatial_key="spatial"):
    """Compute characteristic spacing."""
    coords = np.asarray(adata.obsm[spatial_key], dtype=np.float64)[:, :2]
    coords = coords[np.all(np.isfinite(coords), axis=1)]
    if len(coords) < 2: return np.nan

    
    
    unique_coords, inverse = np.unique(coords, axis=0, return_inverse=True)
    if len(unique_coords) < 2: return np.nan

    tree = cKDTree(unique_coords)
    distances, _ = tree.query(unique_coords, k=2)
    nn_dist = distances[:, 1][inverse]
    nn_dist = nn_dist[np.isfinite(nn_dist) & (nn_dist > 0)]

    return float(np.median(nn_dist)) if len(nn_dist) else np.nan


def compute_raster_diagnostics(adata, grid_meta, spatial_key="spatial"):
    """Compute raster diagnostics."""
    coords = np.asarray(adata.obsm[spatial_key], dtype=np.float64)[:, :2]
    coords = coords[np.all(np.isfinite(coords), axis=1)]

    H, W = grid_meta["H"], grid_meta["W"]
    x_min, y_min = grid_meta["x_min"], grid_meta["y_min"]
    pixel_size_x, pixel_size_y = grid_meta["pixel_size_x"], grid_meta["pixel_size_y"]

    u = np.floor((coords[:, 0] - x_min) / pixel_size_x).astype(np.int64)
    v = np.floor((coords[:, 1] - y_min) / pixel_size_y).astype(np.int64)
    valid = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v = u[valid], v[valid]

    n_rasterized = len(u)
    if n_rasterized == 0:
        return np.nan, 0.0, 0, len(coords)

    
    pixel_id = v * W + u
    _, counts = np.unique(pixel_id, return_counts=True)

    
    collision_rate = float(counts[counts > 1].sum() / n_rasterized)

    # O_H^(i): occupied pixels / HW
    occupancy = float(len(counts) / (H * W))

    return collision_rate, occupancy, n_rasterized, len(coords)



from scipy.spatial import cKDTree

def compute_nn_spacing_pixels(adata, delta_H, spatial_key="spatial"):
    """Compute nn spacing pixels."""
    coords = np.asarray(adata.obsm[spatial_key], dtype=np.float64)[:, :2]
    valid = np.all(np.isfinite(coords), axis=1)
    coords = coords[valid]

    if len(coords) < 2:
        return np.array([], dtype=np.float64)

    
    unique_coords, inverse = np.unique(coords, axis=0, return_inverse=True)

    if len(unique_coords) < 2:
        return np.array([], dtype=np.float64)

    tree = cKDTree(unique_coords)
    distances, _ = tree.query(unique_coords, k=2)

    
    nn_unique = distances[:, 1]

    
    d_nn = nn_unique[inverse]
    s_H_n = d_nn / float(delta_H)

    return s_H_n[np.isfinite(s_H_n) & (s_H_n > 0)]

