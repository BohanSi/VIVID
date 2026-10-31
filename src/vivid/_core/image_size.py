# Migrated from 1002genertation.ipynb, cell 6. See README.md.
from scipy.spatial import cKDTree

def infer_bin_spacing(adata, spatial_key="spatial"):
    """Infer original bin lattice spacing from median nearest non-zero-neighbor distance."""
    coords = np.asarray(adata.obsm[spatial_key], dtype=np.float64)[:, :2]
    coords = coords[np.all(np.isfinite(coords), axis=1)]
    coords = np.unique(coords, axis=0)

    if len(coords) < 2:
        raise ValueError("Not enough valid spatial coordinates to infer bin spacing.")

    tree = cKDTree(coords)
    dist, _ = tree.query(coords, k=2)
    nn_dist = dist[:, 1]
    nn_dist = nn_dist[np.isfinite(nn_dist) & (nn_dist > 0)]

    if len(nn_dist) == 0:
        raise ValueError("Cannot infer a valid non-zero bin spacing.")

    return float(np.median(nn_dist))
    


allowed_sizes = [512, 1024, 2048]

if config["resolution"] == "singlecell":
    
    N1, N2 = int(adata1.n_obs), int(adata2.n_obs)
    N_bar = (N1 + N2) / 2.0

    sigma_render = 1.0  # pixel
    A_eff = np.pi * (3.0 * sigma_render) ** 2

    # Theoretical minimum size
    H_min_theoretical = int(np.ceil(np.sqrt(N_bar * A_eff)))

    # Choose the smallest allowed size satisfying Lambda_H <= 1
    valid_sizes = []
    for H in allowed_sizes:
        Lambda_H = N_bar * A_eff / H**2
        if Lambda_H <= 1.0:
            valid_sizes.append(H)

    if not valid_sizes:
        raise ValueError(
            f"No allowed raster size satisfies Lambda_H <= 1. "
            f"Theoretical minimum H={H_min_theoretical}, "
            f"but maximum allowed size is {max(allowed_sizes)}."
        )

    H_recommended = min(valid_sizes)
    Lambda_H = N_bar * A_eff / H_recommended**2

    print("\n[Automatic raster-size selection: single-cell]")
    print(f"  endpoint counts N1, N2   = {N1}, {N2}")
    print(f"  mean endpoint count N_bar= {N_bar:.1f}")
    print(f"  sigma_render             = {sigma_render:.4f} pixels")
    print(f"  A_eff = pi(3sigma)^2     = {A_eff:.4f} pixels^2")
    print(f"  theoretical minimum H    = {H_min_theoretical}")
    print(f"  allowed sizes            = {allowed_sizes}")
    print(f"  recommended H = W        = {H_recommended}")
    print(f"  Lambda_H                 = {Lambda_H:.6f}  (required <= 1)")

    #config["target_H"] = H_recommended
    #config["target_W"] = H_recommended


elif config["resolution"] == "bin":

    r = 4
    margin_pixels = int(config.get("margin_pixels", 0))
    allowed_sizes = [512, 1024, 2048]

    # ============================================================
    # Infer original bin lattice spacing ONLY from observed endpoints
    # ============================================================
    d_bin_1 = infer_bin_spacing(adata1, spatial_key="spatial")
    d_bin_2 = infer_bin_spacing(adata2, spatial_key="spatial")
    d_bin = float(np.median([d_bin_1, d_bin_2]))

    # ============================================================
    # Spatial extent: ONLY observed endpoints
    # ============================================================
    coords1 = np.asarray(adata1.obsm["spatial"], dtype=np.float64)[:, :2]
    coords2 = np.asarray(adata2.obsm["spatial"], dtype=np.float64)[:, :2]

    coords1 = coords1[np.all(np.isfinite(coords1), axis=1)]
    coords2 = coords2[np.all(np.isfinite(coords2), axis=1)]

    if len(coords1) == 0 or len(coords2) == 0:
        raise ValueError("adata1 or adata2 contains no valid spatial coordinates.")

    all_coords = np.vstack([coords1, coords2])

    L_x = float(np.max(all_coords[:, 0]) - np.min(all_coords[:, 0]))
    L_y = float(np.max(all_coords[:, 1]) - np.min(all_coords[:, 1]))
    L_max = max(L_x, L_y)

    if L_max <= 0:
        raise ValueError("Invalid spatial extent.")

    # ============================================================
    # Select minimum raster size satisfying:
    #
    # s_H^bin = d_bin / Delta_H
    # floor(s_H^bin) > r
    # ============================================================
    diagnostics = []

    for H in allowed_sizes:
        inner_H = H - 2 * margin_pixels
        if inner_H <= 0:
            continue

        Delta_H = L_max / inner_H
        s_H_bin = d_bin / Delta_H
        floor_s = int(np.floor(s_H_bin + 1e-12))
        valid = floor_s > r

        diagnostics.append((H, Delta_H, s_H_bin, floor_s, valid))

    valid_candidates = [x for x in diagnostics if x[4]]

    if not valid_candidates:
        details = ", ".join(
            f"{H}: s_H^bin={s:.3f}"
            for H, _, s, _, _ in diagnostics
        )
        raise ValueError(
            f"No allowed raster size satisfies floor(s_H^bin) > {r}. "
            f"Candidate results: {details}"
        )

    H_recommended, Delta_H, s_H_bin, floor_s, _ = valid_candidates[0]

    # ============================================================
    # Print
    # ============================================================
    print("\n[Automatic raster-size selection: bin]")
    print(f"  inferred bin spacing:")
    print(f"    endpoint 1              = {d_bin_1:.6g}")
    print(f"    endpoint 2              = {d_bin_2:.6g}")
    print(f"    final d_bin             = {d_bin:.6g}")
    print(f"  endpoint spatial extent  = Lx={L_x:.6g}, Ly={L_y:.6g}")
    print(f"  localization radius r    = {r} pixels")
    print(f"  criterion                = floor(s_H^bin) > {r}")

    print("\n  Candidate raster sizes:")
    for H, delta, s, floor_value, valid in diagnostics:
        print(
            f"    {H:4d} x {H:<4d}: "
            f"Delta_H={delta:.6g}, "
            f"s_H^bin={s:.4f}, "
            f"floor={floor_value}, "
            f"{'PASS' if valid else 'FAIL'}"
        )

    print(f"\n  recommended H = W        = {H_recommended}")
    print(f"  raster pixel size Delta_H= {Delta_H:.6g}")
    print(f"  s_H^bin                  = {s_H_bin:.6f}")
    print(f"  floor(s_H^bin)           = {floor_s}")

    #config["target_H"] = H_recommended
    #config["target_W"] = H_recommended


else:
    raise ValueError(
        "config['resolution'] must be either 'singlecell' or 'bin'."
    )