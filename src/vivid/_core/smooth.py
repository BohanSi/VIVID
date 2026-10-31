# Migrated from 1002genertation.ipynb, cell 22. See README.md.
# ============================================================
# 3. RUN

# ============================================================
gaussian_smoothed_outputs, gaussian_save_root, gaussian_meta, gaussian_summary = gaussian_smooth_and_save_global_density(
    outputs=outputs,
    save_dir=save_dir,
    sigma_pixels=gaussian_sigma_pixels,
    truncate=gaussian_truncate,
    folder_name=gaussian_folder_name,
    intensity_mode=gaussian_intensity_mode,
    save_npz=gaussian_save_npz,
    verbose=True,
)

gaussian_png_dir = gaussian_save_root / "png"
gaussian_npz_dir = gaussian_save_root / "npz"