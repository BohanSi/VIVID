# Migrated from 1002genertation.ipynb, cell 21. See docs/MIGRATION.md.
# ============================================================
# Gaussian smoothing of global T_count
# Save PNG + NPZ
# ============================================================

import re, json
import numpy as np
from pathlib import Path
from PIL import Image
from scipy.ndimage import gaussian_filter


# ============================================================
# 0. Configuration
# ============================================================

gaussian_sigma_pixels = 1.0
gaussian_truncate = 4.0


gaussian_intensity_mode = "folder_max"

gaussian_folder_name = "gaussian_density"
gaussian_save_npz = True


# ============================================================
# 1. Utilities
# ============================================================

def sanitize_name(name):
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))
    return re.sub(r"_+", "_", name).strip("_") or "unknown"


def make_slice_name(slice_name):
    name = sanitize_name(str(slice_name).strip())
    if name.lower().startswith("slice_"): return f"slice_{name[6:]}"
    if name.lower().startswith("slice"): return f"slice_{name[5:].lstrip('_')}"
    return f"slice_{name}"


def gaussian_smooth_preserve_sum(count_map, sigma=1.0, truncate=4.0, eps=1e-12):
    """Gaussian smooth preserve sum."""
    arr = np.asarray(count_map, dtype=np.float32)

    if arr.ndim != 2: raise ValueError(f"count_map  must be  H×W,  current ={arr.shape}")
    if sigma < 0: raise ValueError("sigma  must not be less than 0.")

    original_sum = float(arr.sum())

    if original_sum <= eps: return np.zeros_like(arr, dtype=np.float32)
    if sigma == 0: return arr.copy()

    smoothed = gaussian_filter(
        arr.astype(np.float64),
        sigma=float(sigma),
        mode="constant",
        cval=0.0,
        truncate=float(truncate),
    )

    smoothed = np.maximum(smoothed, 0.0)
    smooth_sum = float(smoothed.sum())

    if smooth_sum > eps: smoothed *= original_sum / smooth_sum

    return smoothed.astype(np.float32)


def density_to_rgb_uint8(density, vmax):
    """
    Gaussian density -> RGB PNG

        density = 0    -> black
        density = vmax -> white
    """
    density = np.asarray(density, dtype=np.float32)

    if vmax <= 0:
        density_norm = np.zeros_like(density, dtype=np.float32)
    else:
        density_norm = np.clip(density / float(vmax), 0.0, 1.0).astype(np.float32)

    gray = np.rint(density_norm * 255.0).astype(np.uint8)
    rgb = np.repeat(gray[:, :, None], 3, axis=2)

    return rgb, density_norm



# ============================================================
# 2. Gaussian smoothing + PNG/NPZ saving
# ============================================================

def gaussian_smooth_and_save_global_density(
    outputs,
    save_dir,
    sigma_pixels=1.0,
    truncate=4.0,
    folder_name="gaussian_density",
    intensity_mode="folder_max",
    save_npz=True,
    verbose=True,
):
    """Gaussian smooth and save global density."""
    if len(outputs) == 0: raise ValueError("outputs  is empty .")
    if intensity_mode not in ["folder_max", "slice_max"]: raise ValueError("intensity_mode  must be  folder_max  or  slice_max.")

    root = Path(save_dir).resolve() / folder_name
    png_dir = root / "png"
    npz_dir = root / "npz"

    png_dir.mkdir(parents=True, exist_ok=True)
    if save_npz: npz_dir.mkdir(parents=True, exist_ok=True)

    smoothed_outputs = {}
    expected_shape = None

    if verbose:
        print("="*100)
        print("GLOBAL GAUSSIAN DENSITY")
        print("="*100)
        print("sigma          :", sigma_pixels)
        print("truncate       :", truncate)
        print("intensity_mode :", intensity_mode)
        print("PNG directory  :", png_dir)
        print("NPZ directory  :", npz_dir)
        print("="*100)

    # ========================================================
    # A. Gaussian smoothing
    # ========================================================

    for slice_name, result in outputs.items():
        if "T_count" not in result: raise KeyError(f"{slice_name}: outputs  not found in  T_count.")

        T_count = np.asarray(result["T_count"], dtype=np.float32)

        if T_count.ndim != 2: raise ValueError(f"{slice_name}: T_count  must be  H×W,  current ={T_count.shape}")

        if expected_shape is None: expected_shape = T_count.shape
        elif T_count.shape != expected_shape: raise ValueError(f"{slice_name}: shape={T_count.shape},  expected ={expected_shape}")

        density = gaussian_smooth_preserve_sum(
            T_count,
            sigma=sigma_pixels,
            truncate=truncate,
        )

        smoothed_outputs[slice_name] = {
            "T_count": T_count,
            "gaussian_density": density,
            "original_count": float(T_count.sum()),
            "smoothed_count": float(density.sum()),
            "density_max": float(density.max()),
        }

        if verbose:
            print(
                f"{slice_name}: cells={T_count.sum():.0f} | "
                f"gaussian_sum={density.sum():.6f} | "
                f"max={density.max():.6f}"
            )

    # ========================================================
    # B. Determine intensity scale
    # ========================================================

    global_vmax = None

    if intensity_mode == "folder_max":
        global_vmax = max(float(x["gaussian_density"].max()) for x in smoothed_outputs.values())

    if verbose and global_vmax is not None:
        print("\nGlobal vmax:", global_vmax)

    # ========================================================
    # C. Save PNG + NPZ
    # ========================================================

    summary = {}

    for slice_name, result in smoothed_outputs.items():
        T_count = result["T_count"]
        density = result["gaussian_density"]

        vmax = float(global_vmax) if intensity_mode == "folder_max" else float(density.max())
        rgb, density_norm = density_to_rgb_uint8(density, vmax=vmax)

        stem = make_slice_name(slice_name)
        png_path = png_dir / f"{stem}.png"
        npz_path = npz_dir / f"{stem}.npz"

        Image.fromarray(rgb, mode="RGB").save(png_path)

        if save_npz:
            np.savez_compressed(
                npz_path,

                
                gaussian_density=density.astype(np.float32),

                
                T_count=T_count.astype(np.float32),

                
                density_norm=density_norm.astype(np.float32),

                
                rgb=rgb.astype(np.uint8),

                slice_name=np.array(str(slice_name)),
                vmax=np.array(vmax, dtype=np.float32),
                sigma_pixels=np.array(float(sigma_pixels), dtype=np.float32),
                truncate=np.array(float(truncate), dtype=np.float32),
                original_count=np.array(result["original_count"], dtype=np.float32),
                smoothed_count=np.array(result["smoothed_count"], dtype=np.float32),
            )

        result["density_norm"] = density_norm
        result["rgb"] = rgb
        result["vmax"] = vmax
        result["png_path"] = png_path
        result["npz_path"] = npz_path if save_npz else None

        summary[slice_name] = {
            "png_path": str(png_path),
            "npz_path": str(npz_path) if save_npz else None,
            "original_count": result["original_count"],
            "smoothed_count": result["smoothed_count"],
            "density_max": result["density_max"],
            "vmax": vmax,
        }

        if verbose:
            print(f"Saved {slice_name}: {png_path.name} + {npz_path.name}")

    # ========================================================
    # D. Metadata
    # ========================================================

    H, W = expected_shape

    metadata = {
        "representation": "global_gaussian_density",
        "H": int(H),
        "W": int(W),
        "sigma_pixels": float(sigma_pixels),
        "truncate": float(truncate),
        "boundary_mode": "constant",
        "boundary_cval": 0.0,
        "mass_preservation": True,
        "intensity_mode": str(intensity_mode),
        "global_vmax": None if global_vmax is None else float(global_vmax),
        "png_mapping": "gray = clip(gaussian_density / vmax, 0, 1) * 255",
        "png_mode": "RGB",
        "png_dtype": "uint8",
        "npz_gaussian_density_dtype": "float32",
        "slice_names": [str(x) for x in outputs.keys()],
        "num_slices": int(len(outputs)),
        "note": "NPZ gaussian_density is the Gaussian-smoothed floating-point density and should be used for numerical analysis. PNG is only its visualization.",
    }

    with open(root / "gaussian_meta.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    with open(root / "gaussian_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if verbose:
        print("\n"+"="*100)
        print("GAUSSIAN DENSITY FINISHED")
        print("="*100)
        print("Root  :", root)
        print("PNG   :", png_dir)
        print("NPZ   :", npz_dir)
        print("Shape :", f"{W} × {H}")
        print("Slices:", len(outputs))
        print("="*100)

    return smoothed_outputs, root, metadata, summary