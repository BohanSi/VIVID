"""Stepwise public API for VIVID spatial slice generation."""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
import json
import re
import shutil
import sys
import tempfile

from ._defaults import DEFAULTS


def _safe_id(value):
    value = str(value)
    if not re.fullmatch(r"[\w.-]+", value) or value in {".", ".."}:
        raise ValueError("Identifiers must contain only letters, numbers, _, - or .")
    return value


def _validate(adata, celltype_key, spatial_key):
    import numpy as np
    from scipy import sparse
    if adata.n_obs < 2 or adata.n_vars == 0:
        raise ValueError("Each endpoint must have at least two observations and one feature.")
    if not adata.var_names.is_unique or not adata.obs_names.is_unique:
        raise ValueError("Observation and feature names must be unique within each slice.")
    if celltype_key not in adata.obs or adata.obs[celltype_key].isna().any():
        raise ValueError(f"Missing or null cell type labels: {celltype_key}")
    coords = np.asarray(adata.obsm[spatial_key])
    if coords.shape != (adata.n_obs, 2) or not np.isfinite(coords).all():
        raise ValueError("Aligned spatial coordinates must be finite with shape (n_obs, 2).")
    values = adata.X.data if sparse.issparse(adata.X) else np.asarray(adata.X)
    if not np.isfinite(values).all():
        raise ValueError("Expression contains NaN/Inf.")
    if np.unique(coords, axis=0).shape[0] < 2:
        raise ValueError("Each slice needs at least two distinct spatial locations.")


def read_inputs(left_path, right_path=None, *, slice_names=None,
                celltype_key="cell_type", spatial_key="spatial"):
    """Read explicit h5ad paths in the supplied physical order.

    With a path sequence, slice_names is required and the result is an ordered
    name -> AnnData dictionary. Names are never inferred from filenames or sorted.
    The legacy two-positional-path form returns a (left, right) tuple.
    Expression and coordinates are not modified by this loader.
    """
    import anndata as ad
    if right_path is not None:
        if slice_names is not None:
            raise ValueError("Use a path list with slice_names for manually named inputs.")
        left, right = ad.read_h5ad(left_path), ad.read_h5ad(right_path)
        for obj in (left, right):
            _validate(obj, celltype_key, spatial_key)
        print(f"Loaded endpoints: {left.n_obs:,} / {right.n_obs:,} observations.")
        return left, right
    if isinstance(left_path, (str, Path)):
        raise ValueError("Provide at least two h5ad paths as a list and define slice_names.")
    paths = list(left_path)
    if slice_names is None or isinstance(slice_names, str):
        raise ValueError("Manually define slice_names in the same order as the input paths.")
    names = [_safe_id(name) for name in slice_names]
    if len(paths) < 2 or len(names) != len(paths) or len(names) != len(set(names)):
        raise ValueError("Provide at least two paths and one unique manual name per path.")
    result = {}
    for path, name in zip(paths, names):
        obj = ad.read_h5ad(path)
        _validate(obj, celltype_key, spatial_key)
        result[name] = obj
        print(f"Loaded {name}: {obj.n_obs:,} observations | {Path(path)}")
    return result


@dataclass
class Session:
    """A private computation state. Use as a context manager to release scratch files."""
    output_dir: Path
    config: dict
    left_id: str
    right_id: str
    save_intermediates: bool
    _state: dict = field(repr=False)
    _temporary: object = field(default=None, repr=False)
    stage: str = "configured"
    results: dict = field(default_factory=dict)
    saved_paths: list = field(default_factory=list)
    _closed: bool = False

    @property
    def work_dir(self):
        return Path(self._state["save_dir"])

    @property
    def endpoints(self):
        return tuple(self._state["anndata_dict"].values())

    def close(self):
        """Remove temporary intermediates; in-memory predictions remain available."""
        if self._temporary is not None:
            self._temporary.cleanup()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def configure(left, right=None, *, dataset_name="my_dataset", left_id="1", right_id="2",
              target_number=1, resolution="singlecell", generation_z=False,
              left_omics="RNA", right_omics="RNA", same_omics=None,
              encoding="gnn", interpolation="RIFE", celltype_key="cell_type",
              spatial_key="spatial", save_root="save", save_intermediates=False,
              model_root=None, backend_python=None, seed=0, optimization_solver="gurobi", advanced=None, slice_omics=None):
    """Configure a pair or ordered named inputs, retaining notebook defaults.

    For a mapping, return a SeriesSession with target_number per adjacent pair.
    Names and physical order come from the mapping; no filename sorting occurs.

    model_root contains sibling RIFE/, GIMM-VFI/, FILM/ directories.
    backend_python selects a separate backend environment's absolute Python path.
    advanced overrides numerical settings such as num_epochs or z_section_thickness.
    optimization_solver defaults to gurobi; scipy and auto are explicit alternatives.
    """
    if right is None:
        from .series import configure_series
        return configure_series(
            left, dataset_name=dataset_name, target_number=target_number,
            resolution=resolution, generation_z=generation_z,
            left_omics=left_omics, right_omics=right_omics, same_omics=same_omics,
            encoding=encoding, interpolation=interpolation, celltype_key=celltype_key,
            spatial_key=spatial_key, save_root=save_root,
            save_intermediates=save_intermediates, model_root=model_root,
            backend_python=backend_python, seed=seed, optimization_solver=optimization_solver,
            advanced=advanced, slice_omics=slice_omics,
        )
    if slice_omics is not None:
        raise ValueError("slice_omics is for named input sequences; use left_omics/right_omics for a pair.")
    import numpy as np
    from ._utils import set_global_seed, make_run_directory
    for obj in (left, right):
        _validate(obj, celltype_key, spatial_key)
    if isinstance(target_number, bool) or not isinstance(target_number, int) or target_number < 1:
        raise ValueError("target_number must be a positive integer.")
    resolution, encoding = resolution.lower(), encoding.lower()
    interpolation = {"RIFE": "RIFE", "GIMM": "GIMM", "GIMM-VFI": "GIMM", "FILM": "FILM"}.get(interpolation.upper())
    if resolution not in {"singlecell", "bin"} or encoding not in {"gnn", "pca", "spherical"} or interpolation is None:
        raise ValueError("Use singlecell/bin; gnn/pca/spherical; RIFE/GIMM-VFI/FILM.")
    dataset_name, left_id, right_id = map(_safe_id, (dataset_name, left_id, right_id))
    if left_id == right_id:
        raise ValueError("Left and right slice identifiers must differ.")
    left_omics, right_omics = str(left_omics), str(right_omics)
    if not left_omics.strip() or not right_omics.strip():
        raise ValueError("Omics names cannot be empty.")
    same_omics = left_omics == right_omics if same_omics is None else bool(same_omics)
    if not same_omics and left_omics == right_omics:
        raise ValueError("Different omics require distinct left_omics/right_omics names.")
    if same_omics and not len(left.var_names.intersection(right.var_names)):
        raise ValueError("Same-omics endpoints need common features.")
    if optimization_solver not in {"auto", "gurobi", "scipy"}:
        raise ValueError("optimization_solver must be auto, gurobi, or scipy.")
    cfg = dict(DEFAULTS)
    protected = {"resolution", "generation_z", "left_omics", "right_omics", "same_omics", "image_encoding_method", "target_number", "seed", "optimization_solver"}
    if advanced and protected.intersection(advanced):
        raise ValueError("Set public options through named arguments, rather than advanced.")
    cfg.update(advanced or {})
    cfg.update(resolution=resolution, generation_z=bool(generation_z), left_omics=left_omics,
               right_omics=right_omics, same_omics=same_omics, image_encoding_method=encoding,
               target_number=target_number, seed=int(seed), optimization_solver=optimization_solver)
    # Copies isolate user objects from rasterization, type assignment, and z metadata.
    endpoints = [left.copy(), right.copy()]
    for obj in endpoints:
        obj.obsm["spatial"] = np.asarray(obj.obsm[spatial_key], dtype=np.float64).copy()
        obj.obs[celltype_key] = obj.obs[celltype_key].astype(str)
    if model_root is None:
        model_root = Path(__file__).resolve().parents[2]
    model_root = Path(model_root).expanduser().resolve()
    backend_dir = model_root / {"RIFE": "RIFE", "GIMM": "GIMM-VFI", "FILM": "FILM"}[interpolation]
    if backend_python is None:
        if interpolation != "RIFE":
            raise ValueError("Set backend_python to the GIMM-VFI/FILM environment's Python path.")
        backend_python = sys.executable
    python_path = Path(backend_python).expanduser().resolve()
    if not python_path.is_file():
        raise FileNotFoundError(f"Backend Python not found: {python_path}")
    out = make_run_directory(save_root, dataset_name, encoding, interpolation)
    temporary = None
    if save_intermediates:
        work = out / "intermediates"
        work.mkdir()
    else:
        temporary = tempfile.TemporaryDirectory(prefix="vivid_")
        work = Path(temporary.name)
    state = dict(__name__="vivid._session", config=cfg, dataset_name=dataset_name,
                 celltype_col=celltype_key, seed=int(seed), image_encoding_method=encoding,
                 save_dir=str(work), anndata_dict=dict(zip(("slice_1", "slice_2"), endpoints)),
                 real_adata_dict=dict(zip(("slice_1", "slice_2"), endpoints)),
                 adata1=endpoints[0], adata2=endpoints[1], interpolation_model_names=[interpolation],
                 backend_dir=str(backend_dir), backend_python=str(python_path),
                 blend_mode="distance", use_neighborhood_pixels=True, use_mask=False,
                 mask_mode="center_neighborhood_mean", optimization_alpha=0.5,
                 rife_random_init=False, rife_random_init_seed=0, rife_zero_correspondence=False,
                 rife_perturb_fraction=0.0, rife_perturb_seed=0, rife_uniform_mask=False)
    set_global_seed(seed)
    session = Session(out, cfg, left_id, right_id, bool(save_intermediates), state, temporary)
    print(f"Save directory: {out}\nEncoding: {encoding.upper()} | Interpolation: {interpolation} | Targets: {target_number}")
    return session


def _require(session, stage):
    if session._closed:
        raise RuntimeError("This session's working files are closed. Create a new session.")
    if session.stage != stage:
        raise RuntimeError(f"Expected stage {stage!r}, current stage is {session.stage!r}.")


def _run(session, *stages):
    """Run packaged Python kernels with isolated globals, retaining function semantics.

    No notebook is opened, no caller globals are read, and no source is downloaded.
    A shared per-session namespace preserves the original kernels' global references.
    """
    log_path = session.work_dir / "execution.log"
    try:
        with log_path.open("a", encoding="utf-8") as log, contextlib.redirect_stdout(log):
            for name in stages:
                resource = files("vivid._core").joinpath(name + ".py")
                source = resource.read_text(encoding="utf-8")
                exec(compile(source, str(resource), "exec"), session._state)
    except Exception:
        print(f"Stage failed; log: {log_path}")
        print(log_path.read_text(encoding="utf-8")[-4000:])
        raise


def select_image_size(session):
    """Apply the notebook's endpoint-only single-cell/bin resolution criterion."""
    from .series import SeriesSession
    if isinstance(session, SeriesSession):
        return session.run_step("select_image_size")
    _require(session, "configured")
    print("Selecting image size from the two observed endpoints...")
    _run(session, "imports", "image_size")
    size = int(session._state["H_recommended"])
    session._state.update(target_HW=size, patch_size=128 if size < 2048 else 256)
    session.config.update(target_H=size, target_W=size)
    session.stage = "sized"
    print(f"Selected canvas: {size} × {size}")
    return size


def encode(session):
    """Rasterize, smooth density, and encode cell type/context as latent RGB."""
    from .series import SeriesSession
    if isinstance(session, SeriesSession):
        return session.run_step("encode")
    _require(session, "sized")
    print("Encoding: rasterizing cell types and constructing the shared grid...")
    _run(session, "diagnostics", "raster")
    s = session._state
    s["outputs"], s["grid_meta"], s["celltypes"] = s["build_fixed_shape_density_tensors"](
        anndata_dict=s["anndata_dict"], spatial_key="spatial", celltype_key=s["celltype_col"],
        target_H=s["target_HW"], target_W=s["target_HW"], margin_pixels=session.config["margin_pixels"])
    s["save_all_density_outputs"](s["outputs"], s["grid_meta"], s["celltypes"], save_dir=s["save_dir"], save_extra=True)
    print("Encoding: smoothing global density and computing latent RGB images...")
    _run(session, "gaussian", "smooth", "graph_data", "graph_model", "encode_gnn", "encode_static")
    session.stage = "encoded"
    print("Encoding complete. Use vivid.plot_encoding(session) to inspect both endpoints.")
    return session


def check_backend(session):
    """Check inference scripts and user-downloaded weights before expensive work."""
    from .series import SeriesSession
    if isinstance(session, SeriesSession):
        return all(check_backend(pair) for pair in session.pairs)
    root = Path(session._state["backend_dir"])
    name = session._state["interpolation_model_names"][0]
    required = {
        "RIFE": ["inference_img_flow.py", "train_log/RIFE_HDv3.py", "train_log/IFNet_HDv3.py", "train_log/flownet.pkl"],
        "GIMM": ["src/video_Nx_gimm_vivid.py", "configs/gimmvfi/gimmvfi_r_arb.yaml", "pretrained_ckpt/gimmvfi_r_arb_lpips.pt", "pretrained_ckpt/raft-things.pth"],
        "FILM": ["run_film_pair_from_jupyter.py", "pretrained_models/film_net/Style/saved_model/saved_model.pb", "pretrained_models/film_net/Style/saved_model/variables/variables.index"],
    }[name]
    missing = [str(root / p) for p in required if not (root / p).is_file()]
    if missing:
        raise FileNotFoundError("Missing backend files; see README weight instructions:\n" + "\n".join(missing))
    print(f"{name}: inference script and weight files found. Python: {session._state['backend_python']}")
    return True


def interpolate(session):
    """Run one backend in its own Python process and validate optical flow output."""
    from .series import SeriesSession
    if isinstance(session, SeriesSession):
        return session.run_step("interpolate")
    _require(session, "encoded")
    check_backend(session)
    name = session._state["interpolation_model_names"][0]
    print(f"Interpolating encoded endpoints with {name}...")
    _run(session, "interpolate_" + {"RIFE": "rife", "GIMM": "gimm", "FILM": "film"}[name])
    session.stage = "interpolated"
    print(f"Interpolation complete; selected frames: {session._state['target_img']}")
    return session


def reconstruct(session):
    """Preserve density/count, flow, integer type allocation, coordinates and expression."""
    from .series import SeriesSession
    if isinstance(session, SeriesSession):
        return session.run_step("reconstruct")
    _require(session, "interpolated")
    solver_name = {"gurobi": "Gurobi", "scipy": "SciPy/HiGHS",
                   "auto": "the configured solver"}[session.config["optimization_solver"]]
    for text, stages in [
        ("Detecting sites and conserving the interpolated total cell count", ("detect",)),
        ("Mapping flow and estimating cell type proportions", ("proportions",)),
        (f"Optimizing integer cell type allocations with {solver_name}", ("optimize",)),
        ("Reconstructing single-cell or bin coordinates", ("coordinates",)),
        ("Assigning optional continuous z coordinates", ("depth",)),
        ("Building target AnnData objects", ("anndata",)),
        ("Predicting expression using flow and microenvironment fallback", ("expression_imports", "expression")),
    ]:
        print(text + "...")
        _run(session, *stages)
    import numpy as np
    from scipy import sparse
    objects = [a for a in session._state["pred_expr_adata_dict"].values()
               if a.uns.get("record_kind") == "predicted_middle"]
    objects.sort(key=lambda a: int(a.uns["image_index"]))
    if len(objects) != session.config["target_number"]:
        raise RuntimeError(f"Expected {session.config['target_number']} targets, got {len(objects)}.")
    for i, obj in enumerate(objects, 1):
        arrays = [obj.X, *obj.layers.values(), *[v for k, v in obj.obsm.items() if k.startswith("pred_")]]
        for arr in arrays:
            values = arr.data if sparse.issparse(arr) else np.asarray(arr)
            if not np.isfinite(values).all():
                raise RuntimeError(f"Target {i}: prediction contains NaN/Inf.")
        obj.uns["vivid"] = dict(left_id=session.left_id, right_id=session.right_id,
                                  target_index=i, parameters=dict(session.config),
                                  optimization_backends_used=list(session._state.get("optimization_backends_used", [])))
        key = f"{session.left_id}_{session.right_id}_{i}"
        session.results[key] = obj
    session.stage = "reconstructed"
    print(f"Reconstruction complete: {len(session.results)} target AnnData objects.")
    return session.results


def _serializable(value):
    """Normalize AnnData metadata while retaining summaries inside the final h5ad."""
    import numpy as np
    import pandas as pd
    if value is None:
        return "null"
    if isinstance(value, dict):
        return {str(k): _serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        if not value:
            return np.array([], dtype=str)
        if any(isinstance(x, (dict, list, tuple)) for x in value):
            return json.dumps(value, default=lambda x: x.item() if isinstance(x, np.generic) else str(x), ensure_ascii=False)
        return np.asarray(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.DataFrame):
        return value.copy()
    return value


def save_results(session):
    """Write leftID_rightID_targetIndex.h5ad; remove temporary work after success."""
    from .series import SeriesSession
    if isinstance(session, SeriesSession):
        return session.run_step("save_results")
    _require(session, "reconstructed")
    destinations = [session.output_dir / (key + ".h5ad") for key in session.results]
    for dest in destinations:
        if dest.exists():
            raise FileExistsError(dest)
    pending = []
    try:
        for (key, obj), dest in zip(session.results.items(), destinations):
            scratch = session.output_dir / ("." + dest.name + ".tmp")
            pending.append(scratch)
            a = obj.copy()
            a.uns = _serializable(dict(a.uns))
            a.write_h5ad(scratch, compression="gzip")
        for scratch, dest in zip(pending, destinations):
            scratch.rename(dest)
    finally:
        for scratch in pending:
            scratch.unlink(missing_ok=True)
    session.saved_paths = destinations
    session.stage = "saved"
    if session.save_intermediates:
        metadata = dict(config=session.config, left_id=session.left_id, right_id=session.right_id,
                        backend_dir=session._state["backend_dir"], backend_python=session._state["backend_python"])
        (session.work_dir / "run.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    session.close()
    print(f"Saved {len(destinations)} h5ad files to {session.output_dir}")
    return destinations
