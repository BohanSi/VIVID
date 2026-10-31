# Validation record

Checked on 2026-10-31 in the existing Linux environment (Python 3.12, PyTorch 2.6, CPU fallback). This differs from the recommended clean Python 3.10 dependency range; these results do not establish compatibility for every dependency/GPU combination.

Actual MB1 loading confirmed 16,861 and 22,840 observations with 1,120 shared features; automatic canvas selection returned 1024×1024. The endpoint plot was generated in temporary validation storage.

Final regression result: **42 tests passed**, including the opt-in full RIFE + SciPy reconstruction. The default suite skips four model-dependent integration cases. Wheel and source distribution builds succeeded through `setuptools.build_meta` (the local environment has no `build` frontend).

## Checks performed

- Explicit h5ad path lists and manually named input dictionaries preserve the supplied order, including deliberately unsorted labels. Duplicate names and mismatched path/name counts are rejected.
- Three observed input files are run through the full GNN/RIFE/HiGHS pipeline with one and two targets per interval, producing two and four correctly named final files respectively. A three-input continuous-z case also completes with its pair-local coordinate scope recorded. Tests cover pair-specific visualizations, count/metadata checks, output reloading and scratch cleanup.
- Additional sequence checks cover shared output directories, retained pair subfolders, pair isolation, per-pair image sizes, named alignment to one reference, and resuming partially completed stages. The original numerical kernels are unchanged by the sequence wrapper.
- The revised MB1 tutorial's explicit-path loading and optional preprocessing cells were executed against the actual local files, retaining manual names `6` and `8`.


- Input and stage validation; single-cell/bin canvas selection and application; single-omics OT alignment; caller input isolation; h5ad naming/round trip; intermediate retention and cleanup; spatial plotting; notebook validation and Python syntax.
- Optimizer tests compare HiGHS against exhaustive enumeration on a small integer problem for alpha=0, 0.5 and 1, with and without global hard constraints. They check the actual candidate support, the objective and exact count conservation.
- Auto selection tests cover both an absent Gurobi installation and a Gurobi license error; explicit Gurobi mode propagates those errors.
- Real RIFE integration on 48+48 synthetic observations, three types and five features: GNN encoding (two debugging epochs), 512×512 canvas, one target, interpolation, density/count conservation, type proportions, HiGHS integer allocation, coordinates, AnnData, expression reconstruction and final h5ad saving. The final directory contains only `2_4_1.h5ad`. The short training run checks execution, not prediction quality.
- The initial auto-mode integration also completed despite the local Gurobi HostID mismatch, by falling back to HiGHS. The opt-in integration test now explicitly selects SciPy to exercise the license-free path.
- MB1 preprocessing is compared against the original concatenate/inner-join, normalize_total(10,000), log1p sequence on test data. The actual local MB1 endpoints are checked separately for loading, preprocessing, automatic size selection and input visualization.
- 137 migrated top-level definitions are tracked in `kernel_manifest.json`. Structural comparison masks string literals and removes docstrings; only `optimize_celltypes` has a non-text structural change, to dispatch the solver and record its identity. Original/current full hashes retain the distinction between text edits and unchanged definitions.
- Package wheel/source builds and English-only explanation checks are included in the final checks. No private data or weights are copied into the repository.

## Limits

The local Gurobi license fails with a HostID mismatch, so a real Gurobi-vs-HiGHS numerical comparison has not been performed. Equivalent formulations can return different allocations in tied cases, especially when the network LP chooses a different candidate support.

Actual MB1 full-scale prediction is not established by the synthetic integration test. Real-data biological accuracy, continuous-z and different-omics end-to-end results still need dataset-specific evaluation. GIMM-VFI and FILM source/configuration are included but their complete inference runs have not been validated here. Downloaded weights, CUDA and independent environments require local verification.

## Reproduce

Run from the repository root after installation:

```bash
python -m pytest -q
# Real RIFE weights required; no Gurobi license required:
VIVID_INTEGRATION=full python -m pytest -q
# Stop after flow/type proportions:
VIVID_INTEGRATION=flow python -m pytest tests/test_integration.py -q
python -m build
```

`VIVID_MODEL_ROOT=/path/to/model/root` can point to another local checkout containing `RIFE/` with weights. Default tests never download models and skip real-backend inference. On systems with read-only Python installation caches, set `NUMBA_CACHE_DIR` and `MPLCONFIGDIR` to writable directories before running Scanpy/plotting checks.

## Notebook import regression

The original first cell imported `vivid` before discovering the repository's `src/` directory. Earlier checks used PYTHONPATH and therefore missed the uninstalled-checkout startup failure. The initialization cell now discovers the root first, prepends its source directory, imports the checkout, verifies its location and prints the current kernel executable.

Five added checks exercise the actual initialization cell in fresh Python subprocesses with PYTHONPATH removed: repository root, notebooks directory, original parent workspace, and an unrelated directory with REPO_ROOT explicitly supplied; the fifth checks that the configuration cell exposes a manual optimizer variable. The revised initialization/loading/preprocessing cells also executed against actual MB1 sections 6 and 8 from the original parent workspace without PYTHONPATH, preserving the expected 16,861/22,840 observations and 1,120 features.

The tutorial now explicitly sets OPTIMIZATION_SOLVER="gurobi" by default. Users may deliberately select "scipy" or optional "auto"; the selected value is passed into configuration. A change requires a new configured session and rerunning subsequent stages. Source discovery does not replace installing dependencies in the selected notebook kernel or downloading interpolation weights.

The current full regression suite completed with **42 passed** (56.84 seconds), including real RIFE and explicitly selected SciPy reconstruction for two- and three-input cases.

## Visualization, output paths and default solver

The API, tutorial and main environment now default to Gurobi, and README requires Gurobi installation and valid licensing before the default workflow. Existing full-inference checks explicitly select SciPy because the local Gurobi license still has a HostID mismatch; they do not establish licensed Gurobi inference on this machine.

New plotting checks verify case-insensitive sorted-name assignment of the supplied 48-color palette, a one-row input/prediction layout, two encoding panels without a type-RGB legend, flipped image y direction, single-line interpolation titles and a compact feature colorbar. Executing each of the five plotting calls through IPython's real inline backend captures exactly one PNG rich result and zero additional automatic figures. This verifies the inline display path without requiring a separate network kernel. An actual MB1 input preview uses 23 types and the first 23 supplied colors.

Output directories use the dataset, uppercase encoding, backend and local `MonDD_HH_MM` time (for example `MB1_GNN_RIFE_Oct14_20_32`), without random identifiers. A regression test checks the filename pattern, printed absolute path and Gurobi default. The retained explicit alternatives remain available. Numerical reconstruction kernels are unchanged.

The palette regression uses mixed-case labels `z`, `A`, `m` and verifies the legend order `A`, `m`, `z` for both input and prediction plots. Original label spelling is retained.

The Gurobi tutorial no longer advertises alternative optimizers. Reconstruction progress reports the configured solver only (Gurobi for the default), and adjacent-pair progress/image titles use `left / right` without arrows. Stale notebook outputs were cleared.

The final folder-naming update passed 19 targeted workflow/series checks (three model-dependent cases skipped), including output naming, printed destinations and final h5ad save/cleanup.
