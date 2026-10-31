# Migration from the research notebook

The source is `1002genertation.ipynb` (original spelling). The installed package does not load or execute the notebook. `source_manifest.json` records the original notebook and cell hashes.

The public interface is in `api.py`, `examples.py`, `alignment.py`, `plotting.py` and `optimization.py`. A Session owns private stage state. Numerical stages in `_core/*.py` execute in that namespace to retain the original functions' shared-variable and `globals()` semantics. No caller globals or user-supplied code strings are executed. Use separate processes for concurrent sessions; thread safety is not provided.

| Original cell | Stage | Retained computation |
| --- | --- | --- |
| 6 | image_size | Single-cell/bin size criteria, now applied to encoding |
| 13–14 | diagnostics, raster | Shared grid, type density and coordinate conversion |
| 21–22 | gaussian, smooth | Mass-preserving Gaussian density |
| 24–27 | graph_data/model, encode_* | GNN/PCA/spherical RGB encoding |
| 30, 33, 36 | interpolate_* | Backend frame selection and endpoint flow interfaces |
| 39 | detect | RGB watershed, flow density and integer count conservation |
| 45 | proportions | Local/neighborhood flow-based type proportions |
| 47 | optimize | Integer type allocation and optional hard global totals |
| 50, 52 | coordinates, depth | Single-cell/bin coordinates and optional continuous z |
| 57, 60–61 | anndata, expression* | AnnData construction, expression propagation and fallback |

Dataset-specific branching, machine paths, global AnnData discovery, exploratory displays and the old save entry point were removed. Inputs, stage checks, temporary storage, output naming and visualization are handled by the public layer. Alignment is the supplied OT + Procrustes method wrapped for exactly two observed endpoints; no middle reference is loaded.

The MB1 loader reproduces sections 6/8, the inner feature join, total normalization to 10,000 and log1p from the original notebook. Corrected coordinates are used directly. The original fixed MB1 canvas of 1024 is replaced by endpoint-only automatic selection, so results need not match a manually configured old run bit for bit.

## Optional optimizer

The original `opt_solve_gurobi` and Gurobi network LP remain available. The optimization stage now dispatches through `opt_solve_selected`, with `auto`, `gurobi` or `scipy`. Gurobi import is optional. Auto falls back on missing installation or Gurobi runtime errors (including license failures); other algorithm errors are not silently swallowed.

SciPy/HiGHS solves the same network-flow warm problem and builds candidates with the original helper. It models integer candidate allocations and absolute deviations explicitly. Exact site/global constraints and the weighted site/patch L1 objective remain. Different tied network optima can produce different candidate support. SciPy does not expose MIP-start injection or Gurobi-specific tuning. No new rounding-only approximation replaces integer allocation.

English-only documentation, comments, docstrings and diagnostics replace the original Chinese annotations. These textual changes do not intentionally alter numerical kernels. `kernel_manifest.json` records original/current AST hashes and a structural hash with docstrings removed and string literals masked, making the scope of textual changes explicit. The solver-dispatch change in `optimize_celltypes` is recorded separately; hashes are not evidence of biological or output equivalence.

Intermediate files are still needed internally by the original algorithm. The save flag controls retention, not whether scratch files are ever created. Nested summaries are serialized into final h5ad metadata when necessary, without creating extra default output files.

## Ordered inputs and adjacent intervals

`read_inputs(paths, slice_names=...)` reads explicit h5ad paths and returns a dictionary in supplied physical order. Slice names are manual and never inferred from basenames. `configure(inputs, ...)` creates a `SeriesSession`, with one isolated original pair Session per consecutive interval. No nonadjacent pairs are generated. A target count K means K targets per interval, for a total of (N−1)×K.

The numerical kernels and kernel manifest are unchanged by this orchestration. Each pair independently selects its canvas and fits its encoder; shared observed sections are copied into each pair to avoid cross-pair mutation. The wrapper runs the same stage across every pair and combines results in physical order. It can resume a failed stage by skipping pairs that already completed it. All h5ad files share one dataset directory; retained intermediates use numbered pair subdirectories. Predictions are named with manual endpoint labels and a target index that restarts in each interval.

Named-sequence alignment registers all supplied observed slices to the first observed slice. Explicit MB1 preprocessing is a separate optional call; reading paths alone never normalizes data. Existing two-endpoint APIs and the MB1 convenience loader remain available. The original z algorithm is still pair-local and is labeled as such in result metadata; the sequence wrapper does not infer a global z registration.

## Display and default configuration updates

Gurobi is the default solver; SciPy and auto-fallback are manual alternatives. README and the main environment include mandatory Gurobi setup for the default workflow. Output directory names use `{dataset_name}_{ENCODING}_{interpolation_model}_{MonDD_HH_MM}`, for example `MB1_GNN_RIFE_Oct14_20_32`, with English month abbreviations and local time. The absolute destination is printed; existing same-minute directories are not overwritten.

Plot functions detach returned Figures from pyplot's automatic inline queue so a final-expression call displays once. This does not prevent saving the Figure or explicitly displaying an assigned Figure. Input/prediction type colors now follow the supplied palette in case-insensitive sorted-name order; RGB encoding/interpolation displays are flipped vertically without changing image files used by the model. Encoding displays omit type-RGB legend panels; interpolation titles and feature colorbars are compact.
