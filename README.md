<p align="center">
  <img src="assets/VIVID.png" alt="VIVID logo" width="300">
</p>

# Reconstructing missing spatial-omics sections with video-derived correspondence priors

**VIVID** reconstructs unmeasured spatial-omics sections between two aligned observed endpoints by transferring correspondence knowledge learned from natural videos. It encodes spatial annotations, local tissue context and occupancy into images for a frozen pretrained interpolation backbone, then uses inferred target-to-endpoint correspondence fields to recover spatial locations, annotations and molecular profiles. VIVID supports reconstruction at multiple depths in cell- and bin-resolved spatial omics, including transcriptomic and proteomic measurements.

<p align="center">
  <img src="assets/Figure1.png" alt="Figure 1. Overview of the VIVID reconstruction framework" width="1000">
</p>

*Figure 1. Overview of VIVID.*

## Requirements

- Linux and Python 3.10 are recommended. Use a compatible PyTorch (2.0–2.5) and torchvision (0.15–0.20) installation; an NVIDIA GPU is recommended for interpolation.
- Input files in AnnData (`.h5ad`) format, containing spatial coordinates in `obsm["spatial"]`, annotation labels in `obs` and molecular features in `X`. Observed sections must share an aligned coordinate system and a common annotation scheme. Same-modality inputs must have matching molecular features.
- Pretrained weights for the selected interpolation backbone (RIFE by default).
- A valid [Gurobi license](https://www.gurobi.com/academia/academic-program-and-licenses/) for the default integer-optimization solver. Alternatively, choose `optimization_solver="scipy"` to use HiGHS without a Gurobi license.

## Installation

Clone the repository and install VIVID from its root directory:

```bash
git clone https://github.com/BohanSi/VIVID.git
cd VIVID
conda create -n vivid python=3.10 -y
conda activate vivid
# If needed, install a CUDA-compatible PyTorch/torchvision pair first.
python -m pip install -e ".[tutorial,gurobi]"
python -m ipykernel install --user --name vivid --display-name "Python (VIVID)"
```

You can also create the main environment from `environments/vivid.yml`. Install PyTorch for your system using the [official instructions](https://pytorch.org/get-started/locally/). To use HiGHS instead of Gurobi, the package can be installed with `python -m pip install -e ".[tutorial]"` and configured with `optimization_solver="scipy"`.

## Pretrained interpolation models

Download the selected model weights and place them at the following paths within the repository. **Weights are not included.**

| Backbone | Weights | Expected location |
| --- | --- | --- |
| **RIFE (default)** | [HD weights](https://drive.google.com/file/d/1APIzVeI-4ZZCEuIRE1m6WYfSCaOsi_7_/view) | `RIFE/train_log/flownet.pkl` |
| GIMM-VFI (R-P) | [Official model repository](https://huggingface.co/GSean/GIMM-VFI/tree/main) | `GIMM-VFI/pretrained_ckpt/gimmvfi_r_arb_lpips.pt` and `GIMM-VFI/pretrained_ckpt/raft-things.pth` |
| FILM (Style) | [Pretrained model folder](https://drive.google.com/drive/folders/1q8110-qp225asX3DQvZnfLfJPkCHmDpy) | `FILM/pretrained_models/film_net/Style/saved_model/` |

Keep the included inference adapters when adding weights. GIMM-VFI and FILM require separate environments because of their backend dependencies; see the [GIMM-VFI installation guide](https://github.com/GSeanCDAT/GIMM-VFI#install) and [FILM installation guide](https://github.com/google-research/frame-interpolation#installation). Specify their Python executable through `backend_python` when configuring the selected backend.

## Quick start

The following example reconstructs one intermediate section from **two observed, spatially aligned** `.h5ad` files:

```python
import vivid
from IPython.display import display

inputs = vivid.read_inputs(
    ["/path/to/section_6.h5ad", "/path/to/section_8.h5ad"],
    slice_names=["6", "8"],
    celltype_key="cell_type",
)
display(vivid.plot_inputs(inputs))

# If needed: align observed sections before reconstruction (single-omics only).
# inputs = vivid.align_inputs(inputs)

run = vivid.configure(
    inputs,
    dataset_name="my_dataset",
    target_number=1,
    resolution="singlecell",     # "singlecell" or "bin"
    generation_z=False,
    left_omics="RNA",
    right_omics="RNA",
    encoding="gnn",             # "gnn", "pca" or "spherical"
    interpolation="RIFE",       # "RIFE", "GIMM-VFI" or "FILM"
    optimization_solver="gurobi",  # Or "scipy" (no Gurobi license)
    save_root="save",
    save_intermediates=False,
)

vivid.check_backend(run)
image_sizes = vivid.select_image_size(run)
vivid.encode(run)
display(vivid.plot_encoding(run, pair_index=0))
vivid.interpolate(run)
display(vivid.plot_interpolation(run, pair_index=0))
predictions = vivid.reconstruct(run)
display(vivid.plot_predictions(run))
paths = vivid.save_results(run)
```

`target_number` specifies the number of reconstructed sections **per adjacent observed pair**. Results are saved as `.h5ad` files under the configured `save_root`. For cross-omics endpoints, provide the corresponding modality names and a shared annotation scheme; all input endpoints must be spatially aligned beforehand.

## Tutorial

See the step-by-step [VIVID tutorial](notebooks/VIVID_tutorial.ipynb) for data loading, optional alignment, parameter configuration, image encoding, interpolation, hierarchical reconstruction, visualization and output files. Edit the input paths in the notebook to use your own data.

For implementation details and checks, see [Migration notes](docs/MIGRATION.md), [Validation](docs/VALIDATION.md) and [Third-party notices](THIRD_PARTY_NOTICES.md).

## Contact

Please do not hesitate to contact **Mr. Bo-Han Si** ([sibohan@mails.ccnu.edu.cn](mailto:sibohan@mails.ccnu.edu.cn)) or **Dr. Xiao-Fei Zhang** ([zhangxf@ccnu.edu.cn](mailto:zhangxf@ccnu.edu.cn)) to seek any clarifications regarding any contents or operation of the archive.
