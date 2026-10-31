
<p align="center">
  <img src="./VIVID.png" alt="VIVID logo" width="300">
</p>

# Reconstructing missing spatial-omics sections with video-derived correspondence priors

**VIVID** reconstructs unmeasured spatial-omics sections between two aligned observed endpoints by transferring correspondence knowledge learned from natural videos. It encodes spatial annotations, local tissue context and occupancy into images for a frozen pretrained interpolation backbone, then uses inferred target-to-endpoint correspondence fields to recover spatial locations, annotations and molecular profiles. VIVID supports reconstruction at multiple depths in cell- and bin-resolved spatial omics, including transcriptomic and proteomic measurements.

<p align="center">
  <img src="./Figure1.png" alt="Figure 1. Overview of the VIVID reconstruction framework" width="1000">
</p>

*Figure 1. Overview of VIVID.*

## Requirements

- Linux and Python 3.10 are recommended. Use compatible PyTorch (2.0–2.5) and torchvision (0.15–0.20) installations; an NVIDIA GPU is recommended for interpolation.
- Input files in AnnData (`.h5ad`) format, containing spatial coordinates in `obsm["spatial"]`, annotation labels in `obs` and molecular features in `X`. Observed sections must share an aligned coordinate system and a common annotation scheme. Same-modality inputs must have matching molecular features.
- Pretrained weights for the selected interpolation backbone (RIFE by default).
- Gurobi Optimizer with a valid license for integer optimization.

> **Note:** Gurobi (`gurobipy`) is commercial optimization software required by VIVID. Students, faculty and staff at eligible academic institutions can obtain a free, full-featured academic license for non-commercial research and educational use. For details, see the [Gurobi Academic Licensing Program](https://www.gurobi.com/academics). Installing `gurobipy` does not automatically activate a full academic license.

## Installation

Clone the repository and install VIVID from its root directory:

```bash
git clone https://github.com/BohanSi/VIVID.git
cd VIVID

conda create -n vivid python=3.10 -y
conda activate vivid

# Install compatible PyTorch and torchvision versions first if needed.
python -m pip install -e ".[tutorial,gurobi]"

# Register the Jupyter kernel.
python -m ipykernel install --user --name vivid --display-name "Python (VIVID)"
```

Alternatively, create the main environment using `environments/vivid.yml`. For GPU-specific PyTorch installations, follow the [official PyTorch instructions](https://pytorch.org/get-started/locally/).

### Gurobi license configuration

Gurobi must be installed and licensed in the Python environment selected by Jupyter before running the tutorial. Use `optimization_solver="gurobi"` for the documented workflow. VIVID requires a properly configured Gurobi license. Academic users can request a free license through the [Gurobi User Portal](https://portal.gurobi.com/).

For an Academic Named-User License, follow the instructions provided by Gurobi to activate the license on your machine. Activation generally requires connecting to an academic network. Other supported license types can be configured according to the official documentation.

After activation, verify that Gurobi is available in the VIVID environment:

```bash
conda activate vivid

python - <<'PY'
import gurobipy as gp

with gp.Model("license_test") as model:
    x = model.addVar(vtype=gp.GRB.INTEGER, lb=0)
    model.addConstr(x >= 1)
    model.setObjective(x, gp.GRB.MINIMIZE)
    model.optimize()
    assert model.Status == gp.GRB.OPTIMAL

print("Gurobi license check passed.")
PY
```

The test confirms basic solver operation. Ensure that the configured license supports the optimization problem sizes required by your datasets.

## Pretrained interpolation models

Download the selected pretrained model weights and place them at the corresponding locations within the repository. **Pretrained weights are not included.**

| Backbone | Weights | Expected location |
| --- | --- | --- |
| **RIFE (default)** | [HD weights](https://drive.google.com/file/d/1APIzVeI-4ZZCEuIRE1m6WYfSCaOsi_7_/view) | `RIFE/train_log/flownet.pkl` |
| GIMM-VFI (R-P) | [Official model repository](https://huggingface.co/GSean/GIMM-VFI/tree/main) | `GIMM-VFI/pretrained_ckpt/gimmvfi_r_arb_lpips.pt` and `GIMM-VFI/pretrained_ckpt/raft-things.pth` |
| FILM (Style) | [Pretrained model folder](https://drive.google.com/drive/folders/1q8110-qp225asX3DQvZnfLfJPkCHmDpy) | `FILM/pretrained_models/film_net/Style/saved_model/` |

Keep the included inference adapters when adding pretrained weights.

RIFE uses the main VIVID environment by default. GIMM-VFI and FILM require separate environments due to their backend dependencies. Refer to the [GIMM-VFI installation guide](https://github.com/GSeanCDAT/GIMM-VFI#install) and [FILM installation guide](https://github.com/google-research/frame-interpolation#installation). Specify the corresponding Python executable through `backend_python` when using these backbones.

## Tutorial

A step-by-step Jupyter notebook is provided to demonstrate the complete VIVID reconstruction workflow:

**[VIVID Tutorial](notebooks/VIVID_tutorial.ipynb)**

The tutorial covers data loading, optional spatial alignment, parameter configuration, spatial-omics image representation, correspondence inference, hierarchical reconstruction, visualization and result saving.

Users can modify the input paths and configuration parameters to apply VIVID to their own spatial-omics datasets.

See [Third-party notices](THIRD_PARTY_NOTICES.md) for model-source attribution and license conditions.

## Usage notes

- Supply observed h5ad paths and unique manual slice names in physical order. VIVID predicts only between adjacent observed sections: N inputs and K targets per interval produce (N−1)×K targets. All observed sections must share an aligned x/y coordinate frame.
- Input loading does not normalize expression. The tutorial includes explicit MB1 common-feature selection, normalization to 10,000 and log1p; skip that preprocessing when inputs already use the intended scale. Use assay-appropriate preprocessing for other datasets.
- Run directories follow `{dataset_name}_{ENCODING}_{interpolation_model}_{MonDD_HH_MM}`, for example `MB1_GNN_RIFE_Oct14_20_32`. The absolute path is printed. Existing same-minute directories are not overwritten. Target files are named `{left_name}_{right_name}_{target_index}.h5ad`.
- By default, only final h5ad files are retained after successful saving. Enable `save_intermediates=True` to retain process artifacts under `intermediates/`. Call `run.close()` after debugging an interrupted run to clean temporary files.
- Encoders and canvas sizes are selected independently for adjacent intervals. Optional continuous z coordinates retain the original pair-local convention; they do not establish a globally registered z stack. Same-modality fused expression is in `X`; different-modality predictions are stored separately in `obsm["pred_<modality>"]`, with feature names in `uns["left_feature_names"]` and `uns["right_feature_names"]`.




## Contact

Please do not hesitate to contact **Mr. Bo-Han Si** ([sibohan@mails.ccnu.edu.cn](mailto:sibohan@mails.ccnu.edu.cn)) or **Dr. Xiao-Fei Zhang** ([zhangxf@ccnu.edu.cn](mailto:zhangxf@ccnu.edu.cn)) to seek any clarifications regarding any contents or operation of the archive.
