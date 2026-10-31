"""Seed and configuration helpers retained from model/utils.py."""
import os, random, json
from pathlib import Path
import numpy as np
import torch

def save_config(config, save_dir, filename="config.json"):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    config_path = save_dir / filename

    config_to_save = {}

    for k, v in config.items():
        if isinstance(v, Path):
            config_to_save[k] = str(v)
        else:
            config_to_save[k] = v

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_to_save, f, indent=4, ensure_ascii=False)

    print(f"Config saved to: {config_path}")
    return config_path

def set_global_seed(seed=1234): #, deterministic=True):
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    '''if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        try:
            torch.use_deterministic_algorithms(True)
        except Exception as e:
            print(f"[Warning] torch.use_deterministic_algorithms failed: {e}")
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    #'''


def make_run_directory(save_root, dataset_name, encoding, interpolation):
    """Create dataset_ENCODING_BACKEND_MonDD_HH_MM without overwriting runs."""
    from datetime import datetime
    encoding = str(encoding).upper()
    backend = {"RIFE": "RIFE", "GIMM": "GIMM-VFI", "GIMM-VFI": "GIMM-VFI",
               "FILM": "FILM"}.get(str(interpolation).upper())
    if encoding not in {"GNN", "PCA", "SPHERICAL"} or backend is None:
        raise ValueError("Use GNN/PCA/spherical encoding and RIFE/GIMM-VFI/FILM interpolation.")
    now = datetime.now()
    months = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
    stamp = months[now.month-1] + now.strftime("%d_%H_%M")
    path = Path(save_root).expanduser().resolve() / f"{dataset_name}_{encoding}_{backend}_{stamp}"
    try:
        path.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise FileExistsError(
            f"Save directory already exists: {path}. The requested name has minute precision; "
            "choose a different dataset_name or start in a later minute. Existing results were not overwritten."
        ) from None
    return path
