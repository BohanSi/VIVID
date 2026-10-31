"""Small notebook-friendly visualizations; never save files implicitly."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image


CELLTYPE_COLORS = (
    "#f7e425", "#f99a3e", "#db5c68", "#ff6347", "#b83289", "#fa9bb4",
    "#ef599e", "#ff1493", "#2a0593", "#0000ff", "#0cf3ff", "#40bfff",
    "#738cff", "#a659ff", "#f30cff", "#5dae66", "#0092b6", "#008000",
    "#e7d150", "#878478", "#48526c", "#dac6a1", "#bb7967", "#f6c85f",
    "#f28e2b", "#e15759", "#c44e52", "#b07aa1", "#9c755f", "#8c564b",
    "#d37295", "#af7aa1", "#9467bd", "#6f4e7c", "#4e79a7", "#1f77b4",
    "#17becf", "#00a6a6", "#2ca02c", "#59a14f", "#8cd17d", "#b6992d",
    "#79706e", "#bab0ac", "#7f7f7f", "#393b79", "#637939", "#843c39",
)


def _finish(fig):
    # A returned open figure is rendered both as a cell result and by the inline
    # backend. Detach it from pyplot so the rich cell result displays it once.
    plt.close(fig)
    return fig


def plot_inputs(left, right=None, *, celltype_key="cell_type", spatial_key="spatial",
                names=None, point_size=3, invert_y=False):
    """Show named observed slices (or two endpoints) with a shared type palette."""
    if right is None:
        if not hasattr(left, "items") or len(left) < 2:
            raise ValueError("Provide at least two named input slices.")
        objects = list(left.values())
        names = list(left) if names is None else names
    else:
        objects = [left, right]
        names = ("Left endpoint", "Right endpoint") if names is None else names
    if len(names) != len(objects):
        raise ValueError("Provide one plot title per input slice.")
    return _spatial(objects, names, celltype_key, spatial_key, point_size, invert_y)


def _spatial(objects, names, celltype_key, spatial_key, point_size, invert_y):
    types = sorted({str(t) for a in objects for t in a.obs[celltype_key]},
                   key=lambda name: (name.casefold(), name))
    if len(types) > len(CELLTYPE_COLORS):
        raise ValueError(f"The supplied palette supports {len(CELLTYPE_COLORS)} types; found {len(types)}.")
    colors = dict(zip(types, CELLTYPE_COLORS[:len(types)]))
    fig, axes = plt.subplots(1, len(objects), figsize=(5 * len(objects), 5), squeeze=False, sharex=True, sharey=True)
    for ax, obj, title in zip(axes.flat, objects, names):
        xy = np.asarray(obj.obsm[spatial_key])
        labels = obj.obs[celltype_key].astype(str).to_numpy()
        for ct in types:
            mask = labels == ct
            ax.scatter(xy[mask, 0], xy[mask, 1], s=point_size, color=colors[ct], linewidths=0)
        ax.set(title=title, aspect="equal", xticks=[], yticks=[])
    if invert_y:
        axes.flat[0].invert_yaxis()
    handles = [Line2D([], [], marker="o", linestyle="", color=colors[t], label=t, markersize=4) for t in types]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1, .5), frameon=False)
    fig.tight_layout()
    return _finish(fig)


def plot_encoding(session, *, pair_index=0):
    """Show encoded endpoints only, oriented to match the spatial scatter plots."""
    from .series import SeriesSession
    if isinstance(session, SeriesSession):
        return plot_encoding(session.pairs[pair_index])
    if session._closed or session.stage not in {"encoded", "interpolated", "reconstructed"}:
        raise RuntimeError("Call encode before plotting and plot before save_results closes scratch files.")
    entries = list(session._state["summary"].items())
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, (_, record), name in zip(axes, entries, (session.left_id, session.right_id)):
        with Image.open(record["png_path"]) as im:
            ax.imshow(np.asarray(im).copy(), origin="upper")
        ax.invert_yaxis()
        ax.set_title(name)
        ax.axis("off")
    fig.tight_layout()
    return _finish(fig)


def plot_interpolation(session, *, max_targets=5, pair_index=0):
    """Show inverted-y target images with one single-line title per panel."""
    from .series import SeriesSession
    if isinstance(session, SeriesSession):
        return plot_interpolation(session.pairs[pair_index], max_targets=max_targets)
    if session._closed or session.stage not in {"interpolated", "reconstructed"}:
        raise RuntimeError("Call interpolate first; plot before saving closes working files.")
    if max_targets < 1:
        raise ValueError("max_targets must be positive.")
    s = session._state
    model = s["interpolation_model_names"][0]
    root = session.work_dir / (model + "_result")
    selected = set(int(i) for i in s["target_img"])
    paths = []
    import re
    for p in root.rglob("img*.png"):
        match = re.fullmatch(r"img(\d+)\.png", p.name)
        if match and int(match[1]) in selected:
            paths.append((int(match[1]), p))
    paths.sort()
    if not paths:
        raise FileNotFoundError("Selected interpolated RGB frames were not found.")
    indices = np.unique(np.linspace(0, len(paths)-1, min(max_targets, len(paths)), dtype=int))
    fig, axes = plt.subplots(1, len(indices), figsize=(5*len(indices), 5), squeeze=False)
    for ax, j in zip(axes.flat, indices):
        _, path = paths[j]
        with Image.open(path) as im:
            ax.imshow(np.asarray(im).copy(), origin="upper")
        ax.invert_yaxis()
        ax.set_title(f"{session.left_id} / {session.right_id} | Target {j+1}")
        ax.axis("off")
    fig.tight_layout()
    return _finish(fig)


def plot_predictions(session, *, max_targets=4, point_size=3, invert_y=False):
    """Compare endpoints and reconstructed targets with a consistent type palette."""
    from .series import SeriesSession
    if isinstance(session, SeriesSession):
        if not session.results:
            raise RuntimeError("Call reconstruct first.")
        if max_targets < 1:
            raise ValueError("max_targets must be positive.")
        all_keys = list(session.results)
        selected = {all_keys[i] for i in np.unique(np.linspace(
            0, len(all_keys)-1, min(max_targets, len(all_keys)), dtype=int))}
        names = list(session.inputs)
        objects, titles = [], []
        observed = [p.endpoints[0] for p in session.pairs] + [session.pairs[-1].endpoints[1]]
        for i, name in enumerate(names):
            objects.append(observed[i])
            titles.append(name)
            if i < len(session.pairs):
                for key, obj in session.pairs[i].results.items():
                    if key in selected:
                        objects.append(obj)
                        titles.append(key)
        return _spatial(objects, titles, session.pairs[0]._state["celltype_col"],
                        "spatial", point_size, invert_y)
    if not session.results:
        raise RuntimeError("Call reconstruct first.")
    entries = list(session.results.items())
    indices = np.unique(np.linspace(0, len(entries)-1, min(max_targets, len(entries)), dtype=int))
    chosen = [entries[i] for i in indices]
    left, right = session.endpoints
    objects = [left] + [a for _, a in chosen] + [right]
    names = [f"Left {session.left_id}"] + [key for key, _ in chosen] + [f"Right {session.right_id}"]
    return _spatial(objects, names, session._state["celltype_col"], "spatial", point_size, invert_y)


def plot_feature(adata, feature, *, omics=None, point_size=4):
    """Plot a gene/protein from X, or from pred_<omics> for multimodal outputs."""
    from scipy import sparse
    if omics is None:
        values = adata[:, [feature]].X
    else:
        side = "left" if adata.uns["vivid"]["parameters"]["left_omics"] == omics else "right"
        names = list(adata.uns[f"{side}_feature_names"])
        values = adata.obsm[f"pred_{omics}"][:, names.index(feature)]
    values = values.toarray() if sparse.issparse(values) else np.asarray(values)
    xy = np.asarray(adata.obsm["spatial"])
    fig, ax = plt.subplots(figsize=(5, 5), facecolor="black")
    ax.set_facecolor("black")
    points = ax.scatter(xy[:, 0], xy[:, 1], c=values.ravel(), s=point_size, cmap="viridis", linewidths=0)
    ax.set(title=feature, aspect="equal", xticks=[], yticks=[])
    colorbar = fig.colorbar(points, ax=ax, label="Predicted expression",
                            fraction=0.035, pad=0.025, shrink=0.65, aspect=25)
    ax.title.set_color("white")
    colorbar.ax.tick_params(labelsize=8)
    colorbar.set_label("Predicted expression", fontsize=9, color="white")
    for panel in (ax, colorbar.ax):
        panel.set_facecolor("black")
        panel.tick_params(axis="both", colors="white")
        panel.xaxis.label.set_color("white")
        panel.yaxis.label.set_color("white")
        panel.xaxis.get_offset_text().set_color("white")
        panel.yaxis.get_offset_text().set_color("white")
        for spine in panel.spines.values():
            spine.set_edgecolor("white")
    colorbar.outline.set_edgecolor("white")
    fig.tight_layout()
    return _finish(fig)
