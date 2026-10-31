"""Explicit example loaders; private datasets are never bundled or downloaded."""
from pathlib import Path


def read_mb1(data_dir, *, normalize=True):
    """Load the original aligned MB1 endpoints 6 and 8.

    Reproduce the original inner feature join, normalize_total(target_sum=1e4),
    and log1p. Set normalize=False only if inputs already have that expression
    scale. This function does not load section 7 or perform spatial alignment.
    """
    from .api import read_inputs
    root = Path(data_dir).expanduser()
    left, right = read_inputs(root/'slice_6_corrected_spatial.h5ad',
                              root/'slice_8_corrected_spatial.h5ad', celltype_key='cell_type')
    common = left.var_names.intersection(right.var_names)
    if not len(common):
        raise ValueError('MB1 endpoints have no common features.')
    left, right = left[:,common].copy(), right[:,common].copy()
    if normalize:
        import scanpy as sc
        for obj in (left, right):
            sc.pp.normalize_total(obj, target_sum=1e4)
            sc.pp.log1p(obj)
    print(f'MB1: sections 6 and 8 | {len(common):,} shared features | normalize={normalize}')
    return left, right


def preprocess_inputs(inputs, *, target_sum=1e4, log1p=True, common_features=True):
    """Return copies with explicit MB1-style preprocessing for same-omics inputs.

    Optionally retain the common feature intersection before normalization,
    reproducing the original MB1 inner join. For other data, choose preprocessing
    appropriate to the assay externally; never log-transform twice.
    """
    if len(inputs) < 2:
        raise ValueError('At least two named observed slices are required.')
    objects = {name: obj.copy() for name, obj in inputs.items()}
    if common_features:
        common = next(iter(objects.values())).var_names
        for obj in objects.values():
            common = common.intersection(obj.var_names)
        if not len(common):
            raise ValueError('The inputs have no common features.')
        objects = {name: obj[:, common].copy() for name, obj in objects.items()}
    import scanpy as sc
    for obj in objects.values():
        if target_sum is not None:
            sc.pp.normalize_total(obj, target_sum=target_sum)
        if log1p:
            sc.pp.log1p(obj)
    return objects
