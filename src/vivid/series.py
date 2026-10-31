"""Adjacent-pair orchestration without changes to the numerical kernels."""
from collections.abc import Mapping
import contextlib
import io
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SeriesSession:
    """Ordered observed slices and isolated sessions for consecutive pairs.

    target_number is the count per adjacent interval. Numerical models are fitted
    independently for each interval; all final predictions share one output folder.
    """
    inputs: dict
    pairs: list
    output_dir: Path
    config: dict
    stage: str = 'configured'
    results: dict = field(default_factory=dict)
    saved_paths: list = field(default_factory=list)
    _closed: bool = False

    @property
    def slice_names(self):
        return tuple(self.inputs)

    @property
    def save_intermediates(self):
        return self.pairs[0].save_intermediates

    def close(self):
        for pair in self.pairs:
            pair.close()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def run_step(self, name):
        from . import api
        stages = {
            'select_image_size': ('configured', 'sized'),
            'encode': ('sized', 'encoded'),
            'interpolate': ('encoded', 'interpolated'),
            'reconstruct': ('interpolated', 'reconstructed'),
            'save_results': ('reconstructed', 'saved'),
        }
        if self._closed:
            raise RuntimeError('This series is closed. Create a new session.')
        before, after = stages[name]
        if self.stage != before:
            raise RuntimeError(f'Expected stage {before!r}, current stage is {self.stage!r}.')
        # Preflight every destination before writing any new pair.
        if name == 'save_results':
            for pair in self.pairs:
                if pair.stage != 'saved':
                    for key in pair.results:
                        path = self.output_dir / f'{key}.h5ad'
                        if path.exists():
                            raise FileExistsError(path)
        sizes = {}
        for i, pair in enumerate(self.pairs):
            print(f'[{i+1}/{len(self.pairs)}] {pair.left_id} / {pair.right_id}: {name}')
            # Allow resuming an interrupted stage without rerunning completed pairs.
            if pair.stage != after:
                getattr(api, name)(pair)
            if name == 'select_image_size':
                sizes[(pair.left_id, pair.right_id)] = pair.config['target_H']
            if name == 'reconstruct':
                for obj in pair.results.values():
                    obj.uns['vivid']['input_slice_names'] = list(self.slice_names)
                    obj.uns['vivid']['pair_index'] = i+1
                    obj.uns['vivid']['z_coordinate_scope'] = 'adjacent_pair'
        self.stage = after
        if name == 'reconstruct':
            self.results = {key: obj for pair in self.pairs for key, obj in pair.results.items()}
            expected = (len(self.inputs)-1)*self.config['target_number']
            if len(self.results) != expected:
                raise RuntimeError(f'Expected {expected} total targets, got {len(self.results)}.')
            return self.results
        if name == 'save_results':
            self.saved_paths = [path for pair in self.pairs for path in pair.saved_paths]
            self._closed = True
            return self.saved_paths
        return sizes if name == 'select_image_size' else self


def configure_series(inputs, *, slice_omics=None, **kwargs):
    """Use mapping insertion order as physical section order, without sorting names."""
    from .api import configure, _safe_id, _validate
    if not isinstance(inputs, Mapping) or len(inputs) < 2:
        raise ValueError('Provide an ordered mapping containing at least two observed slices.')
    names = [_safe_id(name) for name in inputs]
    if len(names) != len(set(names)):
        raise ValueError('Slice names must be unique after conversion to strings.')
    objects = list(inputs.values())
    for obj in objects:
        _validate(obj, kwargs.get('celltype_key', 'cell_type'), kwargs.get('spatial_key', 'spatial'))
    if slice_omics is None:
        left_omics, right_omics = kwargs.get('left_omics', 'RNA'), kwargs.get('right_omics', 'RNA')
        if left_omics != right_omics:
            raise ValueError('For a series with different omics, specify slice_omics for every input.')
        omics = [left_omics]*len(names)
    elif isinstance(slice_omics, Mapping):
        if set(slice_omics) != set(names):
            raise ValueError('slice_omics keys must match the manually supplied slice names.')
        omics = [slice_omics[name] for name in names]
    else:
        if isinstance(slice_omics, str) or len(slice_omics) != len(names):
            raise ValueError('slice_omics must contain one modality per observed slice.')
        omics = list(slice_omics)
    target = kwargs.get('target_number', 1)
    if isinstance(target, bool) or not isinstance(target, int) or target < 1:
        raise ValueError('target_number must be a positive integer per adjacent pair.')
    keys = [f'{a}_{b}_{j}' for a,b in zip(names,names[1:]) for j in range(1,target+1)]
    if len(keys) != len(set(keys)):
        raise ValueError('These slice names produce ambiguous output filenames; choose distinct names.')
    dataset = _safe_id(kwargs.get('dataset_name', 'my_dataset'))
    save_root = Path(kwargs.pop('save_root', 'save')).expanduser().resolve()
    from ._utils import make_run_directory
    out = make_run_directory(save_root, dataset, kwargs.get("encoding", "gnn"),
                             kwargs.get("interpolation", "RIFE"))
    pairs = []
    try:
        for i in range(len(names)-1):
            opts = dict(kwargs)
            opts.update(left_id=names[i], right_id=names[i+1], left_omics=omics[i], right_omics=omics[i+1])
            with contextlib.redirect_stdout(io.StringIO()):
                pair = configure(objects[i], objects[i+1], save_root=out, **opts)
            pairs.append(pair)
            old_out = pair.output_dir
            if pair.save_intermediates:
                work = out/'intermediates'/f'pair_{i+1:04d}'
                work.parent.mkdir(exist_ok=True)
                pair.work_dir.rename(work)
                pair._state['save_dir'] = str(work)
            pair.output_dir = out
            old_out.rmdir()
    except Exception:
        for pair in pairs:
            pair.close()
        # Preserve retained diagnostics on failure; remove only an empty directory.
        if not any(out.iterdir()):
            out.rmdir()
        raise
    cfg = dict(pairs[0].config)
    cfg.update(n_input_slices=len(names), slice_names=names, slice_omics=omics)
    result = SeriesSession(dict(zip(names, objects)), pairs, out, cfg)
    print(f'Save directory: {out}\nObserved slices: {names}\n'
          f'Adjacent pairs: {len(pairs)} | Targets per pair: {target} | Total targets: {len(pairs)*target}')
    return result
