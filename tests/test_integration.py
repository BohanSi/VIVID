"""Opt-in real-backend smoke test, never downloads weights.

VIVID_INTEGRATION=flow runs through flow/type proportions without Gurobi.
VIVID_INTEGRATION=full additionally reconstructs and saves real predictions.
VIVID_MODEL_ROOT may point to an existing model checkout containing weights.
"""
import os
from pathlib import Path
import pytest

MODE = os.environ.get('VIVID_INTEGRATION', '')

@pytest.mark.skipif(MODE not in {'flow', 'full'}, reason='Opt-in: requires RIFE weights; full uses license-free SciPy/HiGHS')
def test_real_rife_workflow(tmp_path):
    import numpy as np
    import pandas as pd
    import anndata as ad
    import vivid
    from vivid.api import _run
    rng = np.random.default_rng(7)
    def endpoint():
        a = ad.AnnData(rng.poisson(3,(48,5)).astype('float32'),
            obs=pd.DataFrame({'cell_type':pd.Categorical(['A','B','C']*16)},
                             index=[f'c{i}' for i in range(48)]))
        a.obsm['spatial'] = rng.uniform(0,100,(48,2))
        return a
    model_root = os.environ.get('VIVID_MODEL_ROOT', str(Path(__file__).resolve().parents[1]))
    with vivid.configure(endpoint(), endpoint(), model_root=model_root, save_root=tmp_path,
                         left_id='2', right_id='4', optimization_solver='scipy', advanced={'num_epochs':2}) as run:
        vivid.check_backend(run)
        vivid.select_image_size(run)
        vivid.encode(run)
        vivid.interpolate(run)
        if MODE == 'flow':
            _run(run, 'detect', 'proportions')
            table = run._state['fused_celltype_prop_df']
            assert len(table) > 0
            assert np.isfinite(table.filter(like='prop__').to_numpy()).all()
        else:
            results = vivid.reconstruct(run)
            assert len(results) == 1
            assert next(iter(results.values())).n_obs > 0
            paths = vivid.save_results(run)
            assert paths[0].name == '2_4_1.h5ad'
            assert len(list(run.output_dir.iterdir())) == 1
