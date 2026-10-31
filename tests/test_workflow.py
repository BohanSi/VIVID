import ast
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import anndata as ad
import pytest
import vivid

ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture
def pair():
    rng = np.random.default_rng(7)
    xy = rng.uniform(0, 100, (30, 2))
    x = rng.poisson(3, (30, 4)).astype('float32')
    def make(coords):
        a = ad.AnnData(x.copy(), obs=pd.DataFrame({'cell_type':pd.Categorical(['A','B','C']*10)}, index=[f'c{i}' for i in range(30)]))
        a.obsm['spatial'] = coords.copy()
        return a
    return make(xy), make(xy + [7, -3])

@pytest.mark.parametrize('resolution', ['singlecell', 'bin'])
def test_size_and_state(pair, tmp_path, resolution):
    left,right=pair
    with vivid.configure(left,right,resolution=resolution,save_root=tmp_path) as run:
        with pytest.raises(RuntimeError): vivid.encode(run)
        size=vivid.select_image_size(run)
        assert size in (512,1024,2048)
        assert run.config['target_H']==run._state['target_HW']==size
        assert run.stage=='sized'
        assert list(run.output_dir.iterdir())==[]
    assert not run.work_dir.exists()


def test_alignment(pair):
    left,right=pair
    original=right.obsm['spatial'].copy()
    a,b=vivid.align_inputs(left,right,verbose=False)
    np.testing.assert_allclose(a.obsm['spatial'],b.obsm['spatial'],atol=1e-7)
    np.testing.assert_array_equal(right.obsm['spatial'],original)
    assert b.uns['alignment']['role']=='model_input'
    with pytest.raises(ValueError): vivid.align_inputs(left,right,same_omics=False)

@pytest.mark.parametrize('keep',[False,True])
def test_h5ad_names_cleanup_roundtrip(pair,tmp_path,keep):
    run=vivid.configure(*pair,left_id='2',right_id='4',save_root=tmp_path,save_intermediates=keep)
    work=run.work_dir
    (work/'scratch.txt').write_text('temporary')
    result=pair[0].copy()
    result.uns['summary']=[{'n':3}]
    run.results={'2_4_1':result}
    run.stage='reconstructed'
    paths=vivid.save_results(run)
    assert [p.name for p in paths]==['2_4_1.h5ad']
    loaded=ad.read_h5ad(paths[0])
    np.testing.assert_array_equal(loaded.X,result.X)
    assert loaded.uns['summary']=='[{"n": 3}]'
    assert work.exists()==keep
    assert set(p.name for p in run.output_dir.iterdir())==({'2_4_1.h5ad','intermediates'} if keep else {'2_4_1.h5ad'})


def test_input_and_backend_errors(pair,tmp_path):
    a,b=pair
    with pytest.raises(ValueError): vivid.configure(a,b,target_number=0,save_root=tmp_path)
    with pytest.raises(ValueError): vivid.configure(a,b,left_id='../2',save_root=tmp_path)
    with vivid.configure(a,b,save_root=tmp_path,model_root=tmp_path) as run:
        with pytest.raises(FileNotFoundError,match='Missing backend'): vivid.check_backend(run)
    a.X[0,0]=np.nan
    with pytest.raises(ValueError,match='NaN'): vivid.configure(a,b,save_root=tmp_path)


def test_plot_pair(pair):
    import matplotlib.pyplot as plt
    fig=vivid.plot_inputs(*pair)
    assert len(fig.axes)==2
    assert fig.axes[0].get_xlim()==fig.axes[1].get_xlim()
    plt.close(fig)


def test_tutorial():
    import nbformat
    nb=nbformat.read(ROOT/'notebooks/VIVID_tutorial.ipynb',as_version=4)
    nbformat.validate(nb)
    for cell in nb.cells:
        if cell.cell_type=='code': ast.parse(cell.source)


def test_gurobi_default_and_readable_output_timestamp(pair,tmp_path,capsys):
    import re
    from vivid import configure
    with configure(*pair,dataset_name='MB1',save_root=tmp_path) as run:
        assert run.config['optimization_solver']=='gurobi'
        assert re.fullmatch(r'MB1_GNN_RIFE_(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\d{2}_\d{2}_\d{2}',run.output_dir.name)
        assert f'Save directory: {run.output_dir}' in capsys.readouterr().out
