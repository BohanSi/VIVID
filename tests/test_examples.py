from pathlib import Path
import re
import numpy as np
import anndata as ad
import pandas as pd
import pytest
import vivid


def test_mb1_loader_matches_original_preprocessing(tmp_path):
    import scanpy as sc
    def make(x,features):
        a=ad.AnnData(np.asarray(x,dtype='float32'),obs=pd.DataFrame({'cell_type':['A','B']},index=['c1','c2']))
        a.var_names=features
        a.obsm['spatial']=np.array([[0.,0.],[1.,1.]])
        return a
    l=make([[2,4,6],[4,3,5]],['g0','g1','g2'])
    r=make([[8,6,2],[6,2,1]],['g1','g2','g3'])
    l.write_h5ad(tmp_path/'slice_6_corrected_spatial.h5ad')
    r.write_h5ad(tmp_path/'slice_8_corrected_spatial.h5ad')
    expected=ad.concat([l,r],axis=0,index_unique='-')
    sc.pp.normalize_total(expected,target_sum=1e4)
    sc.pp.log1p(expected)
    left,right=vivid.read_mb1(tmp_path)
    np.testing.assert_allclose(left.X,expected.X[:2])
    np.testing.assert_allclose(right.X,expected.X[2:])
    np.testing.assert_array_equal(left.obsm['spatial'],l.obsm['spatial'])
    unscaled,_=vivid.read_mb1(tmp_path,normalize=False)
    np.testing.assert_array_equal(unscaled.X,l[:,['g1','g2']].X)
    assert not (tmp_path/'slice_7_corrected_spatial.h5ad').exists()


def test_repository_explanations_are_english():
    root=Path(__file__).resolve().parents[1]
    for p in root.rglob('*'):
        if p.suffix in {'.py','.md','.ipynb','.yml','.toml'} and not {'build', '.ipynb_checkpoints', '.pytest_cache'}.intersection(p.parts):
            assert not re.search(r'[\u4e00-\u9fff]',p.read_text()),p
