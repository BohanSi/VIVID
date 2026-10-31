"""Named paths, physical ordering, adjacent targets, and output isolation."""
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
import pytest
import vivid


@pytest.fixture
def slices():
    rng=np.random.default_rng(8)
    result={}
    for name in ['section_z','section_a','section_m']:
        a=ad.AnnData(rng.poisson(3,(48,5)).astype('float32'),
            obs=pd.DataFrame({'cell_type':pd.Categorical(['A','B','C']*16)},index=[f'c{i}' for i in range(48)]))
        a.obsm['spatial']=rng.uniform(0,100,(48,2))
        result[name]=a
    return result


def test_named_paths_preserve_order(slices,tmp_path):
    paths=[]
    for i,obj in enumerate(slices.values()):
        path=tmp_path/f'unrelated_filename_{i}.h5ad'
        obj.write_h5ad(path);paths.append(path)
    loaded=vivid.read_inputs(paths,slice_names=list(slices))
    assert list(loaded)==list(slices)
    for name in slices:
        np.testing.assert_array_equal(loaded[name].X,slices[name].X)
    with pytest.raises(ValueError):vivid.read_inputs(paths)
    with pytest.raises(ValueError):vivid.read_inputs(paths,slice_names=['a','a','b'])
    with pytest.raises(ValueError):vivid.read_inputs(paths,slice_names=['a','b'])


@pytest.mark.parametrize('keep',[False,True])
@pytest.mark.parametrize('targets',[1,2])
def test_adjacent_configuration_and_save(slices,tmp_path,keep,targets):
    with vivid.configure(slices,target_number=targets,save_root=tmp_path,
                         save_intermediates=keep,slice_omics=['RNA']*3) as run:
        assert [(p.left_id,p.right_id) for p in run.pairs]==[
            ('section_z','section_a'),('section_a','section_m')]
        assert len(run.pairs)==2
        assert all(p.output_dir==run.output_dir for p in run.pairs)
        works=[p.work_dir for p in run.pairs]
        for p in run.pairs:
            for i in range(1,targets+1):
                key=f'{p.left_id}_{p.right_id}_{i}'
                p.results[key]=next(iter(slices.values())).copy()
            p.stage='reconstructed'
        run.results={k:a for p in run.pairs for k,a in p.results.items()}
        run.stage='reconstructed'
        paths=vivid.save_results(run)
        assert [p.name for p in paths]==[
            f'section_z_section_a_{i}.h5ad' for i in range(1,targets+1)]+[
            f'section_a_section_m_{i}.h5ad' for i in range(1,targets+1)]
        assert all(p.exists()==keep for p in works)
        assert len(list(run.output_dir.iterdir()))==2*targets+int(keep)
        if keep:
            assert len(list((run.output_dir/'intermediates').iterdir()))==2


def test_series_size_plot_and_input_isolation(slices,tmp_path):
    with vivid.configure(slices,save_root=tmp_path) as run:
        sizes=vivid.select_image_size(run)
        assert list(sizes)==[('section_z','section_a'),('section_a','section_m')]
        assert run.stage=='sized'
        assert all(p.stage=='sized' for p in run.pairs)
        fig=vivid.plot_inputs(slices)
        assert [a.get_title() for a in fig.axes]==list(slices)
        import matplotlib.pyplot as plt
        plt.close(fig)
        run.pairs[0].endpoints[1].X[:]=0
        assert np.any(run.pairs[1].endpoints[0].X)
        assert np.any(slices['section_a'].X)


def test_series_validation(slices,tmp_path):
    with pytest.raises(ValueError):vivid.configure(slices,save_root=tmp_path,target_number=0)
    with pytest.raises(ValueError):vivid.configure(slices,save_root=tmp_path,left_omics='RNA',right_omics='Protein')
    with pytest.raises(ValueError):vivid.configure(slices,save_root=tmp_path,slice_omics=['RNA'])
    with vivid.configure(slices,save_root=tmp_path,slice_omics=['RNA','Protein','RNA']) as run:
        assert not any(p.config['same_omics'] for p in run.pairs)


def test_resume_partial_series_stage(slices,tmp_path,monkeypatch):
    from vivid import api
    with vivid.configure(slices,save_root=tmp_path) as run:
        # First pair completed a stage before a later pair failed.
        run.pairs[0].stage='sized';run.pairs[0].config['target_H']=512
        original=api.select_image_size
        calls=[]
        def track(pair):
            calls.append(pair.left_id)
            return original(pair)
        monkeypatch.setattr(api,'select_image_size',track)
        run.run_step('select_image_size')
        assert calls==['section_a']


def test_align_named_observations_to_one_reference(slices):
    anchor=next(iter(slices.values()))
    inputs={name:anchor.copy() for name in slices}
    inputs['section_a'].obsm['spatial'] += [7,-3]
    inputs['section_m'].obsm['spatial'] += [-2,5]
    aligned=vivid.align_inputs(inputs,verbose=False)
    assert list(aligned)==list(inputs)
    for obj in aligned.values():
        np.testing.assert_allclose(obj.obsm['spatial'],aligned['section_z'].obsm['spatial'],atol=1e-7)
    np.testing.assert_allclose(inputs['section_a'].obsm['spatial']-anchor.obsm['spatial'],np.tile([7,-3],(anchor.n_obs,1)))


@pytest.mark.parametrize('targets,generation_z',[(1,False),(2,False),(1,True)])
def test_real_three_input_pipeline(slices,tmp_path,targets,generation_z):
    import os
    if os.environ.get('VIVID_INTEGRATION')!='full':
        pytest.skip('Opt-in: full RIFE weights and SciPy reconstruction')
    model_root=os.environ.get('VIVID_MODEL_ROOT',str(Path(__file__).resolve().parents[1]))
    import matplotlib.pyplot as plt
    paths=[]
    for i,obj in enumerate(slices.values()):
        path=tmp_path/f'observed_file_{i}.h5ad'
        obj.write_h5ad(path);paths.append(path)
    slices=vivid.read_inputs(paths,slice_names=list(slices))
    with vivid.configure(slices,save_root=tmp_path,model_root=model_root,target_number=targets,
                         generation_z=generation_z,optimization_solver='scipy',advanced={'num_epochs':2}) as run:
        vivid.check_backend(run)
        vivid.select_image_size(run)
        vivid.encode(run)
        fig=vivid.plot_encoding(run,pair_index=1)
        assert fig.axes[0].get_title()=='section_a'
        plt.close(fig)
        vivid.interpolate(run)
        plt.close(vivid.plot_interpolation(run,pair_index=1))
        results=vivid.reconstruct(run)
        expected=[f'{a}_{b}_{i}' for a,b in [('section_z','section_a'),('section_a','section_m')] for i in range(1,targets+1)]
        assert list(results)==expected
        for obj in results.values():
            assert obj.uns['vivid']['input_slice_names']==list(slices)
            assert obj.uns['vivid']['optimization_backends_used']==['scipy']*targets
            if generation_z:
                assert np.isfinite(obj.obs['z']).all()
                assert obj.uns['vivid']['z_coordinate_scope']=='adjacent_pair'
        plt.close(vivid.plot_predictions(run))
        paths=vivid.save_results(run)
        assert [p.stem for p in paths]==expected
        assert set(p.name for p in run.output_dir.iterdir())=={p.name for p in paths}
        for p in paths:
            saved=ad.read_h5ad(p)
            assert saved.n_obs>0
        assert all(not pair.work_dir.exists() for pair in run.pairs)
