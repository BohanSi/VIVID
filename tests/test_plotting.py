"""Palette, orientation, figure lifecycle and real Jupyter display checks."""
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from PIL import Image
import vivid
from vivid.plotting import CELLTYPE_COLORS



def data():
    a=ad.AnnData(np.array([[1.,2.],[3.,4.],[5.,6.]],dtype='float32'),
                 obs=pd.DataFrame({'cell_type':['z','A','m']},index=['c1','c2','c3']))
    a.var_names=['gene1','gene2']
    a.obsm['spatial']=np.array([[0.,0.],[1.,0.],[0.,1.]])
    return a


def session(tmp_path):
    a=data()
    run=vivid.configure(a,a,left_id='L',right_id='R',save_root=tmp_path)
    image=np.zeros((10,10,3),dtype='uint8');image[5:]=255
    p=tmp_path/'encoded.png';Image.fromarray(image).save(p)
    run._state['summary']={'slice_1':{'png_path':str(p)},'slice_2':{'png_path':str(p)}}
    run._state['target_img']=[1]
    frames=run.work_dir/'RIFE_result'/'pair'/'frames';frames.mkdir(parents=True)
    Image.fromarray(image).save(frames/'img1.png')
    run.stage='reconstructed';run.results={'L_R_1':a.copy()}
    return run


def test_palette_sorted_and_closed_figures(tmp_path):
    a=data()
    with session(tmp_path) as run:
        for fig in [vivid.plot_inputs(a,a),vivid.plot_predictions(run)]:
            assert not plt.fignum_exists(fig.number)
            assert all(ax.get_subplotspec().get_gridspec().nrows==1 for ax in fig.axes)
            for ax in fig.axes:
                assert [to_hex(c.get_facecolors()[0]) for c in ax.collections]==list(CELLTYPE_COLORS[:3])
            assert [t.get_text() for t in fig.legends[0].get_texts()]==['A','m','z']


def test_rgb_orientation_titles_and_no_color_legend(tmp_path):
    with session(tmp_path) as run:
        fig=vivid.plot_encoding(run)
        assert len(fig.axes)==2
        assert not fig.legends
        assert fig._suptitle is None
        assert all(not ax.yaxis_inverted() for ax in fig.axes)
        assert not plt.fignum_exists(fig.number)
        fig=vivid.plot_interpolation(run)
        assert len(fig.axes)==1
        assert fig._suptitle is None
        assert fig.axes[0].get_title()=='L / R | Target 1'
        assert not fig.axes[0].yaxis_inverted()
        assert not plt.fignum_exists(fig.number)


def test_feature_colorbar_is_compact():
    fig=vivid.plot_feature(data(),'gene1')
    fig.canvas.draw()
    ax,bar=fig.axes
    assert bar.get_position().height < .8*ax.get_position().height
    assert bar.yaxis.label.get_fontsize()==9
    assert not plt.fignum_exists(fig.number)


def test_one_image_per_cell_with_ipython_inline(tmp_path):
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output
    from IPython.core.displayhook import CapturingDisplayHook
    from matplotlib_inline.backend_inline import configure_inline_support
    previous_backend = plt.get_backend()
    shell = InteractiveShell.instance()
    previous_hook = shell.display_trap.hook
    plt.switch_backend("module://matplotlib_inline.backend_inline")
    configure_inline_support(shell, "module://matplotlib_inline.backend_inline")
    try:
        with session(tmp_path) as run:
            shell.user_ns.update(vivid=vivid, run=run, a=data())
            calls = ['vivid.plot_inputs(a,a,point_size=0.5)', 'vivid.plot_encoding(run)',
                     'vivid.plot_interpolation(run)', 'vivid.plot_predictions(run,point_size=0.5)',
                     'vivid.plot_feature(a,"gene1",point_size=0.5)']
            for call in calls:
                records = []
                shell.display_trap.hook = CapturingDisplayHook(shell, records)
                with capture_output() as captured:
                    result = shell.run_cell(call, store_history=True)
                assert result.error_in_exec is None
                rich = sum('image/png' in item['data'] for item in records)
                automatic = sum('image/png' in item.data for item in captured.outputs)
                assert rich == 1 and automatic == 0, (call, rich, automatic)
    finally:
        shell.display_trap.hook = previous_hook
        plt.switch_backend(previous_backend)
