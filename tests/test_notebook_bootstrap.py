"""Exercise the actual first cell in fresh interpreters without PYTHONPATH."""
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('location', ['repository', 'notebooks', 'parent', 'external'])
def test_first_cell_without_editable_install(location, tmp_path):
    notebook=json.loads((ROOT/'notebooks/VIVID_tutorial.ipynb').read_text())
    code=''.join(next(c['source'] for c in notebook['cells'] if c['cell_type']=='code'))
    cwd={'repository':ROOT,'notebooks':ROOT/'notebooks','parent':ROOT.parent,'external':tmp_path}[location]
    if location=='external':
        code=code.replace('REPO_ROOT = None', f'REPO_ROOT = {str(ROOT)!r}',1)
    env=dict(os.environ)
    env.pop('PYTHONPATH',None)
    env['MPLCONFIGDIR']=str(tmp_path/'matplotlib')
    env['NUMBA_CACHE_DIR']=str(tmp_path/'numba')
    code+='\nassert ROOT == Path('+repr(str(ROOT))+')\n'
    completed=subprocess.run([sys.executable,'-c',code],cwd=cwd,env=env,
                             capture_output=True,text=True,timeout=60)
    assert completed.returncode==0,completed.stdout+'\n'+completed.stderr
    assert 'VIVID source:' in completed.stdout
    assert str(ROOT/'src/vivid/__init__.py') in completed.stdout


def test_tutorial_manual_solver_setting():
    notebook=json.loads((ROOT/'notebooks/VIVID_tutorial.ipynb').read_text())
    cell=next(''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code' and 'run = vivid.configure' in ''.join(c['source']))
    assert 'OPTIMIZATION_SOLVER = "gurobi"' in cell
    assert 'optimization_solver=OPTIMIZATION_SOLVER' in cell
