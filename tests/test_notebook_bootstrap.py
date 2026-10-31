"""Exercise the actual first cell in fresh interpreters without PYTHONPATH."""
import ast
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


def test_tutorial_uses_required_gurobi():
    notebook=json.loads((ROOT/'notebooks/VIVID_tutorial.ipynb').read_text())
    cell=next(''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code' and 'run = vivid.configure' in ''.join(c['source']))
    tree=ast.parse(cell)
    call=next(n for n in ast.walk(tree) if isinstance(n,ast.Call)
              and isinstance(n.func,ast.Attribute) and n.func.attr=='configure')
    solver=next(k.value for k in call.keywords if k.arg=='optimization_solver')
    if isinstance(solver,ast.Name):
        assignments={target.id:node.value.value for node in tree.body
                     if isinstance(node,ast.Assign) and isinstance(node.value,ast.Constant)
                     for target in node.targets if isinstance(target,ast.Name)}
        assert assignments[solver.id]=='gurobi'
    else:
        assert isinstance(solver,ast.Constant) and solver.value=='gurobi'
