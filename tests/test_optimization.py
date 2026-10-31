import ast
from pathlib import Path
from itertools import product
import numpy as np
import pytest
from scipy import sparse
from vivid.optimization import solve_scipy, solve_with_backend

@pytest.fixture
def helpers():
    source=Path(__file__).resolve().parents[1]/'src/vivid/_core/optimize.py'
    tree=ast.parse(source.read_text())
    names={'opt_balanced_round','opt_build_network_arcs','opt_build_candidate_data','opt_local_rounding'}
    scope={'np':np}
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[]),str(source),'exec'),scope)
    return dict(build_arcs=scope['opt_build_network_arcs'],build_candidates=scope['opt_build_candidate_data'],local_rounding=scope['opt_local_rounding'])


def problem():
    probs=np.array([[.8,.2],[.2,.8],[.55,.45]])
    n=np.array([1,2,2]);expected=n[:,None]*probs
    A=sparse.csr_matrix([[1,1,0],[0,1,1]],dtype=float)
    return dict(probs=probs,n=n,expected=expected,global_target=np.array([3,2]),
                patch_matrix=A,patch_reference=A@expected,patch_weight=np.array([.5/12,.5/16]))


@pytest.mark.parametrize('hard',[False,True])
@pytest.mark.parametrize('alpha',[0.,.5,1.])
def test_highs_matches_enumeration(helpers,hard,alpha):
    d=problem()
    # These patch weights represent alpha-dependent normalization in production.
    d['patch_weight']*=alpha/.5
    z,info=solve_scipy(d,**helpers,alpha=alpha,enforce_global_hard_constraint=hard,mip_gap=0)
    def objective(x):
        return (1-alpha)*np.abs(x-d['expected']).sum()/(2*d['n'].sum())+np.sum(d['patch_weight'][:,None]*np.abs(d['patch_matrix']@x-d['patch_reference']))
    # Compare on the exact candidate support used by the solve, not a relaxed LP.
    from vivid.optimization import _network_warm
    warm=(_network_warm(d['probs'],d['n'],d['global_target'],helpers['build_arcs'],1e-6)
          if hard else helpers['local_rounding'](d['probs'],d['n']))
    candidate=helpers['build_candidates'](d['probs'],d['n'],d['global_target'],warm,hard)['candidate_mask']
    possible=[]
    for first in product(*[range(int(n)+1) for n in d['n']]):
        x=np.column_stack([first,d['n']-first])
        if hard and not np.array_equal(x.sum(axis=0),d['global_target']):continue
        if alpha>0 and np.any(x[~candidate]):continue
        possible.append(objective(x))
    assert objective(z)==pytest.approx(min(possible),abs=1e-8)
    assert info['objective']==pytest.approx(objective(z))
    assert np.array_equal(z.sum(axis=1),d['n'])
    if hard: assert np.array_equal(z.sum(axis=0),d['global_target'])


def test_auto_without_gurobi(helpers,monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules,'gurobipy',None)
    def unavailable(*args,**kwargs):raise AssertionError('Gurobi must not be called')
    z,info=solve_with_backend(problem(),backend='auto',gurobi_solver=unavailable,**helpers)
    assert info['solver']=='scipy'
    assert z.sum()==5
    with pytest.raises(ImportError):
        solve_with_backend(problem(),backend='gurobi',gurobi_solver=unavailable,**helpers)


def test_auto_license_error(helpers,monkeypatch):
    import types,sys
    class LicenseError(Exception):pass
    monkeypatch.setitem(sys.modules,'gurobipy',types.SimpleNamespace(GurobiError=LicenseError))
    def unavailable(*args,**kwargs):raise LicenseError('No license')
    z,info=solve_with_backend(problem(),backend='auto',gurobi_solver=unavailable,**helpers)
    assert info['solver']=='scipy'
    with pytest.raises(LicenseError):
        solve_with_backend(problem(),backend='gurobi',gurobi_solver=unavailable,**helpers)
