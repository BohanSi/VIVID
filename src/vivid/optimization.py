"""Optional Gurobi dispatch and a license-free SciPy/HiGHS formulation.

Both backends minimize the original weighted site/patch L1 objective on the
same candidate support, with integer site allocations and optional global totals.
"""
from __future__ import annotations
import time
import warnings
import numpy as np
from scipy import sparse
from scipy.optimize import linprog, milp, Bounds, LinearConstraint


def _validate_allocation(raw, n, target, hard, tolerance):
    rounded = np.rint(raw).astype(np.int64)
    fractionality = float(np.max(np.abs(raw - rounded), initial=0))
    if fractionality > tolerance + 1e-8:
        raise RuntimeError(f"Solver returned a nonintegral allocation: {fractionality:.3g}")
    if np.any(rounded < 0) or not np.array_equal(rounded.sum(axis=1), n):
        raise RuntimeError("Solver allocation violates nonnegativity or site totals.")
    if hard and not np.array_equal(rounded.sum(axis=0), target):
        raise RuntimeError("Solver allocation violates global type totals.")
    return rounded, fractionality


def _network_warm(probs, n, target, build_arcs, tolerance):
    """Solve the original integral network LP using the HiGHS dual simplex."""
    S, K = probs.shape
    if int(target.sum()) != int(n.sum()):
        raise ValueError("Global targets must sum to the total cell count.")
    arcs = build_arcs(probs, n, 1 / (2 * int(n.sum())))
    keep = target[arcs['k']] > 0
    i, k = arcs['i'][keep], arcs['k'][keep]
    L = len(i)
    columns = np.arange(L)
    A = sparse.coo_matrix((np.r_[np.ones(L), -np.ones(L)],
                          (np.r_[i, S+k], np.r_[columns, columns])), shape=(S+K, L)).tocsr()
    result = linprog(arcs['cost'][keep], A_eq=A, b_eq=np.r_[n, -target],
                     bounds=np.column_stack([np.zeros(L), arcs['capacity'][keep]]), method='highs-ds')
    if not result.success:
        raise RuntimeError(f"HiGHS network LP failed: {result.message}")
    if np.max(np.abs(result.x - np.rint(result.x)), initial=0) > tolerance:
        raise RuntimeError("HiGHS network LP returned fractional arc flows.")
    warm = np.zeros((S,K), dtype=np.int64)
    np.add.at(warm, (i,k), np.rint(result.x).astype(np.int64))
    return _validate_allocation(warm, n, target, True, tolerance)[0]


def solve_scipy(data, *, build_arcs, build_candidates, local_rounding,
                alpha=0.5, enforce_global_hard_constraint=True, mip_gap=1e-2,
                time_limit=None, output=False, integrality_tol=1e-6, **unused):
    """Solve the original sparse allocation problem with scipy.optimize.milp.

    Absolute deviations use continuous epigraph variables. This is an equivalent
    linearization of the original convex PWL objective, not a rounding heuristic.
    HiGHS cannot accept the original MIP start through SciPy; the network solution
    is still used to construct exactly the same kind of feasible candidate support.
    """
    started = time.perf_counter()
    probs, n, expected, target = (data[k] for k in ('probs','n','expected','global_target'))
    n, target = np.asarray(n, dtype=np.int64), np.asarray(target, dtype=np.int64)
    S,K = expected.shape
    N = int(n.sum())
    if N <= 0 or not 0 <= alpha <= 1:
        raise ValueError("Positive total count and alpha in [0, 1] are required.")
    hard = bool(enforce_global_hard_constraint)
    warm = (_network_warm(probs,n,target,build_arcs,integrality_tol) if hard
            else local_rounding(probs,n))
    warm_runtime = time.perf_counter() - started
    if alpha == 0:
        objective = float(np.abs(warm-expected).sum() / (2*N))
        return warm, dict(solver='scipy', status=0, objective=objective,
            runtime=warm_runtime, wall_runtime=warm_runtime, warm_runtime=warm_runtime,
            mip_gap=0., obj_bound=objective, node_count=0., n_variables=0,
            n_constraints=0, max_fractionality=0., candidate_topk=np.nan)
    candidate = build_candidates(probs,n,target,warm,hard)
    ci,ck = candidate['cand_i'],candidate['cand_k']
    J = len(ci)
    Q = data['patch_matrix'].shape[0]
    # z (integer), local |z-e|, patch |A z - reference|.
    local_scale = (1-alpha)/(2*N)
    L = J if local_scale > 0 else 0
    P = Q*K
    nv = J+L+P
    site = sparse.coo_matrix((np.ones(J),(ci,np.arange(J))),shape=(S,J)).tocsr()
    glob = sparse.coo_matrix((np.ones(J),(ck,np.arange(J))),shape=(K,J)).tocsr()
    equality = sparse.vstack([site,glob],format='csr') if hard else site
    rhs = np.r_[n,target] if hard else n
    rows=[sparse.hstack([equality,sparse.csr_matrix((len(rhs),L+P))],format='csr')]
    lower=[rhs.astype(float)];upper=[rhs.astype(float)]
    if L:
        eye=sparse.eye(J,format='csr');zero=sparse.csr_matrix((J,P))
        rows.extend([sparse.hstack([eye,-eye,zero]),sparse.hstack([-eye,-eye,zero])])
        lower.extend([np.full(J,-np.inf)]*2)
        upper.extend([expected[ci,ck],-expected[ci,ck]])
    if P:
        # Expand the patch incidence matrix only at existing candidate pairs.
        expand=sparse.coo_matrix((np.ones(J),(ci*K+ck,np.arange(J))),shape=(S*K,J)).tocsr()
        B=sparse.kron(data['patch_matrix'],sparse.eye(K),format='csr') @ expand
        zero=sparse.csr_matrix((P,L));eye=sparse.eye(P,format='csr')
        rows.extend([sparse.hstack([B,zero,-eye]),sparse.hstack([-B,zero,-eye])])
        lower.extend([np.full(P,-np.inf)]*2)
        ref=np.asarray(data['patch_reference']).ravel()
        upper.extend([ref,-ref])
    constraint_matrix=sparse.vstack(rows,format='csc')
    c=np.r_[np.zeros(J),np.full(L,local_scale),np.repeat(data['patch_weight'],K)]
    bounds=Bounds(np.zeros(nv),np.r_[candidate['candidate_ub'],np.full(L+P,np.inf)])
    integrality=np.r_[np.ones(J,dtype=int),np.zeros(L+P,dtype=int)]
    options=dict(disp=bool(output),mip_rel_gap=float(mip_gap))
    if time_limit is not None: options['time_limit']=float(time_limit)
    solve_start=time.perf_counter()
    result=milp(c,integrality=integrality,bounds=bounds,
                constraints=LinearConstraint(constraint_matrix,np.concatenate(lower),np.concatenate(upper)),options=options)
    runtime=time.perf_counter()-solve_start
    if result.x is None or result.status not in (0,1):
        raise RuntimeError(f"HiGHS MILP produced no usable incumbent: {result.message}")
    raw=np.zeros((S,K));raw[ci,ck]=result.x[:J]
    allocation,fractionality=_validate_allocation(raw,n,target,hard,integrality_tol)
    if result.status==1:
        warnings.warn(f"HiGHS stopped at a limit; returning a validated feasible incumbent: {result.message}",RuntimeWarning)
    objective=float(local_scale*np.abs(allocation-expected).sum()
                    + np.sum(np.asarray(data['patch_weight'])[:,None]*np.abs(data['patch_matrix']@allocation-data['patch_reference'])))
    constant=float(local_scale*expected[~candidate['candidate_mask']].sum())
    bound=getattr(result,'mip_dual_bound',None)
    bound=float(bound+constant) if bound is not None else np.nan
    return allocation,dict(solver='scipy',status=int(result.status),message=result.message,
        objective=objective,runtime=runtime,wall_runtime=time.perf_counter()-started,
        build_runtime=solve_start-started-warm_runtime,warm_runtime=warm_runtime,
        mip_gap=max(0.,objective-bound)/max(abs(objective),1e-12) if np.isfinite(bound) else np.nan,
        obj_bound=bound,node_count=float(getattr(result,'mip_node_count',0) or 0),
        n_variables=nv,n_constraints=constraint_matrix.shape[0],n_integer_variables=J,
        max_fractionality=fractionality,**{k:v for k,v in candidate.items() if np.isscalar(v)})


def solve_with_backend(data, *, backend, gurobi_solver, **kwargs):
    """Auto falls back on missing Gurobi or Gurobi runtime/license errors only."""
    helpers = {k:kwargs.pop(k) for k in ('build_arcs','build_candidates','local_rounding')}
    if backend != 'scipy':
        try:
            import gurobipy as gp
        except ImportError:
            if backend == 'gurobi':
                raise ImportError("Install vivid-spatial[gurobi] and configure a Gurobi license.") from None
            print("Gurobi is not installed; using SciPy/HiGHS.")
        else:
            try:
                allocation, info = gurobi_solver(data, **kwargs)
                info['solver']='gurobi'
                return allocation,info
            except gp.GurobiError as exc:
                if backend == 'gurobi': raise
                print(f"Gurobi unavailable ({exc}); using SciPy/HiGHS.")
    return solve_scipy(data,**helpers,**kwargs)
