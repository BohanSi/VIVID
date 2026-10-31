"""Optional single-omics OT + Procrustes alignment supplied with VIVID."""
import numpy as np
from scipy import sparse
import ot


def center_spatial_coords(adata, spatial_key="spatial_raw"):
    coords = np.asarray(adata.obsm[spatial_key], dtype=np.float64)
    adata.obsm["spatial"] = coords - coords.mean(axis=0)
    return adata


def apply_rigid_transform(coords, R, t):
    return np.asarray(coords) @ R.T + t


def generalized_procrustes_analysis(X, Y, pi, allow_reflection=False):
    """Estimate a rigid Y -> X transform from an OT coupling."""
    mass = pi.sum()
    muX = (pi.sum(axis=1) @ X) / mass
    muY = (pi.sum(axis=0) @ Y) / mass
    H = (Y - muY).T @ pi.T @ (X - muX)
    U, _, Vt = np.linalg.svd(H)
    D = np.eye(2)
    if not allow_reflection:
        D[-1, -1] = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ D @ U.T
    t = muX - muY @ R.T
    return R, t


def rigid_body_transformation_invariant_OT(
    sliceA, sliceB, down_sampling_number=2000, alpha=0.5,
    iter_num=15, tolerance=1e-6, spatial_key="spatial",
    random_seed=0, allow_reflection=False,
    initial_angles=(0, 90, 180, 270), verbose=True,
):
    """OT + Procrustes rigid registration. Transform applies to centered coordinates."""
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1]")
    if iter_num < 1 or down_sampling_number < 2 or len(initial_angles) == 0:
        raise ValueError("Invalid iterations, sample count, or initial angles")
    common_genes = sliceA.var_names.intersection(sliceB.var_names)
    if len(common_genes) == 0:
        raise ValueError("Endpoints have no common features")
    X_full = np.asarray(sliceA.obsm[spatial_key], dtype=np.float64)
    Y_full = np.asarray(sliceB.obsm[spatial_key], dtype=np.float64)
    if X_full.ndim != 2 or Y_full.ndim != 2 or X_full.shape[1] != 2 or Y_full.shape[1] != 2:
        raise ValueError("Spatial coordinates must have shape (N, 2)")
    nA, nB = sliceA.n_obs, sliceB.n_obs
    if min(nA, nB) == 0:
        raise ValueError("Endpoints cannot be empty")
    X_full = X_full - X_full.mean(axis=0)
    Y_full = Y_full - Y_full.mean(axis=0)
    rng = np.random.default_rng(random_seed)
    if nA + nB > down_sampling_number:
        kA = int(np.clip(round(down_sampling_number * nA / (nA + nB)), 1, down_sampling_number - 1))
        kA = min(kA, nA)
        kB = min(nB, down_sampling_number - kA)
        idxA = np.sort(rng.choice(nA, kA, replace=False))
        idxB = np.sort(rng.choice(nB, kB, replace=False))
    else:
        idxA, idxB = np.arange(nA), np.arange(nB)
    X, Y = X_full[idxA], Y_full[idxB]
    EX, EY = sliceA[idxA, common_genes].X, sliceB[idxB, common_genes].X
    EX = EX.toarray() if sparse.issparse(EX) else np.asarray(EX)
    EY = EY.toarray() if sparse.issparse(EY) else np.asarray(EY)
    if not all(np.isfinite(v).all() for v in (X, Y, EX, EY)):
        raise ValueError("Expression or coordinates contain NaN/Inf")

    def normalize_cost(C):
        values = C[C > 0]
        return C / (np.median(values) if values.size else 1.0)

    expr_cost = normalize_cost(ot.dist(EX, EY, metric="sqeuclidean"))
    initial_cost = ot.dist(X, Y, metric="sqeuclidean")
    positive = initial_cost[initial_cost > 0]
    spatial_scale = np.median(positive) if positive.size else 1.0
    a, b = np.full(len(X), 1 / len(X)), np.full(len(Y), 1 / len(Y))

    def solve_ot(R, t):
        spatial_cost = ot.dist(X, apply_rigid_transform(Y, R, t), metric="sqeuclidean") / spatial_scale
        C = (1 - alpha) * expr_cost + alpha * spatial_cost
        pi = ot.emd(a, b, C)
        return pi, float(np.sum(pi * C))

    best = None
    for angle in initial_angles:
        theta = np.deg2rad(angle)
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        t = X.mean(axis=0) - Y.mean(axis=0) @ R.T
        pi, objective = solve_ot(R, t)
        for iteration in range(iter_num):
            R_new, t_new = generalized_procrustes_analysis(X, Y, pi, allow_reflection)
            pi_new, objective_new = solve_ot(R_new, t_new)
            improvement = objective - objective_new
            R, t, pi = R_new, t_new, pi_new
            previous_objective, objective = objective, objective_new
            if abs(improvement) <= tolerance * max(1.0, abs(previous_objective)):
                break
        result = (objective, R.copy(), t.copy(), pi.copy(), angle, iteration + 1)
        if best is None or objective < best[0]:
            best = result
    objective, R, t, pi, angle, n_iter = best
    info = dict(objective=float(objective), rotation_degrees=float(np.degrees(np.arctan2(R[1, 0], R[0, 0]))),
                determinant=float(np.linalg.det(R)), initial_angle=angle, iterations=n_iter,
                n_sample_A=len(X), n_sample_B=len(Y), n_common_genes=len(common_genes), alpha=alpha, random_seed=random_seed)
    if verbose:
        print(f"Cells: {len(X)}/{len(Y)} | Genes: {len(common_genes)} | Angle: {info['rotation_degrees']:.2f}° | Objective: {objective:.6f} | Iterations: {n_iter}")
    return pi, R, t, info


def align_inputs(left, right=None, *, same_omics=True, spatial_key="spatial", **kwargs):
    """Return aligned copies of the two observed endpoints. Only for single omics.

    Original coordinates are retained as spatial_raw. No middle/ground-truth slice
    is accepted. Any external alignment method may replace this optional step.
    """
    if right is None:
        if not hasattr(left, "items") or len(left) < 2:
            raise ValueError("Provide at least two named input slices.")
        if not same_omics:
            raise ValueError("Default alignment supports single omics only.")
        items = list(left.items())
        name0, anchor = items[0]
        aligned = {}
        for name, obj in items[1:]:
            first, other = align_inputs(anchor, obj, same_omics=True,
                                        spatial_key=spatial_key, **kwargs)
            aligned.setdefault(name0, first)
            aligned[name] = other
        return aligned
    if not same_omics:
        raise ValueError("Default OT alignment supports single omics only; supply externally aligned endpoints.")
    # Alignment itself needs expression and coordinates, not cell type labels.
    left, right = left.copy(), right.copy()
    for obj in (left, right):
        coords = np.asarray(obj.obsm[spatial_key], dtype=float)
        if obj.n_obs < 2 or coords.shape != (obj.n_obs, 2) or not np.isfinite(coords).all():
            raise ValueError("Alignment requires finite (N, 2) coordinates and at least two observations.")
        if "spatial_raw" not in obj.obsm:
            obj.obsm["spatial_raw"] = coords.copy()
        obj.obsm["spatial"] = coords - coords.mean(axis=0)
    defaults = dict(down_sampling_number=5000, alpha=0.5, iter_num=15, tolerance=1e-6,
                    random_seed=0, allow_reflection=False, initial_angles=(0, 90, 180, 270), verbose=True)
    defaults.update(kwargs)
    _, R, t, info = rigid_body_transformation_invariant_OT(left, right, spatial_key="spatial", **defaults)
    right.obsm["spatial"] = apply_rigid_transform(right.obsm["spatial"], R, t)
    right.uns["alignment"] = dict(R=R, t=t, info=info, role="model_input", transform_coordinates="centered")
    assert np.allclose(R.T @ R, np.eye(2), atol=1e-8)
    if not defaults["allow_reflection"]:
        assert np.isclose(np.linalg.det(R), 1, atol=1e-8)
    return left, right
