# Migrated from 1002genertation.ipynb, cell 47. See docs/MIGRATION.md.
import json, time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, coo_matrix
from tqdm.auto import tqdm
from IPython.display import display

if config.get("optimization_solver", "gurobi") == "scipy":
    gp = GRB = None
else:
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError:
        gp = GRB = None


# ============================================================
# 1. Utilities
# ============================================================

def opt_balanced_round(x, total):
    x = np.maximum(np.asarray(x, dtype=float), 0)
    z = np.floor(x).astype(np.int64)
    diff = int(total - z.sum())
    if diff < 0: raise RuntimeError("balanced rounding  has a negative remainder ")
    if diff: z[np.argsort(-(x - z), kind="stable")[:diff]] += 1
    if z.sum() != total: raise RuntimeError("balanced rounding  total count is incorrect ")
    return z


def opt_patch_starts(length, patch_size, stride):
    if patch_size > length: raise ValueError(f"patch_size={patch_size} > image length={length}")
    starts = list(range(0, length - patch_size + 1, stride))
    if not starts or starts[-1] != length - patch_size: starts.append(length - patch_size)
    return starts


def opt_abs_pwl_points(upper, target, scale):
    upper = float(upper)
    target = float(np.clip(target, 0, upper))
    scale = float(scale)

    if target <= 1e-12: return [0, upper], [0, scale * upper]
    if target >= upper - 1e-12: return [0, upper], [scale * target, 0]

    return [0, target, upper], [scale * target, 0, scale * (upper - target)]


def opt_check_cell_count_conservation(df, integer_count_results, model_names):
    target = {}

    for pair, frames in integer_count_results.items():
        for frame in frames:
            image_index = int(frame["image_index"])

            if "estimated_count" in frame:
                target[(str(pair), image_index)] = int(frame["estimated_count"])
            else:
                with np.load(frame["npz_path"]) as f:
                    target[(str(pair), image_index)] = int(np.asarray(f["count_map"]).sum())

    current = df.assign(
        _model=df["model"].astype(str), _pair=df["pair"].astype(str), _image_index=df["image_index"].astype(int)
    ).groupby(["_model", "_pair", "_image_index"], sort=False, observed=True)["cell_number"].sum()

    rows = []

    for model in map(str, model_names):
        for (pair, image_index), expected in target.items():
            observed = int(current.get((model, pair, image_index), 0))
            rows.append({
                "model": model, "pair": pair, "image_index": image_index,
                "count_map_cells": expected, "fused_table_cells": observed,
                "difference": observed - expected, "match": observed == expected,
            })

    check = pd.DataFrame(rows)
    bad = check.loc[~check["match"]]

    if len(bad):
        display(bad)
        raise RuntimeError(f"{len(bad)}  items  model/frame  lost cells .")

    return check


# ============================================================
# 2. Patch data
# ============================================================

def opt_build_patch_data(frame_df, n, expected, H, W, patch_size, patch_stride, alpha):
    if alpha <= 0: return csr_matrix((0, len(frame_df))), np.empty((0, expected.shape[1])), np.empty(0), pd.DataFrame()

    P = int(patch_size)
    stride = P // 2 if patch_stride is None else int(patch_stride)

    if P <= 0 or stride <= 0 or stride > P: raise ValueError("patch_size / patch_stride  invalid configuration ")

    u = frame_df["center_pixel_u"].to_numpy(dtype=np.int64)
    v = frame_df["center_pixel_v"].to_numpy(dtype=np.int64)

    if np.any((u < 0) | (u >= W) | (v < 0) | (v >= H)): raise ValueError(" contains site center outside image bounds ")
    if len(np.unique(v * W + u)) != len(u): raise RuntimeError(" same frame contains duplicates site center")

    site_grid = np.full((H, W), -1, dtype=np.int32)
    site_grid[v, u] = np.arange(len(frame_df), dtype=np.int32)

    indices, refs, meta = [], [], []

    for y0 in opt_patch_starts(H, P, stride):
        for x0 in opt_patch_starts(W, P, stride):
            idx = site_grid[y0:y0 + P, x0:x0 + P].ravel()
            idx = idx[idx >= 0].astype(np.int64, copy=False)

            if not len(idx): continue

            Nq = int(n[idx].sum())
            if Nq <= 0: continue

            indices.append(idx)
            refs.append(expected[idx].sum(axis=0))
            meta.append({
                "patch_id": len(meta), "x0": x0, "x1": x0 + P, "y0": y0, "y1": y0 + P,
                "n_sites": len(idx), "n_cells": Nq,
            })

    if not indices: raise RuntimeError("alpha>0,  but none were constructed patch")

    rows = np.concatenate([np.full(len(idx), q, dtype=np.int64) for q, idx in enumerate(indices)])
    cols = np.concatenate(indices)

    A = csr_matrix((np.ones(len(rows), dtype=float), (rows, cols)), shape=(len(indices), len(frame_df)))
    reference = np.vstack(refs).astype(float)
    weight = np.asarray([alpha / (2 * len(indices) * x["n_cells"]) for x in meta], dtype=float)

    return A, reference, weight, pd.DataFrame(meta)


# ============================================================
# 3. Precompute frame data
# ============================================================

def opt_prepare_frame(frame_df, outputs, celltypes, alpha, patch_size, patch_stride, count_map_cache, global_prop_cache):
    frame_df = frame_df.copy().reset_index(drop=True)

    model = str(frame_df["model"].iloc[0])
    pair = str(frame_df["pair"].iloc[0])
    image = str(frame_df["image"].iloc[0])
    image_index = int(frame_df["image_index"].iloc[0])

    left_tag, right_tag = ctp_parse_pair(pair)
    K = len(celltypes)

    def get_maps(tag):
        key = ctp_resolve_output_key(outputs, tag)
        if str(key) not in count_map_cache: count_map_cache[str(key)] = ctp_get_count_maps(outputs[key], K)
        return count_map_cache[str(key)]

    def get_global(tag):
        key = ctp_resolve_output_key(outputs, tag)

        if str(key) not in global_prop_cache:
            counts = get_maps(tag).sum(axis=(1, 2))
            if counts.sum() <= 0: raise ValueError(f"{tag}:  observed total cell count is 0")
            global_prop_cache[str(key)] = counts / counts.sum()

        return global_prop_cache[str(key)]

    left_maps, right_maps = get_maps(left_tag), get_maps(right_tag)

    if left_maps.shape != right_maps.shape: raise ValueError(f"{pair}:  observed left and right  count_maps shape  inconsistent ")

    _, H, W = left_maps.shape

    ts = frame_df["interp_fraction"].dropna().unique()
    if len(ts) != 1: raise ValueError(f"{pair}/img{image_index}: interp_fraction  not unique : {ts}")

    t = float(ts[0])

    left_global, right_global = get_global(left_tag), get_global(right_tag)
    global_reference = (1 - t) * left_global + t * right_global
    global_reference /= global_reference.sum()

    prop_cols = [f"prop__{ctp_safe_name(ct)}" for ct in celltypes]
    ctp_required_columns(frame_df, prop_cols, "fused_celltype_prop_df")

    probs = np.maximum(frame_df[prop_cols].to_numpy(dtype=float), 0)

    fallback = probs.sum(axis=1) <= 1e-12
    if "has_type_evidence" in frame_df.columns: fallback |= ~frame_df["has_type_evidence"].fillna(False).astype(bool).to_numpy()

    probs[fallback] = global_reference
    probs /= probs.sum(axis=1, keepdims=True)

    n = frame_df["cell_number"].to_numpy(dtype=np.int64)
    if np.any(n <= 0): raise ValueError(" optimization input site contains  cell_number<=0")

    N = int(n.sum())
    expected = n[:, None] * probs

    global_expected = N * global_reference
    global_target = opt_balanced_round(global_expected, N)

    patch_matrix, patch_reference, patch_weight, patch_meta = opt_build_patch_data(
        frame_df, n, expected, H, W, patch_size, patch_stride, alpha
    )

    return {
        "frame_df": frame_df,
        "model": model, "pair": pair, "image": image, "image_index": image_index, "t": t,
        "left_global": left_global, "right_global": right_global,
        "global_reference": global_reference, "global_expected": global_expected, "global_target": global_target,
        "probs": probs, "n": n, "expected": expected, "fallback": fallback, "N": N,
        "patch_matrix": patch_matrix, "patch_reference": patch_reference,
        "patch_weight": patch_weight, "patch_meta": patch_meta,
    }


def opt_prepare_all_frames(fused_celltype_prop_df, outputs, celltypes, model_names, alpha, patch_size, patch_stride):
    required = [
        "model", "pair", "image", "image_index", "site_id",
        "cell_number", "center_pixel_u", "center_pixel_v", "interp_fraction",
    ]

    ctp_required_columns(fused_celltype_prop_df, required, "fused_celltype_prop_df")

    model_names = [str(x) for x in model_names]

    df = fused_celltype_prop_df.loc[
        fused_celltype_prop_df["model"].astype(str).isin(model_names)
        & (fused_celltype_prop_df["cell_number"].astype(int) > 0)
    ].copy().reset_index(drop=True)

    if not len(df): raise ValueError(" no optimizable site")

    count_map_cache, global_prop_cache, prepared = {}, {}, []
    groups = df.groupby(["model", "pair", "image_index"], sort=False, observed=True).indices

    for _, idx in tqdm(groups.items(), total=len(groups), desc="Preparing optimization inputs"):
        prepared.append(
            opt_prepare_frame(
                df.iloc[np.asarray(idx, dtype=np.int64)], outputs, celltypes, alpha,
                patch_size, patch_stride, count_map_cache, global_prop_cache
            )
        )

    return df, prepared


# ============================================================
# 4. Exact site-L1 network warm solution
# ============================================================

def opt_build_network_arcs(probs, n, scale):
    expected = n[:, None] * probs
    expected = np.where(np.abs(expected - np.rint(expected)) <= 1e-12, np.rint(expected), expected)

    q = np.floor(expected).astype(np.int64)
    frac = expected - q
    nmat = np.broadcast_to(n[:, None], expected.shape)

    all_i, all_k, all_cap, all_cost = [], [], [], []

    def add(mask, capacity, cost):
        if not np.any(mask): return
        i, k = np.where(mask)
        all_i.append(i); all_k.append(k)
        all_cap.append(capacity[mask].astype(float) if np.ndim(capacity) else np.full(len(i), float(capacity)))
        all_cost.append(cost[mask].astype(float) if np.ndim(cost) else np.full(len(i), float(cost)))

    add(q > 0, q, -float(scale))
    add(q < nmat, np.ones_like(q), float(scale) * (1 - 2 * frac))

    remainder = nmat - q - 1
    add(remainder > 0, remainder, float(scale))

    if not all_i: raise RuntimeError(" cannot construct network warm arcs")

    return {
        "i": np.concatenate(all_i).astype(np.int64),
        "k": np.concatenate(all_k).astype(np.int64),
        "capacity": np.concatenate(all_cap).astype(float),
        "cost": np.concatenate(all_cost).astype(float),
        "expected": expected,
    }


def opt_solve_network_warm(probs, n, target, threads=0, lp_method=1, presolve=-1, integrality_tol=1e-6):
    S, K = probs.shape
    N = int(n.sum())
    scale = 1 / (2 * N)

    if int(target.sum()) != N: raise ValueError("global target  sum  !=  total cell count ")

    arcs = opt_build_network_arcs(probs, n, scale)
    i, k, capacity, cost = arcs["i"], arcs["k"], arcs["capacity"], arcs["cost"]

    
    keep = target[k] > 0
    i, k, capacity, cost = i[keep], k[keep], capacity[keep], cost[keep]

    L = len(i)
    cols = np.arange(L, dtype=np.int64)

    A = coo_matrix(
        (
            np.concatenate([np.ones(L), -np.ones(L)]),
            (
                np.concatenate([i, S + k]),
                np.concatenate([cols, cols]),
            ),
        ),
        shape=(S + K, L),
    ).tocsr()

    b = np.concatenate([n.astype(float), -target.astype(float)])

    model = gp.Model("hard_global_network_warm")
    model.Params.OutputFlag = 0
    model.Params.Threads = int(threads)
    model.Params.Method = int(lp_method)
    model.Params.Presolve = int(presolve)

    flow = model.addMVar(L, lb=0, ub=capacity, vtype=GRB.CONTINUOUS)
    model.addConstr(A @ flow == b)

    constant = float(scale * arcs["expected"].sum())
    model.setObjective(cost @ flow + constant, GRB.MINIMIZE)

    start = time.perf_counter()
    model.optimize()
    runtime = time.perf_counter() - start

    if model.Status != GRB.OPTIMAL:
        status = int(model.Status)
        model.dispose()
        raise RuntimeError(f"network warm LP failed , status={status}")

    values = np.asarray(flow.X, dtype=float)
    max_fractionality = float(np.abs(values - np.rint(values)).max()) if len(values) else 0

    if max_fractionality > integrality_tol:
        model.dispose()
        raise RuntimeError(f"network warm solution is nonintegral , max_fractionality={max_fractionality:.3e}")

    values = np.rint(values).astype(np.int64)
    warm = np.zeros((S, K), dtype=np.int64)
    np.add.at(warm, (i, k), values)

    if not np.array_equal(warm.sum(axis=1), n): raise RuntimeError("network warm row sums are incorrect ")
    if not np.array_equal(warm.sum(axis=0), target): raise RuntimeError("network warm column sums are incorrect ")

    model.dispose()
    return warm, runtime


def opt_local_rounding(probs, n):
    expected = n[:, None] * probs
    allocation = np.zeros_like(expected, dtype=np.int64)

    for s in range(len(n)):
        allocation[s] = opt_balanced_round(expected[s], int(n[s]))

    return allocation


# ============================================================
# 5. Sparse candidate support
# ============================================================

def opt_build_candidate_data(probs, n, target, warm, enforce_global_hard_constraint):
    S, K = probs.shape
    candidate_topk = max(1, (K + 1) // 2)

    top_idx = np.argsort(-probs, axis=1, kind="stable")[:, :candidate_topk]

    top_mask = np.zeros((S, K), dtype=bool)
    top_mask[np.arange(S)[:, None], top_idx] = True

    candidate_mask = top_mask | (warm > 0)

    if enforce_global_hard_constraint: candidate_mask[:, target <= 0] = False

    if np.any((warm > 0) & ~candidate_mask): raise RuntimeError("candidate support removed warm-positive variables ")

    per_site = candidate_mask.sum(axis=1)
    if np.any(per_site == 0): raise RuntimeError(f"{int((per_site == 0).sum())}  items site missing candidate types ")

    cand_i, cand_k = np.where(candidate_mask)
    candidate_pairs = list(zip(cand_i.astype(int), cand_k.astype(int)))

    ub = n[cand_i].astype(np.int64)
    if enforce_global_hard_constraint: ub = np.minimum(ub, target[cand_k])

    if np.any(ub <= 0): raise RuntimeError("candidate upper bound must  > 0")

    return {
        "candidate_mask": candidate_mask,
        "cand_i": cand_i,
        "cand_k": cand_k,
        "candidate_pairs": candidate_pairs,
        "candidate_ub": ub,
        "candidate_topk": candidate_topk,
        "n_full_variables": int(S * K),
        "n_topk_candidates": int((top_mask & candidate_mask).sum()),
        "n_warm_support_added": int(((warm > 0) & ~top_mask).sum()),
        "n_candidate_variables": int(candidate_mask.sum()),
        "candidate_density": float(candidate_mask.mean()),
        "min_candidates_per_site": int(per_site.min()),
        "mean_candidates_per_site": float(per_site.mean()),
        "max_candidates_per_site": int(per_site.max()),
    }


# ============================================================
# 6. Optimized sparse Gurobi
# ============================================================

def opt_solve_gurobi(
    data, alpha=0.5, enforce_global_hard_constraint=True,
    mip_gap=1e-2, time_limit=None, threads=0, output=False,
    mip_focus=3, lp_method=1, mip_root_method=2, presolve=-1, integrality_tol=1e-6
):
    probs, n, expected, target = data["probs"], data["n"], data["expected"], data["global_target"]
    A, patch_ref, patch_weight = data["patch_matrix"], data["patch_reference"], data["patch_weight"]

    S, K = expected.shape
    N = int(n.sum())
    Q = A.shape[0]

    
    if alpha == 0:
        start = time.perf_counter()

        if enforce_global_hard_constraint:
            allocation, warm_runtime = opt_solve_network_warm(
                probs, n, target, threads, lp_method, presolve, integrality_tol
            )
        else:
            allocation = opt_local_rounding(probs, n)
            warm_runtime = time.perf_counter() - start

        objective = float(np.abs(allocation - expected).sum() / (2 * N))

        return allocation, {
            "status": int(GRB.OPTIMAL), "objective": objective,
            "runtime": warm_runtime, "wall_runtime": warm_runtime,
            "warm_runtime": warm_runtime, "mip_gap": 0.0, "obj_bound": objective,
            "node_count": 0.0, "n_variables": 0, "n_constraints": 0,
            "max_fractionality": 0.0, "candidate_topk": np.nan,
        }

    # --------------------------------------------------------
    # High-quality warm solution
    # --------------------------------------------------------

    warm_start = time.perf_counter()

    if enforce_global_hard_constraint:
        warm, _ = opt_solve_network_warm(probs, n, target, threads, lp_method, presolve, integrality_tol)
    else:
        warm = opt_local_rounding(probs, n)

    warm_runtime = time.perf_counter() - warm_start

    candidate = opt_build_candidate_data(probs, n, target, warm, enforce_global_hard_constraint)

    candidate_mask = candidate["candidate_mask"]
    cand_i, cand_k = candidate["cand_i"], candidate["cand_k"]
    candidate_pairs = candidate["candidate_pairs"]
    cand_ub = candidate["candidate_ub"]

    candidate_types_by_site = [np.flatnonzero(candidate_mask[s]).astype(int).tolist() for s in range(S)]
    candidate_sites_by_type = [np.flatnonzero(candidate_mask[:, k]).astype(int).tolist() for k in range(K)]

    binary_mask = cand_ub == 1
    integer_mask = cand_ub > 1

    binary_pairs = [candidate_pairs[j] for j in np.flatnonzero(binary_mask)]
    integer_pairs = [candidate_pairs[j] for j in np.flatnonzero(integer_mask)]

    # --------------------------------------------------------
    # Main MIP
    # --------------------------------------------------------

    build_start = time.perf_counter()

    model = gp.Model(f"celltype_{data['model']}_{data['pair']}_{data['image_index']}")
    model.Params.OutputFlag = int(output)
    model.Params.MIPGap = float(mip_gap)
    model.Params.Threads = int(threads)
    model.Params.MIPFocus = int(mip_focus)
    model.Params.Presolve = int(presolve)

    if mip_root_method is not None: model.Params.Method = int(mip_root_method)
    if time_limit is not None: model.Params.TimeLimit = float(time_limit)

    Z = {}

    if binary_pairs:
        Z.update(model.addVars(binary_pairs, vtype=GRB.BINARY, name="ZB"))

    if integer_pairs:
        ub_dict = {candidate_pairs[j]: float(cand_ub[j]) for j in np.flatnonzero(integer_mask)}
        Z.update(model.addVars(integer_pairs, lb=0, ub=ub_dict, vtype=GRB.INTEGER, name="ZI"))

    # Site totals
    for s in range(S):
        model.addConstr(gp.quicksum(Z[s, k] for k in candidate_types_by_site[s]) == int(n[s]), name=f"site_{s}")

    # Optional hard global totals
    if enforce_global_hard_constraint:
        for k in range(K):
            sites = candidate_sites_by_type[k]

            if sites:
                model.addConstr(gp.quicksum(Z[s, k] for s in sites) == int(target[k]), name=f"type_{k}")
            elif target[k] != 0:
                raise RuntimeError(f"type={k}: global target={target[k]},  but missing candidate site")

    # --------------------------------------------------------
    # Site L1 objective
    #
    # No site_dev variables:
    # linear where possible, PWL only when necessary
    # --------------------------------------------------------

    local_scale = float(1 - alpha) / (2 * N)
    excluded_constant = float(local_scale * expected[~candidate_mask].sum())
    linear_constant = 0.0
    linear_expr = gp.LinExpr()
    pwl_specs = []

    if local_scale > 0:
        for j, (s, k) in enumerate(candidate_pairs):
            var = Z[s, k]
            ub = float(cand_ub[j])
            e = float(expected[s, k])

            if ub <= 1 + 1e-12:
                f0 = local_scale * abs(e)
                f1 = local_scale * abs(1 - e)
                linear_constant += f0
                linear_expr.add(var, f1 - f0)

            elif e <= 1e-12:
                linear_expr.add(var, local_scale)

            elif e >= ub - 1e-12:
                linear_constant += local_scale * e
                linear_expr.add(var, -local_scale)

            else:
                pwl_specs.append((var, [0, e, ub], [local_scale * e, 0, local_scale * (ub - e)]))

    model.setObjective(excluded_constant + linear_constant + linear_expr, GRB.MINIMIZE)

    for var, xpts, ypts in pwl_specs:
        model.setPWLObj(var, xpts, ypts)

    # --------------------------------------------------------
    # Patch L1
    #
    # patch_total + PWL objective
    # only one linking constraint per patch/type
    # --------------------------------------------------------

    patch_upper = np.repeat(data["patch_meta"]["n_cells"].to_numpy(float)[:, None], K, axis=1)
    patch_total = model.addMVar((Q, K), lb=0, ub=patch_upper, vtype=GRB.CONTINUOUS, name="patch_total")

    for q in range(Q):
        idx = A.indices[A.indptr[q]:A.indptr[q + 1]]

        for k in range(K):
            active = idx[candidate_mask[idx, k]]
            patch_var = patch_total[q, k].item()

            model.addConstr(
                patch_var == gp.quicksum(Z[int(s), k] for s in active) if len(active) else patch_var == 0,
                name=f"patch_{q}_{k}",
            )

            xpts, ypts = opt_abs_pwl_points(
                data["patch_meta"]["n_cells"].iloc[q],
                patch_ref[q, k],
                patch_weight[q],
            )

            model.setPWLObj(patch_var, xpts, ypts)

    # --------------------------------------------------------
    # Warm start
    # --------------------------------------------------------

    model.update()

    for s, k in candidate_pairs: Z[s, k].Start = float(warm[s, k])

    patch_start = np.asarray(A.dot(warm), dtype=float)
    patch_total.setAttr(GRB.Attr.Start, patch_start)

    build_runtime = time.perf_counter() - build_start

    # --------------------------------------------------------
    # Solve
    # --------------------------------------------------------

    solve_start = time.perf_counter()
    model.optimize()
    wall_runtime = time.perf_counter() - solve_start

    if model.SolCount <= 0:
        status = int(model.Status)
        model.dispose()
        raise RuntimeError(f"Gurobi  no feasible solution , status={status}")

    z_raw = np.asarray([Z[s, k].X for s, k in candidate_pairs], dtype=float)
    z_int = np.rint(z_raw).astype(np.int64)

    max_fractionality = float(np.abs(z_raw - z_int).max()) if len(z_raw) else 0

    if max_fractionality > integrality_tol + 1e-8:
        print(f"Warning: max_fractionality={max_fractionality:.3e}")

    allocation = np.zeros((S, K), dtype=np.int64)
    allocation[cand_i, cand_k] = z_int

    if not np.array_equal(allocation.sum(axis=1), n):
        model.dispose()
        raise RuntimeError("Gurobi result violates  site cell-number constraint")

    if enforce_global_hard_constraint and not np.array_equal(allocation.sum(axis=0), target):
        model.dispose()
        raise RuntimeError("Gurobi result violates  global hard constraint")

    info = {
        "status": int(model.Status),
        "objective": float(model.ObjVal),
        "runtime": float(model.Runtime),
        "wall_runtime": float(wall_runtime),
        "build_runtime": float(build_runtime),
        "warm_runtime": float(warm_runtime),

        "mip_gap": float(model.MIPGap),
        "obj_bound": float(model.ObjBound),
        "node_count": float(model.NodeCount),
        "n_variables": int(model.NumVars),
        "n_constraints": int(model.NumConstrs),
        "n_integer_variables": int(model.NumIntVars),
        "n_binary_variables": int(model.NumBinVars),
        "max_fractionality": max_fractionality,

        "n_local_linear_variables": int(len(candidate_pairs) - len(pwl_specs)),
        "n_local_pwl_variables": int(len(pwl_specs)),
        **{k: v for k, v in candidate.items() if k not in {"candidate_mask", "cand_i", "cand_k", "candidate_pairs", "candidate_ub"}},
    }

    model.dispose()

    return allocation, info


# ============================================================
# 7. Extract optimization results
# ============================================================

def opt_build_outputs(data, allocation, solver_info, celltypes, alpha, enforce_global_hard_constraint):
    frame_df = data["frame_df"]
    n, expected, probs, N = data["n"], data["expected"], data["probs"], data["N"]

    patch_count = np.asarray(data["patch_matrix"].dot(allocation)) if data["patch_matrix"].shape[0] else np.empty((0, len(celltypes)))
    patch_ref, patch_meta = data["patch_reference"], data["patch_meta"]

    site_L1 = float(np.abs(allocation - expected).sum() / (2 * N))

    if len(patch_meta):
        patch_n = patch_meta["n_cells"].to_numpy(dtype=float)
        patch_L1 = np.abs(patch_count / patch_n[:, None] - patch_ref / patch_n[:, None]).sum(axis=1)
        mean_patch_L1 = float(patch_L1.mean())
    else:
        patch_L1, mean_patch_L1 = np.empty(0), 0.0

    objective = float((1 - alpha) * site_L1 + alpha * mean_patch_L1 / 2)

    extra = {
        "type_prop_fallback_global": data["fallback"].astype(bool),
        "allocation_total": allocation.sum(axis=1),
    }

    for k, ct in enumerate(celltypes):
        safe = ctp_safe_name(ct)
        extra[f"prop__normalized__{safe}"] = probs[:, k]
        extra[f"expected__{safe}"] = expected[:, k]
        extra[f"alloc__{safe}"] = allocation[:, k]
        extra[f"optimized_prop__{safe}"] = allocation[:, k] / n

    site_df = pd.concat([frame_df.reset_index(drop=True), pd.DataFrame(extra)], axis=1)

    optimized_global = allocation.sum(axis=0)

    global_df = pd.DataFrame({
        "model": data["model"], "pair": data["pair"], "image": data["image"], "image_index": data["image_index"],
        "interp_fraction": data["t"], "cell_type": celltypes,
        "global_hard_constraint_enabled": bool(enforce_global_hard_constraint),
        "left_real_prop": data["left_global"], "right_real_prop": data["right_global"],
        "global_reference_prop": data["global_reference"],
        "global_expected_count": data["global_expected"], "global_integer_target": data["global_target"],
        "optimized_count": optimized_global, "count_difference": optimized_global - data["global_target"],
        "optimized_prop": optimized_global / N,
    })

    patch_df = patch_meta.copy()

    if len(patch_df):
        patch_df["model"] = data["model"]; patch_df["pair"] = data["pair"]
        patch_df["image"] = data["image"]; patch_df["image_index"] = data["image_index"]
        patch_df["patch_L1"] = patch_L1

        for k, ct in enumerate(celltypes):
            safe = ctp_safe_name(ct)
            patch_df[f"reference_count__{safe}"] = patch_ref[:, k]
            patch_df[f"optimized_count__{safe}"] = patch_count[:, k].astype(np.int64)
            patch_df[f"reference_prop__{safe}"] = patch_ref[:, k] / patch_df["n_cells"]
            patch_df[f"optimized_prop__{safe}"] = patch_count[:, k] / patch_df["n_cells"]

    summary = {
        "model": data["model"], "pair": data["pair"], "image": data["image"], "image_index": data["image_index"],
        "interp_fraction": data["t"], "n_sites": len(frame_df), "n_cells": N,
        "n_fallback_sites": int(data["fallback"].sum()), "n_patches": len(patch_meta), "alpha": float(alpha),

        "global_hard_constraint_enabled": bool(enforce_global_hard_constraint),
        "global_constraint_satisfied": bool(np.array_equal(optimized_global, data["global_target"])),

        "candidate_topk": solver_info.get("candidate_topk", np.nan),
        "n_full_variables": solver_info.get("n_full_variables", np.nan),
        "n_candidate_variables": solver_info.get("n_candidate_variables", np.nan),
        "n_warm_support_added": solver_info.get("n_warm_support_added", np.nan),
        "candidate_density": solver_info.get("candidate_density", np.nan),
        "mean_candidates_per_site": solver_info.get("mean_candidates_per_site", np.nan),

        "site_L1_normalized": site_L1,
        "mean_patch_L1": mean_patch_L1,
        "objective": objective,

        "warm_start_runtime": solver_info.get("warm_runtime", 0),
        "model_build_runtime": solver_info.get("build_runtime", 0),

        "solver_status": solver_info["status"],
        "solver_objective": solver_info["objective"],
        "solver_runtime": solver_info["runtime"],
        "solver_wall_runtime": solver_info["wall_runtime"],
        "solver_mip_gap": solver_info["mip_gap"],
        "solver_obj_bound": solver_info["obj_bound"],
        "solver_node_count": solver_info["node_count"],
        "solver_n_variables": solver_info["n_variables"],
        "solver_n_constraints": solver_info["n_constraints"],
        "solver_n_integer_variables": solver_info.get("n_integer_variables", 0),
        "solver_n_binary_variables": solver_info.get("n_binary_variables", 0),

        "n_local_linear_variables": solver_info.get("n_local_linear_variables", 0),
        "n_local_pwl_variables": solver_info.get("n_local_pwl_variables", 0),

        "max_fractionality": solver_info["max_fractionality"],
    }

    return site_df, global_df, patch_df, summary


# ============================================================
# 8. Complete pipeline
# ============================================================

def opt_solve_selected(data, **kwargs):
    from vivid.optimization import solve_with_backend
    allocation, info = solve_with_backend(
        data, backend=config.get("optimization_solver", "gurobi"),
        gurobi_solver=opt_solve_gurobi, build_arcs=opt_build_network_arcs,
        build_candidates=opt_build_candidate_data, local_rounding=opt_local_rounding,
        **kwargs,
    )
    globals().setdefault("optimization_backends_used", []).append(info["solver"])
    return allocation, info


def optimize_celltypes(
    save_dir, interpolation_model_names, fused_celltype_prop_df, outputs, celltypes,
    integer_count_results=None, alpha=0.5, patch_size=128, patch_stride=None,
    enforce_global_hard_constraint=True, mip_gap=1e-2, time_limit=None,
    solver_threads=0, solver_output=False, solver_mip_focus=3,
    solver_lp_method=1, solver_mip_root_method=2, solver_presolve=-1,
    solver_integrality_tol=1e-6, output_folder_name=None
):
    if not 0 <= float(alpha) <= 1: raise ValueError("alpha  must be in  [0,1]")

    celltypes = [str(x) for x in celltypes]
    model_names = [str(x) for x in interpolation_model_names]
    candidate_topk = max(1, (len(celltypes) + 1) // 2)

    print(f"Cell types={len(celltypes)} | candidate top-k={candidate_topk}")

    df, prepared_frames = opt_prepare_all_frames(
        fused_celltype_prop_df, outputs, celltypes, model_names, float(alpha), patch_size, patch_stride
    )

    cell_count_check_df = (
        opt_check_cell_count_conservation(df, integer_count_results, model_names)
        if integer_count_results is not None else pd.DataFrame()
    )

    print(f"Prepared {len(prepared_frames)} frames | sites={len(df):,} | cells={int(df['cell_number'].sum()):,}")

    site_results, global_results, patch_results, summaries = [], [], [], []

    for data in tqdm(prepared_frames, desc="Integer allocation optimization"):
        allocation, solver_info = opt_solve_selected(
            data,
            alpha=float(alpha),
            enforce_global_hard_constraint=enforce_global_hard_constraint,
            mip_gap=mip_gap,
            time_limit=time_limit,
            threads=solver_threads,
            output=solver_output,
            mip_focus=solver_mip_focus,
            lp_method=solver_lp_method,
            mip_root_method=solver_mip_root_method,
            presolve=solver_presolve,
            integrality_tol=solver_integrality_tol,
        )

        site_df, global_df, patch_df, summary = opt_build_outputs(
            data, allocation, solver_info, celltypes, float(alpha), enforce_global_hard_constraint
        )

        summary["solver_backend"] = solver_info.get("solver", "gurobi")
        site_results.append(site_df); global_results.append(global_df)
        if len(patch_df): patch_results.append(patch_df)
        summaries.append(summary)

    final_site_type_df = pd.concat(site_results, ignore_index=True, copy=False) if site_results else pd.DataFrame()
    final_global_reference_df = pd.concat(global_results, ignore_index=True, copy=False) if global_results else pd.DataFrame()
    final_patch_reference_df = pd.concat(patch_results, ignore_index=True, copy=False) if patch_results else pd.DataFrame()
    final_optimization_summary_df = pd.DataFrame(summaries)

    alloc_cols = [f"alloc__{ctp_safe_name(ct)}" for ct in celltypes]

    if len(final_site_type_df):
        allocation_sum = final_site_type_df[alloc_cols].sum(axis=1).to_numpy(dtype=np.int64)
        original_n = final_site_type_df["cell_number"].to_numpy(dtype=np.int64)
        if not np.array_equal(allocation_sum, original_n): raise RuntimeError(" final site allocation row sum validation failed ")

    if output_folder_name is None:
        output_folder_name = "optimized_celltype_reconstruction_hardglobal" if enforce_global_hard_constraint else "optimized_celltype_reconstruction_no_global_hard"

    out_root = Path(save_dir).resolve() / output_folder_name
    out_root.mkdir(parents=True, exist_ok=True)

    final_site_type_df.to_csv(out_root / "optimized_site_celltypes.csv", index=False, encoding="utf-8-sig")
    final_global_reference_df.to_csv(out_root / "global_reference.csv", index=False, encoding="utf-8-sig")
    final_patch_reference_df.to_csv(out_root / "patch_reference.csv", index=False, encoding="utf-8-sig")
    final_optimization_summary_df.to_csv(out_root / "optimization_summary.csv", index=False, encoding="utf-8-sig")

    if len(cell_count_check_df): cell_count_check_df.to_csv(out_root / "cell_count_conservation_check.csv", index=False, encoding="utf-8-sig")

    config_out = {
        "alpha": float(alpha),
        "patch_size": int(patch_size),
        "patch_stride": int(patch_size // 2 if patch_stride is None else patch_stride),
        "global_hard_constraint_enabled": bool(enforce_global_hard_constraint),

        "candidate_rule": "top-half local probability + exact warm support",
        "candidate_topk": candidate_topk,

        "mip_gap": float(mip_gap),
        "solver_mip_focus": int(solver_mip_focus),
        "solver_lp_method": int(solver_lp_method),
        "solver_mip_root_method": int(solver_mip_root_method),
        "solver_presolve": int(solver_presolve),

        "celltypes": celltypes,
    }

    with open(out_root / "optimization_config.json", "w", encoding="utf-8") as f:
        json.dump(config_out, f, ensure_ascii=False, indent=2)

    print(
        f"Optimization finished | frames={len(final_optimization_summary_df)} | "
        f"sites={len(final_site_type_df):,} | cells={int(final_site_type_df['cell_number'].sum()):,} | output={out_root}"
    )

    return final_site_type_df, final_global_reference_df, final_patch_reference_df, final_optimization_summary_df


# ============================================================
# 9. Parameters
# ============================================================

optimization_celltypes = [str(x) for x in celltypes]
enforce_global_hard_constraint = bool(config["enforce_global_hard_constraint"])

patch_stride = None

solver_mip_gap = 1e-2
solver_time_limit = None
solver_threads = 0
solver_output = False

solver_mip_focus = 3
solver_lp_method = 1
solver_mip_root_method = 2
solver_presolve = -1
solver_integrality_tol = 1e-6


# ============================================================
# 10. Run
# ============================================================

final_site_type_df, final_global_reference_df, final_patch_reference_df, final_optimization_summary_df = optimize_celltypes(
    save_dir, interpolation_model_names, fused_celltype_prop_df, outputs, optimization_celltypes,
    integer_count_results=integer_count_results, alpha=optimization_alpha, patch_size=patch_size, patch_stride=patch_stride,
    enforce_global_hard_constraint=enforce_global_hard_constraint, mip_gap=solver_mip_gap,
    time_limit=solver_time_limit, solver_threads=solver_threads, solver_output=solver_output,
    solver_mip_focus=solver_mip_focus, solver_lp_method=solver_lp_method,
    solver_mip_root_method=solver_mip_root_method, solver_presolve=solver_presolve,
    solver_integrality_tol=solver_integrality_tol
)


# ============================================================
# 11. Per-model dictionaries
# ============================================================

final_site_type_dict = {str(m): g.reset_index(drop=True) for m, g in final_site_type_df.groupby("model", sort=False)} if len(final_site_type_df) else {}
final_global_reference_dict = {str(m): g.reset_index(drop=True) for m, g in final_global_reference_df.groupby("model", sort=False)} if len(final_global_reference_df) else {}
final_patch_reference_dict = {str(m): g.reset_index(drop=True) for m, g in final_patch_reference_df.groupby("model", sort=False)} if len(final_patch_reference_df) and "model" in final_patch_reference_df.columns else {}
final_optimization_summary_dict = {str(m): g.reset_index(drop=True) for m, g in final_optimization_summary_df.groupby("model", sort=False)} if len(final_optimization_summary_df) else {}


# ============================================================
# 12. Quick check
# ============================================================

