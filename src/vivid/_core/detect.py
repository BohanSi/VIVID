# Migrated from 1002genertation.ipynb, cell 39. See README.md.
import re, json, time, shutil
import numpy as np
import pandas as pd
from functools import lru_cache
from pathlib import Path
from PIL import Image
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from IPython.display import display


# ============================================================
# 1. BASIC UTILITIES
# ============================================================

def fg_natural_key(path):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)",Path(path).name)]

def fg_img_index(path):
    m=re.search(r"img(\d+)",Path(path).stem,flags=re.I)
    return int(m.group(1)) if m else None

def fg_build_gaussian_npz_lookup(gaussian_summary):
    lookup={}
    for slice_name,info in gaussian_summary.items():
        if info.get("npz_path") is None: continue
        stem=Path(info["npz_path"]).stem
        if stem in lookup: raise ValueError(f" duplicate  Gaussian stem: {stem}")
        lookup[stem]={"slice_name":str(slice_name),"npz_path":Path(info["npz_path"]).resolve()}
    return lookup

def fg_load_gaussian_density(npz_path):
    with np.load(npz_path) as f:
        if "gaussian_density" not in f.files: raise KeyError(f"{npz_path}:  missing  gaussian_density")
        density=np.asarray(f["gaussian_density"],dtype=np.float32)
    if density.ndim!=2: raise ValueError(f"{npz_path}: gaussian_density shape={density.shape}")
    return density

def fg_get_pair_frames(pair_info):
    frames=pair_info.get("frames",pair_info.get("selected_frames",None))
    if frames is None: raise KeyError("global_flow_results[pair]  missing  frames / selected_frames")
    return frames

def fg_get_frame_image_path(pair_info,frame,image_index):
    for key in ["image_file","image_path","intermediate_image","intermediate_image_path"]:
        if frame.get(key) is not None:
            p=Path(frame[key])
            if p.exists(): return p.resolve()
    for key in ["output_dir","pair_dir","result_dir"]:
        if pair_info.get(key) is not None:
            p=Path(pair_info[key])/f"img{int(image_index)}.png"
            if p.exists(): return p.resolve()
    flow_file=Path(frame["flow_file"])
    candidates=[flow_file.parent.parent/f"img{int(image_index)}.png",flow_file.parent.parent.parent/f"img{int(image_index)}.png"]
    for p in candidates:
        if p.exists(): return p.resolve()
    raise FileNotFoundError(f" cannot find  img{image_index}  corresponding intermediate image .")


# ============================================================
# 2. BILINEAR SAMPLING
# ============================================================

def fg_bilinear_sample(image,x,y):
    image=np.asarray(image,dtype=np.float32); x=np.asarray(x,dtype=np.float32); y=np.asarray(y,dtype=np.float32)
    if image.ndim!=2: raise ValueError(f"image  must be  H×W,  current ={image.shape}")
    if x.shape!=y.shape: raise ValueError(f"x/y shape inconsistent : {x.shape} vs {y.shape}")

    H,W=image.shape
    finite=np.isfinite(x)&np.isfinite(y); valid=finite&(x>=0)&(x<=W-1)&(y>=0)&(y<=H-1)
    sampled=np.zeros(x.shape,dtype=np.float32)
    if not np.any(valid): return sampled,valid

    xv=x[valid].astype(np.float64); yv=y[valid].astype(np.float64)
    x0=np.floor(xv).astype(np.int64); y0=np.floor(yv).astype(np.int64)
    x1=np.minimum(x0+1,W-1); y1=np.minimum(y0+1,H-1)
    wx=xv-x0; wy=yv-y0

    sampled[valid]=((1-wx)*(1-wy)*image[y0,x0]+wx*(1-wy)*image[y0,x1]+(1-wx)*wy*image[y1,x0]+wx*wy*image[y1,x1]).astype(np.float32)
    return sampled,valid


# ============================================================
# 3. FLOW -> REAL GAUSSIAN -> INTERMEDIATE DENSITY
# ============================================================

def fg_build_one_intermediate_density(flow_file,left_density,right_density,blend_mode="distance",renormalize_invalid=True,eps=1e-12):
    flow_file=Path(flow_file)

    with np.load(flow_file) as f:
        required=["flow_to_left","flow_to_right","ratio"]; missing=[k for k in required if k not in f.files]
        if missing: raise KeyError(f"{flow_file}:  missing  {missing}")

        flow_left=np.asarray(f["flow_to_left"],dtype=np.float32); flow_right=np.asarray(f["flow_to_right"],dtype=np.float32); ratio=float(f["ratio"])
        valid_left_flow=np.asarray(f["valid_to_left"],dtype=bool) if "valid_to_left" in f.files else np.isfinite(flow_left).all(axis=-1)
        valid_right_flow=np.asarray(f["valid_to_right"],dtype=bool) if "valid_to_right" in f.files else np.isfinite(flow_right).all(axis=-1)
        mask_left=np.asarray(f["mask_to_left"],dtype=np.float32) if "mask_to_left" in f.files else None
        mask_right=np.asarray(f["mask_to_right"],dtype=np.float32) if "mask_to_right" in f.files else None

    if flow_left.ndim!=3 or flow_left.shape[-1]!=2: raise ValueError(f"flow_to_left shape={flow_left.shape}")
    if flow_right.shape!=flow_left.shape: raise ValueError(" left and right  FLOW shape  inconsistent ")

    H,W=flow_left.shape[:2]
    if left_density.shape!=(H,W): raise ValueError(f"left_density={left_density.shape}, flow={(H,W)}")
    if right_density.shape!=(H,W): raise ValueError(f"right_density={right_density.shape}, flow={(H,W)}")

    base_x=np.arange(W,dtype=np.float32)[None,:]; base_y=np.arange(H,dtype=np.float32)[:,None]
    left_x=flow_left[:,:,0]+base_x; left_y=flow_left[:,:,1]+base_y
    right_x=flow_right[:,:,0]+base_x; right_y=flow_right[:,:,1]+base_y

    sampled_left,coord_valid_left=fg_bilinear_sample(left_density,left_x,left_y)
    sampled_right,coord_valid_right=fg_bilinear_sample(right_density,right_x,right_y)
    valid_left=valid_left_flow&coord_valid_left; valid_right=valid_right_flow&coord_valid_right

    if blend_mode=="distance":
        wl=valid_left.astype(np.float32)*(1.0-ratio); wr=valid_right.astype(np.float32)*ratio
    elif blend_mode=="mask":
        if mask_left is None or mask_right is None: raise KeyError(f"{flow_file}: blend_mode='mask'  but missing  mask_to_left/right")
        if mask_left.shape!=(H,W) or mask_right.shape!=(H,W): raise ValueError("mask shape  and  FLOW  inconsistent ")
        wl=np.clip(mask_left,0.0,1.0)*valid_left.astype(np.float32); wr=np.clip(mask_right,0.0,1.0)*valid_right.astype(np.float32)
    else: raise ValueError("blend_mode  must be  'distance'  or  'mask'")

    denom=wl+wr; target=np.zeros((H,W),dtype=np.float32); valid_any=denom>eps
    if renormalize_invalid: target[valid_any]=(wl[valid_any]*sampled_left[valid_any]+wr[valid_any]*sampled_right[valid_any])/denom[valid_any]
    else: target=wl*sampled_left+wr*sampled_right

    target=np.maximum(target,0.0).astype(np.float32); target_sum=float(target.sum())

    return {
        "target_density":target,"left_sampled_density":sampled_left,"right_sampled_density":sampled_right,
        "left_x":left_x,"left_y":left_y,"right_x":right_x,"right_y":right_y,
        "left_valid":valid_left,"right_valid":valid_right,"left_weight":wl,"right_weight":wr,
        "ratio":ratio,"blend_mode":blend_mode,"estimated_count_float":target_sum,"estimated_count":int(np.floor(target_sum+0.5))
    }


# ============================================================
# 4. BUILD ALL FLOW-MIGRATED GAUSSIAN MAPS
# ============================================================

def build_all_intermediate_gaussian_targets(global_flow_results,gaussian_summary,save_dir,blend_mode="distance",visualization_vmax=None,save_mapping_arrays=True,target_image_indices=None,verbose=True):
    if blend_mode not in ["distance","mask"]: raise ValueError("blend_mode  must be  distance  or  mask")
    if not save_mapping_arrays: raise ValueError(" this pipeline requires  save_mapping_arrays=True")

    gaussian_lookup=fg_build_gaussian_npz_lookup(gaussian_summary)
    out_root=Path(save_dir).resolve()/"intermediate_gaussian_density"/blend_mode; out_root.mkdir(parents=True,exist_ok=True)
    results={}; target_image_indices=None if target_image_indices is None else set(int(x) for x in target_image_indices)
    if verbose: print("\nTarget image filtering:",sorted(target_image_indices) if target_image_indices is not None else "ALL")

    for pair_name,pair_info in global_flow_results.items():
        left_stem=Path(pair_info["left_image"]).stem; right_stem=Path(pair_info["right_image"]).stem
        if left_stem not in gaussian_lookup: raise KeyError(f"{pair_name}: Gaussian  not found in  {left_stem}")
        if right_stem not in gaussian_lookup: raise KeyError(f"{pair_name}: Gaussian  not found in  {right_stem}")

        left_info=gaussian_lookup[left_stem]; right_info=gaussian_lookup[right_stem]
        left_density=fg_load_gaussian_density(left_info["npz_path"]); right_density=fg_load_gaussian_density(right_info["npz_path"])
        pair_out=out_root/pair_name; pair_out.mkdir(parents=True,exist_ok=True); pair_results=[]

        if verbose:
            print("\n"+"="*100); print("PAIR :",pair_name); print("Left :",left_info["slice_name"])
            print("Right:",right_info["slice_name"]); print("Blend:",blend_mode); print("="*100)

        frames=fg_get_pair_frames(pair_info)

        if target_image_indices is not None:
            available_indices={int(frame["image_index"]) for frame in frames}; missing=target_image_indices-available_indices
            if missing: raise KeyError(f"{pair_name}:  cannot find target frames: {sorted(missing)}; available={sorted(available_indices)}")
            frames=[frame for frame in frames if int(frame["image_index"]) in target_image_indices]

        for frame in frames:
            image_index=int(frame["image_index"]); flow_file=Path(frame["flow_file"])
            image_path=fg_get_frame_image_path(pair_info,frame,image_index)

            mapped=fg_build_one_intermediate_density(flow_file,left_density,right_density,blend_mode=blend_mode,renormalize_invalid=True)
            target=mapped["target_density"]; stem=f"img{image_index}"; npz_path=pair_out/f"{stem}.npz"; png_path=pair_out/f"{stem}.png"

            save_dict={
                "target_density":target.astype(np.float32),"ratio":np.array(mapped["ratio"],dtype=np.float32),"blend_mode":np.array(str(blend_mode)),
                "estimated_count_float":np.array(mapped["estimated_count_float"],dtype=np.float64),"estimated_count":np.array(mapped["estimated_count"],dtype=np.int64),
                "left_slice_name":np.array(left_info["slice_name"]),"right_slice_name":np.array(right_info["slice_name"])
            }

            if save_mapping_arrays:
                save_dict.update({
                    "left_sampled_density":mapped["left_sampled_density"].astype(np.float32),"right_sampled_density":mapped["right_sampled_density"].astype(np.float32),
                    "left_x":mapped["left_x"].astype(np.float32),"left_y":mapped["left_y"].astype(np.float32),
                    "right_x":mapped["right_x"].astype(np.float32),"right_y":mapped["right_y"].astype(np.float32),
                    "left_valid":mapped["left_valid"].astype(bool),"right_valid":mapped["right_valid"].astype(bool),
                    "left_weight":mapped["left_weight"].astype(np.float32),"right_weight":mapped["right_weight"].astype(np.float32)
                })

            np.savez_compressed(npz_path,**save_dict)

            vmax=float(target.max()) if visualization_vmax is None else float(visualization_vmax)
            gray=np.rint(np.clip(target/vmax,0,1)*255).astype(np.uint8) if vmax>0 else np.zeros(target.shape,dtype=np.uint8)
            Image.fromarray(np.repeat(gray[:,:,None],3,axis=2),mode="RGB").save(png_path)

            record={
                "pair":pair_name,"image_index":image_index,"image":image_path.name,"intermediate_image_path":image_path,"flow_file":flow_file,
                "ratio":mapped["ratio"],"interp_fraction":mapped["ratio"],"blend_mode":blend_mode,
                "target_npz_path":npz_path,"target_png_path":png_path,
                "estimated_count_float":mapped["estimated_count_float"],"estimated_count":mapped["estimated_count"],
                "density_sum":mapped["estimated_count_float"],"density_max":float(target.max())
            }

            pair_results.append(record)
            if verbose: print(f"img{image_index} | t={mapped['ratio']:.4f} | FLOW KDE sum={mapped['estimated_count_float']:.4f} | global cells={mapped['estimated_count']:,} | image={image_path.name}")

            del mapped,target,save_dict

        results[pair_name]=pair_results

    return results,out_root


# ============================================================
# 5-8. RGB / MORPHOLOGY UTILITIES
# ============================================================

def gd_load_detection_rgb(path):
    return (np.asarray(Image.open(path).convert("RGB"),dtype=np.float32)/255.0).astype(np.float32)

def gd_resolve_background_mode(save_dir,background_mode=None):
    if background_mode is not None: mode=str(background_mode).lower()
    else:
        meta_path=Path(save_dir).resolve()/"gaussian_density_image"/"gaussian_meta.json"; mode="white"
        if meta_path.exists():
            with open(meta_path,"r",encoding="utf-8") as f: mode=str(json.load(f).get("background_mode","white")).lower()
    if mode not in ["black","white"]: raise ValueError("background_mode  must be  black  or  white")
    return mode

def gd_rgb_foreground_score(rgb,background_mode="white"):
    rgb=np.clip(np.asarray(rgb,dtype=np.float32),0.0,1.0)
    if background_mode=="white": score=np.linalg.norm(1.0-rgb,axis=2)/np.sqrt(3.0)
    elif background_mode=="black": score=np.linalg.norm(rgb,axis=2)/np.sqrt(3.0)
    else: raise ValueError("background_mode  must be  white  or  black")
    return score.astype(np.float32)

def gd_rgb_color_gradient(rgb):
    rgb=np.clip(np.asarray(rgb,dtype=np.float32),0.0,1.0); H,W,_=rgb.shape
    grad=np.zeros((H,W),dtype=np.float32)
    diff_ud=np.linalg.norm(rgb[1:]-rgb[:-1],axis=2); grad[1:]=np.maximum(grad[1:],diff_ud); grad[:-1]=np.maximum(grad[:-1],diff_ud)
    diff_lr=np.linalg.norm(rgb[:,1:]-rgb[:,:-1],axis=2); grad[:,1:]=np.maximum(grad[:,1:],diff_lr); grad[:,:-1]=np.maximum(grad[:,:-1],diff_lr)
    return grad

def gd_structure(connectivity=8):
    if connectivity==4: return np.array([[0,1,0],[1,1,1],[0,1,0]],dtype=bool)
    if connectivity==8: return np.ones((3,3),dtype=bool)
    raise ValueError("connectivity must be 4 or 8")

def gd_remove_small_components(mask,min_area=1,connectivity=8):
    if min_area<=1: return mask.copy()
    labels,n=ndi.label(mask,structure=gd_structure(connectivity))
    if n==0: return mask.copy()
    areas=np.bincount(labels.ravel(),minlength=n+1); keep=np.flatnonzero(areas>=int(min_area)); keep=keep[keep!=0]
    return np.isin(labels,keep)

def gd_normalize_in_mask(x,mask,eps=1e-8):
    out=np.zeros_like(x,dtype=np.float32); vals=x[mask]
    if vals.size==0: return out
    vmin=float(vals.min()); vmax=float(vals.max())
    if vmax-vmin<=eps: out[mask]=1.0
    else: out[mask]=(vals-vmin)/(vmax-vmin)
    return out


# ============================================================
# 9. PEAK MARKERS
# ============================================================

def gd_build_markers(score,fg_mask,peak_min_distance=1,peak_threshold_abs=0.05,peak_smooth_sigma=0.8,connectivity=4):
    score_peak=ndi.gaussian_filter(score,sigma=float(peak_smooth_sigma)) if peak_smooth_sigma is not None and peak_smooth_sigma>0 else score
    score_norm=gd_normalize_in_mask(score_peak,fg_mask)

    peak_coords=peak_local_max(score_norm,min_distance=int(peak_min_distance),threshold_abs=float(peak_threshold_abs),
                                labels=fg_mask.astype(np.uint8),exclude_border=False)

    markers=np.zeros(score.shape,dtype=np.int32)
    if len(peak_coords): markers[peak_coords[:,0],peak_coords[:,1]]=np.arange(1,len(peak_coords)+1,dtype=np.int32)

    cc,ncc=ndi.label(fg_mask,structure=gd_structure(connectivity))
    if ncc==0: return markers

    has_marker=np.zeros(ncc+1,dtype=bool); marker_cc=np.unique(cc[markers>0]); has_marker[marker_cc]=True
    marker_id=int(markers.max())+1

    for cid in np.where(~has_marker[1:])[0]+1:
        ys,xs=np.where(cc==cid)
        if len(ys)==0: continue
        k=int(np.argmax(score[ys,xs])); markers[int(ys[k]),int(xs[k])]=marker_id; marker_id+=1

    return markers


# ============================================================
# 10. RGB WATERSHED
# ============================================================

def gd_watershed_rgb(detect_rgb,background_mode="white",fg_threshold=0.01,min_fg_component_area=1,
                     peak_min_distance=1,peak_threshold_abs=0.05,peak_smooth_sigma=0.8,
                     color_grad_weight=1.5,connectivity=4,crop_to_foreground=True,crop_pad=8):

    detect_rgb=np.clip(np.asarray(detect_rgb,dtype=np.float32),0.0,1.0)
    if detect_rgb.ndim!=3 or detect_rgb.shape[2]!=3: raise ValueError(f"detect_rgb  must be  H×W×3,  current ={detect_rgb.shape}")

    H,W,_=detect_rgb.shape
    score_full=gd_rgb_foreground_score(detect_rgb,background_mode)
    fg_full=score_full>float(fg_threshold)
    fg_full=gd_remove_small_components(fg_full,min_fg_component_area,connectivity)

    if not np.any(fg_full):
        return np.zeros((H,W),dtype=np.int32),score_full,fg_full,{"input_foreground_pixels":0,"n_markers":0,"n_watershed_regions":0}

    if crop_to_foreground:
        ys,xs=np.where(fg_full)
        y0=max(0,int(ys.min())-int(crop_pad)); y1=min(H,int(ys.max())+int(crop_pad)+1)
        x0=max(0,int(xs.min())-int(crop_pad)); x1=min(W,int(xs.max())+int(crop_pad)+1)
    else: y0,y1,x0,x1=0,H,0,W

    score=score_full[y0:y1,x0:x1]; fg=fg_full[y0:y1,x0:x1]; rgb_crop=detect_rgb[y0:y1,x0:x1]
    markers=gd_build_markers(score,fg,peak_min_distance,peak_threshold_abs,peak_smooth_sigma,connectivity)
    n_markers=int(markers.max())

    if n_markers==0:
        return np.zeros((H,W),dtype=np.int32),score_full,fg_full,{"input_foreground_pixels":int(fg_full.sum()),"n_markers":0,"n_watershed_regions":0}

    score_norm=gd_normalize_in_mask(score,fg)

    if color_grad_weight is not None and color_grad_weight>0:
        color_grad=gd_rgb_color_gradient(rgb_crop); color_grad_norm=gd_normalize_in_mask(color_grad,fg)
        elevation=(1.0-score_norm)+float(color_grad_weight)*color_grad_norm
    else: elevation=1.0-score_norm

    labels_crop=watershed(elevation, markers=markers, mask=fg, ).astype(np.int32)
    labels=np.zeros((H,W),dtype=np.int32); labels[y0:y1,x0:x1]=labels_crop

    info={"input_foreground_pixels":int(fg_full.sum()),"n_markers":n_markers,"n_watershed_regions":int(labels_crop.max())}
    return labels,score_full,fg_full,info


# ============================================================
# 11. REGION -> CENTER
# ============================================================

def gd_extract_region_centers(labels,score,min_region_area=1):
    labels=np.asarray(labels,dtype=np.int32); score=np.asarray(score,dtype=np.float32)
    H,W=labels.shape; n=int(labels.max())

    if n==0: return np.empty((0,2),dtype=np.int64),np.empty(0,dtype=np.int32)

    flat_lab=labels.ravel(); flat_score=score.ravel(); valid=flat_lab>0
    ids=flat_lab[valid]; vals=flat_score[valid]; inds=np.flatnonzero(valid)

    areas=np.bincount(ids,minlength=n+1); keep=areas>=int(min_region_area); keep[0]=False
    max_scores=np.full(n+1,-np.inf,dtype=np.float32); np.maximum.at(max_scores,ids,vals)

    candidate=keep[ids]&(vals==max_scores[ids])
    cand_inds=inds[candidate]; cand_ids=ids[candidate]

    if len(cand_inds)==0: return np.empty((0,2),dtype=np.int64),np.empty(0,dtype=np.int32)

    order=np.argsort(cand_ids,kind="mergesort"); cand_inds=cand_inds[order]; cand_ids=cand_ids[order]
    _,first=np.unique(cand_ids,return_index=True)

    center_inds=cand_inds[first]; region_ids=cand_ids[first]
    y,x=np.unravel_index(center_inds,(H,W))
    return np.column_stack([y,x]).astype(np.int64),region_ids.astype(np.int32)


# ============================================================
# 12-13. FAST NEIGHBORHOOD
# ============================================================

@lru_cache(maxsize=32)
def _gd_make_disk_offsets_cached(radius_pixels):
    r=float(radius_pixels); ir=int(np.ceil(r))
    dy,dx=np.mgrid[-ir:ir+1,-ir:ir+1]
    dist=np.sqrt(dx.astype(np.float64)**2+dy.astype(np.float64)**2)
    keep=dist<=r+1e-8
    return np.column_stack([dy[keep],dx[keep],dist[keep]]).astype(np.float64)

def gd_make_disk_offsets(radius_pixels):
    return _gd_make_disk_offsets_cached(float(radius_pixels)).copy()

def gd_expand_neighborhood_pixels(shape,centers_yx,offsets):
    H,W=map(int,shape); centers=np.asarray(centers_yx,dtype=np.int64); offsets=np.asarray(offsets,dtype=np.float64)
    n_sites=len(centers); n_offsets=len(offsets)

    if n_sites==0 or n_offsets==0:
        owner_count=np.zeros((H,W),dtype=np.uint16); empty_i=np.empty(0,dtype=np.int64); empty_f=np.empty(0,dtype=np.float64)
        return empty_i,empty_i.copy(),empty_i.copy(),empty_i.copy(),empty_i.copy(),empty_f,owner_count

    dy0=offsets[:,0].astype(np.int64); dx0=offsets[:,1].astype(np.int64); dist0=offsets[:,2].astype(np.float64)

    site_ids=np.repeat(np.arange(n_sites,dtype=np.int64),n_offsets)
    dy=np.tile(dy0,n_sites); dx=np.tile(dx0,n_sites); dist=np.tile(dist0,n_sites)

    yy=np.repeat(centers[:,0],n_offsets)+dy
    xx=np.repeat(centers[:,1],n_offsets)+dx

    valid=(yy>=0)&(yy<H)&(xx>=0)&(xx<W)

    if not np.all(valid):
        site_ids=site_ids[valid]; yy=yy[valid]; xx=xx[valid]
        dy=dy[valid]; dx=dx[valid]; dist=dist[valid]

    flat_idx=yy*W+xx
    counts=np.bincount(flat_idx,minlength=H*W)

    max_owner=int(counts.max()) if counts.size else 0
    if max_owner>np.iinfo(np.uint16).max: raise OverflowError(f"owner_count={max_owner}  exceeds  uint16  upper bound ")

    owner_count=counts.reshape(H,W).astype(np.uint16,copy=False)
    return site_ids,yy,xx,dy,dx,dist,owner_count

def gd_build_shared_owner_count(shape,centers_yx,radius_pixels,use_neighborhood_pixels=True):
    offsets=gd_make_disk_offsets(radius_pixels) if use_neighborhood_pixels else np.array([[0.0,0.0,0.0]],dtype=np.float64)
    _,_,_,_,_,_,owner_count=gd_expand_neighborhood_pixels(shape,centers_yx,offsets)
    return owner_count,offsets


# ============================================================
# 14. ROUNDING
# ============================================================

def gd_round_half_up_nonnegative(x):
    x=np.asarray(x,dtype=np.float64)
    if np.any(~np.isfinite(x)): raise ValueError("Gaussian density  contains  NaN  or  Inf")
    if np.any(x<0): raise ValueError("Gaussian density  must be nonnegative ")
    return np.floor(x+0.5).astype(np.int64)

def gd_balanced_round_to_total(expected_mass,target_total,tie_break_score=None):
    expected_mass=np.asarray(expected_mass,dtype=np.float64); target_total=int(target_total)

    if expected_mass.ndim!=1: raise ValueError("expected_mass  must be a one-dimensional array ")
    if np.any(~np.isfinite(expected_mass)): raise ValueError("expected_mass  contains  NaN / Inf")
    if np.any(expected_mass<0): raise ValueError("expected_mass  must be nonnegative ")
    if target_total<0: raise ValueError("target_total  must be nonnegative ")

    n=len(expected_mass)

    if n==0:
        if target_total!=0: raise ValueError(f" missing site,  but global target_total={target_total}")
        return np.empty(0,dtype=np.int64)

    base=np.floor(expected_mass).astype(np.int64)
    frac=expected_mass-base
    remaining=int(target_total-base.sum())

    if remaining<0 or remaining>n:
        raise RuntimeError(f"Balanced rounding failed : target_total={target_total}, floor_sum={int(base.sum())}, remaining={remaining}, n_sites={n}, expected_sum={expected_mass.sum():.8f}")

    if remaining==0: return base

    if tie_break_score is None: tie_break_score=np.zeros(n,dtype=np.float64)
    else:
        tie_break_score=np.asarray(tie_break_score,dtype=np.float64)
        if tie_break_score.shape!=(n,): raise ValueError("tie_break_score shape inconsistent ")
        tie_break_score=np.nan_to_num(tie_break_score,nan=-np.inf,posinf=np.inf,neginf=-np.inf)

    site_order=np.arange(n,dtype=np.int64)
    order=np.lexsort((site_order,-tie_break_score,-frac))
    base[order[:remaining]]+=1

    if int(base.sum())!=target_total: raise RuntimeError(f"Balanced rounding resulting total ={int(base.sum())},  target ={target_total}")
    return base


# ============================================================
# 15. FLOW PIXEL UTILITIES
# ============================================================

def gd_continuous_to_nearest_pixel(x,y,W,H):
    x=np.asarray(x,dtype=np.float64); y=np.asarray(y,dtype=np.float64)
    finite=np.isfinite(x)&np.isfinite(y); valid=finite&(x>=0)&(x<=W-1)&(y>=0)&(y<=H-1)

    u=np.zeros(len(x),dtype=np.int64); v=np.zeros(len(y),dtype=np.int64)

    if np.any(finite):
        u[finite]=np.floor(x[finite]+0.5).astype(np.int64)
        v[finite]=np.floor(y[finite]+0.5).astype(np.int64)

    return np.clip(u,0,W-1),np.clip(v,0,H-1),valid

@lru_cache(maxsize=128)
def _gd_endpoint_indices_cached(parent_str):
    parent=Path(parent_str); indices=[]

    for p in parent.glob("img*.png"):
        idx=fg_img_index(p)
        if idx is not None: indices.append(idx)

    if len(indices)>=2: return int(min(indices)),int(max(indices))
    return 0,0

def gd_endpoint_indices_from_intermediate_image(image_path):
    return _gd_endpoint_indices_cached(str(Path(image_path).parent.resolve()))

def _gd_bincount(site_ids,values,n_sites):
    return np.bincount(site_ids,weights=np.asarray(values,dtype=np.float64),minlength=n_sites)

def _gd_bool_count(site_ids,mask,n_sites):
    return np.bincount(site_ids,weights=np.asarray(mask,dtype=np.int8),minlength=n_sites).astype(np.int64)

def _gd_unique_valid_support_counts_by_site(site_ids,u,v,valid_mask,support_mask,n_sites,H,W):
    valid_mask=np.asarray(valid_mask,dtype=bool); support_mask=np.asarray(support_mask,dtype=bool)
    zeros=np.zeros(n_sites,dtype=np.int64)

    if n_sites==0 or not np.any(valid_mask): return zeros.copy(),zeros.copy()

    idx=np.flatnonzero(valid_mask)
    sid=site_ids[idx].astype(np.int64,copy=False)
    code=v[idx].astype(np.int64,copy=False)*int(W)+u[idx].astype(np.int64,copy=False)

    pixel_space=np.int64(H)*np.int64(W)
    combined=sid*pixel_space+code

    unique_combined,inverse=np.unique(combined,return_inverse=True)
    unique_sid=(unique_combined//pixel_space).astype(np.int64,copy=False)

    valid_unique=np.bincount(unique_sid,minlength=n_sites).astype(np.int64)
    unique_has_support=np.zeros(len(unique_combined),dtype=bool)

    np.logical_or.at(unique_has_support,inverse,support_mask[idx])

    support_sid=unique_sid[unique_has_support]
    support_unique=np.bincount(support_sid,minlength=n_sites).astype(np.int64) if len(support_sid) else zeros.copy()

    return valid_unique,support_unique

def _gd_const_object(value,n):
    out=np.empty(int(n),dtype=object); out[:]=value
    return out


# ============================================================
# 16. FAST ONE FRAME
#



# ============================================================

def gd_build_flow_mapped_site_tables(
    pair_name,frame,model_name="RIFE",
    use_neighborhood_pixels=True,support_radius_pixels=4.0,
    background_mode="white",detect_fg_threshold=0.01,min_fg_component_area=1,
    peak_min_distance=1,peak_threshold_abs=0.05,peak_smooth_sigma=0.8,
    color_grad_weight=1.5,connectivity=4,min_region_area=1,
    crop_to_foreground=True,crop_pad=8,real_density_threshold=0.0,output_nonwhite_threshold=0.03,
):
    image_index=int(frame["image_index"]); image_path=Path(frame["intermediate_image_path"]); ratio=float(frame["ratio"])

    detect_rgb=gd_load_detection_rgb(image_path)

    labels,detect_score,fg_mask,watershed_info=gd_watershed_rgb(
        detect_rgb=detect_rgb,background_mode=background_mode,
        fg_threshold=detect_fg_threshold,min_fg_component_area=min_fg_component_area,
        peak_min_distance=peak_min_distance,peak_threshold_abs=peak_threshold_abs,
        peak_smooth_sigma=peak_smooth_sigma,color_grad_weight=color_grad_weight,
        connectivity=connectivity,crop_to_foreground=crop_to_foreground,crop_pad=crop_pad)

    centers_yx,region_ids=gd_extract_region_centers(labels=labels,score=detect_score,min_region_area=min_region_area)

    with np.load(frame["target_npz_path"]) as f:
        needed=["target_density","left_sampled_density","right_sampled_density","left_x","left_y","right_x","right_y","left_valid","right_valid"]
        missing=[k for k in needed if k not in f.files]
        if missing: raise KeyError(f"{frame['target_npz_path']}:  missing  {missing}")

        target=np.asarray(f["target_density"],dtype=np.float32)
        left_density=np.asarray(f["left_sampled_density"],dtype=np.float32); right_density=np.asarray(f["right_sampled_density"],dtype=np.float32)

        left_x=np.asarray(f["left_x"],dtype=np.float32); left_y=np.asarray(f["left_y"],dtype=np.float32)
        right_x=np.asarray(f["right_x"],dtype=np.float32); right_y=np.asarray(f["right_y"],dtype=np.float32)

        left_valid=np.asarray(f["left_valid"],dtype=bool); right_valid=np.asarray(f["right_valid"],dtype=bool)

        left_weight=np.asarray(f["left_weight"],dtype=np.float32) if "left_weight" in f.files else np.full(target.shape,1.0-ratio,dtype=np.float32)
        right_weight=np.asarray(f["right_weight"],dtype=np.float32) if "right_weight" in f.files else np.full(target.shape,ratio,dtype=np.float32)

    H,W=target.shape

    if detect_rgb.shape[:2]!=(H,W): raise ValueError(f" interpolation RGB shape={detect_rgb.shape[:2]}  and  FLOW Gaussian shape={(H,W)}  inconsistent ")

    offsets=gd_make_disk_offsets(support_radius_pixels) if use_neighborhood_pixels else np.array([[0.0,0.0,0.0]],dtype=np.float64)
    site_ids,yy,xx,dy,dx,dist,owner_count=gd_expand_neighborhood_pixels((H,W),centers_yx,offsets)

    n_sites=len(centers_yx); n_pixels=len(site_ids)
    left_real_index,right_real_index=gd_endpoint_indices_from_intermediate_image(image_path)

    center_y=centers_yx[:,0].astype(np.int64); center_x=centers_yx[:,1].astype(np.int64)

    if n_pixels:
        owners=owner_count[yy,xx].astype(np.int64)
        if np.any(owners<=0): raise RuntimeError(" found  support pixel owner_count <= 0")

        share=1.0/owners.astype(np.float64)

        raw_density=target[yy,xx].astype(np.float64)
        allocated_density=raw_density*share

        lden=left_density[yy,xx].astype(np.float64); rden=right_density[yy,xx].astype(np.float64)
        lv=left_valid[yy,xx]; rv=right_valid[yy,xx]

        lx=left_x[yy,xx].astype(np.float64); ly=left_y[yy,xx].astype(np.float64)
        rx=right_x[yy,xx].astype(np.float64); ry=right_y[yy,xx].astype(np.float64)

        lu,lv_pix,lcoord=gd_continuous_to_nearest_pixel(lx,ly,W,H)
        ru,rv_pix,rcoord=gd_continuous_to_nearest_pixel(rx,ry,W,H)

        lmap=lv&lcoord; rmap=rv&rcoord

        l_support=lmap&(lden>float(real_density_threshold))
        r_support=rmap&(rden>float(real_density_threshold))

        center_y_px=center_y[site_ids]; center_x_px=center_x[site_ids]

        is_center=(yy==center_y_px)&(xx==center_x_px)
        is_shared=owners>1

        pixel_rgb=detect_rgb[yy,xx].astype(np.float64)
        pixel_detection_score=detect_score[yy,xx].astype(np.float64)

        left_weight_px=left_weight[yy,xx].astype(np.float64)
        right_weight_px=right_weight[yy,xx].astype(np.float64)

    else:
        owners=np.empty(0,dtype=np.int64); share=np.empty(0,dtype=np.float64)
        raw_density=np.empty(0,dtype=np.float64); allocated_density=np.empty(0,dtype=np.float64)

        lden=rden=lx=ly=rx=ry=np.empty(0,dtype=np.float64)
        lv=rv=lmap=rmap=l_support=r_support=is_center=is_shared=np.empty(0,dtype=bool)

        lu=lv_pix=ru=rv_pix=np.empty(0,dtype=np.int64)

        pixel_rgb=np.empty((0,3),dtype=np.float64)
        pixel_detection_score=np.empty(0,dtype=np.float64)
        left_weight_px=right_weight_px=np.empty(0,dtype=np.float64)

    n_support_pixels=np.bincount(site_ids,minlength=n_sites).astype(np.int64) if n_sites else np.empty(0,dtype=np.int64)
    n_shared_pixels=_gd_bool_count(site_ids,is_shared,n_sites) if n_sites else np.empty(0,dtype=np.int64)

    support_raw_density_sum=_gd_bincount(site_ids,raw_density,n_sites) if n_sites else np.empty(0,dtype=np.float64)
    support_allocated_density_sum=_gd_bincount(site_ids,allocated_density,n_sites) if n_sites else np.empty(0,dtype=np.float64)
    support_effective_pixel_count=_gd_bincount(site_ids,share,n_sites) if n_sites else np.empty(0,dtype=np.float64)

    support_max_density=np.zeros(n_sites,dtype=np.float64)
    if n_pixels: np.maximum.at(support_max_density,site_ids,raw_density)

    support_mean_density=np.divide(support_raw_density_sum,n_support_pixels,out=np.zeros(n_sites,dtype=np.float64),where=n_support_pixels>0)
    fraction_shared_pixels=np.divide(n_shared_pixels,n_support_pixels,out=np.zeros(n_sites,dtype=np.float64),where=n_support_pixels>0)

    neighborhood_mass=support_allocated_density_sum.copy()
    independent_count=gd_round_half_up_nonnegative(neighborhood_mass) if n_sites else np.empty(0,dtype=np.int64)

    n_left_valid_mapped_pixels=_gd_bool_count(site_ids,lmap,n_sites) if n_sites else np.empty(0,dtype=np.int64)
    n_right_valid_mapped_pixels=_gd_bool_count(site_ids,rmap,n_sites) if n_sites else np.empty(0,dtype=np.int64)

    n_left_density_support_pixels=_gd_bool_count(site_ids,l_support,n_sites) if n_sites else np.empty(0,dtype=np.int64)
    n_right_density_support_pixels=_gd_bool_count(site_ids,r_support,n_sites) if n_sites else np.empty(0,dtype=np.int64)

    left_has_real_support=n_left_density_support_pixels>0
    right_has_real_support=n_right_density_support_pixels>0
    eligible=left_has_real_support|right_has_real_support

    n_left_unique_real_pixels,n_left_unique_density_support_pixels=_gd_unique_valid_support_counts_by_site(
        site_ids,lu,lv_pix,lmap,l_support,n_sites,H,W)

    n_right_unique_real_pixels,n_right_unique_density_support_pixels=_gd_unique_valid_support_counts_by_site(
        site_ids,ru,rv_pix,rmap,r_support,n_sites,H,W)

    global_density_sum=float(np.asarray(target,dtype=np.float64).sum())
    global_target_count=int(gd_round_half_up_nonnegative([global_density_sum])[0])

    if n_sites:
        eligible_raw_mass=np.where(eligible,neighborhood_mass,0.0)

        raw_local_mass_total=float(neighborhood_mass.sum())
        eligible_local_mass=float(eligible_raw_mass.sum())

        if global_density_sum<=1e-12:
            mass_scale_factor=1.0
            allocation_mass=np.zeros(n_sites,dtype=np.float64)
            balanced_count=np.zeros(n_sites,dtype=np.int64)
            count_mass_coverage=1.0

        else:
            if eligible_local_mass<=1e-12:
                raise RuntimeError(
                    f"{model_name} | {pair_name} | img{image_index}: global FLOW KDE={global_density_sum:.6f}, "
                    f" but all eligible site local density mass are all 0. indicates watershed position detection and FLOW Gaussian severely inconsistent ."
                )

            mass_scale_factor=global_density_sum/eligible_local_mass
            allocation_mass=eligible_raw_mass*mass_scale_factor

            allocation_sum=float(allocation_mass.sum())
            if allocation_sum>0: allocation_mass*=global_density_sum/allocation_sum

            count_mass_coverage=eligible_local_mass/global_density_sum

            center_detection_score=detect_score[center_y,center_x].astype(np.float64)
            balanced_count=gd_balanced_round_to_total(allocation_mass,global_target_count,center_detection_score)

        allocation_mass_total=float(allocation_mass.sum())
        independent_total=int(independent_count.sum())
        balanced_total=int(balanced_count.sum())

        if not np.isclose(allocation_mass_total,global_density_sum,rtol=1e-10,atol=1e-8):
            raise RuntimeError(f"Global continuous mass conservation failed: allocation={allocation_mass_total:.12f}, global={global_density_sum:.12f}")

        if balanced_total!=global_target_count:
            raise RuntimeError(f"Global integer count conservation failed: balanced={balanced_total}, target={global_target_count}")

    else:
        raw_local_mass_total=0.0; eligible_local_mass=0.0; allocation_mass_total=0.0
        mass_scale_factor=1.0; count_mass_coverage=0.0 if global_density_sum>0 else 1.0
        independent_total=0; balanced_total=0

        eligible_raw_mass=np.empty(0,dtype=np.float64)
        allocation_mass=np.empty(0,dtype=np.float64)
        balanced_count=np.empty(0,dtype=np.int64)

        if global_target_count>0:
            raise RuntimeError(f"{model_name} | {pair_name} | img{image_index}: global FLOW KDE corresponding to {global_target_count} cells ,  but watershed none detected site.")

    covered_mask=owner_count>0
    covered_density=float(np.asarray(target[covered_mask],dtype=np.float64).sum()) if np.any(covered_mask) else 0.0
    allocated_density_total=float(np.asarray(allocated_density,dtype=np.float64).sum())

    absolute_error=abs(covered_density-allocated_density_total)
    relative_error=absolute_error/max(abs(covered_density),1e-12)

    if absolute_error>1e-5*max(1.0,abs(covered_density)):
        raise RuntimeError(
            f"Neighborhood density conservation failed: covered={covered_density:.8f}, "
            f"allocated={allocated_density_total:.8f}, error={absolute_error:.3e}"
        )

    if n_sites:
        center_rgb_all=detect_rgb[center_y,center_x]

        strict_nonbackground=np.any(center_rgb_all<1.0,axis=1) if background_mode=="white" else np.any(center_rgb_all>0.0,axis=1)
        output_strict_nonwhite_pixels=int(strict_nonbackground.sum())

        center_scores=gd_rgb_foreground_score(center_rgb_all.reshape(-1,1,3),background_mode).reshape(-1)
        output_score_nonwhite_pixels=int((center_scores>float(output_nonwhite_threshold)).sum())

        center_detection_score=detect_score[center_y,center_x].astype(np.float64)
        center_rgb64=center_rgb_all.astype(np.float64)
        center_density=target[center_y,center_x].astype(np.float64)

    else:
        output_strict_nonwhite_pixels=0; output_score_nonwhite_pixels=0
        center_detection_score=np.empty(0,dtype=np.float64)
        center_rgb64=np.empty((0,3),dtype=np.float64)
        center_density=np.empty(0,dtype=np.float64)

    site_df=pd.DataFrame({
        "model":_gd_const_object(str(model_name),n_sites),"pair":_gd_const_object(str(pair_name),n_sites),"image":_gd_const_object(image_path.name,n_sites),
        "image_index":np.full(n_sites,image_index,dtype=np.int64),"interp_fraction":np.full(n_sites,ratio,dtype=np.float64),
        "site_id":np.arange(n_sites,dtype=np.int64),"region_id":region_ids.astype(np.int64,copy=False),
        "center_pixel_u":center_x,"center_pixel_v":center_y,

        "center_r":center_rgb64[:,0] if n_sites else np.empty(0),
        "center_g":center_rgb64[:,1] if n_sites else np.empty(0),
        "center_b":center_rgb64[:,2] if n_sites else np.empty(0),

        "center_detection_score":center_detection_score,"center_gray":center_detection_score,"center_density":center_density,

        "support_radius_pixels":np.full(n_sites,float(support_radius_pixels if use_neighborhood_pixels else 0.0)),
        "use_neighborhood_pixels":np.full(n_sites,bool(use_neighborhood_pixels),dtype=bool),

        "n_support_pixels":n_support_pixels,"n_shared_pixels":n_shared_pixels,"fraction_shared_pixels":fraction_shared_pixels,

        "support_raw_density_sum":support_raw_density_sum,
        "support_allocated_density_sum":support_allocated_density_sum,
        "support_effective_pixel_count":support_effective_pixel_count,
        "support_max_density":support_max_density,
        "support_mean_density":support_mean_density,

        "neighborhood_mass":neighborhood_mass,
        "raw_neighborhood_mass":neighborhood_mass,

        "cell_number_independent_round":independent_count,

        "left_has_real_support_prelim":left_has_real_support,
        "right_has_real_support_prelim":right_has_real_support,
        "count_allocation_eligible":eligible,

        "cell_number":balanced_count,

        "eligible_neighborhood_mass":eligible_raw_mass,
        "allocation_mass":allocation_mass,

        "global_density_sum":np.full(n_sites,global_density_sum),
        "global_target_count":np.full(n_sites,global_target_count,dtype=np.int64),
        "global_mass_scale_factor":np.full(n_sites,mass_scale_factor),
        "count_mass_coverage":np.full(n_sites,count_mass_coverage),

        "cell_number_adjustment":balanced_count-independent_count,
    })

    flow_screen_precomputed_df=pd.DataFrame({
        "model":_gd_const_object(str(model_name),n_sites),
        "pair":_gd_const_object(str(pair_name),n_sites),
        "image":_gd_const_object(image_path.name,n_sites),

        "image_index":np.full(n_sites,image_index,dtype=np.int64),
        "site_id":np.arange(n_sites,dtype=np.int64),

        "n_mid_support_pixels":n_support_pixels,

        "n_left_valid_mapped_pixels":n_left_valid_mapped_pixels,
        "n_right_valid_mapped_pixels":n_right_valid_mapped_pixels,

        "n_left_unique_real_pixels":n_left_unique_real_pixels,
        "n_right_unique_real_pixels":n_right_unique_real_pixels,

        "n_left_density_support_pixels":n_left_density_support_pixels,
        "n_right_density_support_pixels":n_right_density_support_pixels,

        "n_left_unique_density_support_pixels":n_left_unique_density_support_pixels,
        "n_right_unique_density_support_pixels":n_right_unique_density_support_pixels,

        "left_has_real_support":left_has_real_support,
        "right_has_real_support":right_has_real_support,

        "mapped_real_no_density_support":~eligible,
        "keep_after_flow_screen":eligible,

        "n_left_nonblack_mapped_pixels":n_left_density_support_pixels,
        "n_right_nonblack_mapped_pixels":n_right_density_support_pixels,

        "n_left_unique_nonblack_real_pixels":n_left_unique_density_support_pixels,
        "n_right_unique_nonblack_real_pixels":n_right_unique_density_support_pixels,

        "mapped_real_all_black":~eligible,
    })

    pixel_df=pd.DataFrame({
        "model":_gd_const_object(str(model_name),n_pixels),
        "pair":_gd_const_object(str(pair_name),n_pixels),
        "image":_gd_const_object(image_path.name,n_pixels),

        "image_index":np.full(n_pixels,image_index,dtype=np.int64),
        "interp_fraction":np.full(n_pixels,ratio,dtype=np.float64),

        "site_id":site_ids,
        "region_id":region_ids[site_ids].astype(np.int64,copy=False) if n_pixels else np.empty(0,dtype=np.int64),

        "center_pixel_u":center_x[site_ids] if n_pixels else np.empty(0,dtype=np.int64),
        "center_pixel_v":center_y[site_ids] if n_pixels else np.empty(0,dtype=np.int64),

        "pixel_u":xx,"pixel_v":yy,
        "offset_u":dx,"offset_v":dy,
        "distance_to_center":dist,

        "is_center_pixel":is_center,

        "owner_count":owners,
        "is_shared_pixel":is_shared,
        "share":share,

        "detect_r":pixel_rgb[:,0] if n_pixels else np.empty(0),
        "detect_g":pixel_rgb[:,1] if n_pixels else np.empty(0),
        "detect_b":pixel_rgb[:,2] if n_pixels else np.empty(0),

        "detection_score":pixel_detection_score,
        "raw_gray":pixel_detection_score,
        "allocated_gray":pixel_detection_score*share,

        "raw_density":raw_density,
        "allocated_density":allocated_density,

        "left_real_density":lden,
        "right_real_density":rden,

        "left_weight":left_weight_px,
        "right_weight":right_weight_px,

        "left_real_x":lx,"left_real_y":ly,
        "right_real_x":rx,"right_real_y":ry,

        "left_real_pixel_u":lu,
        "left_real_pixel_v":lv_pix,

        "right_real_pixel_u":ru,
        "right_real_pixel_v":rv_pix,

        "left_flow_valid":lv,
        "right_flow_valid":rv,

        "left_mapping_valid":lmap,
        "right_mapping_valid":rmap,

        "left_real_support":l_support,
        "right_real_support":r_support,
        "any_real_support":l_support|r_support,

        "left_real_nonblack":l_support,
        "right_real_nonblack":r_support,
        "any_real_nonblack":l_support|r_support,

        "left_real_index":np.full(n_pixels,left_real_index,dtype=np.int64),
        "right_real_index":np.full(n_pixels,right_real_index,dtype=np.int64),

        "_keep_after_flow_screen_internal":eligible[site_ids] if n_pixels else np.empty(0,dtype=bool),
    })

    summary={
        "model":str(model_name),"pair":str(pair_name),"image":image_path.name,
        "image_index":image_index,"interp_fraction":ratio,

        "height":int(H),"width":int(W),

        "input_foreground_pixels":int(watershed_info["input_foreground_pixels"]),
        "n_markers":int(watershed_info["n_markers"]),
        "n_watershed_regions":int(watershed_info["n_watershed_regions"]),
        "n_regions_used_for_output":int(n_sites),

        "output_score_nonwhite_pixels":int(output_score_nonwhite_pixels),
        "output_strict_nonwhite_pixels":int(output_strict_nonwhite_pixels),
        "output_nonwhite_threshold":float(output_nonwhite_threshold),

        "use_neighborhood_pixels":bool(use_neighborhood_pixels),
        "support_radius_pixels":float(support_radius_pixels if use_neighborhood_pixels else 0.0),

        "n_foreground_pixels":int(fg_mask.sum()),
        "n_detected_sites":int(n_sites),

        "n_covered_pixels":int(covered_mask.sum()),
        "n_overlap_pixels":int((owner_count>1).sum()),
        "maximum_owner_count":int(owner_count.max()) if owner_count.size else 0,

        "covered_density":covered_density,
        "allocated_density":allocated_density_total,
        "absolute_error":absolute_error,
        "relative_error":relative_error,

        "global_density_sum":global_density_sum,
        "global_target_count":global_target_count,

        "raw_local_mass_total":raw_local_mass_total,
        "eligible_local_mass_total":eligible_local_mass,

        "count_mass_coverage":count_mass_coverage,
        "global_mass_scale_factor":mass_scale_factor,

        "allocation_mass_total":allocation_mass_total,
        "allocation_mass_error":allocation_mass_total-global_density_sum,

        "independent_round_count":independent_total,
        "independent_round_error":independent_total-global_target_count,

        "balanced_round_count":balanced_total,
        "balanced_round_error":balanced_total-global_target_count,

        "total_neighborhood_mass":raw_local_mass_total,
        "estimated_count":balanced_total,
    }

    
    
    return site_df,pixel_df,flow_screen_precomputed_df,summary,labels,owner_count,target


# ============================================================
# 17. FAST FLOW SCREENING
#


# ============================================================

def gd_screen_sites_by_flow_density(center_df,mapped_pixel_df,precomputed_screen_df=None):
    keys=["model","pair","image","image_index","site_id"]

    screen_cols=[
        "n_mid_support_pixels",
        "n_left_valid_mapped_pixels","n_right_valid_mapped_pixels",
        "n_left_unique_real_pixels","n_right_unique_real_pixels",
        "n_left_density_support_pixels","n_right_density_support_pixels",
        "n_left_unique_density_support_pixels","n_right_unique_density_support_pixels",
        "left_has_real_support","right_has_real_support",
        "mapped_real_no_density_support","keep_after_flow_screen",
        "n_left_nonblack_mapped_pixels","n_right_nonblack_mapped_pixels",
        "n_left_unique_nonblack_real_pixels","n_right_unique_nonblack_real_pixels",
        "mapped_real_all_black"
    ]

    precomputed=precomputed_screen_df

    if len(center_df) and isinstance(precomputed,pd.DataFrame) and len(precomputed)==len(center_df) and all(c in precomputed.columns for c in keys+screen_cols):
        summary_df=precomputed[keys+screen_cols].copy(deep=False)

        for k in keys:
            if not np.array_equal(center_df[k].to_numpy(),summary_df[k].to_numpy()):
                raise RuntimeError(f" precomputed  FLOW screening  of  key={k}  and  center_df  order mismatch ")

        result=center_df.copy(deep=False)

        for c in screen_cols:
            result[c]=summary_df[c].to_numpy(copy=False)

    elif not len(center_df):
        summary_df=pd.DataFrame(columns=keys+screen_cols)
        result=center_df.copy()

    else:
        
        records=[]

        for key,g in mapped_pixel_df.groupby(keys,sort=False):
            lv=g["left_mapping_valid"].to_numpy(dtype=bool)
            rv=g["right_mapping_valid"].to_numpy(dtype=bool)

            ls=g["left_real_support"].to_numpy(dtype=bool)
            rs=g["right_real_support"].to_numpy(dtype=bool)

            model,pair,image,image_index,site_id=key

            left_unique=np.unique(g.loc[lv,["left_real_pixel_u","left_real_pixel_v"]].to_numpy(dtype=np.int64),axis=0) if lv.any() else np.empty((0,2),dtype=np.int64)
            right_unique=np.unique(g.loc[rv,["right_real_pixel_u","right_real_pixel_v"]].to_numpy(dtype=np.int64),axis=0) if rv.any() else np.empty((0,2),dtype=np.int64)

            left_unique_support=np.unique(g.loc[ls,["left_real_pixel_u","left_real_pixel_v"]].to_numpy(dtype=np.int64),axis=0) if ls.any() else np.empty((0,2),dtype=np.int64)
            right_unique_support=np.unique(g.loc[rs,["right_real_pixel_u","right_real_pixel_v"]].to_numpy(dtype=np.int64),axis=0) if rs.any() else np.empty((0,2),dtype=np.int64)

            keep_site=bool(ls.any() or rs.any())

            records.append({
                "model":str(model),"pair":str(pair),"image":str(image),
                "image_index":int(image_index),"site_id":int(site_id),

                "n_mid_support_pixels":int(len(g)),

                "n_left_valid_mapped_pixels":int(lv.sum()),
                "n_right_valid_mapped_pixels":int(rv.sum()),

                "n_left_unique_real_pixels":int(len(left_unique)),
                "n_right_unique_real_pixels":int(len(right_unique)),

                "n_left_density_support_pixels":int(ls.sum()),
                "n_right_density_support_pixels":int(rs.sum()),

                "n_left_unique_density_support_pixels":int(len(left_unique_support)),
                "n_right_unique_density_support_pixels":int(len(right_unique_support)),

                "left_has_real_support":bool(ls.any()),
                "right_has_real_support":bool(rs.any()),

                "mapped_real_no_density_support":bool(not keep_site),
                "keep_after_flow_screen":keep_site,

                "n_left_nonblack_mapped_pixels":int(ls.sum()),
                "n_right_nonblack_mapped_pixels":int(rs.sum()),

                "n_left_unique_nonblack_real_pixels":int(len(left_unique_support)),
                "n_right_unique_nonblack_real_pixels":int(len(right_unique_support)),

                "mapped_real_all_black":bool(not keep_site),
            })

        summary_df=pd.DataFrame(records)
        result=center_df.merge(summary_df,on=keys,how="left",validate="one_to_one")

    if len(result) and result["keep_after_flow_screen"].isna().any():
        raise RuntimeError(f"{int(result['keep_after_flow_screen'].isna().sum())}  items site no corresponding FLOW mapping")

    if len(result):
        result["keep_after_flow_screen"]=result["keep_after_flow_screen"].astype(bool)
        result["mapped_real_no_density_support"]=result["mapped_real_no_density_support"].astype(bool)

        if "count_allocation_eligible" in result.columns:
            prelim=result["count_allocation_eligible"].to_numpy(dtype=bool)
            final=result["keep_after_flow_screen"].to_numpy(dtype=bool)

            if not np.array_equal(prelim,final):
                raise RuntimeError(f"count_allocation_eligible  and final FLOW screening inconsistent ,  total {int(np.sum(prelim!=final))} items site.")

    keep_mask=result["keep_after_flow_screen"].to_numpy(dtype=bool) if len(result) else np.empty(0,dtype=bool)

    kept=result.loc[keep_mask].copy().reset_index(drop=True) if len(result) else result.copy()
    removed=result.loc[~keep_mask].copy().reset_index(drop=True) if len(result) else result.copy()

    
    for _df in [result,kept,removed,summary_df]:
        if isinstance(_df,pd.DataFrame): _df.attrs.clear()

    return result,kept,removed,summary_df


# ============================================================
# 17.1 FAST COMBINED CSV
# ============================================================

def gd_concat_frame_csvs(csv_paths,out_path):
    paths=[Path(p) for p in csv_paths]; out_path=Path(out_path)

    if not paths:
        pd.DataFrame().to_csv(out_path,index=False,encoding="utf-8-sig")
        return

    if len(paths)==1:
        shutil.copyfile(paths[0],out_path)
        return

    bom=b"\xef\xbb\xbf"

    with open(out_path,"wb") as fout:
        for i,p in enumerate(paths):
            with open(p,"rb") as fin:
                if i==0:
                    shutil.copyfileobj(fin,fout,length=16*1024*1024)
                else:
                    head=fin.read(3)
                    if head!=bom: fin.seek(0)
                    fin.readline()
                    shutil.copyfileobj(fin,fout,length=16*1024*1024)


# ============================================================
# 18. MAIN
# ============================================================

def optimize_all_intermediate_gaussian_frames(
    intermediate_gaussian_results,save_dir,
    sigma_pixels=1.0,truncate=4.0,use_neighborhood_pixels=True,support_radius_pixels=None,
    detect_fg_threshold=0.01,min_fg_component_area=1,peak_min_distance=1,
    peak_threshold_abs=0.05,peak_smooth_sigma=0.8,color_grad_weight=1.5,
    connectivity=4,min_region_area=1,crop_to_foreground=True,crop_pad=8,
    background_mode=None,real_density_threshold=0.0,output_nonwhite_threshold=0.03,
    model_name="RIFE",target_image_indices=None,verbose=True,
):
    start=time.perf_counter()

    target_image_indices=None if target_image_indices is None else set(int(x) for x in target_image_indices)

    if verbose:
        print("\nTarget frames for cell-site detection:",sorted(target_image_indices) if target_image_indices is not None else "ALL")

    background_mode=gd_resolve_background_mode(save_dir,background_mode)

    if support_radius_pixels is None:
        support_radius_pixels=float(sigma_pixels)*float(truncate)

    out_root=Path(save_dir).resolve()/"intermediate_integer_count_maps"
    out_root.mkdir(parents=True,exist_ok=True)

    all_results={}
    summary_records=[]

    site_parts=[]
    screen_parts=[]
    pixel_parts=[]
    pixel_keep_parts=[]

    frame_site_csv_paths=[]
    frame_pixel_csv_paths=[]

    for pair_name,frames in intermediate_gaussian_results.items():

        if target_image_indices is not None:
            available_indices={int(frame["image_index"]) for frame in frames}
            missing=target_image_indices-available_indices

            if missing:
                raise KeyError(f"{pair_name}: intermediate_gaussian_results missing target frames: {sorted(missing)}")

            frames=[frame for frame in frames if int(frame["image_index"]) in target_image_indices]

        pair_out=out_root/pair_name
        pair_out.mkdir(parents=True,exist_ok=True)

        pair_results=[]

        for frame in frames:
            frame_start=time.perf_counter()
            image_index=int(frame["image_index"])

            if verbose:
                print("\n"+"#"*100)
                print(f"{model_name} | {pair_name} | img{image_index}")
                print("RGB watershed + FLOW Gaussian + global count conservation")
                print("use_neighborhood_pixels =",use_neighborhood_pixels)
                print("#"*100)

            
            
            (
                site_df,
                pixel_df,
                screen_precomputed,
                summary,
                labels,
                owner_count,
                target,
            )=gd_build_flow_mapped_site_tables(
                pair_name=pair_name,
                frame=frame,
                model_name=model_name,

                use_neighborhood_pixels=use_neighborhood_pixels,
                support_radius_pixels=support_radius_pixels,

                background_mode=background_mode,

                detect_fg_threshold=detect_fg_threshold,
                min_fg_component_area=min_fg_component_area,

                peak_min_distance=peak_min_distance,
                peak_threshold_abs=peak_threshold_abs,
                peak_smooth_sigma=peak_smooth_sigma,

                color_grad_weight=color_grad_weight,

                connectivity=connectivity,
                min_region_area=min_region_area,

                crop_to_foreground=crop_to_foreground,
                crop_pad=crop_pad,

                real_density_threshold=real_density_threshold,
                output_nonwhite_threshold=output_nonwhite_threshold,
            )

            pixel_keep_mask=pixel_df.pop("_keep_after_flow_screen_internal").to_numpy(dtype=bool,copy=False) if "_keep_after_flow_screen_internal" in pixel_df.columns else np.zeros(len(pixel_df),dtype=bool)

            count_map=np.zeros(target.shape,dtype=np.int32)

            if len(site_df):
                center_v=site_df["center_pixel_v"].to_numpy(dtype=np.int64)
                center_u=site_df["center_pixel_u"].to_numpy(dtype=np.int64)
                count_map[center_v,center_u]=site_df["cell_number"].to_numpy(dtype=np.int32)

            estimated_count=int(count_map.sum())

            if estimated_count!=int(summary["global_target_count"]):
                raise RuntimeError(
                    f"{model_name} | {pair_name} | img{image_index}: "
                    f"count_map={estimated_count}, global_target={summary['global_target_count']}"
                )

            stem=f"img{image_index}"
            npz_path=pair_out/f"{stem}.npz"
            frame_out=pair_out/stem

            frame_out.mkdir(parents=True,exist_ok=True)

            np.savez_compressed(
                npz_path,

                count_map=count_map,
                target_density=target.astype(np.float32),

                watershed_labels=labels.astype(np.int32),
                owner_count=owner_count,

                estimated_count=np.array(estimated_count,dtype=np.int64),
                estimated_count_float=np.array(float(summary["global_density_sum"]),dtype=np.float64),

                global_density_sum=np.array(float(summary["global_density_sum"]),dtype=np.float64),
                global_target_count=np.array(int(summary["global_target_count"]),dtype=np.int64),

                raw_local_mass_total=np.array(float(summary["raw_local_mass_total"]),dtype=np.float64),
                eligible_local_mass_total=np.array(float(summary["eligible_local_mass_total"]),dtype=np.float64),

                count_mass_coverage=np.array(float(summary["count_mass_coverage"]),dtype=np.float64),
                global_mass_scale_factor=np.array(float(summary["global_mass_scale_factor"]),dtype=np.float64),

                allocation_mass_total=np.array(float(summary["allocation_mass_total"]),dtype=np.float64),

                independent_round_count=np.array(int(summary["independent_round_count"]),dtype=np.int64),
                balanced_round_count=np.array(int(summary["balanced_round_count"]),dtype=np.int64),

                ratio=np.array(frame["ratio"],dtype=np.float32),
                use_neighborhood_pixels=np.array(bool(use_neighborhood_pixels)),
            )

            site_csv=frame_out/"centers_and_support.csv"
            pixel_csv=frame_out/"neighborhood_flow_mapping.csv"

            site_df.to_csv(site_csv,index=False,encoding="utf-8-sig")
            pixel_df.to_csv(pixel_csv,index=False,encoding="utf-8-sig")

            np.save(frame_out/"watershed_labels.npy",labels.astype(np.int32))
            np.save(frame_out/"neighborhood_owner_count.npy",owner_count)

            frame_site_csv_paths.append(site_csv)
            frame_pixel_csv_paths.append(pixel_csv)

            runtime=time.perf_counter()-frame_start

            record={
                "model":str(model_name),"pair":pair_name,
                "image":frame["image"],"image_index":image_index,

                "ratio":float(frame["ratio"]),
                "interp_fraction":float(frame["ratio"]),

                "estimated_count_float":float(summary["global_density_sum"]),
                "estimated_count":estimated_count,

                "global_density_sum":float(summary["global_density_sum"]),
                "global_target_count":int(summary["global_target_count"]),

                "raw_local_mass_total":float(summary["raw_local_mass_total"]),
                "eligible_local_mass_total":float(summary["eligible_local_mass_total"]),

                "count_mass_coverage":float(summary["count_mass_coverage"]),
                "global_mass_scale_factor":float(summary["global_mass_scale_factor"]),

                "allocation_mass_total":float(summary["allocation_mass_total"]),
                "allocation_mass_error":float(summary["allocation_mass_error"]),

                "independent_round_count":int(summary["independent_round_count"]),
                "independent_round_error":int(summary["independent_round_error"]),

                "balanced_round_count":int(summary["balanced_round_count"]),
                "balanced_round_error":int(summary["balanced_round_error"]),

                "n_candidates":int(len(site_df)),
                "n_detected_sites":int(len(site_df)),

                "n_nonzero_count_pixels":int((count_map>0).sum()),
                "max_count_per_pixel":int(count_map.max()) if count_map.size else 0,

                "status":"RGB_WATERSHED_FLOW_GLOBAL_BALANCED",

                "runtime":float(runtime),
                "optimization_runtime":0.0,
                "iterations":0,

                "npz_path":str(npz_path),
                "use_neighborhood_pixels":bool(use_neighborhood_pixels),
            }

            pair_results.append(record)
            summary_records.append({**summary,**record})

            if len(site_df):
                site_parts.append(site_df)
                screen_parts.append(screen_precomputed)

            if len(pixel_df):
                pixel_parts.append(pixel_df)
                pixel_keep_parts.append(pixel_keep_mask)

            if verbose:
                positive_sites=int((site_df["cell_number"]>0).sum()) if len(site_df) else 0

                print(
                    f"foreground={summary['input_foreground_pixels']:,} | "
                    f"markers={summary['n_markers']:,} | "
                    f"watershed={summary['n_watershed_regions']:,} | "
                    f"sites={len(site_df):,} | "
                    f"global_KDE={summary['global_density_sum']:.3f} -> {summary['global_target_count']:,} cells | "
                    f"local_raw={summary['raw_local_mass_total']:.3f} | "
                    f"coverage={summary['count_mass_coverage']:.4f} | "
                    f"scale={summary['global_mass_scale_factor']:.4f} | "
                    f"independent={summary['independent_round_count']:,} ({summary['independent_round_error']:+d}) | "
                    f"balanced={estimated_count:,} | "
                    f"positive_sites={positive_sites:,} | "
                    f"runtime={runtime:.3f}s"
                )

            del labels,owner_count,target,count_map

        all_results[pair_name]=pair_results

    # ========================================================
    # COMBINED OUTPUTS
    # ========================================================

    global_center_site_df=pd.concat(site_parts,ignore_index=True,copy=False) if site_parts else pd.DataFrame()
    global_neighborhood_pixel_df=pd.concat(pixel_parts,ignore_index=True,copy=False) if pixel_parts else pd.DataFrame()

    global_center_cellnumber_df=global_center_site_df.copy(deep=False)

    precomputed_screen_df=pd.concat(screen_parts,ignore_index=True,copy=False) if screen_parts else pd.DataFrame()

    
    if len(global_center_cellnumber_df):
        (
            flow_site_all_df,
            flow_screened_site_df,
            flow_removed_site_df,
            flow_screen_summary_df,
        )=gd_screen_sites_by_flow_density(
            center_df=global_center_cellnumber_df,
            mapped_pixel_df=global_neighborhood_pixel_df,
            precomputed_screen_df=precomputed_screen_df,
        )
    else:
        flow_site_all_df=pd.DataFrame()
        flow_screened_site_df=pd.DataFrame()
        flow_removed_site_df=pd.DataFrame()
        flow_screen_summary_df=pd.DataFrame()

    if len(flow_removed_site_df):
        removed_cells=int(flow_removed_site_df["cell_number"].sum())

        if removed_cells!=0:
            raise RuntimeError(
                f"FLOW screening removed site still contains {removed_cells} items cell."
                f" indicates count_allocation_eligible and final screening criteria differ ."
            )

    global_center_cellnumber_dict={
        model:g.reset_index(drop=True)
        for model,g in global_center_cellnumber_df.groupby("model",sort=False)
    } if len(global_center_cellnumber_df) else {}

    # ========================================================
    # PIXEL KEEP STATUS
    # ========================================================

    flow_mapped_pixel_df=global_neighborhood_pixel_df.copy(deep=False)

    if len(flow_mapped_pixel_df):
        keep_all=np.concatenate(pixel_keep_parts) if pixel_keep_parts else np.zeros(len(flow_mapped_pixel_df),dtype=bool)

        if len(keep_all)!=len(flow_mapped_pixel_df):
            raise RuntimeError("pixel keep mask length and global pixel table inconsistent ")

        flow_mapped_pixel_df["keep_after_flow_screen"]=keep_all

    # ========================================================
    # SAVE
    # ========================================================

    gd_concat_frame_csvs(frame_site_csv_paths,out_root/"all_centers_and_support.csv")
    gd_concat_frame_csvs(frame_pixel_csv_paths,out_root/"all_neighborhood_flow_mapping.csv")

    flow_site_all_df.to_csv(out_root/"all_sites_flow_screening.csv",index=False,encoding="utf-8-sig")
    flow_screened_site_df.to_csv(out_root/"retained_sites.csv",index=False,encoding="utf-8-sig")
    flow_removed_site_df.to_csv(out_root/"removed_no_real_density_support_sites.csv",index=False,encoding="utf-8-sig")

    flow_mapped_pixel_df.to_csv(out_root/"flow_mapped_pixel_df.csv",index=False,encoding="utf-8-sig")
    flow_screen_summary_df.to_csv(out_root/"flow_screen_summary.csv",index=False,encoding="utf-8-sig")

    summary_df=pd.DataFrame(summary_records)
    summary_df.to_csv(out_root/"optimization_summary.csv",index=False,encoding="utf-8-sig")

    if verbose:
        print("\n"+"="*100)
        print("ALL INTERMEDIATE FRAMES FINISHED")
        print("="*100)

        print("Position detection      : RGB 3-channel watershed")
        print("Cell-number source      : FLOW-migrated REAL Gaussian density")
        print("Integer allocation      : global mass calibration + balanced rounding")

        print("use_neighborhood_pixels :",use_neighborhood_pixels)
        print("support_radius_pixels   :",support_radius_pixels)

        print("Detected sites          :",f"{len(global_center_site_df):,}")

        print(
            "Estimated cells         :",
            f"{int(global_center_cellnumber_df['cell_number'].sum()):,}" if len(global_center_cellnumber_df) else "0",
        )

        print("FLOW retained sites     :",f"{len(flow_screened_site_df):,}")
        print("FLOW removed sites      :",f"{len(flow_removed_site_df):,}")

        if len(flow_screened_site_df):
            print("FLOW retained cells     :",f"{int(flow_screened_site_df['cell_number'].sum()):,}")

        print("Total runtime           :",f"{time.perf_counter()-start:.3f}s")
        print("Output                  :",out_root)

        print("="*100)

    # ========================================================
    # PANDAS 3.x SAFETY
    #
    
    
    # ========================================================

    for _df in [
        global_center_site_df,
        global_neighborhood_pixel_df,
        global_center_cellnumber_df,
        precomputed_screen_df,
        flow_site_all_df,
        flow_screened_site_df,
        flow_removed_site_df,
        flow_mapped_pixel_df,
        flow_screen_summary_df,
        summary_df,
    ]:
        if isinstance(_df,pd.DataFrame): _df.attrs.clear()

    for _df in global_center_cellnumber_dict.values():
        if isinstance(_df,pd.DataFrame): _df.attrs.clear()

    # ========================================================
    # EXPOSE GLOBAL VARIABLES
    # ========================================================

    globals()["global_center_site_df"]=global_center_site_df
    globals()["global_neighborhood_pixel_df"]=global_neighborhood_pixel_df

    globals()["global_center_cellnumber_df"]=global_center_cellnumber_df
    globals()["global_center_cellnumber_dict"]=global_center_cellnumber_dict

    globals()["flow_site_all_df"]=flow_site_all_df
    globals()["flow_screened_site_df"]=flow_screened_site_df
    globals()["flow_removed_site_df"]=flow_removed_site_df

    globals()["flow_mapped_pixel_df"]=flow_mapped_pixel_df
    globals()["flow_screen_summary_df"]=flow_screen_summary_df

    return all_results,out_root,summary_df


# ============================================================
# 19. RUN
# ============================================================

intermediate_gaussian_results,intermediate_gaussian_root=build_all_intermediate_gaussian_targets(
    global_flow_results=global_flow_results,
    gaussian_summary=gaussian_summary,
    save_dir=save_dir,
    blend_mode=blend_mode,
    visualization_vmax=None,
    save_mapping_arrays=True,
    target_image_indices=target_img,
    verbose=True,
)

(
    integer_count_results,
    integer_count_root,
    integer_count_summary_df,
)=optimize_all_intermediate_gaussian_frames(
    intermediate_gaussian_results=intermediate_gaussian_results,
    save_dir=save_dir,

    sigma_pixels=gaussian_sigma_pixels,
    truncate=gaussian_truncate,

    use_neighborhood_pixels=use_neighborhood_pixels,
    support_radius_pixels=None,

    detect_fg_threshold=config["detect_fg_threshold"],
    min_fg_component_area=config["min_fg_component_area"],

    peak_min_distance=config["peak_min_distance"],
    peak_threshold_abs=config["peak_threshold_abs"],
    peak_smooth_sigma=config["peak_smooth_sigma"],

    color_grad_weight=config["color_grad_weight"],

    connectivity=config["connectivity"],
    min_region_area=config["min_region_area"],

    crop_to_foreground=True,
    crop_pad=8,

    background_mode=None,
    real_density_threshold=0.0,

    output_nonwhite_threshold=config.get("output_nonwhite_threshold",0.03),

    model_name=str(interpolation_model_names[0]),
    target_image_indices=target_img,
    verbose=True,
)


