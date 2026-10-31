import cv2
import torch
import argparse
import warnings
import numpy as np
import random 
from pathlib import Path
from torch.nn import functional as F
from model.warplayer import warp


# ============================================================
# 0. Basic setup
# ============================================================

warnings.filterwarnings("ignore")

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)

if torch.cuda.is_available():
    torch.backends.cudnn.enabled=True
    torch.backends.cudnn.benchmark=True


# ============================================================
# 1. Arguments
# ============================================================

parser=argparse.ArgumentParser(
    description="RIFE interpolation + matched global endpoint FLOW/MASK"
)

parser.add_argument("--img",dest="img",nargs=2,required=True)
parser.add_argument("--exp",default=4,type=int)
parser.add_argument("--model",dest="modelDir",type=str,default="train_log")

parser.add_argument("--random_init",action="store_true")
parser.add_argument(
    "--random_init_seed",
    type=int,
    default=0,
    help="Seed used only for random initialization of RIFE.",
)
parser.add_argument(
    "--zero_correspondence",
    action="store_true",
    help="Disable RIFE spatial correspondence by forcing flow to zero.",
)

parser.add_argument(
    "--perturb_fraction",
    type=float,
    default=0.0,
    help=(
        "Fraction of spatial positions whose "
        "final RIFE correspondence vectors are shuffled."
    ),
)

parser.add_argument(
    "--perturb_seed",
    type=int,
    default=None,
    help=(
        "Independent seed used only for "
        "correspondence perturbation."
    ),
)

parser.add_argument(
    "--uniform_mask",
    action="store_true",
    help=(
        "Replace the learned RIFE spatial blending mask "
        "with direct temporal weighting."
    ),
)


parser.add_argument("--zero_weights", action="store_true")

        


# ------------------------------------------------------------
# Backward compatibility only.
#



# ------------------------------------------------------------

parser.add_argument("--save_flow",action="store_true",help=argparse.SUPPRESS)
parser.add_argument("--save_mask",action="store_true",help=argparse.SUPPRESS)

args=parser.parse_args()


if args.exp < 0:
    raise ValueError(
        "--exp  must  >= 0"
    )


if not (
    0.0
    <= float(args.perturb_fraction)
    <= 1.0
):
    raise ValueError(
        "--perturb_fraction  must be in  [0,1]"
    )


if (
    float(args.perturb_fraction) > 0.0
    and args.perturb_seed is None
):
    raise ValueError(
        "--perturb_seed  must be in  "
        "--perturb_fraction > 0  explicitly provide when "
    )


if (
    args.zero_correspondence
    and float(args.perturb_fraction) > 0.0
):
    raise ValueError(
        "--zero_correspondence  and  correspondence "
        "perturbation  cannot be enabled together "
    )

# ============================================================
# 2. Load RIFE model
# ============================================================

try:
    try:
        try:
            from model.RIFE_HDv2 import Model
            model=Model()
            model.load_model(args.modelDir,-1)
            model_name="RIFE_HDv2"
            print("Loaded v2.x HD model.")

        except Exception:
            from train_log.RIFE_HDv3 import Model
            model=Model()
            model.load_model(args.modelDir,-1)
            model_name="RIFE_HDv3"
            print("Loaded v3.x HD model.")

    except Exception:
        from model.RIFE_HD import Model
        model=Model()
        model.load_model(args.modelDir,-1)
        model_name="RIFE_HD"
        print("Loaded v1.x HD model.")

except Exception:
    from model.RIFE import Model
    model=Model()
    model.load_model(args.modelDir,-1)
    model_name="RIFE"
    print("Loaded ArXiv-RIFE model.")


# ============================================================
# RIFE parameter ablation
# ============================================================

if args.zero_weights and args.random_init:
    raise ValueError("--zero_weights and --random_init cannot be used together")

if args.zero_weights:
    with torch.no_grad():
        for param in model.flownet.parameters():
            param.zero_()

    params = list(model.flownet.parameters())
    assert params, "RIFE network has no parameters"
    assert all(torch.count_nonzero(p).item() == 0 for p in params), \
        "Some RIFE parameters are not zero"

    print(f"Zero-initialized {model_name}: all RIFE network parameters verified zero")

elif args.random_init:
    random_init_seed = int(args.random_init_seed)

    random.seed(random_init_seed)
    np.random.seed(random_init_seed)
    torch.manual_seed(random_init_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_init_seed)

    ModelClass = type(model)
    del model
    model = ModelClass()

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    print(f"Randomly initialized {model_name} | random_init_seed={random_init_seed}")

else:
    print(f"Using pretrained {model_name}")
        


model.eval()
model.device()


model.eval()
model.device()


# ============================================================
# 3. Read input images
# ============================================================

is_exr=args.img[0].lower().endswith(".exr") and args.img[1].lower().endswith(".exr")

if is_exr:
    arr0=cv2.imread(args.img[0],cv2.IMREAD_COLOR|cv2.IMREAD_ANYDEPTH)
    arr1=cv2.imread(args.img[1],cv2.IMREAD_COLOR|cv2.IMREAD_ANYDEPTH)

    if arr0 is None or arr1 is None: raise FileNotFoundError(f" cannot read input image : {args.img}")
    if arr0.shape!=arr1.shape: raise ValueError(f" input endpoint image sizes differ : {arr0.shape} vs {arr1.shape}")

    img0=torch.tensor(arr0.transpose(2,0,1),dtype=torch.float32,device=device).unsqueeze(0)
    img1=torch.tensor(arr1.transpose(2,0,1),dtype=torch.float32,device=device).unsqueeze(0)

else:
    arr0=cv2.imread(args.img[0],cv2.IMREAD_UNCHANGED)
    arr1=cv2.imread(args.img[1],cv2.IMREAD_UNCHANGED)

    if arr0 is None or arr1 is None: raise FileNotFoundError(f" cannot read input image : {args.img}")

    if arr0.ndim==2: arr0=np.repeat(arr0[:,:,None],3,axis=2)
    if arr1.ndim==2: arr1=np.repeat(arr1[:,:,None],3,axis=2)

    if arr0.shape[2]==4: arr0=arr0[:,:,:3]
    if arr1.shape[2]==4: arr1=arr1[:,:,:3]

    if arr0.shape!=arr1.shape: raise ValueError(f" input endpoint image sizes differ : {arr0.shape} vs {arr1.shape}")

    img0=torch.tensor(arr0.transpose(2,0,1),dtype=torch.float32,device=device).div(255.0).unsqueeze(0)
    img1=torch.tensor(arr1.transpose(2,0,1),dtype=torch.float32,device=device).div(255.0).unsqueeze(0)


# ============================================================
# 4. Pad to multiple of 32
# ============================================================

_,_,h,w=img0.shape

ph=((h-1)//32+1)*32
pw=((w-1)//32+1)*32

padding=(0,pw-w,0,ph-h)

img0=F.pad(img0,padding)
img1=F.pad(img1,padding)

print("\n"+"="*100)
print("RIFE INPUT")
print("="*100)
print(f"Input size  : {w} × {h}")
print(f"Padded size : {pw} × {ph}")
print(f"exp         : {args.exp}")
print(f"segments    : {2**args.exp}")
print(f"model       : {model_name}")
print(f"device      : {device}")
print("="*100)


# ============================================================
# 5. Utilities
# ============================================================

def collect_tensors(obj):
    """Collect tensors."""
    tensors=[]

    if torch.is_tensor(obj):
        tensors.append(obj)

    elif isinstance(obj,(list,tuple)):
        for x in obj: tensors.extend(collect_tensors(x))

    elif isinstance(obj,dict):
        for x in obj.values(): tensors.extend(collect_tensors(x))

    return tensors


def resize_flow_to(flow,H,W):
    """Resize flow to."""

    flow=flow[:,:4].float()

    fh,fw=flow.shape[-2:]

    if (fh,fw)==(H,W): return flow

    sx=float(W)/float(fw)
    sy=float(H)/float(fh)

    flow=F.interpolate(flow,size=(H,W),mode="bilinear",align_corners=False)
    flow=flow.clone()

    flow[:,0]*=sx
    flow[:,2]*=sx
    flow[:,1]*=sy
    flow[:,3]*=sy

    return flow


def resize_mask_to(mask,H,W):
    """Resize mask to."""
    mask=mask.float()

    if mask.shape[-2:]!=(H,W):
        mask=F.interpolate(mask,size=(H,W),mode="bilinear",align_corners=False)

    return mask


# ============================================================
# 6. Extract local FLOW + RIFE MASK
# ============================================================

def extract_flow_mask_from_flownet_output(output,H,W):
    """Extract flow mask from flownet output."""

    if not isinstance(output,(list,tuple)):
        raise RuntimeError(
            f"model.flownet  output is not  list/tuple,  current type ={type(output)}"
        )

    if len(output)<2:
        raise RuntimeError(
            "model.flownet  insufficient output length ,  cannot extract both  flow  and  mask."
        )

    # --------------------------------------------------------
    # Flow
    # --------------------------------------------------------

    flow_candidates=[
        t for t in collect_tensors(output[0])
        if t.ndim==4 and t.shape[1]>=4
    ]

    if not flow_candidates:
        raise RuntimeError(
            " cannot recover from  model.flownet output[0]  found in  >=4-channel flow tensor."
        )

    flow=resize_flow_to(flow_candidates[-1],H,W)

    # --------------------------------------------------------
    # Mask
    # --------------------------------------------------------

    mask_candidates=[
        t for t in collect_tensors(output[1])
        if t.ndim==4 and t.shape[1]==1
    ]

    if not mask_candidates:
        mask_candidates=[
            t for t in collect_tensors(output[1:])
            if t.ndim==4 and t.shape[1]==1
        ]

    if not mask_candidates:
        raise RuntimeError(
            " cannot recover from  model.flownet  found in output  1-channel RIFE mask."
        )

    mask=resize_mask_to(mask_candidates[-1],H,W)

    # --------------------------------------------------------
    
    
    # --------------------------------------------------------

    mask_min=float(mask.min().detach().cpu())
    mask_max=float(mask.max().detach().cpu())

    if mask_min<-1e-4 or mask_max>1.0+1e-4:
        print("  Detected mask outside [0,1], applying sigmoid.")
        mask=torch.sigmoid(mask)

    mask=torch.clamp(mask,0.0,1.0)

    print(
        f"  local RIFE mask: "
        f"min={mask.min().item():.4f}, "
        f"mean={mask.mean().item():.4f}, "
        f"max={mask.max().item():.4f}"
    )

    return flow,mask


# ============================================================
# 7. RIFE inference + hook FLOW/MASK
# ============================================================

def inference_with_flow_mask(model,left_img,right_img, perturb_fraction=0.0, perturb_seed=None, uniform_mask=False,):
    """Inference with flow mask."""

    captured=[]

    def hook_fn(module,inputs,output): captured.append(output)

    handle=model.flownet.register_forward_hook(hook_fn)

    try:
        if args.zero_correspondence:
            if model_name != "RIFE_HDv3":
                raise RuntimeError(
                    "--zero_correspondence is currently implemented "
                    "only for RIFE_HDv3."
                )
    
            middle = model.inference(
                left_img,
                right_img,
                zero_correspondence=True,
                uniform_mask=uniform_mask,
            )
        elif float(perturb_fraction) > 0.0:

            if model_name != "RIFE_HDv3":
                raise RuntimeError(
                    "correspondence perturbation is currently "
                    "implemented only for RIFE_HDv3."
                )

            if perturb_seed is None:
                raise ValueError(
                    "perturb_seed is required."
                )

            middle = model.inference(
                left_img,
                right_img,
                perturb_fraction=float(
                    perturb_fraction
                ),
                perturb_seed=int(
                    perturb_seed
                ),
                uniform_mask=uniform_mask,
            )
            
        else:
            middle = model.inference(
                left_img,
                right_img,
                uniform_mask=uniform_mask,
            )

    finally:
        handle.remove()

    if not captured:
        raise RuntimeError(
            " not captured  model.flownet  of  forward  output ."
        )

    flow,mask=extract_flow_mask_from_flownet_output(
        captured[-1],
        left_img.shape[-2],
        left_img.shape[-1],
    )

    if flow.shape[1]!=4:
        raise RuntimeError(
            f" final  local flow  must be 4 channels,  current shape={tuple(flow.shape)}"
        )

    if mask.shape[1]!=1:
        raise RuntimeError(
            f" final  local mask  must be 1 channel,  current shape={tuple(mask.shape)}"
        )

    return middle,flow,mask


# ============================================================
# 8. Recursive FLOW composition
# ============================================================

def compose_backward_flow(local_flow,parent_to_global):
    """
    current -> global endpoint

    F_current->global(p)
      =
    F_current->parent(p)
      +
    F_parent->global(
        p + F_current->parent(p)
    )
    """

    return local_flow+warp(parent_to_global,local_flow)


# ============================================================
# 9. Recursive endpoint MASK composition
# ============================================================

def compose_endpoint_weights(
    local_mask,
    local_left_flow,
    local_right_flow,

    left_parent_weight_to_left,
    left_parent_weight_to_right,

    right_parent_weight_to_left,
    right_parent_weight_to_right,

    current_ratio,
    eps=1e-8,
):
    """Compose endpoint weights."""

    local_mask=torch.clamp(local_mask.float(),0.0,1.0)
    local_right_mask=1.0-local_mask

    # --------------------------------------------------------
    
    
    # --------------------------------------------------------

    lp_to_left=warp(left_parent_weight_to_left,local_left_flow)
    lp_to_right=warp(left_parent_weight_to_right,local_left_flow)

    rp_to_left=warp(right_parent_weight_to_left,local_right_flow)
    rp_to_right=warp(right_parent_weight_to_right,local_right_flow)

    # --------------------------------------------------------
    # Local RIFE blending
    # --------------------------------------------------------

    raw_left=local_mask*lp_to_left+local_right_mask*rp_to_left
    raw_right=local_mask*lp_to_right+local_right_mask*rp_to_right

    weight_sum=raw_left+raw_right
    valid=weight_sum>eps

    normalized_left=raw_left/torch.clamp(weight_sum,min=eps)
    normalized_right=raw_right/torch.clamp(weight_sum,min=eps)

    # --------------------------------------------------------
    
    
    # --------------------------------------------------------

    fallback_left=torch.full_like(normalized_left,1.0-float(current_ratio))
    fallback_right=torch.full_like(normalized_right,float(current_ratio))

    weight_to_left=torch.where(valid,normalized_left,fallback_left)
    weight_to_right=torch.where(valid,normalized_right,fallback_right)

    weight_to_left=torch.clamp(weight_to_left,0.0,1.0)
    weight_to_right=torch.clamp(weight_to_right,0.0,1.0)

    # final normalization
    final_sum=weight_to_left+weight_to_right

    weight_to_left=weight_to_left/torch.clamp(final_sum,min=eps)
    weight_to_right=weight_to_right/torch.clamp(final_sum,min=eps)

    return weight_to_left,weight_to_right


# ============================================================
# 10. Generate one recursive midpoint
# ============================================================

def make_mid_node(left_node,right_node):
    """Make mid node."""

    current_ratio=0.5*(left_node["ratio"]+right_node["ratio"])

    local_perturb_seed = None

    if float(args.perturb_fraction) > 0.0:
        ratio_code = int(
            round(
                current_ratio
                * 1_000_000
            )
        )
    
        local_perturb_seed = int(args.perturb_seed)
    
    # --------------------------------------------------------
    # RIFE interpolation
    # --------------------------------------------------------

    middle,flow,mask=inference_with_flow_mask(
        model,
        left_node["img"],
        right_node["img"],
        perturb_fraction=float(
            args.perturb_fraction
        ),
        perturb_seed=local_perturb_seed,
        uniform_mask=args.uniform_mask,
    )

    local_left=flow[:,:2]
    local_right=flow[:,2:4]

    # --------------------------------------------------------
    
    # --------------------------------------------------------

    if left_node["to_left"] is None:
        raise RuntimeError(
            " left parent has no mapping to  original left  of  global flow."
        )

    if right_node["to_right"] is None:
        raise RuntimeError(
            " right parent has no mapping to  original right  of  global flow."
        )

    # --------------------------------------------------------
    # Compose current -> original endpoints FLOW
    # --------------------------------------------------------

    to_left=compose_backward_flow(
        local_left,
        left_node["to_left"],
    )

    to_right=compose_backward_flow(
        local_right,
        right_node["to_right"],
    )

    # --------------------------------------------------------
    
    # --------------------------------------------------------

    for key in ["weight_to_left","weight_to_right"]:
        if left_node.get(key) is None: raise RuntimeError(f" left parent is missing  {key}")
        if right_node.get(key) is None: raise RuntimeError(f" right parent is missing  {key}")

    # --------------------------------------------------------
    # Compose current -> original endpoints MASK
    # --------------------------------------------------------

    weight_to_left,weight_to_right=compose_endpoint_weights(
        local_mask=mask,
        local_left_flow=local_left,
        local_right_flow=local_right,

        left_parent_weight_to_left=left_node["weight_to_left"],
        left_parent_weight_to_right=left_node["weight_to_right"],

        right_parent_weight_to_left=right_node["weight_to_left"],
        right_parent_weight_to_right=right_node["weight_to_right"],

        current_ratio=current_ratio,
    )

    sum_error=torch.abs(
        weight_to_left+weight_to_right-1.0
    ).max().item()

    print(
        f"  global endpoint mask: "
        f"left(mean)={weight_to_left.mean().item():.4f}, "
        f"right(mean)={weight_to_right.mean().item():.4f}, "
        f"max sum error={sum_error:.3e}"
    )

    return {
        "img":middle,
        "ratio":current_ratio,

        "to_left":to_left,
        "to_right":to_right,

        "weight_to_left":weight_to_left,
        "weight_to_right":weight_to_right,

        "perturb_seed": (
            None
            if local_perturb_seed is None
            else int(local_perturb_seed)
        ),
    }


# ============================================================
# 11. Initial real endpoints
# ============================================================

zero_flow=torch.zeros(
    (1,2,ph,pw),
    dtype=img0.dtype,
    device=img0.device,
)

one_weight=torch.ones(
    (1,1,ph,pw),
    dtype=img0.dtype,
    device=img0.device,
)

zero_weight=torch.zeros(
    (1,1,ph,pw),
    dtype=img0.dtype,
    device=img0.device,
)

left_real={
    "img":img0,
    "ratio":0.0,
    "to_left":zero_flow,
    "to_right":None,
    "weight_to_left":one_weight,
    "weight_to_right":zero_weight,
}

right_real={
    "img":img1,
    "ratio":1.0,
    "to_left":None,
    "to_right":zero_flow,
    "weight_to_left":zero_weight,
    "weight_to_right":one_weight,
}


# ============================================================
# 12. Recursive RIFE interpolation
# ============================================================

nodes=[left_real,right_real]

for level in range(args.exp):

    print(
        f"\nInterpolation level "
        f"{level+1}/{args.exp}"
    )

    tmp=[]

    for j in range(len(nodes)-1):

        left_node=nodes[j]
        right_node=nodes[j+1]

        mid_node=make_mid_node(
            left_node,
            right_node,
        )

        print(
            f"  "
            f"{left_node['ratio']:.6f}"
            f" -> "
            f"{mid_node['ratio']:.6f}"
            f" -> "
            f"{right_node['ratio']:.6f}"
        )

        tmp.append(left_node)
        tmp.append(mid_node)

    tmp.append(nodes[-1])

    nodes=tmp


# ============================================================
# 13. Save interpolated images
# ============================================================

for i,node in enumerate(nodes):

    img=node["img"]

    if is_exr:
        out=img[0].detach().cpu().numpy().transpose(1,2,0)[:h,:w]

        ok=cv2.imwrite(
            f"img{i}.exr",
            out,
            [
                cv2.IMWRITE_EXR_TYPE,
                cv2.IMWRITE_EXR_TYPE_HALF,
            ],
        )

    else:
        out=(
            img[0]
            .clamp(0.0,1.0)
            .mul(255.0)
            .byte()
            .detach()
            .cpu()
            .numpy()
            .transpose(1,2,0)[:h,:w]
        )

        ok=cv2.imwrite(
            f"img{i}.png",
            out,
        )

    if not ok:
        raise RuntimeError(
            f" cannot save interpolated image  img{i}"
        )

print(
    f"\nSaved {len(nodes)} images: "
    f"img0 ... img{len(nodes)-1}"
)


# ============================================================
# 14. Save matched global FLOW + MASK
# ============================================================

flow_dir=Path("flow")
flow_dir.mkdir(parents=True,exist_ok=True)


def flow_to_numpy(flow):
    return (
        flow[0]
        .detach()
        .cpu()
        .numpy()
        .transpose(1,2,0)[:h,:w]
        .astype(np.float32)
    )


def mask_to_numpy(mask):
    return (
        mask[0,0]
        .detach()
        .cpu()
        .numpy()[:h,:w]
        .astype(np.float32)
    )


for i,node in enumerate(nodes):

    
    if i==0 or i==len(nodes)-1: continue

    # --------------------------------------------------------
    # Matched endpoint FLOW + MASK
    # --------------------------------------------------------

    flow_to_left=flow_to_numpy(
        node["to_left"]
    )

    flow_to_right=flow_to_numpy(
        node["to_right"]
    )

    mask_to_left=mask_to_numpy(
        node["weight_to_left"]
    )

    mask_to_right=mask_to_numpy(
        node["weight_to_right"]
    )

    ratio=np.float32(
        node["ratio"]
    )

    # --------------------------------------------------------
    # Shape checks
    # --------------------------------------------------------

    if flow_to_left.shape!=(h,w,2):
        raise ValueError(
            f"img{i}: flow_to_left "
            f"shape={flow_to_left.shape}, "
            f"expected={(h,w,2)}"
        )

    if flow_to_right.shape!=(h,w,2):
        raise ValueError(
            f"img{i}: flow_to_right "
            f"shape={flow_to_right.shape}, "
            f"expected={(h,w,2)}"
        )

    if mask_to_left.shape!=(h,w):
        raise ValueError(
            f"img{i}: mask_to_left "
            f"shape={mask_to_left.shape}, "
            f"expected={(h,w)}"
        )

    if mask_to_right.shape!=(h,w):
        raise ValueError(
            f"img{i}: mask_to_right "
            f"shape={mask_to_right.shape}, "
            f"expected={(h,w)}"
        )

    # --------------------------------------------------------
    # Finite checks
    # --------------------------------------------------------

    if not np.isfinite(flow_to_left).all(): raise ValueError(f"img{i}: flow_to_left contains NaN/Inf")
    if not np.isfinite(flow_to_right).all(): raise ValueError(f"img{i}: flow_to_right contains NaN/Inf")
    if not np.isfinite(mask_to_left).all(): raise ValueError(f"img{i}: mask_to_left contains NaN/Inf")
    if not np.isfinite(mask_to_right).all(): raise ValueError(f"img{i}: mask_to_right contains NaN/Inf")

    # --------------------------------------------------------
    # MASK checks
    # --------------------------------------------------------

    if mask_to_left.min()<-1e-4 or mask_to_left.max()>1.0+1e-4:
        raise ValueError(
            f"img{i}: mask_to_left outside [0,1], "
            f"range=({mask_to_left.min():.6f}, "
            f"{mask_to_left.max():.6f})"
        )

    if mask_to_right.min()<-1e-4 or mask_to_right.max()>1.0+1e-4:
        raise ValueError(
            f"img{i}: mask_to_right outside [0,1], "
            f"range=({mask_to_right.min():.6f}, "
            f"{mask_to_right.max():.6f})"
        )

    mask_sum_error=float(
        np.max(
            np.abs(
                mask_to_left
                +mask_to_right
                -1.0
            )
        )
    )

    if mask_sum_error>1e-4:
        raise ValueError(
            f"img{i}: "
            f"mask_to_left + mask_to_right != 1, "
            f"max error={mask_sum_error:.3e}"
        )

    # --------------------------------------------------------
    # Save ONE matched NPZ per intermediate image
    # --------------------------------------------------------

    np.savez_compressed(
        flow_dir/f"img{i}.npz",

        flow_to_left=flow_to_left,
        mask_to_left=mask_to_left,

        flow_to_right=flow_to_right,
        mask_to_right=mask_to_right,

        ratio=ratio,
    )

    print(
        f"Saved flow/img{i}.npz | "
        f"t={float(ratio):.6f} | "
        f"flow={flow_to_left.shape} | "
        f"mask={mask_to_left.shape} | "
        f"mask sum error={mask_sum_error:.3e}"
    )


# ============================================================
# 15. Final
# ============================================================

print("\n"+"="*100)
print("RIFE FINISHED")
print("="*100)

print(
    f"Images:"
    f"\n  img0 ... img{len(nodes)-1}"
)

print(
    f"\nIntermediate images:"
    f"\n  {max(len(nodes)-2,0)}"
)

print(
    f"\nMatched FLOW + MASK NPZ:"
    f"\n  {flow_dir.resolve()}"
)

print(
    "\nNPZ fields:"
    "\n  flow_to_left"
    "\n  mask_to_left"
    "\n  flow_to_right"
    "\n  mask_to_right"
    "\n  ratio"
)

print("="*100)