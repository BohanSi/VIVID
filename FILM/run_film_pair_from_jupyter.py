import os
import glob
import argparse
import numpy as np
import tensorflow as tf
from eval import util

# ============================================================

# ============================================================

parser = argparse.ArgumentParser()
parser.add_argument("--frame1", required=True)
parser.add_argument("--frame2", required=True)
parser.add_argument("--output_dir", required=True)
parser.add_argument("--model_path", required=True)
parser.add_argument("--k", type=int, required=True)
parser.add_argument("--align", type=int, default=64)
parser.add_argument("--block_height", type=int, default=1)
parser.add_argument("--block_width", type=int, default=1)
args = parser.parse_args()

if not os.path.isfile(args.frame1): raise FileNotFoundError(args.frame1)
if not os.path.isfile(args.frame2): raise FileNotFoundError(args.frame2)
if not os.path.isdir(args.model_path): raise FileNotFoundError(args.model_path)
if args.k < 1: raise ValueError(f"k  must  >= 1,  currently  {args.k}")
if args.block_height != 1 or args.block_width != 1: raise ValueError("FLOW  mode requires  block_height=1  and  block_width=1.")

# ============================================================

# ============================================================

os.makedirs(args.output_dir, exist_ok=True)
flow_dir = os.path.join(args.output_dir, "flow")
os.makedirs(flow_dir, exist_ok=True)

for path in glob.glob(os.path.join(args.output_dir, "img*.png")): os.remove(path)
for path in glob.glob(os.path.join(flow_dir, "img*.npz")): os.remove(path)

print("FRAME1 =", args.frame1)
print("FRAME2 =", args.frame2)
print("MODEL_PATH =", args.model_path)
print("OUTPUT_DIR =", args.output_dir)
print("K =", args.k)

# ============================================================

# ============================================================

model = tf.compat.v2.saved_model.load(args.model_path)
printed_keys = False

# ============================================================
# 3. padding / crop
# ============================================================

def pad_to_align(image, align):
    h, w = image.shape[:2]
    h_pad = align - h % align if h % align else 0
    w_pad = align - w % align if w % align else 0
    top, left = h_pad // 2, w_pad // 2

    x = tf.image.pad_to_bounding_box(
        image[np.newaxis, ...].astype(np.float32),
        top, left, h + h_pad, w + w_pad
    )

    return x, (top, left, h, w)

def crop4(x, crop):
    top, left, h, w = crop
    return tf.image.crop_to_bounding_box(x, top, left, h, w)

# ============================================================

# ============================================================

def finest_flow(result, key):
    if key not in result: raise KeyError(f"SavedModel  missing  {key};  actual  keys = {list(result.keys())}")

    flat = tf.nest.flatten(result[key])
    if not flat: raise RuntimeError(f"{key}  is empty ")

    flow = tf.convert_to_tensor(flat[0])
    if flow.shape.rank != 4 or flow.shape[-1] != 2: raise RuntimeError(f"{key} finest flow  invalid shape : {flow.shape}")

    return flow

# ============================================================

# ============================================================

def in_bounds(flow):
    h, w = flow.shape[:2]
    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")

    qx = xx + flow[..., 0]
    qy = yy + flow[..., 1]

    return np.isfinite(flow).all(axis=-1) & (qx >= 0) & (qx <= w - 1) & (qy >= 0) & (qy <= h - 1)

# ============================================================

# ============================================================

def bilinear(field, x, y):
    field = np.asarray(field, dtype=np.float32)
    if field.ndim == 2: field = field[..., np.newaxis]

    h, w = field.shape[:2]

    x = np.clip(x, 0, w - 1)
    y = np.clip(y, 0, h - 1)

    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)

    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)

    wx = (x - x0)[..., np.newaxis]
    wy = (y - y0)[..., np.newaxis]

    return (
        field[y0, x0] * (1 - wx) * (1 - wy)
        + field[y0, x1] * wx * (1 - wy)
        + field[y1, x0] * (1 - wx) * wy
        + field[y1, x1] * wx * wy
    )

# ============================================================
# 7. Flow composition
#
# F_AC(p) = F_AB(p) + F_BC(p + F_AB(p))
# ============================================================

def compose(flow_ab, valid_ab, flow_bc, valid_bc):
    h, w = flow_ab.shape[:2]
    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")

    qx = xx + flow_ab[..., 0]
    qy = yy + flow_ab[..., 1]

    inside = (qx >= 0) & (qx <= w - 1) & (qy >= 0) & (qy <= h - 1)

    sampled_flow_bc = bilinear(flow_bc, qx, qy)
    sampled_valid_bc = bilinear(valid_bc.astype(np.float32), qx, qy)[..., 0] > 0.999

    flow_ac = flow_ab + sampled_flow_bc
    valid_ac = valid_ab & inside & sampled_valid_bc & np.isfinite(flow_ac).all(axis=-1)

    flow_ac = np.where(np.isfinite(flow_ac), flow_ac, 0).astype(np.float32)

    return flow_ac, valid_ac

# ============================================================

# ============================================================

def infer_mid(frame_left, frame_right):
    global printed_keys

    x0, crop = pad_to_align(frame_left, args.align)
    x1, _ = pad_to_align(frame_right, args.align)

    result = model(
        {"x0": x0, "x1": x1, "time": np.array([[0.5]], dtype=np.float32)},
        training=False
    )

    if not printed_keys:
        print("SavedModel output keys =", list(result.keys()))
        printed_keys = True

    image = crop4(result["image"], crop)[0].numpy().astype(np.float32)

    forward = crop4(finest_flow(result, "forward_flow_pyramid"), crop)[0].numpy().astype(np.float32)
    backward = crop4(finest_flow(result, "backward_flow_pyramid"), crop)[0].numpy().astype(np.float32)

    
    flow_to_left = 0.5 * backward

    
    flow_to_right = 0.5 * forward

    valid_to_left = in_bounds(flow_to_left)
    valid_to_right = in_bounds(flow_to_right)

    return image, flow_to_left, flow_to_right, valid_to_left, valid_to_right

# ============================================================

# ============================================================

# ============================================================

# ============================================================

def save_flow(index, t, flow_left, flow_right, valid_left, valid_right):
    flow_left_save = flow_left.copy()
    flow_right_save = flow_right.copy()

    flow_left_save[~valid_left] = np.nan
    flow_right_save[~valid_right] = np.nan

    np.savez_compressed(
        os.path.join(flow_dir, f"img{index}.npz"),
        flow_to_left=flow_left_save.astype(np.float32),
        flow_to_right=flow_right_save.astype(np.float32),
        valid_to_left=valid_left.astype(np.uint8),
        valid_to_right=valid_right.astype(np.uint8),

        
        ratio=np.float32(t),

        
        time=np.float32(t),

        flow_channel_order=np.array("dx_dy"),
    )

    print(
        f"Saved img{index}.npz | ratio={t:.6f} | "
        f"valid_left={valid_left.mean():.4f} | "
        f"valid_right={valid_right.mean():.4f}"
    )

# ============================================================

# ============================================================

denom = 2 ** args.k

left_image = util.read_image(args.frame1).astype(np.float32)
right_image = util.read_image(args.frame2).astype(np.float32)

if left_image.shape != right_image.shape:
    raise ValueError(f" endpoint image sizes differ : {left_image.shape} vs {right_image.shape}")

h, w = left_image.shape[:2]

zero_flow = np.zeros((h, w, 2), dtype=np.float32)
all_valid = np.ones((h, w), dtype=bool)

left_node = {
    "image": left_image,
    "t": 0.0,
    "flow_to_left": zero_flow.copy(),
    "valid_to_left": all_valid.copy(),
    "flow_to_right": None,
    "valid_to_right": None,
}

right_node = {
    "image": right_image,
    "t": 1.0,
    "flow_to_left": None,
    "valid_to_left": None,
    "flow_to_right": zero_flow.copy(),
    "valid_to_right": all_valid.copy(),
}

util.write_image(os.path.join(args.output_dir, "img0.png"), left_image)
util.write_image(os.path.join(args.output_dir, f"img{denom}.png"), right_image)

# ============================================================

# ============================================================

def recurse(left, right, depth):
    if depth == 0: return

    mid_image, flow_mid_left_local, flow_mid_right_local, valid_mid_left_local, valid_mid_right_local = infer_mid(
        left["image"], right["image"]
    )

    
    flow_mid_global_left, valid_mid_global_left = compose(
        flow_mid_left_local,
        valid_mid_left_local,
        left["flow_to_left"],
        left["valid_to_left"],
    )

    
    flow_mid_global_right, valid_mid_global_right = compose(
        flow_mid_right_local,
        valid_mid_right_local,
        right["flow_to_right"],
        right["valid_to_right"],
    )

    mid_t = 0.5 * (left["t"] + right["t"])
    mid_index = int(round(mid_t * denom))

    util.write_image(os.path.join(args.output_dir, f"img{mid_index}.png"), mid_image)

    save_flow(
        mid_index,
        mid_t,
        flow_mid_global_left,
        flow_mid_global_right,
        valid_mid_global_left,
        valid_mid_global_right,
    )

    mid_node = {
        "image": mid_image,
        "t": mid_t,
        "flow_to_left": flow_mid_global_left,
        "valid_to_left": valid_mid_global_left,
        "flow_to_right": flow_mid_global_right,
        "valid_to_right": valid_mid_global_right,
    }

    recurse(left, mid_node, depth - 1)
    recurse(mid_node, right, depth - 1)

# ============================================================

# ============================================================

recurse(left_node, right_node, args.k)

# ============================================================

# ============================================================

expected_images = [os.path.join(args.output_dir, f"img{i}.png") for i in range(denom + 1)]
expected_flows = [os.path.join(flow_dir, f"img{i}.npz") for i in range(1, denom)]

missing_images = [p for p in expected_images if not os.path.isfile(p)]
missing_flows = [p for p in expected_flows if not os.path.isfile(p)]

if missing_images: raise RuntimeError(f" missing images : {missing_images}")
if missing_flows: raise RuntimeError(f" missing  FLOW: {missing_flows}")

print(f"\n complete : {denom + 1}  images , {denom - 1}  intermediate frames  FLOW.")
print("FLOW  define : flow[y, x] = (dx, dy)")
print(" left coordinates : (x_left, y_left) = (x, y) + flow_to_left[y, x]")
print(" right coordinates : (x_right, y_right) = (x, y) + flow_to_right[y, x]")
print("FLOW_DIR =", flow_dir)

