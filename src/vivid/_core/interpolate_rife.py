# Migrated from 1002genertation.ipynb, cell 30. See docs/MIGRATION.md.
if interpolation_model_names == ["RIFE"]:  
    
    def find_min_n_by_search(K, threshold=0.02):
        """Find min n by search."""
    
        if K <= 0: return 0, 0, 1, [], 0.0
    
        n = 1
    
        while True:
    
            segments = 2 ** n
            total_frames = segments - 1
    
            actual_positions = np.arange(1, segments, dtype=float) / segments
            target_positions = np.arange(1, K + 1, dtype=float) / (K + 1)
    
            target_img, errors = [], []
    
            for target in target_positions:
                nearest_idx = int(np.argmin(np.abs(actual_positions - target)))
                target_img.append(nearest_idx + 1)
                errors.append(float(abs(actual_positions[nearest_idx] - target)))
    
            max_error = max(errors)
    
            if max_error <= threshold or n > 20: break
    
            n += 1
    
        return n, total_frames, segments, target_img, max_error
    
    
    K = int(config['target_number'])
    rife_position_threshold = 0.02 
    
    target_rife_exp, total_frames, segments, target_img, max_error = find_min_n_by_search(
        K=K,
        threshold=rife_position_threshold,
    )
    
    print("\n" + "=" * 100)
    print("RIFE configuration")
    print("target K              =", K)
    print("RIFE exp              =", target_rife_exp)
    print("segments              =", segments)
    print("intermediate frames   =", total_frames)
    print("target img indices    =", target_img)
    print("target ratios         =", [round(i / segments, 6) for i in target_img])
    print("maximum ratio error   =", max_error)
    print("=" * 100)
    # ============================================================
    # 6. RIFE interpolation on AE latent RGB images
    # ============================================================
    
    import re, sys, shutil, subprocess
    import numpy as np
    from pathlib import Path
    from PIL import Image
    
    
    # ============================================================
    # 6.0 Paths
    # ============================================================
    
    RIFE_dir = Path(backend_dir).resolve()
    rife_model_dir = RIFE_dir / "train_log"
    rife_flow_script = RIFE_dir / "inference_img_flow.py"
    
    
    latent_input_dir = Path(save_dir).resolve() / "latent_images"
    
    
    rife_global_output_root = Path(save_dir).resolve() / "RIFE_result"
    
    
    # ============================================================
    # 6.1 Utilities
    # ============================================================
    
    def run_cmd(cmd, cwd):
        print("-"*100)
        print("Command:")
        print(" ".join(map(str, cmd)))
        print("cwd:", cwd)
        print("-"*100)
    
        result = subprocess.run(
            [str(x) for x in cmd], cwd=str(cwd),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
        )
    
        print(result.stdout)
    
        if result.returncode != 0:
            raise RuntimeError(f"RIFE failed, returncode={result.returncode}\n{' '.join(map(str, cmd))}")
    
        return result
    
    
    def validate_image_group(image_files, check_nonwhite=True):
        """Validate image group."""
        if len(image_files) == 0: raise ValueError("image_files  is empty .")
    
        expected_size = None
    
        for p in image_files:
            p = Path(p)
    
            if not p.exists(): raise FileNotFoundError(f" input image not found : {p}")
    
            with Image.open(p) as img:
                if img.mode != "RGB": raise ValueError(f"{p.name}: image mode={img.mode},  expected  RGB")
    
                size = img.size
    
                if expected_size is None: expected_size = size
                elif size != expected_size: raise ValueError(f"{p.name}: size={size},  expected ={expected_size}")
    
                if check_nonwhite:
                    arr = np.asarray(img, dtype=np.uint8)
                    n_nonwhite = int(np.any(arr != 255, axis=-1).sum())
    
                    if n_nonwhite == 0: raise ValueError(f"{p.name}:  image is entirely white ,  no valid  latent RGB  pixels .")
    
                    print(f"  {p.name}: size={size}, nonwhite_pixels={n_nonwhite}")
    
        return expected_size
    
    
    def validate_flow_npz(flow_file, expected_h=None, expected_w=None, expected_ratio=None, atol=1e-4):
        """Validate flow npz."""
        required_keys = ["flow_to_left", "mask_to_left", "flow_to_right", "mask_to_right", "ratio"]
    
        with np.load(flow_file) as data:
            missing = [x for x in required_keys if x not in data.files]
            if missing: raise KeyError(f"{flow_file}:  missing  {missing}")
    
            flow_left = np.asarray(data["flow_to_left"], dtype=np.float32)
            flow_right = np.asarray(data["flow_to_right"], dtype=np.float32)
            mask_left = np.asarray(data["mask_to_left"], dtype=np.float32)
            mask_right = np.asarray(data["mask_to_right"], dtype=np.float32)
            ratio = float(data["ratio"])
    
        if flow_left.ndim != 3 or flow_left.shape[-1] != 2: raise ValueError(f"{flow_file}: flow_to_left shape={flow_left.shape},  expected  H×W×2")
        if flow_right.shape != flow_left.shape: raise ValueError(f"{flow_file}: flow_to_right shape={flow_right.shape} != {flow_left.shape}")
        if mask_left.shape != flow_left.shape[:2]: raise ValueError(f"{flow_file}: mask_to_left shape={mask_left.shape} != {flow_left.shape[:2]}")
        if mask_right.shape != flow_left.shape[:2]: raise ValueError(f"{flow_file}: mask_to_right shape={mask_right.shape} != {flow_left.shape[:2]}")
    
        if expected_h is not None and expected_w is not None:
            if flow_left.shape[:2] != (expected_h, expected_w):
                raise ValueError(f"{flow_file}: flow size={flow_left.shape[:2]},  expected ={(expected_h, expected_w)}")
    
        if not np.isfinite(flow_left).all(): raise ValueError(f"{flow_file}: flow_to_left  contains  NaN/Inf")
        if not np.isfinite(flow_right).all(): raise ValueError(f"{flow_file}: flow_to_right  contains  NaN/Inf")
        if not np.isfinite(mask_left).all(): raise ValueError(f"{flow_file}: mask_to_left  contains  NaN/Inf")
        if not np.isfinite(mask_right).all(): raise ValueError(f"{flow_file}: mask_to_right  contains  NaN/Inf")
    
        if mask_left.min() < -atol or mask_left.max() > 1 + atol: raise ValueError(f"{flow_file}: mask_to_left  outside  [0,1]")
        if mask_right.min() < -atol or mask_right.max() > 1 + atol: raise ValueError(f"{flow_file}: mask_to_right  outside  [0,1]")
    
        mask_sum_error = float(np.max(np.abs(mask_left + mask_right - 1.0)))
        if mask_sum_error > atol: raise ValueError(f"{flow_file}: mask_to_left + mask_to_right != 1, max error={mask_sum_error:.3e}")
    
        if expected_ratio is not None and abs(ratio - float(expected_ratio)) > 1e-5:
            raise ValueError(f"{flow_file}: ratio={ratio:.8f},  expected ={expected_ratio:.8f}")
    
        return {
            "flow_shape": tuple(flow_left.shape),
            "mask_shape": tuple(mask_left.shape),
            "ratio": ratio,
            "mask_sum_error": mask_sum_error,
        }
    
    
    def rife_natural_key(path):
        return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", Path(path).name)]
    
    
    def rife_clean_tag(path):
        return re.sub(r"[^0-9A-Za-z_.-]+", "_", Path(path).stem).strip("_")
    
    
    def rife_pair_name(left, right):
        return f"{rife_clean_tag(left)}_to_{rife_clean_tag(right)}"
    
    
    
    # ============================================================
    # 6.2 Basic settings
    # ============================================================
    
    python_exe = backend_python
    
    if not RIFE_dir.exists(): raise FileNotFoundError(f" cannot find  RIFE_dir: {RIFE_dir}")
    if not rife_model_dir.exists(): raise FileNotFoundError(f" cannot find  RIFE model: {rife_model_dir}")
    if not rife_flow_script.exists(): raise FileNotFoundError(f" cannot find  inference_img_flow.py: {rife_flow_script}")
    if not latent_input_dir.exists(): raise FileNotFoundError(f" cannot find  latent RGB  input directory : {latent_input_dir}")
    
    if "target_rife_exp" not in globals(): raise NameError(" define first  target_rife_exp")
    if "overwrite_rife_result" not in globals(): overwrite_rife_result = True
    
    rife_exp = int(target_rife_exp)
    
    
    
    if rife_exp < 1: raise ValueError("target_rife_exp  must  >= 1")
    
    segments = 2 ** rife_exp
    num_intermediate_frames = segments - 1
    
    rife_global_output_root.mkdir(parents=True, exist_ok=True)
    
    
    
    
    
    # ============================================================
    # 6.3 Collect AE latent RGB images
    # ============================================================
    
    if "summary" not in globals(): raise NameError(" cannot find  summary,  run first  create_dominant_type_latent_images(...)")
    if "outputs" not in globals(): raise NameError(" cannot find  outputs")
    
    slice_names = list(outputs.keys())
    image_files = []
    
    for slice_name in slice_names:
        if slice_name not in summary: raise KeyError(f"{slice_name}  not found in  latent image summary  in ")
    
        png_path = Path(summary[slice_name]["png_path"]).resolve()
    
        if not png_path.exists(): raise FileNotFoundError(f"{slice_name}  of  latent RGB  not found : {png_path}")
    
        image_files.append(png_path)
    
    if len(image_files) < 2: raise ValueError("RIFE  requires at least two observed slice images .")
    
    W, H = validate_image_group(image_files, check_nonwhite=True)
    
    slice_pairs = list(zip(image_files[:-1], image_files[1:]))
    
    
    print("\n"+"="*100)
    print("RIFE INTERPOLATION ON AE LATENT RGB")
    print("="*100)
    print("Input :", latent_input_dir)
    print("Output:", rife_global_output_root)
    print("Images:", len(image_files))
    print("Pairs :", len(slice_pairs))
    print("Size  :", f"{W} × {H}")
    print("exp   :", rife_exp)
    print("segments:", segments)
    print("intermediate frames per pair:", num_intermediate_frames)
    print("="*100)
    
    print("\nInput slice order:")
    for i, (slice_name, image_file) in enumerate(zip(slice_names, image_files), start=1):
        print(f"  {i:3d} | {slice_name} -> {image_file.name}")
    
    print("\nInterpolation pairs:")
    for img1, img2 in slice_pairs:
        print(" ", img1.name, "->", img2.name)
    
    
    # ============================================================
    # 6.4 Run RIFE
    # ============================================================
    
    global_flow_results = {}
    
    for pair_idx, (img1, img2) in enumerate(slice_pairs, start=1):
    
        pair_name = rife_pair_name(img1, img2)
        output_dir = rife_global_output_root / pair_name
    
        if output_dir.exists() and overwrite_rife_result: shutil.rmtree(output_dir)
        if output_dir.exists(): raise FileExistsError(f" output directory already exists : {output_dir}")
    
        output_dir.mkdir(parents=True, exist_ok=True)
    
        print("\n"+"#"*100)
        print(f"PAIR {pair_idx}/{len(slice_pairs)}")
        print(f"{img1.name} -> {img2.name}")
        print("Output:", output_dir)
        print("#"*100)
    
        infer_cmd = [
            python_exe,
            str(rife_flow_script),
            "--img", str(img1), str(img2),
            "--exp", str(rife_exp),
            "--model", str(rife_model_dir),
        ]
    
        # Randomly initialized RIFE
        if rife_random_init:
            infer_cmd += [
                "--random_init",
                "--random_init_seed", str(rife_random_init_seed),
            ]
        
        # No correspondence
        if rife_zero_correspondence:
            infer_cmd += [
                "--zero_correspondence",
            ]
        
        # Correspondence perturbation
        if float(rife_perturb_fraction) > 0.0:
            infer_cmd += [
                "--perturb_fraction", str(rife_perturb_fraction),
                "--perturb_seed", str(rife_perturb_seed),
            ]
        
        # Uniform mask
        if rife_uniform_mask:
            infer_cmd += [
                "--uniform_mask",
            ]
        
        run_cmd(infer_cmd, cwd=output_dir)
    
        # ========================================================
        # A. Check interpolated RGB images
        # ========================================================
    
        generated_images = sorted(output_dir.glob("img*.png"), key=rife_natural_key)
        expected_image_count = segments + 1
    
        if len(generated_images) != expected_image_count:
            raise RuntimeError(
                f"{pair_name}:  number of generated images ={len(generated_images)}, "
                f"exp={rife_exp}  expected when ={expected_image_count}"
            )
    
        expected_names = [f"img{i}.png" for i in range(segments + 1)]
        actual_names = [p.name for p in generated_images]
    
        if actual_names != expected_names:
            raise RuntimeError(f"{pair_name}:  unexpected image indices \n actual ={actual_names}\n expected ={expected_names}")
    
        generated_size = validate_image_group(generated_images, check_nonwhite=False)
    
        if generated_size != (W, H):
            raise RuntimeError(f"{pair_name}: RIFE output size ={generated_size},  input size ={(W,H)}")
    
        # ========================================================
        # B. Check FLOW + MASK
        # ========================================================
    
        flow_dir = output_dir / "flow"
    
        if not flow_dir.exists(): raise RuntimeError(f"{pair_name}:  not generated  flow  directory ")
    
        generated_npz = sorted(flow_dir.glob("img*.npz"), key=rife_natural_key)
    
        if len(generated_npz) != num_intermediate_frames:
            raise RuntimeError(
                f"{pair_name}: NPZ count ={len(generated_npz)}, "
                f" expected ={num_intermediate_frames}"
            )
    
        # ========================================================
        # C. Image <-> flow/mask exact matching
        # ========================================================
    
        frame_results = []
    
        for img_idx in range(1, segments):
    
            image_file = output_dir / f"img{img_idx}.png"
            flow_file = flow_dir / f"img{img_idx}.npz"
    
            if not image_file.exists(): raise FileNotFoundError(f" missing interpolated image : {image_file}")
            if not flow_file.exists(): raise FileNotFoundError(f" missing corresponding  FLOW+MASK: {flow_file}")
    
            expected_ratio = img_idx / segments
    
            info = validate_flow_npz(
                flow_file=flow_file,
                expected_h=H,
                expected_w=W,
                expected_ratio=expected_ratio,
            )
    
            frame_results.append({
                "image_index": int(img_idx),
                "ratio": float(info["ratio"]),
                "image_file": image_file,
                "flow_file": flow_file,
            })
    
            print(
                f"  img{img_idx} | "
                f"t={info['ratio']:.6f} | "
                f"flow={info['flow_shape']} | "
                f"mask={info['mask_shape']} | "
                f"mask error={info['mask_sum_error']:.2e}"
            )
    
        global_flow_results[pair_name] = {
            "left_image": img1,
            "right_image": img2,
            "output_dir": output_dir,
            "flow_dir": flow_dir,
            "frames": frame_results,
        }
    
        print(f"\nFinished: {pair_name}")
        print("Images      :", output_dir)
        print("FLOW + MASK :", flow_dir)