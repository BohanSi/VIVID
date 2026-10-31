# Migrated from 1002genertation.ipynb, cell 36. See README.md.
if interpolation_model_names == ["FILM"]:

    import re, sys, shutil, subprocess
    import numpy as np
    from pathlib import Path
    from PIL import Image


    # ============================================================
    # 1. Determine FILM recursive depth
    #
    
    
    # ============================================================

    def find_min_n_by_search(K, threshold=0.02):
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


    K = int(config["target_number"])
    film_position_threshold = 0.02

    film_K, total_frames, segments, target_img, max_error = find_min_n_by_search(
        K=K,
        threshold=film_position_threshold,
    )

    print("\n" + "=" * 100)
    print("FILM configuration")
    print("target K              =", K)
    print("FILM recursive depth  =", film_K)
    print("segments              =", segments)
    print("intermediate frames   =", total_frames)
    print("target img indices    =", target_img)
    print("target ratios         =", [round(i / segments, 6) for i in target_img])
    print("maximum ratio error   =", max_error)
    print("=" * 100)


    # ============================================================
    # 2. Paths
    # ============================================================

    FILM_dir = Path(backend_dir).resolve()
    film_model_path = (FILM_dir / "pretrained_models/film_net/Style/saved_model").resolve()
    film_runner_path = FILM_dir / "run_film_pair_from_jupyter.py"

    latent_input_dir = Path(save_dir).resolve() / "latent_images"
    film_global_output_root = Path(save_dir).resolve() / "FILM_result"

    overwrite_film_result = True

    for path, name in [
        (Path(backend_python), "FILM python"),
        (FILM_dir, "FILM_dir"),
        (film_model_path, "FILM model"),
        (film_runner_path, "FILM runner"),
        (latent_input_dir, "latent RGB input"),
    ]:
        if not path.exists(): raise FileNotFoundError(f"{name} not found : {path}")

    if blend_mode == "mask":
        raise ValueError("FILM currently missing RIFE style learned mask,  use  blend_mode='distance'")

    if use_mask:
        raise ValueError("FILM currently missing RIFE style learned mask,  set  use_mask=False")


    # ============================================================
    # 3. Utilities
    # ============================================================

    def run_cmd(cmd, cwd):
        cmd = [str(x) for x in cmd]

        print("-" * 100)
        print("Command:")
        print(" ".join(cmd))
        print("cwd:", cwd)
        print("-" * 100)

        result = subprocess.run(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        print(result.stdout)

        if result.returncode != 0:
            raise RuntimeError(
                f"FILM failed, returncode={result.returncode}\n"
                f"{' '.join(cmd)}"
            )

        return result


    def validate_image_group(image_files, check_nonwhite=True):
        if not image_files: raise ValueError("image_files is empty ")

        expected_size = None

        for p in map(Path, image_files):
            if not p.exists(): raise FileNotFoundError(p)

            with Image.open(p) as img:
                if img.mode != "RGB":
                    raise ValueError(f"{p.name}: image mode={img.mode},  expected RGB")

                if expected_size is None:
                    expected_size = img.size
                elif img.size != expected_size:
                    raise ValueError(
                        f"{p.name}: size={img.size},  expected ={expected_size}"
                    )

                if check_nonwhite:
                    arr = np.asarray(img, dtype=np.uint8)

                    if not np.any(arr != 255):
                        raise ValueError(f"{p.name}:  image is entirely white ")

        return expected_size


    def validate_film_flow_npz(flow_file, expected_h=None, expected_w=None, expected_ratio=None, atol=1e-5):
        with np.load(flow_file) as f:
            required = ["flow_to_left", "flow_to_right", "valid_to_left", "valid_to_right", "ratio"]
            missing = [k for k in required if k not in f.files]
            if missing: raise KeyError(f"{flow_file}:  missing  {missing}")
    
            flow_left = np.asarray(f["flow_to_left"], dtype=np.float32)
            flow_right = np.asarray(f["flow_to_right"], dtype=np.float32)
            valid_left = np.asarray(f["valid_to_left"], dtype=bool)
            valid_right = np.asarray(f["valid_to_right"], dtype=bool)
            ratio = float(f["ratio"])
    
        if flow_left.ndim != 3 or flow_left.shape[-1] != 2:
            raise ValueError(f"{flow_file}: flow_to_left shape={flow_left.shape}")
    
        if flow_right.shape != flow_left.shape:
            raise ValueError(f"{flow_file}:  left and right FLOW shape inconsistent ")
    
        if valid_left.shape != flow_left.shape[:2]:
            raise ValueError(f"{flow_file}: valid_to_left shape={valid_left.shape}")
    
        if valid_right.shape != flow_left.shape[:2]:
            raise ValueError(f"{flow_file}: valid_to_right shape={valid_right.shape}")
    
        if expected_h is not None and expected_w is not None and flow_left.shape[:2] != (expected_h, expected_w):
            raise ValueError(
                f"{flow_file}: FLOW size={flow_left.shape[:2]}, "
                f" expected ={(expected_h, expected_w)}"
            )
    
        
        if not np.isfinite(flow_left[valid_left]).all():
            raise ValueError(f"{flow_file}: valid_to_left=True positions contain NaN/Inf")
    
        if not np.isfinite(flow_right[valid_right]).all():
            raise ValueError(f"{flow_file}: valid_to_right=True positions contain NaN/Inf")
    
        if expected_ratio is not None and abs(ratio - float(expected_ratio)) > atol:
            raise ValueError(
                f"{flow_file}: ratio={ratio:.8f}, "
                f" expected ={expected_ratio:.8f}"
            )
    
        return {
            "flow_shape": tuple(flow_left.shape),
            "ratio": ratio,
            "valid_left_fraction": float(valid_left.mean()),
            "valid_right_fraction": float(valid_right.mean()),
            "invalid_left_fraction": float((~valid_left).mean()),
            "invalid_right_fraction": float((~valid_right).mean()),
        }


    def film_clean_tag(path):
        return re.sub(
            r"[^0-9A-Za-z_.-]+",
            "_",
            Path(path).stem,
        ).strip("_")


    def film_pair_name(left, right):
        return f"{film_clean_tag(left)}_to_{film_clean_tag(right)}"


    # ============================================================
    # 4. Collect latent RGB images
    #
    
    
    # ============================================================

    if "summary" not in globals():
        raise NameError(" cannot find summary,  generate first latent RGB images")

    if "outputs" not in globals():
        raise NameError(" cannot find outputs")

    slice_names = list(outputs.keys())
    image_files = []

    for slice_name in slice_names:
        if slice_name not in summary:
            raise KeyError(
                f"{slice_name} not found in latent image summary in "
            )

        png_path = Path(summary[slice_name]["png_path"]).resolve()

        if not png_path.exists():
            raise FileNotFoundError(
                f"{slice_name} latent RGB not found : {png_path}"
            )

        image_files.append(png_path)

    if len(image_files) < 2:
        raise ValueError("FILM requires at least two observed slice images ")

    W, H = validate_image_group(
        image_files,
        check_nonwhite=True,
    )

    slice_pairs = list(zip(
        image_files[:-1],
        image_files[1:],
    ))

    num_intermediate_frames = segments - 1

    film_global_output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\n" + "=" * 100)
    print("FILM INTERPOLATION ON LATENT RGB")
    print("=" * 100)
    print("Input :", latent_input_dir)
    print("Output:", film_global_output_root)
    print("Images:", len(image_files))
    print("Pairs :", len(slice_pairs))
    print("Size  :", f"{W} × {H}")
    print("K     :", film_K)
    print("segments:", segments)
    print("intermediate frames per pair:", num_intermediate_frames)
    print("=" * 100)


    # ============================================================
    # 5. FILM inference
    #
    
    # ============================================================

    global_flow_results = {}

    for pair_idx, (img1, img2) in enumerate(
        slice_pairs,
        start=1,
    ):
        pair_name = film_pair_name(img1, img2)
        output_dir = film_global_output_root / pair_name

        if output_dir.exists() and overwrite_film_result:
            shutil.rmtree(output_dir)

        if output_dir.exists():
            raise FileExistsError(
                f" output directory already exists : {output_dir}"
            )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        print(
            f"\n{'#' * 100}\n"
            f"PAIR {pair_idx}/{len(slice_pairs)}\n"
            f"{img1.name} -> {img2.name}\n"
            f"Output: {output_dir}\n"
            f"{'#' * 100}"
        )

        film_cmd = [
            backend_python,
            str(film_runner_path),

            "--frame1", str(img1),
            "--frame2", str(img2),

            "--output_dir", str(output_dir),
            "--model_path", str(film_model_path),

            "--k", str(film_K),

            "--align", "64",
            "--block_height", "1",
            "--block_width", "1",
        ]

        run_cmd(
            film_cmd,
            cwd=FILM_dir,
        )

        # --------------------------------------------------------
        # Check images
        # --------------------------------------------------------

        generated_images = [
            output_dir / f"img{i}.png"
            for i in range(segments + 1)
        ]

        missing_images = [
            p for p in generated_images
            if not p.exists()
        ]

        if missing_images:
            raise FileNotFoundError(
                f"{pair_name}:  missing interpolated images : {missing_images}"
            )

        generated_size = validate_image_group(
            generated_images,
            check_nonwhite=False,
        )

        if generated_size != (W, H):
            raise RuntimeError(
                f"{pair_name}: FILM output size ={generated_size}, "
                f" input size ={(W, H)}"
            )

        # --------------------------------------------------------
        # Check FLOW
        # --------------------------------------------------------

        flow_dir = output_dir / "flow"

        if not flow_dir.exists():
            raise RuntimeError(
                f"{pair_name}:  missing flow directory "
            )

        generated_npz = [
            flow_dir / f"img{i}.npz"
            for i in range(1, segments)
        ]

        missing_npz = [
            p for p in generated_npz
            if not p.exists()
        ]

        if missing_npz:
            raise FileNotFoundError(
                f"{pair_name}:  missing FLOW file : {missing_npz}"
            )

        frame_results = []

        for img_idx in range(1, segments):
            image_file = output_dir / f"img{img_idx}.png"
            flow_file = flow_dir / f"img{img_idx}.npz"
            expected_ratio = img_idx / segments

            info = validate_film_flow_npz(
                flow_file,
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
                f"valid_left={info['valid_left_fraction']:.4f} | "
                f"valid_right={info['valid_right_fraction']:.4f}"
            )

        global_flow_results[pair_name] = {
            "left_image": img1,
            "right_image": img2,
            "output_dir": output_dir,
            "flow_dir": flow_dir,
            "frames": frame_results,
        }

        print(f"\nFinished: {pair_name}")
        print("Images:", output_dir)
        print("FLOW  :", flow_dir)


    print(
        f"\nFILM inference completed | "
        f"pairs={len(global_flow_results)} | "
        f"target frames={target_img}"
    )