# Migrated from 1002genertation.ipynb, cell 26. See docs/MIGRATION.md.
if config["image_encoding_method"] == "gnn": 

    # ============================================================
    # 5. RUN TRAINING
    # ============================================================
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ae_save_dir = Path(save_dir) / "dominant_type_ae_results"
    ae_save_dir.mkdir(parents=True, exist_ok=True)
    
    weight_path = ae_save_dir / "dominant_type_autoencoder.pt"
    
    knn_k = int(config.get("knn_k", 20))
    context_eta = float(config.get("context_eta", 0.20))
    
    model, history = train_context_graph_autoencoder(
        anndata_dict=anndata_dict,
        celltypes=celltypes,
        spatial_key="spatial",
        celltype_key=celltype_col,
        slice_names=None,
    
        knn_k=knn_k,
        distance_weighted=True,
    
        val_fraction=config["val_fraction"],
        batch_size=config["batch_size"],
        num_epochs=config["num_epochs"],
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    
        latent_dim=3,
        context_hidden_dims=(128,64),
        decoder_hidden_dims=(64,128),
        eta=context_eta,
        context_scale=1.0,
        dropout=0.0,
    
        w_type=float(config.get("w_type", 1.0)),
        w_env=float(config.get("w_env", 0.5)),
        w_res=float(config.get("w_res", 0.02)),
        w_bg=float(config.get("w_bg", 0.01)),
        m_bg=float(config.get("m_bg", 0.30)),
        use_class_balance=bool(config.get("use_class_balance", True)),
    
        early_stopping=True,
        early_stopping_patience=20,
        early_stopping_min_delta=1e-4,
        restore_best_model=True,
    
        num_workers=0,
        device=device,
        save_path=weight_path,
        seed=config["seed"],
    )
    
    print("\nSaved model:", weight_path)

    # ============================================================
    # 6. Load trained model
    # ============================================================    
    model, ae_checkpoint = load_dominant_type_autoencoder(
        weight_path=weight_path,
        device=device,
        expected_celltypes=celltypes,
    )

    # ============================================================
    # 7. Learned cell-type base prototypes
    # ============================================================
    @torch.no_grad()
    def get_celltype_latent_colors(model, celltypes):
        z = model.get_prototypes().detach().cpu().numpy().astype(np.float32)
        rgb = np.rint(np.clip(z, 0, 1) * 255).astype(np.uint8)
        return z, rgb
    
    celltype_latent, celltype_rgb = get_celltype_latent_colors(model, celltypes)
    
    print("\n"+"="*100)
    print("LEARNED CELL-TYPE BASE PROTOTYPES")
    print("="*100)
    
    for k, ct in enumerate(celltypes):
        print(f"{k:3d} | {ct} | prototype=({celltype_latent[k,0]:.4f}, {celltype_latent[k,1]:.4f}, {celltype_latent[k,2]:.4f}) | RGB=({celltype_rgb[k,0]:3d}, {celltype_rgb[k,1]:3d}, {celltype_rgb[k,2]:3d})")
    
    celltype_color_path = ae_save_dir / "celltype_latent_rgb.json"
    
    with open(celltype_color_path, "w", encoding="utf-8") as f:
        json.dump({
            "representation": "context_aware_celltype_prototype",
            "note": "These are base prototypes. Actual cell RGB is context-dependent.",
            "celltypes": [
                {"index": int(k), "celltype": str(ct), "latent": celltype_latent[k].tolist(), "rgb": celltype_rgb[k].astype(int).tolist()}
                for k, ct in enumerate(celltypes)
            ],
        }, f, ensure_ascii=False, indent=2)
    print("Saved:", celltype_color_path)


    # ============================================================
    # 8. Encode every real cell
    # ============================================================
    
    @torch.no_grad()
    def encode_one_slice_cells(
        model, adata, celltypes,
        spatial_key="spatial", celltype_key="cell_type",
        knn_k=20, distance_weighted=True,
        batch_size=8192, device=None,
    ):
        if device is None: device = next(model.parameters()).device
    
        d = compute_cell_knn_context(
            adata=adata, celltypes=celltypes,
            spatial_key=spatial_key, celltype_key=celltype_key,
            k=knn_k, distance_weighted=distance_weighted,
        )
    
        z_list, delta_list = [], []
        model.eval()
    
        for start in range(0, len(d["labels"]), batch_size):
            end = min(start + batch_size, len(d["labels"]))
            self_x = torch.from_numpy(d["self_x"][start:end]).to(device)
            neighbor_P = torch.from_numpy(d["neighbor_P"][start:end]).to(device)
    
            z, delta = model.encode(self_x, neighbor_P)
            z_list.append(z.cpu().numpy().astype(np.float32))
            delta_list.append(delta.cpu().numpy().astype(np.float32))
    
        z = np.concatenate(z_list, axis=0)
        delta = np.concatenate(delta_list, axis=0)
        rgb = np.rint(np.clip(z, 0, 1) * 255).astype(np.uint8)
    
        d["z"] = z; d["delta"] = delta; d["rgb"] = rgb
        return d
    
    
    # ============================================================
    # 9. Cell RGB -> normalized Gaussian splatting image
    # ============================================================
    
    def render_context_cells_to_rgb(
        coords, z, grid_meta,
        sigma_pixels=1.0, truncate=4.0,
        occupancy_gain=3.0,
        background="white",
        eps=1e-8,
    ):
        H, W = int(grid_meta["H"]), int(grid_meta["W"])
        coords = np.asarray(coords, dtype=np.float32)
        z = np.asarray(z, dtype=np.float32)
    
        u = np.floor((coords[:,0] - grid_meta["x_min"]) / grid_meta["pixel_size_x"]).astype(np.int64)
        v = np.floor((coords[:,1] - grid_meta["y_min"]) / grid_meta["pixel_size_y"]).astype(np.int64)
    
        valid = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        u, v, colors = u[valid], v[valid], np.clip(z[valid], 0, 1)
    
        occ = np.zeros((H,W), dtype=np.float32)
        color_sum = np.zeros((H,W,3), dtype=np.float32)
    
        np.add.at(occ, (v,u), 1.0)
        for c in range(3): np.add.at(color_sum[:,:,c], (v,u), colors[:,c])
    
        if sigma_pixels > 0:
            occ_blur = gaussian_filter(occ.astype(np.float64), sigma=float(sigma_pixels), mode="constant", cval=0.0, truncate=float(truncate)).astype(np.float32)
            color_blur = np.stack([
                gaussian_filter(color_sum[:,:,c].astype(np.float64), sigma=float(sigma_pixels), mode="constant", cval=0.0, truncate=float(truncate))
                for c in range(3)
            ], axis=-1).astype(np.float32)
        else:
            occ_blur, color_blur = occ, color_sum
    
        local_color = color_blur / (occ_blur[:,:,None] + eps)
        support_scale = 2.0 * np.pi * float(sigma_pixels) ** 2 if sigma_pixels > 0 else 1.0
        support = occ_blur * support_scale
        alpha = 1.0 - np.exp(-float(occupancy_gain) * support)
        alpha = np.clip(alpha, 0, 1).astype(np.float32)
    
        if background == "white":
            image = (1.0 - alpha[:,:,None]) + alpha[:,:,None] * local_color
        elif background == "black":
            image = alpha[:,:,None] * local_color
        else:
            raise ValueError("background must be 'white' or 'black'.")
    
        image = np.clip(image, 0, 1)
        rgb_uint8 = np.rint(image * 255).astype(np.uint8)
    
        return rgb_uint8, image.astype(np.float32), alpha, occ_blur, u, v

    # ============================================================
    # 10. Create all context-aware latent RGB images
    # ============================================================
    
    @torch.no_grad()
    def create_dominant_type_latent_images(
        model, anndata_dict, grid_meta, celltypes, save_dir,
        spatial_key="spatial", celltype_key="cell_type",
        slice_names=None, knn_k=20, distance_weighted=True,
        batch_size=8192, device=None,
        gaussian_sigma_pixels=1.0, gaussian_truncate=4.0,
        occupancy_gain=3.0, background="white",
        save_png=True, save_npz=True,
    ):
        if device is None: device = next(model.parameters()).device
    
        save_dir = Path(save_dir); save_dir.mkdir(parents=True, exist_ok=True)
        slice_names = list(anndata_dict.keys()) if slice_names is None else list(slice_names)
    
        celltype_latent, celltype_rgb = get_celltype_latent_colors(model, celltypes)
        summary = {}
    
        for name in tqdm(slice_names, desc="Create context-aware latent RGB", dynamic_ncols=True):
            encoded = encode_one_slice_cells(
                model=model, adata=anndata_dict[name], celltypes=celltypes,
                spatial_key=spatial_key, celltype_key=celltype_key,
                knn_k=knn_k, distance_weighted=distance_weighted,
                batch_size=batch_size, device=device,
            )
    
            rgb_img, rgb_float, alpha, occ_blur, xs, ys = render_context_cells_to_rgb(
                coords=encoded["coords"], z=encoded["z"], grid_meta=grid_meta,
                sigma_pixels=gaussian_sigma_pixels,
                truncate=gaussian_truncate,
                occupancy_gain=occupancy_gain,
                background=background,
            )
    
            safe_name = safe_filename(name)
            png_path = save_dir / f"{safe_name}.png"
            npz_path = save_dir / f"{safe_name}_latent_data.npz"
    
            if save_png: Image.fromarray(rgb_img, mode="RGB").save(png_path)
    
            if save_npz:
                np.savez_compressed(
                    npz_path,
                    coords=encoded["coords"].astype(np.float32),
                    obs_indices=encoded["obs_indices"].astype(np.int64),
                    ys=ys.astype(np.int64),
                    xs=xs.astype(np.int64),
                    labels=encoded["labels"].astype(np.int64),
                    self_x=encoded["self_x"].astype(np.float32),
                    neighbor_P=encoded["neighbor_P"].astype(np.float32),
                    z=encoded["z"].astype(np.float32),
                    delta=encoded["delta"].astype(np.float32),
                    rgb_values=encoded["rgb"].astype(np.uint8),
                    rgb_image=rgb_img.astype(np.uint8),
                    alpha=alpha.astype(np.float32),
                    occupancy_blur=occ_blur.astype(np.float32),
                    celltype_latent=celltype_latent.astype(np.float32),
                    celltype_rgb=celltype_rgb.astype(np.uint8),
                    celltypes=np.asarray(celltypes, dtype=str),
                    slice_name=np.array(str(name)),
                    knn_k=np.array(int(knn_k)),
                    gaussian_sigma_pixels=np.array(float(gaussian_sigma_pixels)),
                    gaussian_truncate=np.array(float(gaussian_truncate)),
                    occupancy_gain=np.array(float(occupancy_gain)),
                    background=np.array(str(background)),
                )
    
            unique_pixels = len(np.unique(ys.astype(np.int64) * int(grid_meta["W"]) + xs.astype(np.int64)))
    
            summary[name] = {
                "png_path": str(png_path),
                "npz_path": str(npz_path),
                "num_cells": int(len(encoded["labels"])),
                "num_occupied_pixels": int(unique_pixels),
                "image_shape": [int(grid_meta["H"]), int(grid_meta["W"]), 3],
                "knn_k": int(knn_k),
                "gaussian_sigma_pixels": float(gaussian_sigma_pixels),
            }
    
        with open(save_dir / "latent_image_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    
        with open(save_dir / "celltype_latent_rgb.json", "w", encoding="utf-8") as f:
            json.dump({
                str(ct): {
                    "index": int(k),
                    "latent": celltype_latent[k].tolist(),
                    "rgb": celltype_rgb[k].astype(int).tolist(),
                    "note": "base prototype; individual cells are context-modulated",
                }
                for k, ct in enumerate(celltypes)
            }, f, ensure_ascii=False, indent=2)
    
        print("\nSaved latent RGB images to:", save_dir)
        return summary, celltype_latent, celltype_rgb
    
    
    # ============================================================
    # 11. Generate images
    # ============================================================
    
    latent_save_dir = Path(save_dir) / "latent_images"
    
    summary, celltype_latent, celltype_rgb = create_dominant_type_latent_images(
        model=model,
        anndata_dict=anndata_dict,
        grid_meta=grid_meta,
        celltypes=celltypes,
        save_dir=latent_save_dir,
    
        spatial_key="spatial",
        celltype_key=celltype_col,
    
        slice_names=None,
        knn_k=int(ae_checkpoint["knn_k"]),
        distance_weighted=bool(ae_checkpoint["distance_weighted"]),
    
        batch_size=8192,
        device=device,
    
        gaussian_sigma_pixels=gaussian_sigma_pixels,
        gaussian_truncate=gaussian_truncate,
        occupancy_gain=float(config.get("rgb_occupancy_gain", 3.0)),
        background="white",
    
        save_png=True,
        save_npz=True,
    )
    
    
    
    
    # ============================================================
    # 12. Show one image
    # ============================================================
    
    import matplotlib.pyplot as plt
    
