# Migrated from 1002genertation.ipynb, cell 27. See docs/MIGRATION.md.
if config["image_encoding_method"] in ["pca", "spherical"]:

    # ============================================================
    # 5. Build cell-type 3D embedding
    # ============================================================
    encoding_method = str(config["image_encoding_method"]).lower()
    encoding_save_dir = Path(save_dir) / f"{encoding_method}_encoding_results"
    encoding_save_dir.mkdir(parents=True, exist_ok=True)

    def minmax_normalize_embedding(embedding, eps=1e-12):
        embedding = np.asarray(embedding, dtype=np.float32)
        if embedding.ndim != 2 or embedding.shape[1] != 3: raise ValueError(f"embedding  must be  K×3,  current  shape={embedding.shape}")
        latent = np.zeros_like(embedding, dtype=np.float32)
        for d in range(3):
            vmin, vmax = float(embedding[:,d].min()), float(embedding[:,d].max())
            latent[:,d] = 0.5 if vmax-vmin <= eps else (embedding[:,d]-vmin)/(vmax-vmin)
        latent = np.clip(latent, 0, 1).astype(np.float32)
        rgb = np.rint(latent*255).astype(np.uint8)
        return latent, rgb

    # ============================================================
    # 5A. PCA encoding
    # Modality-independent PCA:
    # cell-type identity + spatial neighborhood composition -> PCA -> 3D
    # ============================================================
    
    if encoding_method == "pca":
        from sklearn.decomposition import PCA
        from sklearn.neighbors import NearestNeighbors
    
        pca_knn_k = int(config.get("pca_knn_k", 20))
        pca_identity_weight = float(config.get("pca_identity_weight", 1.0))
        pca_neighbor_weight = float(config.get("pca_neighbor_weight", 1.0))
        pca_spatial_key = str(config.get("pca_spatial_key", "spatial"))
    
        def compute_slice_type_context(adata, celltypes, celltype_key="cell_type", spatial_key="spatial", knn_k=20):
            """Compute slice type context."""
            K = len(celltypes); celltype_to_index = {str(ct):k for k,ct in enumerate(celltypes)}
            coords_all = np.asarray(adata.obsm[spatial_key], dtype=np.float32); labels_raw = adata.obs[celltype_key].astype(str).to_numpy()
            labels_all = np.asarray([celltype_to_index.get(str(ct),-1) for ct in labels_raw], dtype=np.int64)
            valid = (labels_all>=0) & np.all(np.isfinite(coords_all), axis=1); coords = coords_all[valid]; labels = labels_all[valid]
            type_context = np.full((K,K), np.nan, dtype=np.float32); type_counts = np.bincount(labels, minlength=K).astype(np.int64)
            if len(labels)<=1: return type_context, type_counts, np.zeros(K,dtype=np.int64)
    
            n_neighbors = min(int(knn_k)+1, len(labels))
            nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean", algorithm="auto").fit(coords)
            neighbor_idx = nn.kneighbors(coords, return_distance=False)
    
            
            neighbor_lists = []
            for i,row in enumerate(neighbor_idx):
                row = row[row!=i]
                if len(row)>knn_k: row = row[:knn_k]
                neighbor_lists.append(row)
    
            cell_neighbor_P = np.zeros((len(labels),K), dtype=np.float32); valid_neighbor = np.zeros(len(labels), dtype=bool)
            for i,row in enumerate(neighbor_lists):
                if len(row)==0: continue
                counts = np.bincount(labels[row], minlength=K).astype(np.float32)
                cell_neighbor_P[i] = counts/max(float(counts.sum()),1.0); valid_neighbor[i] = True
    
            type_valid_cell_counts = np.zeros(K, dtype=np.int64)
            for k in range(K):
                idx = np.flatnonzero((labels==k) & valid_neighbor); type_valid_cell_counts[k] = len(idx)
                if len(idx)>0: type_context[k] = cell_neighbor_P[idx].mean(axis=0)
    
            return type_context, type_counts, type_valid_cell_counts
    
    
        slice_names = list(anndata_dict.keys()); K = len(celltypes)
        slice_context_dict, slice_type_count_dict, slice_valid_neighbor_count_dict = {}, {}, {}
    
        
        for slice_name in tqdm(slice_names, desc="Calculate spatial type context", dynamic_ncols=True):
            context, type_counts, valid_neighbor_counts = compute_slice_type_context(
                anndata_dict[slice_name], celltypes, celltype_key=celltype_col,
                spatial_key=pca_spatial_key, knn_k=pca_knn_k
            )
            slice_context_dict[slice_name] = context
            slice_type_count_dict[slice_name] = type_counts
            slice_valid_neighbor_count_dict[slice_name] = valid_neighbor_counts
    
        # ============================================================
        
        
        
        # ============================================================
        context_sum = np.zeros((K,K), dtype=np.float64); context_slice_count = np.zeros(K, dtype=np.int64)
    
        for slice_name in slice_names:
            context = slice_context_dict[slice_name]
            for k in range(K):
                if np.all(np.isfinite(context[k])):
                    context_sum[k] += context[k]
                    context_slice_count[k] += 1
    
        mean_type_context = np.zeros((K,K), dtype=np.float32)
        for k in range(K):
            if context_slice_count[k]>0: mean_type_context[k] = (context_sum[k]/context_slice_count[k]).astype(np.float32)
    
        
        row_sum = mean_type_context.sum(axis=1, keepdims=True)
        mean_type_context = np.divide(mean_type_context, row_sum, out=np.zeros_like(mean_type_context), where=row_sum>0)
        mean_type_context = np.clip(mean_type_context,0,1).astype(np.float32)
    
        # ============================================================
        # Identity + spatial context
        #
        
        # Context : K×K spatial neighbor composition
        
        # ============================================================
        identity_features = np.eye(K, dtype=np.float32)
        pca_features = np.concatenate([
            pca_identity_weight*identity_features,
            pca_neighbor_weight*mean_type_context
        ], axis=1).astype(np.float32)
    
        # PCA -> 3D
        n_components = min(3,K,pca_features.shape[1])
        pca_model = PCA(n_components=n_components, svd_solver="full")
        embedding_raw = pca_model.fit_transform(pca_features).astype(np.float32)
        if embedding_raw.shape[1]<3: embedding_raw = np.pad(embedding_raw,((0,0),(0,3-embedding_raw.shape[1])),mode="constant")
        embedding_raw = embedding_raw[:,:3]
    
        
        celltype_latent, celltype_rgb = minmax_normalize_embedding(embedding_raw)
    
        print("\n"+"="*100); print("PCA CELL-TYPE SPATIAL-CONTEXT EMBEDDING"); print("="*100)
        print(f"kNN k               : {pca_knn_k}")
        print(f"identity weight     : {pca_identity_weight}")
        print(f"neighbor weight     : {pca_neighbor_weight}")
        print(f"feature shape       : {pca_features.shape}")
        print(f"explained variance  : {pca_model.explained_variance_ratio_}")
        print("\nType occurrence / valid context slices:")
        for k,ct in enumerate(celltypes):
            total_cells = sum(int(slice_type_count_dict[s][k]) for s in slice_names)
            present_slices = sum(int(slice_type_count_dict[s][k]>0) for s in slice_names)
            print(f"{k:3d} | {str(ct):30s} | cells={total_cells:6d} | present_slices={present_slices:3d} | context_slices={int(context_slice_count[k]):3d} | raw=({embedding_raw[k,0]:.4f}, {embedding_raw[k,1]:.4f}, {embedding_raw[k,2]:.4f}) | latent=({celltype_latent[k,0]:.4f}, {celltype_latent[k,1]:.4f}, {celltype_latent[k,2]:.4f}) | RGB=({celltype_rgb[k,0]:3d}, {celltype_rgb[k,1]:3d}, {celltype_rgb[k,2]:3d})")
    
        # ============================================================
        # Save PCA encoding information
        # ============================================================
        np.save(encoding_save_dir/"mean_type_spatial_context.npy", mean_type_context)
        np.save(encoding_save_dir/"pca_input_features.npy", pca_features)
        np.save(encoding_save_dir/"pca_embedding_raw.npy", embedding_raw)
        np.save(encoding_save_dir/"pca_embedding_normalized.npy", celltype_latent)
        np.save(encoding_save_dir/"context_slice_count.npy", context_slice_count)
    
        pd.DataFrame(
            mean_type_context,
            index=[str(ct) for ct in celltypes],
            columns=[str(ct) for ct in celltypes]
        ).to_csv(encoding_save_dir/"mean_type_spatial_context.csv")
    
        feature_columns = [f"identity_{str(ct)}" for ct in celltypes]+[f"neighbor_{str(ct)}" for ct in celltypes]
        pd.DataFrame(
            pca_features,
            index=[str(ct) for ct in celltypes],
            columns=feature_columns
        ).to_csv(encoding_save_dir/"pca_input_features.csv")
    
        type_summary_rows = []
        for k,ct in enumerate(celltypes):
            total_cells = sum(int(slice_type_count_dict[s][k]) for s in slice_names)
            present_slices = sum(int(slice_type_count_dict[s][k]>0) for s in slice_names)
            type_summary_rows.append({
                "celltype":str(ct),
                "total_cells":total_cells,
                "present_slices":present_slices,
                "context_slices":int(context_slice_count[k]),
                "PC1_raw":embedding_raw[k,0], "PC2_raw":embedding_raw[k,1], "PC3_raw":embedding_raw[k,2],
                "dim1_normalized":celltype_latent[k,0], "dim2_normalized":celltype_latent[k,1], "dim3_normalized":celltype_latent[k,2],
                "R":celltype_rgb[k,0], "G":celltype_rgb[k,1], "B":celltype_rgb[k,2]
            })
        pd.DataFrame(type_summary_rows).to_csv(encoding_save_dir/"celltype_pca_rgb.csv", index=False)
    
        
        context_save_dir = encoding_save_dir/"slice_spatial_contexts"; context_save_dir.mkdir(parents=True, exist_ok=True)
        for slice_name in slice_names:
            safe_name = safe_filename(slice_name)
            pd.DataFrame(
                slice_context_dict[slice_name],
                index=[str(ct) for ct in celltypes],
                columns=[str(ct) for ct in celltypes]
            ).to_csv(context_save_dir/f"{safe_name}_spatial_context.csv")
    
            pd.DataFrame({
                "celltype":[str(ct) for ct in celltypes],
                "num_cells":slice_type_count_dict[slice_name],
                "num_cells_with_valid_neighbors":slice_valid_neighbor_count_dict[slice_name]
            }).to_csv(context_save_dir/f"{safe_name}_type_counts.csv",index=False)
    
        representation_name = "celltype_identity_spatial_context_pca"


    # ============================================================
    # 5B. Spherical encoding
    # ============================================================

    else:
        def spherical_codebook(K):
            k = np.arange(K, dtype=np.float64)
            z = 1.0-2.0*(k+0.5)/K; r = np.sqrt(np.maximum(0.0,1.0-z**2))
            theta = k*np.pi*(3.0-np.sqrt(5.0))
            x = r*np.cos(theta); y = r*np.sin(theta)
            return np.stack([x,y,z], axis=1).astype(np.float32)

        K = len(celltypes)
        embedding_raw = spherical_codebook(K)

        
        celltype_latent, celltype_rgb = minmax_normalize_embedding(embedding_raw)

        print("\n"+"="*100); print("SPHERICAL CELL-TYPE EMBEDDING"); print("="*100)
        for k, ct in enumerate(celltypes):
            print(f"{k:3d} | {ct} | raw=({embedding_raw[k,0]:.4f}, {embedding_raw[k,1]:.4f}, {embedding_raw[k,2]:.4f}) | latent=({celltype_latent[k,0]:.4f}, {celltype_latent[k,1]:.4f}, {celltype_latent[k,2]:.4f}) | RGB=({celltype_rgb[k,0]:3d}, {celltype_rgb[k,1]:3d}, {celltype_rgb[k,2]:3d})")

        np.save(encoding_save_dir/"spherical_embedding_raw.npy", embedding_raw)
        np.save(encoding_save_dir/"spherical_embedding_normalized.npy", celltype_latent)

        pd.DataFrame({
            "celltype":[str(ct) for ct in celltypes],
            "x_raw":embedding_raw[:,0], "y_raw":embedding_raw[:,1], "z_raw":embedding_raw[:,2],
            "dim1_normalized":celltype_latent[:,0], "dim2_normalized":celltype_latent[:,1], "dim3_normalized":celltype_latent[:,2],
            "R":celltype_rgb[:,0], "G":celltype_rgb[:,1], "B":celltype_rgb[:,2],
        }).to_csv(encoding_save_dir/"celltype_spherical_rgb.csv", index=False)

        representation_name = "spherical_celltype_codebook"


    # ============================================================
    # 6. Save cell-type latent RGB codebook
    # ============================================================

    celltype_color_path = encoding_save_dir/"celltype_latent_rgb.json"
    with open(celltype_color_path, "w", encoding="utf-8") as f:
        json.dump({
            "representation":representation_name,
            "encoding_method":encoding_method,
            "normalization":"dimension-wise min-max normalization of raw 3D embedding to [0,1]",
            "rgb_conversion":"round(normalized_embedding * 255)",
            "celltypes":[
                {"index":int(k), "celltype":str(ct), "embedding_raw":embedding_raw[k].tolist(), "latent":celltype_latent[k].tolist(), "rgb":celltype_rgb[k].astype(int).tolist()}
                for k, ct in enumerate(celltypes)
            ],
        }, f, ensure_ascii=False, indent=2)

    print("\nSaved:", celltype_color_path)


    # ============================================================
    # 7. Encode every real cell
    # ============================================================

    def encode_one_slice_cells_static(adata, celltypes, celltype_latent, spatial_key="spatial", celltype_key="cell_type"):
        coords_all = np.asarray(adata.obsm[spatial_key], dtype=np.float32)
        labels_raw = adata.obs[celltype_key].astype(str).to_numpy()
        celltype_to_index = {str(ct):k for k,ct in enumerate(celltypes)}
        labels = np.asarray([celltype_to_index.get(str(ct),-1) for ct in labels_raw], dtype=np.int64)
        valid = (labels>=0) & np.all(np.isfinite(coords_all), axis=1)
        obs_indices = np.flatnonzero(valid).astype(np.int64); coords = coords_all[valid]; labels = labels[valid]
        z = celltype_latent[labels].astype(np.float32); rgb = np.rint(np.clip(z,0,1)*255).astype(np.uint8)

        K = len(celltypes)
        self_x = np.zeros((len(labels),K), dtype=np.float32)
        if len(labels)>0: self_x[np.arange(len(labels)),labels] = 1.0
        neighbor_P = np.zeros((len(labels),K), dtype=np.float32); delta = np.zeros((len(labels),3), dtype=np.float32)

        return {"coords":coords.astype(np.float32), "obs_indices":obs_indices, "labels":labels.astype(np.int64), "self_x":self_x, "neighbor_P":neighbor_P, "z":z, "delta":delta, "rgb":rgb}


    # ============================================================
    # 8. Cell RGB -> normalized Gaussian splatting image
    # ============================================================

    def render_context_cells_to_rgb(coords, z, grid_meta, sigma_pixels=1.0, truncate=4.0, occupancy_gain=3.0, background="white", eps=1e-8):
        H, W = int(grid_meta["H"]), int(grid_meta["W"])
        coords = np.asarray(coords, dtype=np.float32); z = np.asarray(z, dtype=np.float32)

        u = np.floor((coords[:,0]-grid_meta["x_min"])/grid_meta["pixel_size_x"]).astype(np.int64)
        v = np.floor((coords[:,1]-grid_meta["y_min"])/grid_meta["pixel_size_y"]).astype(np.int64)
        valid = (u>=0)&(u<W)&(v>=0)&(v<H); u, v, colors = u[valid], v[valid], np.clip(z[valid],0,1)

        occ = np.zeros((H,W), dtype=np.float32); color_sum = np.zeros((H,W,3), dtype=np.float32)
        np.add.at(occ,(v,u),1.0)
        for c in range(3): np.add.at(color_sum[:,:,c],(v,u),colors[:,c])

        if sigma_pixels>0:
            occ_blur = gaussian_filter(occ.astype(np.float64), sigma=float(sigma_pixels), mode="constant", cval=0.0, truncate=float(truncate)).astype(np.float32)
            color_blur = np.stack([gaussian_filter(color_sum[:,:,c].astype(np.float64), sigma=float(sigma_pixels), mode="constant", cval=0.0, truncate=float(truncate)) for c in range(3)], axis=-1).astype(np.float32)
        else:
            occ_blur, color_blur = occ, color_sum

        local_color = color_blur/(occ_blur[:,:,None]+eps)
        support_scale = 2.0*np.pi*float(sigma_pixels)**2 if sigma_pixels>0 else 1.0
        alpha = 1.0-np.exp(-float(occupancy_gain)*occ_blur*support_scale); alpha = np.clip(alpha,0,1).astype(np.float32)

        if background=="white": image = (1.0-alpha[:,:,None])+alpha[:,:,None]*local_color
        elif background=="black": image = alpha[:,:,None]*local_color
        else: raise ValueError("background must be 'white' or 'black'.")

        image = np.clip(image,0,1); rgb_uint8 = np.rint(image*255).astype(np.uint8)
        return rgb_uint8, image.astype(np.float32), alpha, occ_blur, u, v


    # ============================================================
    # 9. Create all latent RGB images
    # ============================================================

    def create_static_latent_images(
        anndata_dict, grid_meta, celltypes, celltype_latent, celltype_rgb, save_dir, encoding_method,
        spatial_key="spatial", celltype_key="cell_type", slice_names=None,
        gaussian_sigma_pixels=1.0, gaussian_truncate=4.0, occupancy_gain=3.0,
        background="white", save_png=True, save_npz=True,
    ):
        save_dir = Path(save_dir); save_dir.mkdir(parents=True, exist_ok=True)
        slice_names = list(anndata_dict.keys()) if slice_names is None else list(slice_names)
        summary = {}

        for name in tqdm(slice_names, desc=f"Create {encoding_method.upper()} latent RGB", dynamic_ncols=True):
            encoded = encode_one_slice_cells_static(anndata_dict[name], celltypes, celltype_latent, spatial_key=spatial_key, celltype_key=celltype_key)

            rgb_img, rgb_float, alpha, occ_blur, xs, ys = render_context_cells_to_rgb(
                coords=encoded["coords"], z=encoded["z"], grid_meta=grid_meta,
                sigma_pixels=gaussian_sigma_pixels, truncate=gaussian_truncate,
                occupancy_gain=occupancy_gain, background=background,
            )

            safe_name = safe_filename(name)
            png_path = save_dir/f"{safe_name}.png"; npz_path = save_dir/f"{safe_name}_latent_data.npz"

            if save_png: Image.fromarray(rgb_img, mode="RGB").save(png_path)

            if save_npz:
                np.savez_compressed(
                    npz_path,
                    coords=encoded["coords"].astype(np.float32),
                    obs_indices=encoded["obs_indices"].astype(np.int64),
                    ys=ys.astype(np.int64), xs=xs.astype(np.int64),
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
                    celltypes=np.asarray(celltypes,dtype=str),
                    slice_name=np.array(str(name)),
                    encoding_method=np.array(str(encoding_method)),
                    knn_k=np.array(-1),
                    gaussian_sigma_pixels=np.array(float(gaussian_sigma_pixels)),
                    gaussian_truncate=np.array(float(gaussian_truncate)),
                    occupancy_gain=np.array(float(occupancy_gain)),
                    background=np.array(str(background)),
                )

            unique_pixels = len(np.unique(ys.astype(np.int64)*int(grid_meta["W"])+xs.astype(np.int64)))

            summary[name] = {
                "png_path":str(png_path), "npz_path":str(npz_path),
                "num_cells":int(len(encoded["labels"])),
                "num_occupied_pixels":int(unique_pixels),
                "image_shape":[int(grid_meta["H"]),int(grid_meta["W"]),3],
                "knn_k":-1,
                "gaussian_sigma_pixels":float(gaussian_sigma_pixels),
                "encoding_method":encoding_method,
            }

        with open(save_dir/"latent_image_summary.json", "w", encoding="utf-8") as f: json.dump(summary, f, ensure_ascii=False, indent=2)

        with open(save_dir/"celltype_latent_rgb.json", "w", encoding="utf-8") as f:
            json.dump({
                str(ct):{"index":int(k), "latent":celltype_latent[k].tolist(), "rgb":celltype_rgb[k].astype(int).tolist(), "note":f"fixed cell-type {encoding_method} encoding"}
                for k,ct in enumerate(celltypes)
            }, f, ensure_ascii=False, indent=2)

        print("\nSaved latent RGB images to:", save_dir)
        return summary, celltype_latent, celltype_rgb


    # ============================================================
    # 10. Generate images
    # ============================================================
    latent_save_dir = Path(save_dir)/"latent_images"

    summary, celltype_latent, celltype_rgb = create_static_latent_images(
        anndata_dict=anndata_dict,
        grid_meta=grid_meta,
        celltypes=celltypes,
        celltype_latent=celltype_latent,
        celltype_rgb=celltype_rgb,
        save_dir=latent_save_dir,
        encoding_method=encoding_method,
        spatial_key="spatial",
        celltype_key=celltype_col,
        slice_names=None,
        gaussian_sigma_pixels=gaussian_sigma_pixels,
        gaussian_truncate=gaussian_truncate,
        occupancy_gain=float(config.get("rgb_occupancy_gain",3.0)),
        background="white",
        save_png=True,
        save_npz=True,
    )


    # ============================================================
    # 11. Show one image
    # ============================================================
