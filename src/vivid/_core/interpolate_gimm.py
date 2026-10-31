# Migrated from 1002genertation.ipynb, cell 33. See docs/MIGRATION.md.
if interpolation_model_names == ["GIMM"]:
    
    K=int(config["target_number"])
    if K<=0: raise ValueError("config['target_number']  must  > 0")
    
    gimm_N=K+1
    segments=gimm_N
    total_frames=K
    target_img=list(range(1,K+1))
    target_ratios=[i/gimm_N for i in target_img]
    
    print("\n"+"="*100)
    print("GIMM-VFI configuration")
    print("target K              =",K)
    print("GIMM N                =",gimm_N)
    print("intermediate frames   =",total_frames)
    print("target img indices    =",target_img)
    print("target ratios         =",[round(x,6) for x in target_ratios])
    print("maximum ratio error   = 0.0")
    print("="*100)


    # ============================================================
    # 6. GIMM-VFI-R-P interpolation on latent RGB images
    #    Output:
    #      GIMM_result/<pair>/frames/img*.png
    #      GIMM_result/<pair>/flow/img*.npz
    # ============================================================
    
    import os, re, shutil, subprocess
    import numpy as np
    from pathlib import Path
    from PIL import Image
    
    # ============================================================
    # 6.0 Paths / environment
    # ============================================================
    
    GIMM_dir=Path(backend_dir).resolve()
    gimm_python=Path(backend_python).resolve()
    gimm_script=GIMM_dir/"src"/"video_Nx_gimm_vivid.py"
    gimm_config=GIMM_dir/"configs"/"gimmvfi"/"gimmvfi_r_arb.yaml"
    gimm_checkpoint=GIMM_dir/"pretrained_ckpt"/"gimmvfi_r_arb_lpips.pt"
    gimm_raft_checkpoint=GIMM_dir/"pretrained_ckpt"/"raft-things.pth"
    
    latent_input_dir=Path(save_dir).resolve()/"latent_images"
    gimm_global_output_root=Path(save_dir).resolve()/"GIMM_result"
    gimm_ds_factor=float(config.get("gimm_ds_factor",1.0))
    
    if "overwrite_gimm_result" not in globals(): overwrite_gimm_result=True
    if abs(gimm_ds_factor-1.0)>1e-12: raise ValueError(" current  VIVID correspondence  output requires  gimm_ds_factor=1.0")
    
    for p,name in [
        (GIMM_dir,"GIMM_dir"),(gimm_python,"gimm_python"),(gimm_script,"gimm_script"),
        (gimm_config,"gimm_config"),(gimm_checkpoint,"gimm_checkpoint"),
        (gimm_raft_checkpoint,"RAFT checkpoint"),(latent_input_dir,"latent_input_dir"),
    ]:
        if not p.exists(): raise FileNotFoundError(f"{name}  not found : {p}")
    
    gimm_env=os.environ.copy()
    # ============================================================
    # 6.1 Utilities
    # ============================================================
    
    def run_cmd(cmd,cwd,env=None):
        print("-"*100); print("Command:"); print(" ".join(map(str,cmd))); print("cwd:",cwd); print("-"*100)
        result=subprocess.run([str(x) for x in cmd],cwd=str(cwd),env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding="utf-8",errors="replace")
        print(result.stdout)
        if result.returncode!=0: raise RuntimeError(f"GIMM-VFI failed, returncode={result.returncode}\n{' '.join(map(str,cmd))}")
        return result
    
    def validate_image_group(image_files,check_nonwhite=True):
        if len(image_files)==0: raise ValueError("image_files  is empty .")
        expected_size=None
        for p in map(Path,image_files):
            if not p.exists(): raise FileNotFoundError(p)
            with Image.open(p) as img:
                if img.mode!="RGB": raise ValueError(f"{p.name}: image mode={img.mode},  expected  RGB")
                if expected_size is None: expected_size=img.size
                elif img.size!=expected_size: raise ValueError(f"{p.name}: size={img.size},  expected ={expected_size}")
                if check_nonwhite:
                    arr=np.asarray(img,dtype=np.uint8)
                    if not np.any(arr!=255): raise ValueError(f"{p.name}:  image is entirely white .")
        return expected_size
    
    def validate_flow_npz(flow_file,expected_h=None,expected_w=None,expected_ratio=None,atol=1e-4):
        required=["flow_to_left","mask_to_left","flow_to_right","mask_to_right","ratio"]
        with np.load(flow_file) as f:
            missing=[x for x in required if x not in f.files]
            if missing: raise KeyError(f"{flow_file}:  missing  {missing}")
            fl=np.asarray(f["flow_to_left"],dtype=np.float32); fr=np.asarray(f["flow_to_right"],dtype=np.float32)
            ml=np.asarray(f["mask_to_left"],dtype=np.float32); mr=np.asarray(f["mask_to_right"],dtype=np.float32); ratio=float(f["ratio"])
        if fl.ndim!=3 or fl.shape[-1]!=2: raise ValueError(f"{flow_file}: flow_to_left shape={fl.shape}")
        if fr.shape!=fl.shape or ml.shape!=fl.shape[:2] or mr.shape!=fl.shape[:2]: raise ValueError(f"{flow_file}: FLOW/MASK shape  inconsistent ")
        if expected_h is not None and expected_w is not None and fl.shape[:2]!=(expected_h,expected_w):
            raise ValueError(f"{flow_file}: flow size={fl.shape[:2]},  expected ={(expected_h,expected_w)}")
        if not all(np.isfinite(x).all() for x in [fl,fr,ml,mr]): raise ValueError(f"{flow_file}:  contains  NaN/Inf")
        mask_sum_error=float(np.max(np.abs(ml+mr-1.0)))
        if mask_sum_error>atol: raise ValueError(f"{flow_file}: mask sum error={mask_sum_error:.3e}")
        if expected_ratio is not None and abs(ratio-float(expected_ratio))>1e-5:
            raise ValueError(f"{flow_file}: ratio={ratio:.8f},  expected ={expected_ratio:.8f}")
        return {"flow_shape":tuple(fl.shape),"mask_shape":tuple(ml.shape),"ratio":ratio,"mask_sum_error":mask_sum_error}
    
    def vfi_natural_key(path): return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)",Path(path).name)]
    def vfi_clean_tag(path): return re.sub(r"[^0-9A-Za-z_.-]+","_",Path(path).stem).strip("_")
    def vfi_pair_name(left,right): return f"{vfi_clean_tag(left)}_to_{vfi_clean_tag(right)}"
    
    
    # ============================================================
    # 6.2 Collect latent RGB images
    # ============================================================
    
    if "summary" not in globals(): raise NameError(" cannot find  summary,  generate first  latent RGB  image ")
    if "outputs" not in globals(): raise NameError(" cannot find  outputs")
    if "gimm_N" not in globals(): raise NameError(" run first  GIMM-VFI timestep  configuration stage ")
    
    slice_names=list(outputs.keys())
    image_files=[]
    for slice_name in slice_names:
        if slice_name not in summary: raise KeyError(f"{slice_name}  not found in  latent image summary  in ")
        p=Path(summary[slice_name]["png_path"]).resolve()
        if not p.exists(): raise FileNotFoundError(p)
        image_files.append(p)
    
    if len(image_files)<2: raise ValueError("GIMM-VFI  requires at least two observed slice images .")
    W,H=validate_image_group(image_files,check_nonwhite=True)
    slice_pairs=list(zip(image_files[:-1],image_files[1:]))
    num_intermediate_frames=gimm_N-1
    gimm_global_output_root.mkdir(parents=True,exist_ok=True)
    
    print("\n"+"="*100)
    print("GIMM-VFI-R-P INTERPOLATION ON LATENT RGB")
    print("="*100)
    print("Input      :",latent_input_dir)
    print("Output     :",gimm_global_output_root)
    print("Checkpoint :",gimm_checkpoint.name)
    print("Images     :",len(image_files))
    print("Pairs      :",len(slice_pairs))
    print("Size       :",f"{W} × {H}")
    print("N          :",gimm_N)
    print("Intermediate frames per pair:",num_intermediate_frames)
    print("="*100)
    
    
    # ============================================================
    # 6.3 Run GIMM-VFI for each adjacent endpoint pair
    # ============================================================
    
    global_flow_results={}
    
    for pair_idx,(img1,img2) in enumerate(slice_pairs,start=1):
        pair_name=vfi_pair_name(img1,img2)
        output_dir=gimm_global_output_root/pair_name
    
        if output_dir.exists() and overwrite_gimm_result: shutil.rmtree(output_dir)
        if output_dir.exists(): raise FileExistsError(f" output directory already exists : {output_dir}")
        output_dir.mkdir(parents=True,exist_ok=True)
    
        
        pair_input_dir=output_dir/"_input"
        pair_input_dir.mkdir(parents=True,exist_ok=True)
        shutil.copy2(img1,pair_input_dir/"img0.png")
        shutil.copy2(img2,pair_input_dir/"img1.png")
    
        print("\n"+"#"*100)
        print(f"PAIR {pair_idx}/{len(slice_pairs)} | {img1.name} -> {img2.name}")
        print("Output:",output_dir)
        print("#"*100)
    
        infer_cmd=[
            gimm_python,str(gimm_script),
            "--source-path",str(pair_input_dir),
            "--output-path",str(output_dir),
            "--ds-factor","1",
            "--N",str(gimm_N),
            "-m",str(gimm_config),
            "-l",str(gimm_checkpoint),
            "--seed",str(int(config["seed"])),
            "--eval",
        ]
        run_cmd(infer_cmd,cwd=GIMM_dir,env=gimm_env)
    
        frames_dir=output_dir/"frames"
        flow_dir=output_dir/"flow"
        if not frames_dir.exists(): raise RuntimeError(f"{pair_name}:  not generated  frames  directory ")
        if not flow_dir.exists(): raise RuntimeError(f"{pair_name}:  not generated  flow  directory ")
    
        generated_images=sorted(frames_dir.glob("img*.png"),key=vfi_natural_key)
        generated_npz=sorted(flow_dir.glob("img*.npz"),key=vfi_natural_key)
        expected_image_count=gimm_N+1
    
        if len(generated_images)!=expected_image_count:
            raise RuntimeError(f"{pair_name}: PNG count ={len(generated_images)},  expected ={expected_image_count}")
        if len(generated_npz)!=num_intermediate_frames:
            raise RuntimeError(f"{pair_name}: NPZ count ={len(generated_npz)},  expected ={num_intermediate_frames}")
    
        expected_names=[f"img{i}.png" for i in range(gimm_N+1)]
        if [p.name for p in generated_images]!=expected_names:
            raise RuntimeError(f"{pair_name}: PNG unexpected indices ")
    
        generated_size=validate_image_group(generated_images,check_nonwhite=False)
        if generated_size!=(W,H): raise RuntimeError(f"{pair_name}: GIMM output size ={generated_size},  input size ={(W,H)}")
    
        frame_results=[]
        for img_idx in range(1,gimm_N):
            image_file=frames_dir/f"img{img_idx}.png"
            flow_file=flow_dir/f"img{img_idx}.npz"
            info=validate_flow_npz(flow_file,H,W,img_idx/gimm_N)
            frame_results.append({"image_index":img_idx,"ratio":float(info["ratio"]),"image_file":image_file,"flow_file":flow_file})
            print(f"  img{img_idx} | t={info['ratio']:.6f} | flow={info['flow_shape']} | mask error={info['mask_sum_error']:.2e}")
    
        global_flow_results[pair_name]={
            "left_image":img1,"right_image":img2,"output_dir":output_dir,
            "frames_dir":frames_dir,"flow_dir":flow_dir,"frames":frame_results,
        }
    
        shutil.rmtree(pair_input_dir,ignore_errors=True)
        print(f"Finished: {pair_name}")
        print("Images:",frames_dir)
        print("FLOW  :",flow_dir)
    
    print("\nGIMM-VFI finished | pairs:",len(global_flow_results))
