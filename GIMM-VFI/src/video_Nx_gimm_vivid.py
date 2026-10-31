# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# Modified GIMM-VFI demo:
# 1) keeps the original preview MP4 outputs;
# 2) saves lossless interpolated PNG frames;
# 3) saves numerical target-to-endpoint correspondence fields as NPZ.
#
# Main NPZ fields for VIVID:
#   flow_to_left   : H x W x 2, GIMM implicit correspondence F_{t->0}
#   flow_to_right  : H x W x 2, GIMM implicit correspondence F_{t->1}
#   ratio          : interpolation fraction t
#
# Additional diagnostic fields:
#   flow_t
#   flow_to_left_refined_mean / flow_to_right_refined_mean
#   flow_to_left_refined_all / flow_to_right_refined_all
#
# NOTE:
# For VIVID numerical correspondence export, use --ds-factor 1.

import os
import argparse
import subprocess
import cv2
import numpy as np
import torch

from PIL import Image
from tqdm import tqdm
from models import create_model
from utils.utils import set_seed, InputPadder
from utils.setup import single_setup
from utils.flow_viz import flow_to_image


def default_parser():
    parser=argparse.ArgumentParser()
    parser.add_argument("-m","--model-config",type=str,default="./configs/flow/videodnn.yaml")
    parser.add_argument("--source-path",type=str,default="")
    parser.add_argument("--output-path",type=str,default="")
    parser.add_argument("--N",type=int,default=8)
    parser.add_argument("--ds-factor",type=float,default=1.0)
    parser.add_argument("-r","--result-path",type=str,default="./results.tmp")
    parser.add_argument("-l","--load-path",type=str,default="")
    parser.add_argument("-p","--postfix",type=str,default="")
    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--eval",action="store_true")
    return parser


def parse_args():
    parser=default_parser()
    args,extra_args=parser.parse_known_args()
    return args,extra_args


def load_image(img_path):
    img=Image.open(img_path)
    raw_img=np.array(img.convert("RGB"))
    img=torch.from_numpy(raw_img.copy()).permute(2,0,1)/255.0
    return img.float().unsqueeze(0)


def tensor_to_bgr(img):
    arr=(img[0].clamp(0,1).detach().cpu().permute(1,2,0).numpy()*255.0).round().astype(np.uint8)
    return arr[:,:,::-1]


def flow_to_numpy(flow):
    
    if flow.ndim==4:
        if flow.shape[0]!=1:
            raise ValueError(f"flow batch size must be 1, got {tuple(flow.shape)}")
        flow=flow[0]

    if flow.ndim!=3 or flow.shape[0]!=2:
        raise ValueError(
            f"flow shape must be 2xHxW or 1x2xHxW, got {tuple(flow.shape)}"
        )

    return flow.detach().cpu().permute(1,2,0).numpy().astype(np.float32)
    

def multiflow_to_numpy(flow):
    return flow[0].detach().cpu().permute(0,2,3,1).numpy().astype(np.float32)


def images_to_video(imgs,output_video_path,fps=15):
    if len(imgs)==0: return
    height,width,_=imgs[0].shape
    if max(height,width//2)>2048:
        frame_dir=os.path.join(os.path.dirname(output_video_path),"video_frames_tmp")
        os.makedirs(frame_dir,exist_ok=True)
        for idx,img in enumerate(imgs): cv2.imwrite(os.path.join(frame_dir,f"{idx:04d}.png"),img)
        subprocess.run([
            "ffmpeg","-y","-framerate",str(fps),"-i",f"{frame_dir}/%04d.png",
            "-c:v","libx264","-pix_fmt","yuv420p",output_video_path
        ],check=True)
        subprocess.run(["rm","-rf",frame_dir],check=True)
    else:
        video=cv2.VideoWriter(output_video_path,cv2.VideoWriter_fourcc(*"mp4v"),fps,(width,height))
        for img in imgs: video.write(img)
        video.release()


if __name__=="__main__":
    args,extra_args=parse_args()
    if args.N<2: raise ValueError("--N  must  >= 2")
    if abs(args.ds_factor-1.0)>1e-12:
        raise ValueError(" for  VIVID  values  correspondence  use when exporting  --ds-factor 1")

    set_seed(args.seed)
    config=single_setup(args,extra_args)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type!="cuda": raise RuntimeError("GIMM-VFI  requires  CUDA")

    os.makedirs(args.output_path,exist_ok=True)
    frame_output_path=os.path.join(args.output_path,"frames")
    flow_output_path=os.path.join(args.output_path,"flow")
    os.makedirs(frame_output_path,exist_ok=True)
    os.makedirs(flow_output_path,exist_ok=True)

    preview_video_path=os.path.join(args.output_path,"output.mp4")
    flow_video_path=os.path.join(args.output_path,"flow.mp4")

    model,_=create_model(config.arch)
    model=model.to(device)

    if args.load_path=="": raise ValueError("--load-path must be specified")

    ckpt=torch.load(args.load_path,map_location="cpu")
    if "ours" in args.load_path:
        ckpt={
            k.replace("module.feature_bone","frame_encoder"):v
            for k,v in ckpt.items()
            if "feature_bone" in k
        }
        model.load_state_dict(ckpt,strict=False)
    else:
        model.load_state_dict(ckpt["state_dict"],strict=True)

    model.eval()

    img_list=sorted([
        x for x in os.listdir(args.source_path)
        if x.lower().endswith((".png",".jpg",".jpeg",".bmp",".tif",".tiff"))
    ])
    if len(img_list)<2: raise ValueError(f"{args.source_path}  requires at least two input images ")

    preview_frames=[]
    original_frames=[]
    flow_visualizations=[]
    frame_counter=0

    print("="*90)
    print("GIMM-VFI inference + VIVID correspondence export")
    print("source       :",args.source_path)
    print("output       :",args.output_path)
    print("PNG frames   :",frame_output_path)
    print("FLOW NPZ     :",flow_output_path)
    print("N            :",args.N)
    print("checkpoint   :",args.load_path)
    print("="*90)

    for j in tqdm(range(len(img_list)-1)):
        img_path0=os.path.join(args.source_path,img_list[j])
        img_path1=os.path.join(args.source_path,img_list[j+1])
        I0=load_image(img_path0)
        I1=load_image(img_path1)

        if I0.shape!=I1.shape:
            raise ValueError(f" input sizes differ : {img_path0} {tuple(I0.shape)} vs {img_path1} {tuple(I1.shape)}")

        if j==0:
            left_preview=tensor_to_bgr(I0)
            original_frames.append(left_preview)
            preview_frames.append(cv2.hconcat([left_preview,left_preview]))
            cv2.imwrite(os.path.join(frame_output_path,f"img{frame_counter}.png"),left_preview)
            frame_counter+=1

        padder=InputPadder(I0.shape,32)
        I0_pad,I1_pad=padder.pad(I0,I1)
        xs=torch.cat((I0_pad.unsqueeze(2),I1_pad.unsqueeze(2)),dim=2).to(device,non_blocking=True)
        batch_size=xs.shape[0]
        s_shape=xs.shape[-2:]

        with torch.no_grad():
            ratios=[i/args.N for i in range(1,args.N)]
            coord_inputs=[
                (model.sample_coord_input(batch_size,s_shape,[t],device=xs.device,upsample_ratio=1.0),None)
                for t in ratios
            ]
            timesteps=[t*torch.ones(batch_size,device=xs.device,dtype=torch.float32) for t in ratios]
            all_outputs=model(xs,coord_inputs,t=timesteps,ds_factor=1.0)

            out_frames=[padder.unpad(x) for x in all_outputs["imgt_pred"]]
            out_flow_t=[padder.unpad(x) for x in all_outputs["flowt"]]
            refined_left=[padder.unpad(x[0]) for x in all_outputs["flowt0_pred"]]
            refined_right=[padder.unpad(x[0]) for x in all_outputs["flowt1_pred"]]

        for ratio,frame,flow_t,ref_left,ref_right in zip(ratios,out_frames,out_flow_t,refined_left,refined_right):
            image_index=frame_counter
            frame_bgr=tensor_to_bgr(frame)
            cv2.imwrite(os.path.join(frame_output_path,f"img{image_index}.png"),frame_bgr)

            flow_t_np=flow_to_numpy(flow_t)
            flow_to_left=(-float(ratio)*flow_t_np).astype(np.float32)
            flow_to_right=((1.0-float(ratio))*flow_t_np).astype(np.float32)

            ref_left_all=multiflow_to_numpy(ref_left)
            ref_right_all=multiflow_to_numpy(ref_right)
            ref_left_mean=ref_left_all.mean(axis=0).astype(np.float32)
            ref_right_mean=ref_right_all.mean(axis=0).astype(np.float32)

            H,W=flow_to_left.shape[:2]
            mask_to_left=np.full((H,W),1.0-float(ratio),dtype=np.float32)
            mask_to_right=np.full((H,W),float(ratio),dtype=np.float32)

            np.savez_compressed(
                os.path.join(flow_output_path,f"img{image_index}.npz"),
                flow_to_left=flow_to_left,
                flow_to_right=flow_to_right,
                mask_to_left=mask_to_left,
                mask_to_right=mask_to_right,
                ratio=np.float32(ratio),
                flow_t=flow_t_np,
                flow_to_left_refined_mean=ref_left_mean,
                flow_to_right_refined_mean=ref_right_mean,
                flow_to_left_refined_all=ref_left_all,
                flow_to_right_refined_all=ref_right_all,
            )

            flow_visualizations.append(flow_to_image(flow_t_np,convert_to_bgr=True))
            preview_frames.append(cv2.hconcat([original_frames[-1],frame_bgr]))

            print(
                f"img{image_index} | t={ratio:.6f} | "
                f"flow={flow_to_left.shape} | refined={ref_left_all.shape}"
            )
            frame_counter+=1

        right_preview=tensor_to_bgr(I1)
        cv2.imwrite(os.path.join(frame_output_path,f"img{frame_counter}.png"),right_preview)
        original_frames.append(right_preview)
        preview_frames.append(cv2.hconcat([right_preview,right_preview]))
        frame_counter+=1

    images_to_video(preview_frames[:-1],preview_video_path,fps=args.N*2)
    images_to_video(flow_visualizations,flow_video_path,fps=args.N*2)

    print("\n"+"="*90)
    print("GIMM-VFI FINISHED")
    print("Preview video :",preview_video_path)
    print("Flow video    :",flow_video_path)
    print("Lossless PNGs :",frame_output_path)
    print("Numerical flow:",flow_output_path)
    print("Saved images  :",frame_counter)
    print("="*90)
