import os
import sys
import math
import time
import argparse
import random
import inspect
import numpy as np

import torch
from torch.utils.data import DataLoader

from dataset_spatial import SpatialTripletDataset


class SummaryWriter:
    def __init__(self, *args, **kwargs):
        pass

    def add_scalar(self, *args, **kwargs):
        pass

    def close(self):
        pass

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def import_practical_model(model_dir):
    """Import practical model."""
    sys.path.insert(0, os.getcwd())

    try:
        from train_log.RIFE_HDv3 import Model
        print("Imported Model from train_log.RIFE_HDv3")
        return Model
    except Exception as e1:
        print(f"Failed to import train_log.RIFE_HDv3: {e1}")

    try:
        from model.RIFE_HDv2 import Model
        print("Imported Model from model.RIFE_HDv2")
        return Model
    except Exception as e2:
        print(f"Failed to import model.RIFE_HDv2: {e2}")

    try:
        from model.RIFE_HD import Model
        print("Imported Model from model.RIFE_HD")
        return Model
    except Exception as e3:
        print(f"Failed to import model.RIFE_HD: {e3}")

    raise ImportError(
        "Cannot import Practical-RIFE Model. "
        " check  train_log/  contains  RIFE_HDv3.py  and  flownet.pkl."
    )


def get_lr(step, total_steps, base_lr, min_lr, warmup_steps):
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step + 1) / float(warmup_steps)

    if total_steps <= warmup_steps:
        return base_lr

    progress = (step - warmup_steps) / float(total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))

    return min_lr + (base_lr - min_lr) * cosine


def call_update(model, imgs, gt, lr, training=True):
    """Call update."""
    if not hasattr(model, "update"):
        raise AttributeError(
            " current  Practical-RIFE Model  missing  update()  function ."
            " this model package may only support inference ,  does not support fine-tuning ."
        )

    sig = inspect.signature(model.update)
    params = sig.parameters

    kwargs = {}

    if "learning_rate" in params:
        kwargs["learning_rate"] = lr

    if "training" in params:
        kwargs["training"] = training

    try:
        return model.update(imgs, gt, **kwargs)
    except TypeError:
        
        try:
            return model.update(imgs, gt, lr)
        except TypeError:
            return model.update(imgs, gt)


@torch.no_grad()
def evaluate(model, val_loader, device):
    model.eval()

    l1_losses = []
    psnrs = []

    for data, _ in val_loader:
        data = data.to(device, non_blocking=True) / 255.0

        imgs = data[:, :6]
        gt = data[:, 6:9]

        output = call_update(
            model=model,
            imgs=imgs,
            gt=gt,
            lr=0,
            training=False,
        )

        if isinstance(output, tuple):
            pred = output[0]
            info = output[1] if len(output) > 1 else {}
        else:
            pred = output
            info = {}

        if isinstance(info, dict) and "loss_l1" in info:
            loss_l1 = info["loss_l1"].detach().item()
        else:
            loss_l1 = torch.mean(torch.abs(pred - gt)).detach().item()

        l1_losses.append(loss_l1)

        mse = torch.mean((pred - gt) ** 2, dim=[1, 2, 3])
        psnr = -10.0 * torch.log10(mse + 1e-8)
        psnrs.extend(psnr.detach().cpu().numpy().tolist())

    return {
        "val_l1": float(np.mean(l1_losses)),
        "val_psnr": float(np.mean(psnrs)),
    }


@torch.no_grad()
def save_visual_examples(model, val_loader, device, save_dir, epoch, max_images=4):
    import cv2

    os.makedirs(save_dir, exist_ok=True)

    model.eval()

    data, _ = next(iter(val_loader))
    data = data.to(device, non_blocking=True) / 255.0

    imgs = data[:, :6]
    gt = data[:, 6:9]

    img0 = imgs[:, :3]
    img1 = imgs[:, 3:6]

    output = call_update(
        model=model,
        imgs=imgs,
        gt=gt,
        lr=0,
        training=False,
    )

    if isinstance(output, tuple):
        pred = output[0]
    else:
        pred = output

    n = min(max_images, data.shape[0])

    for i in range(n):
        show = torch.cat(
            [
                img0[i],
                pred[i].clamp(0, 1),
                gt[i],
                img1[i],
            ],
            dim=2,
        )

        show = show.permute(1, 2, 0).detach().cpu().numpy()
        show = (show * 255.0).clip(0, 255).astype(np.uint8)

        out_path = os.path.join(save_dir, f"epoch_{epoch:04d}_sample_{i}.png")
        cv2.imwrite(out_path, show)


def save_model(model, save_path):
    """Save model."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if hasattr(model, "flownet"):
        torch.save(model.flownet.state_dict(), save_path)
    elif hasattr(model, "net"):
        torch.save(model.net.state_dict(), save_path)
    else:
        torch.save(model.state_dict(), save_path)


def train(args):
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    os.makedirs(args.save_dir, exist_ok=True)

    train_set = SpatialTripletDataset(
        data_root=args.data_root,
        split="train",
        crop_size=args.crop_size,
        augment=True,
    )

    val_set = SpatialTripletDataset(
        data_root=args.data_root,
        split="val",
        crop_size=args.crop_size,
        augment=False,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    print(f"Train samples: {len(train_set)}")
    print(f"Val samples: {len(val_set)}")
    print(f"Device: {device}")

    Model = import_practical_model(args.model_dir)
    model = Model()

    print(f"Loading model from: {args.model_dir}")
    model.load_model(args.model_dir, -1)

    if hasattr(model, "device"):
        model.device()
    else:
        model.to(device)

    if not hasattr(model, "update"):
        raise AttributeError(
            " current  Practical-RIFE  model is missing  update(),  cannot fine-tune directly ."
            " use a model package with training code ,  or only use  Practical-RIFE  for inference ."
        )

    print("Model has update(). Fine-tuning is available.")
    print("update signature:", inspect.signature(model.update))

    writer = SummaryWriter(args.log_dir)

    total_steps = max(1, args.epochs * len(train_loader))
    global_step = 0
    best_val_l1 = float("inf")

    for epoch in range(args.epochs):
        model.train()
        start_time = time.time()

        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device, non_blocking=True) / 255.0

            imgs = data[:, :6]
            gt = data[:, 6:9]

            lr = get_lr(
                step=global_step,
                total_steps=total_steps,
                base_lr=args.lr,
                min_lr=args.min_lr,
                warmup_steps=args.warmup_steps,
            )

            output = call_update(
                model=model,
                imgs=imgs,
                gt=gt,
                lr=lr,
                training=True,
            )

            if isinstance(output, tuple):
                pred = output[0]
                info = output[1] if len(output) > 1 else {}
            else:
                pred = output
                info = {}

            if isinstance(info, dict) and "loss_l1" in info:
                loss_l1 = info["loss_l1"].detach().item()
            else:
                loss_l1 = torch.mean(torch.abs(pred - gt)).detach().item()

            if global_step % args.print_freq == 0:
                print(
                    f"epoch={epoch:03d} "
                    f"step={global_step:06d}/{total_steps:06d} "
                    f"lr={lr:.3e} "
                    f"loss_l1={loss_l1:.6f}"
                )

                writer.add_scalar("train/lr", lr, global_step)
                writer.add_scalar("train/loss_l1", loss_l1, global_step)

            global_step += 1

        metrics = evaluate(model, val_loader, device)

        writer.add_scalar("val/l1", metrics["val_l1"], epoch)
        writer.add_scalar("val/psnr", metrics["val_psnr"], epoch)

        print(
            f"[VAL] epoch={epoch:03d} "
            f"val_l1={metrics['val_l1']:.6f} "
            f"val_psnr={metrics['val_psnr']:.3f} "
            f"time={time.time() - start_time:.1f}s"
        )

        latest_path = os.path.join(args.save_dir, "flownet_latest.pkl")
        save_model(model, latest_path)

        if metrics["val_l1"] < best_val_l1:
            best_val_l1 = metrics["val_l1"]
            best_path = os.path.join(args.save_dir, "flownet_best.pkl")
            save_model(model, best_path)
            print(f"Saved best model to {best_path}")

        if epoch % args.save_vis_freq == 0:
            save_visual_examples(
                model=model,
                val_loader=val_loader,
                device=device,
                save_dir=os.path.join(args.save_dir, "vis"),
                epoch=epoch,
                max_images=args.num_vis_images,
            )

    writer.close()

    print("Training finished.")
    print(f"Best val_l1: {best_val_l1:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--model-dir", type=str, default="./train_log")
    parser.add_argument("--save-dir", type=str, default="./train_log_spatial")
    parser.add_argument("--log-dir", type=str, default="./runs_spatial_practical_rife")
    
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--val-batch-size", type=int, default=1)
    parser.add_argument("--crop-size", type=int, default=224)

    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--min-lr", type=float, default=1e-7)
    parser.add_argument("--warmup-steps", type=int, default=20)

    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--print-freq", type=int, default=1)

    parser.add_argument("--save-vis-freq", type=int, default=1)
    parser.add_argument("--num-vis-images", type=int, default=4)

    parser.add_argument("--seed", type=int, default=1234)

    args = parser.parse_args()

    train(args)