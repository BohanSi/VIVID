import os
import cv2
import torch
import random
import numpy as np
from torch.utils.data import Dataset

cv2.setNumThreads(1)


class SpatialTripletDataset(Dataset):
    def __init__(
        self,
        data_root,
        split="train",
        crop_size=224,
        augment=True,
    ):
        self.data_root = data_root
        self.sequence_root = os.path.join(data_root, "sequences")
        self.crop_size = crop_size
        self.split = split
        self.augment = augment and split == "train"

        if split == "train":
            list_path = os.path.join(data_root, "trainlist.txt")
        elif split in ["val", "validation"]:
            list_path = os.path.join(data_root, "vallist.txt")
        else:
            raise ValueError(f"Unknown split: {split}")

        if not os.path.exists(list_path):
            raise FileNotFoundError(list_path)

        with open(list_path, "r") as f:
            self.meta_data = f.read().splitlines()

        if len(self.meta_data) == 0:
            raise ValueError(f"No samples found in {list_path}")

    def __len__(self):
        return len(self.meta_data)

    def _read_img(self, path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(path)
        return img

    def _random_crop(self, img0, gt, img1):
        h, w, _ = img0.shape
        cs = self.crop_size

        if h < cs or w < cs:
            raise ValueError(
                f"Image size {(h, w)} is smaller than crop_size={cs}. "
                f" reduce  crop_size  or use a larger raster ."
            )

        x = np.random.randint(0, h - cs + 1)
        y = np.random.randint(0, w - cs + 1)

        img0 = img0[x:x + cs, y:y + cs]
        gt = gt[x:x + cs, y:y + cs]
        img1 = img1[x:x + cs, y:y + cs]

        return img0, gt, img1

    def __getitem__(self, index):
        name = self.meta_data[index]
        img_dir = os.path.join(self.sequence_root, name)

        img0 = self._read_img(os.path.join(img_dir, "im1.png"))
        gt = self._read_img(os.path.join(img_dir, "im2.png"))
        img1 = self._read_img(os.path.join(img_dir, "im3.png"))

        if img0.shape != gt.shape or img0.shape != img1.shape:
            raise ValueError(
                f"Shape mismatch in {img_dir}: "
                f"im1={img0.shape}, im2={gt.shape}, im3={img1.shape}"
            )

        timestep = 0.5

        if self.augment:
            img0, gt, img1 = self._random_crop(img0, gt, img1)

            
            if random.random() < 0.5:
                img0 = img0[:, ::-1]
                gt = gt[:, ::-1]
                img1 = img1[:, ::-1]

            
            if random.random() < 0.5:
                img0 = img0[::-1]
                gt = gt[::-1]
                img1 = img1[::-1]

            
            if random.random() < 0.5:
                img0, img1 = img1, img0
                timestep = 1.0 - timestep

            
            p = random.random()
            if p < 0.25:
                img0 = cv2.rotate(img0, cv2.ROTATE_90_CLOCKWISE)
                gt = cv2.rotate(gt, cv2.ROTATE_90_CLOCKWISE)
                img1 = cv2.rotate(img1, cv2.ROTATE_90_CLOCKWISE)
            elif p < 0.50:
                img0 = cv2.rotate(img0, cv2.ROTATE_180)
                gt = cv2.rotate(gt, cv2.ROTATE_180)
                img1 = cv2.rotate(img1, cv2.ROTATE_180)
            elif p < 0.75:
                img0 = cv2.rotate(img0, cv2.ROTATE_90_COUNTERCLOCKWISE)
                gt = cv2.rotate(gt, cv2.ROTATE_90_COUNTERCLOCKWISE)
                img1 = cv2.rotate(img1, cv2.ROTATE_90_COUNTERCLOCKWISE)

        img0 = torch.from_numpy(img0.copy()).permute(2, 0, 1).float()
        img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1).float()
        gt = torch.from_numpy(gt.copy()).permute(2, 0, 1).float()

        timestep = torch.tensor(timestep).reshape(1, 1, 1).float()

        
        
        data = torch.cat([img0, img1, gt], dim=0)

        return data, timestep