"""Evaluate a saved student checkpoint on a validation split and print IoU/Dice.

Usage:
  python scripts/eval_student_checkpoint.py --ckpt models/tmp_light_distill_smoke/light_distill_ep0.pth --val-imgs data/splits/val_road.txt --val-masks data/splits/val_road_masks.txt --batch 4 --H 544 --W 960 --device cpu
"""
import argparse
import os
import sys
from pathlib import Path
import time

try:
    repo_root = str(Path(__file__).resolve().parents[1])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
except Exception:
    pass

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from common.light_models import LightSegNet, count_params


class PairedImageMaskDataset(Dataset):
    def __init__(self, img_list, mask_list, H=544, W=960):
        assert len(img_list) == len(mask_list)
        self.imgs = img_list
        self.masks = mask_list
        self.H = H
        self.W = W

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        p = self.imgs[idx].strip()
        m = self.masks[idx].strip()
        img = cv2.imread(p)
        if img is None:
            raise RuntimeError('failed load image ' + p)
        mask = cv2.imread(m, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError('failed load mask ' + m)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
        img = img.astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.float32)
        img_t = torch.from_numpy(img.transpose(2, 0, 1)).float()
        mask_t = torch.from_numpy(mask).unsqueeze(0).float()
        return img_t, mask_t


def read_list(path):
    with open(path, 'r', encoding='utf8') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    return lines


def validate(model, dl, device):
    model.eval()
    ious = []
    dices = []
    with torch.no_grad():
        for imgs, masks in dl:
            imgs = imgs.to(device)
            masks = masks.to(device)
            logits = model(imgs)
            if logits.shape[-2:] != masks.shape[-2:]:
                logits = F.interpolate(logits, size=(masks.shape[-2], masks.shape[-1]), mode='bilinear', align_corners=False)
            probs = torch.sigmoid(logits)
            probs_np = probs.cpu().numpy()
            masks_np = masks.cpu().numpy()
            for p, g in zip(probs_np, masks_np):
                p = p[0]
                g = g[0].astype(np.uint8)
                inter = (p > 0.5).astype(np.uint8) & g
                union = ((p > 0.5).astype(np.uint8) | g)
                iou = float(inter.sum()) / (float(union.sum()) + 1e-9)
                ious.append(iou)
                p_bin = (p > 0.5).astype(np.float32)
                dice = (2.0 * (p_bin * g).sum()) / (p_bin.sum() + g.sum() + 1e-9)
                dices.append(float(dice))
    return float(np.mean(ious)) if ious else 0.0, float(np.mean(dices)) if dices else 0.0


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--val-imgs', required=True)
    p.add_argument('--val-masks', required=True)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--H', type=int, default=544)
    p.add_argument('--W', type=int, default=960)
    p.add_argument('--device', type=str, default='cpu')
    args = p.parse_args()

    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print('device ->', device)

    val_imgs = read_list(args.val_imgs)
    val_masks = read_list(args.val_masks)
    val_ds = PairedImageMaskDataset(val_imgs, val_masks, H=args.H, W=args.W)
    vdl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    student = LightSegNet(base_ch=16, input_channels=3)
    print('Student params', count_params(student))
    ckpt = torch.load(args.ckpt, map_location='cpu')
    # ckpt might be a raw state_dict or wrapped dict
    if isinstance(ckpt, dict) and 'model' in ckpt:
        state = ckpt['model']
    else:
        state = ckpt
    try:
        student.load_state_dict(state)
    except Exception:
        student.load_state_dict(state, strict=False)
    student.to(device)

    print('Running validation...')
    t0 = time.time()
    mean_iou, mean_dice = validate(student, vdl, device)
    t1 = time.time()
    print(f'Validation done in {t1-t0:.1f}s; mean_iou={mean_iou:.4f} mean_dice={mean_dice:.4f}')
