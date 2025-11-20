"""Train script for LightSegNet (StageA-light).

Expect paired lists of image paths and mask paths (one per line, same order).
Example:
  python scripts/train_light.py --train-imgs data/splits/train_imgs.txt --train-masks data/splits/train_masks.txt \
      --val-imgs data/splits/val_imgs.txt --val-masks data/splits/val_masks.txt --epochs 30 --batch 8

Notes on using stageA-heavy as pretraining/teacher:
- If you have a heavy checkpoint, you can generate pseudo-labels (prob maps) using existing
  `infer_batch` / `infer_image` utilities in `road_crack_hsv_e2e.py` and then train LightSegNet on
  those pseudo-labels (possibly combined with GT masks) for knowledge transfer.
- Directly loading heavy weights into LightSegNet is generally not possible due to architectural
  mismatch; prefer distillation or pseudo-label generation.
"""
import argparse
import os
import time
from typing import List

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from common.light_models import LightSegNet, count_params


def dice_loss_prob(pred, target, eps=1e-6):
    # pred and target are probabilities in [0,1]
    num = 2 * (pred * target).sum(dim=(1, 2, 3)) + eps
    den = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    return 1 - (num / den).mean()


def iou_metric_np(pred, target, thr=0.5):
    pred_b = (pred > thr).astype(np.uint8)
    inter = (pred_b & target).sum()
    union = ((pred_b | target)).sum()
    return float(inter) / (float(union) + 1e-9)


class PairedImageMaskDataset(Dataset):
    def __init__(self, img_list: List[str], mask_list: List[str], H=512, W=512):
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
        # resize to fixed shape
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
            probs = torch.sigmoid(logits)
            probs_np = probs.cpu().numpy()
            masks_np = masks.cpu().numpy()
            for p, g in zip(probs_np, masks_np):
                # p shape (1,H,W)
                p = p[0]
                g = g[0].astype(np.uint8)
                ious.append(iou_metric_np(p, g, thr=0.5))
                # dice
                p_bin = (p > 0.5).astype(np.float32)
                dice = (2.0 * (p_bin * g).sum()) / (p_bin.sum() + g.sum() + 1e-9)
                dices.append(float(dice))
    mean_iou = float(np.mean(ious)) if ious else 0.0
    mean_dice = float(np.mean(dices)) if dices else 0.0
    return mean_iou, mean_dice


def train(args):
    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    train_imgs = read_list(args.train_imgs)
    train_masks = read_list(args.train_masks)
    val_imgs = read_list(args.val_imgs) if args.val_imgs else []
    val_masks = read_list(args.val_masks) if args.val_masks else []

    train_ds = PairedImageMaskDataset(train_imgs, train_masks, H=args.H, W=args.W)
    val_ds = PairedImageMaskDataset(val_imgs, val_masks, H=args.H, W=args.W) if val_imgs else None

    dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    vdl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0) if val_ds is not None else None

    model = LightSegNet(base_ch=args.base_ch, input_channels=3)
    print('LightSegNet params', count_params(model))
    if args.pretrained and os.path.exists(args.pretrained):
        print('loading pretrained', args.pretrained)
        state = torch.load(args.pretrained, map_location='cpu')
        try:
            model.load_state_dict(state)
        except Exception:
            # try relaxed
            model.load_state_dict(state, strict=False)
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))

    bce = nn.BCEWithLogitsLoss()

    best_iou = 0.0
    os.makedirs(args.save_dir, exist_ok=True)

    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        it = 0
        for imgs, masks in dl:
            imgs = imgs.to(device)
            masks = masks.to(device)
            logits = model(imgs)
            loss_bce = bce(logits, masks)
            probs = torch.sigmoid(logits)
            loss_dice = dice_loss_prob(probs, masks)
            loss = args.lambda_bce * loss_bce + args.lambda_dice * loss_dice
            opt.zero_grad()
            loss.backward()
            opt.step()
            running_loss += float(loss.item())
            it += 1
        scheduler.step()
        t1 = time.time()
        avg_loss = running_loss / (it if it > 0 else 1)

        if vdl is not None:
            mean_iou, mean_dice = validate(model, vdl, device)
        else:
            mean_iou, mean_dice = 0.0, 0.0

        print(f'ep {ep} loss {avg_loss:.6f} val_iou {mean_iou:.4f} val_dice {mean_dice:.4f} time {t1-t0:.1f}s')

        # save best
        if mean_iou > best_iou:
            best_iou = mean_iou
            outp = os.path.join(args.save_dir, f'light_best_ep{ep}_miou{mean_iou:.4f}.pth')
            torch.save(model.state_dict(), outp)
            print('saved', outp)
        # periodic save
        if ep % args.save_every == 0:
            outp = os.path.join(args.save_dir, f'light_ep{ep}.pth')
            torch.save(model.state_dict(), outp)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--train-imgs', required=True)
    p.add_argument('--train-masks', required=True)
    p.add_argument('--val-imgs', required=False)
    p.add_argument('--val-masks', required=False)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-3)
    # prefer a 16:9-ish fixed input that is divisible by 16 (suits 4x downsample)
    p.add_argument('--H', type=int, default=544)
    p.add_argument('--W', type=int, default=960)
    p.add_argument('--base_ch', type=int, default=16)
    p.add_argument('--save-dir', type=str, default='models')
    p.add_argument('--save-every', type=int, default=5)
    p.add_argument('--pretrained', type=str, default=None, help='optional path to pretrained weights for model initialization')
    p.add_argument('--device', type=str, default=None)
    p.add_argument('--lambda-bce', type=float, default=1.0)
    p.add_argument('--lambda-dice', type=float, default=1.0)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
