import os
import sys
import math
import argparse
import random
from time import time
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable as V
from tqdm import tqdm

# Allow running from repo root or experiments folder
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(FILE_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
# Also allow importing `networks.*` when running from experiments/
if FILE_DIR not in sys.path:
    sys.path.insert(0, FILE_DIR)

# Reuse training frame and losses from existing code
from experiments.framework import MyFrame
from experiments.loss import (
    dice_bce_loss,
    focal_tversky_loss,
    binary_tversky_loss,
    binary_focal_tversky_loss,
    focal_tversky_edge_loss,
)


# -----------------------------
# Metric helpers (align with eval_experiment1)
# -----------------------------
def proba_from_logits(x: torch.Tensor) -> torch.Tensor:
    """Map model output to probabilities.
    - If input appears to already be probabilities in [0,1], return as-is.
    - If shape (B,1,H,W), apply sigmoid.
    - If shape (B,2,H,W), apply softmax and take class-1.
    - Else, apply sigmoid as a fallback.
    """
    # Accept list/tuple by taking first element as main branch
    if isinstance(x, (list, tuple)) and len(x) > 0:
        x = x[0]
    if x.ndim == 4:
        with torch.no_grad():
            xmin = float(x.min().item())
            xmax = float(x.max().item())
        if xmin >= 0.0 and xmax <= 1.0:
            return x
        if x.size(1) == 1:
            return torch.sigmoid(x)
        if x.size(1) == 2:
            return torch.softmax(x, dim=1)[:, 1:2]
    return torch.sigmoid(x)


def metrics_from_binary(pred: np.ndarray, gt: np.ndarray):
    pred = pred.astype(np.uint8)
    gt = gt.astype(np.uint8)
    TP = int(((pred == 1) & (gt == 1)).sum())
    TN = int(((pred == 0) & (gt == 0)).sum())
    FP = int(((pred == 1) & (gt == 0)).sum())
    FN = int(((pred == 0) & (gt == 1)).sum())
    eps = 1e-6
    acc = (TP + TN) / (TP + TN + FP + FN + eps)
    pre = TP / (TP + FP + eps)
    rec = TP / (TP + FN + eps)
    iou = TP / (TP + FP + FN + eps)
    dice = 2 * TP / (2 * TP + FP + FN + eps)
    fpr = FP / (FP + TN + eps)
    fnr = FN / (FN + TP + eps)
    return acc, pre, rec, iou, dice, fpr, fnr


def normalize_image(img_bgr: np.ndarray) -> np.ndarray:
    # Resize is done outside; normalize to match existing training code: /255 * 3.2 - 1.6
    return (img_bgr.astype(np.float32) / 255.0) * 3.2 - 1.6


def load_mask_binary(mask_gray: np.ndarray) -> np.ndarray:
    # Convert to {0,1} in CHW
    if mask_gray.ndim == 3:
        mask_gray = cv2.cvtColor(mask_gray, cv2.COLOR_BGR2GRAY)
    mask = (mask_gray > 0).astype(np.float32)
    return mask


class SegPairDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str, str]], img_size: Tuple[int, int] = (1024, 1024)):
        self.pairs = pairs
        self.img_size = img_size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        # Resize
        img = cv2.resize(img, self.img_size)
        mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)

        img = normalize_image(img)
        mask = load_mask_binary(mask)

        img = torch.from_numpy(img.transpose(2, 0, 1))  # CHW
        mask = torch.from_numpy(mask[None, ...])  # 1HW
        return img, mask


def _read_lines(path: str) -> List[str]:
    with open(path, 'r', encoding='utf-8') as f:
        return [ln.strip() for ln in f if ln.strip()]


def make_pairs_for_lists(images_list: str, masks_list: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    imgs = _read_lines(images_list)
    masks = _read_lines(masks_list)
    if len(imgs) != len(masks):
        raise RuntimeError('images-list and masks-list must have the same number of lines')
    for ip, mp in zip(imgs, masks):
        # allow relative path from repo root
        if not os.path.isabs(ip):
            ip = os.path.join(REPO_ROOT, ip)
        if not os.path.isabs(mp):
            mp = os.path.join(REPO_ROOT, mp)
        if os.path.exists(ip) and os.path.exists(mp):
            pairs.append((ip, mp))
    if not pairs:
        raise RuntimeError('No pairs found from provided list files')
    return pairs


def make_pairs_for_dataset(dataset: str,
                           root: str = None,
                           images_dir: str = None,
                           masks_dir: str = None) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    ds = dataset.lower()
    # synonyms mapping
    if ds in ('massachusetts', 'mas'):
        if not root:
            raise ValueError("For dataset 'mas', --root must be provided and point to the Massachusetts dataset root.")
        # Support two layouts:
        # 1) <root>/train/images/*.tiff with masks in <root>/train/masks_road/*.tif
        # 2) <root>/tiff/train/*.tiff with masks in <root>/tiff/train_labels/*.tif
        img_root1 = os.path.join(root, 'train', 'images')
        mask_root1 = os.path.join(root, 'train', 'masks_road')
        img_root2 = os.path.join(root, 'tiff', 'train')
        mask_root2 = os.path.join(root, 'tiff', 'train_labels')
        if os.path.isdir(img_root2) and os.path.isdir(mask_root2):
            for name in os.listdir(img_root2):
                if not name.lower().endswith('.tiff'):
                    continue
                stem = os.path.splitext(name)[0]
                img_path = os.path.join(img_root2, name)
                mask_path = os.path.join(mask_root2, f"{stem}.tif")
                if os.path.exists(img_path) and os.path.exists(mask_path):
                    pairs.append((img_path, mask_path))
        else:
            for name in os.listdir(img_root1):
                stem = os.path.splitext(name)[0]
                img_path = os.path.join(img_root1, f"{stem}.tiff")
                mask_path = os.path.join(mask_root1, f"{stem}.tif")
                if os.path.exists(img_path) and os.path.exists(mask_path):
                    pairs.append((img_path, mask_path))
    elif ds in ('road', 'customer'):
        if images_dir is None:
            images_dir = os.path.join(REPO_ROOT, 'data', 'images', 'road')
        if masks_dir is None:
            masks_dir = os.path.join(REPO_ROOT, 'data', 'masks', 'road')
        for name in os.listdir(images_dir):
            stem, _ = os.path.splitext(name)
            # images may be .jpg, masks are .png with same stem
            img_path = os.path.join(images_dir, name)
            mask_png = os.path.join(masks_dir, f"{stem}.png")
            if os.path.exists(img_path) and os.path.exists(mask_png):
                pairs.append((img_path, mask_png))
    elif ds in ('deepglobe',):
        # For DeepGlobe (or future datasets), prefer training via lists for flexibility
        raise RuntimeError("For dataset 'DeepGlobe', please provide --train-images-list and --train-masks-list.")
    else:
        raise ValueError("dataset must be one of: mas|Massachusetts, road|Customer, or DeepGlobe (use lists)")

    if not pairs:
        raise RuntimeError(f"No (image, mask) pairs found for dataset={dataset}. Check paths.")
    return sorted(pairs)


def make_pairs_for_dataset_split(dataset: str,
                                 root: str = None) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Discover train/val pairs according to repo's agreed split layout.
    - road/customer: use data/splits/{train_road,train_road_masks,val_road,val_road_masks}
    - mas/Massachusetts: use <root>/tiff/{train,train_labels,val,val_labels}
    Returns: (train_pairs, val_pairs)
    """
    ds = dataset.lower()
    def _scan_pair_dir(img_dir: str, mask_dir: str) -> List[Tuple[str, str]]:
        res = []
        if not (os.path.isdir(img_dir) and os.path.isdir(mask_dir)):
            return res
        for name in os.listdir(img_dir):
            stem, _ = os.path.splitext(name)
            img_path = os.path.join(img_dir, name)
            for ext in ('.png', '.tif', '.tiff', '.jpg'):
                m = os.path.join(mask_dir, stem + ext)
                if os.path.exists(m):
                    res.append((img_path, m))
                    break
        return sorted(res)

    if ds in ('road', 'customer'):
        base = os.path.join(REPO_ROOT, 'data', 'splits')
        # Prefer list files if present
        tr_img_list = os.path.join(base, 'train_road.txt')
        tr_msk_list = os.path.join(base, 'train_road_masks.txt')
        va_img_list = os.path.join(base, 'val_road.txt')
        va_msk_list = os.path.join(base, 'val_road_masks.txt')
        if all(os.path.isfile(p) for p in (tr_img_list, tr_msk_list, va_img_list, va_msk_list)):
            train_pairs = make_pairs_for_lists(tr_img_list, tr_msk_list)
            val_pairs = make_pairs_for_lists(va_img_list, va_msk_list)
        else:
            # fallback to directory-style splits
            tr_img = os.path.join(base, 'train_road')
            tr_msk = os.path.join(base, 'train_road_masks')
            va_img = os.path.join(base, 'val_road')
            va_msk = os.path.join(base, 'val_road_masks')
            train_pairs = _scan_pair_dir(tr_img, tr_msk)
            val_pairs = _scan_pair_dir(va_img, va_msk)
        if not train_pairs or not val_pairs:
            raise RuntimeError('Road splits not found or empty under data/splits (txt lists or folders).')
        return train_pairs, val_pairs
    elif ds in ('massachusetts', 'mas'):
        if not root:
            # default to repo dataset path
            root = os.path.join(REPO_ROOT, 'data', 'Massachusetts')
        tiff_root = os.path.join(root, 'tiff')
        tr_img = os.path.join(tiff_root, 'train')
        tr_msk = os.path.join(tiff_root, 'train_labels')
        va_img = os.path.join(tiff_root, 'val')
        va_msk = os.path.join(tiff_root, 'val_labels')
        train_pairs = _scan_pair_dir(tr_img, tr_msk)
        val_pairs = _scan_pair_dir(va_img, va_msk)
        if not train_pairs or not val_pairs:
            raise RuntimeError('Massachusetts splits not found or empty under <root>/tiff/*. Please verify dataset root.')
        return train_pairs, val_pairs
    else:
        raise ValueError("dataset must be one of: mas|Massachusetts, road|Customer")


def build_net_factory(net_name: str,
                      img_size: Tuple[int, int] = None,
                      backbone: str = None,
                      pretrained: bool = False,
                      **adapter_kwargs):
    net_name = net_name.strip()
    # Map common names to (module, attr)
    reg = {
        'BiReNet34': ('networks.BiReNet', 'BiReNet34'),
        'BiResUnetPlus': ('networks.BiResUnetPlusAdapter', 'BiResUnetPlusAdapter'),
        'RCFSNet': ('networks.RCFSNet', 'RCFSNet'),
        'DinkNet34': ('networks.DLinkNet', 'DinkNet34'),
        'NLinkNet34': ('networks.Nlinknet', 'NLinkNet34'),
        'DeepLabv3_plus': ('networks.deeplabv3plus', 'DeepLabv3_plus'),
        'DBRANet': ('networks.DBRANet', 'DBRANet'),
        'MACUNet': ('networks.MACUNet', 'MACUNet'),
        'LinkNet34': ('networks.Linknet', 'LinkNet34'),
        'TransRoadNet': ('networks.TransRoadNet', 'swin_s'),
        'U_Net': ('networks.Unet', 'U_Net'),
        'CARNet': ('networks.CARNet', 'DAM_Net_5'),
        'MSMDFFNet': ('networks.MSMDFF_Net', 'MSMDFF_Net_v3_plus'),
        'SSCNet': ('networks.SSCNet', 'SSCNet'),
        'LightSegNet': ('common.light_models', 'LightSegNet'),
    }
    if net_name not in reg:
        raise ValueError(f"Unsupported net '{net_name}'. Supported: {list(reg.keys())}")
    mod_name, attr = reg[net_name]

    # Lazy import with helpful error for optional external deps
    try:
        mod = __import__(mod_name, fromlist=[attr])
    except Exception as e:
        raise RuntimeError(f"Import failed for model '{net_name}' from '{mod_name}'. "
                           f"Please ensure all submodules are available. Details: {e}")

    cls_or_fn = getattr(mod, attr)

    # Special-case Unet to be 1-channel output and add Sigmoid if needed
    if net_name == 'U_Net':
        def factory():
            m = cls_or_fn(in_ch=3, out_ch=1)
            # Ensure sigmoid for BCE loss
            return torch.nn.Sequential(m, torch.nn.Sigmoid())
        return factory

    # Special-case BiReNet to output single tensor (disable tuple during training)
    if net_name == 'BiReNet34':
        def factory():
            m = cls_or_fn(pretrained=bool(pretrained))
            # Ensure training returns single map for BCE + Dice
            if hasattr(m, 'is_Train'):
                m.is_Train = False
            return m
        return factory

    # BiResUnetPlusAdapter returns probability map already
    if net_name == 'BiResUnetPlus':
        def factory():
            bb = (backbone or 'resnet34').lower()
            # support wider set: 18/34/50/101 (core handles channel projection internally)
            if bb not in ('resnet18', 'resnet34', 'resnet50', 'resnet101'):
                bb = 'resnet34'
            # Pass through adapter kwargs for ALT construction (alt_build_mode, hsv stats, etc.)
            return cls_or_fn(out_ch=1,
                              backbone=bb,
                              pretrained=bool(pretrained),
                              alt_build_mode=adapter_kwargs.get('alt_build_mode'),
                              hsv_stats_path=adapter_kwargs.get('hsv_stats_path'),
                              v_window_mode=adapter_kwargs.get('v_window_mode', 'mad'),
                              smooth_temp=adapter_kwargs.get('smooth_temp', 4.0),
                              channel_weights=adapter_kwargs.get('channel_weights'),
                              use_lite_aspp=adapter_kwargs.get('use_lite_aspp'))
        return factory

    # RCFSNet: plumb pretrained flag for local ResNet34 weights
    if net_name == 'RCFSNet':
        def factory():
            return cls_or_fn(num_classes=1, pretrained=bool(pretrained))
        return factory


    # DANet 已移至 legacy，不再支持在当前训练入口中构建

    # CARNet requires img_size to construct attention grids; plumb pretrained flag
    if net_name == 'CARNet':
        def factory():
            size = None
            if img_size is not None:
                # use the larger side as base size (expects int)
                size = int(max(img_size))
            # Fallback to default if size is None
            return cls_or_fn(img_size=size if size is not None else 1024, num_classes=1, pretrained=bool(pretrained))
        return factory

    # MSMDFFNet may depend on external subpackages; instantiate with defaults
    if net_name == 'MSMDFFNet':
        def factory():
            return cls_or_fn(in_c=3, num_classes=1)
        return factory

    # SSCNet: pass through pretrained flag for local EfficientNet weights
    if net_name == 'SSCNet':
        def factory():
            return cls_or_fn(num_classes=1, pretrained=bool(pretrained))
        return factory

    # DinkNet34: plumb pretrained flag
    if net_name == 'DinkNet34':
        def factory():
            return cls_or_fn(num_classes=1, pretrained=bool(pretrained))
        return factory

    # NLinkNet34: plumb pretrained flag
    if net_name == 'NLinkNet34':
        def factory():
            return cls_or_fn(num_classes=1, pretrained=bool(pretrained))
        return factory

    # LinkNet34: plumb pretrained flag
    if net_name == 'LinkNet34':
        def factory():
            return cls_or_fn(num_classes=1, pretrained=bool(pretrained))
        return factory

    # DBRANet: plumb pretrained flag for local ResNet34 weights
    if net_name == 'DBRANet':
        def factory():
            return cls_or_fn(pretrained=bool(pretrained))
        return factory

    # TransRoadNet (swin_s): plumb pretrained flag into underlying SwinTransformer
    if net_name == 'TransRoadNet':
        def factory():
            return cls_or_fn(pretrained=bool(pretrained))
        return factory

    # DeepLabv3_plus: plumb pretrained flag
    if net_name == 'DeepLabv3_plus':
        def factory():
            return cls_or_fn(nInputChannels=3, n_classes=1, os=16, pretrained=bool(pretrained), freeze_bn=False, _print=False)
        return factory

    # LightSegNet produces logits at 1/2 resolution; upsample to img_size and wrap Sigmoid for dice_bce
    if net_name == 'LightSegNet':
        def factory():
            # map backbone token to capacity tiers
            # 18->16 (light), 34->24 (standard), 50->32 (large), 101->40 (x-large)
            bb = (backbone or 'resnet34').lower()
            base_map = {
                'resnet18': 16,
                'resnet34': 24,
                'resnet50': 32,
                'resnet101': 40,
            }
            base_ch = base_map.get(bb, 24)
            m = cls_or_fn(base_ch=base_ch, input_channels=3)
            # torch Upsample expects size=(H,W); img_size is (W,H)
            size_hw = (int(img_size[1]), int(img_size[0])) if img_size is not None else None
            layers = [m]
            if size_hw is not None:
                layers.append(torch.nn.Upsample(size=size_hw, mode='bilinear', align_corners=False))
            layers.append(torch.nn.Sigmoid())
            return torch.nn.Sequential(*layers)
        return factory

    # Default: return class / fn directly
    return cls_or_fn


def parse_args():
    p = argparse.ArgumentParser(description='Experiment 1 Training (Massachusetts / Road)')
    p.add_argument('--bires-eem', action='store_true', help='Enable EEM edge enhancement in BiResUnetPlus encoder (recommended for mas dataset)')
    p.add_argument('--dataset', required=True, help='Dataset selector (mas|Massachusetts, road|Customer, DeepGlobe)')
    p.add_argument('--root', type=str, default=None, help='Massachusetts dataset root (required for --dataset mas)')
    p.add_argument('--images-dir', type=str, default=None, help='Custom images dir (for --dataset road)')
    p.add_argument('--masks-dir', type=str, default=None, help='Custom masks dir (for --dataset road)')
    p.add_argument('--train-images-list', type=str, default=None, help='Optional: path to training images list file')
    p.add_argument('--train-masks-list', type=str, default=None, help='Optional: path to training masks list file')
    p.add_argument('--val-images-list', type=str, default=None, help='Optional: path to validation images list file')
    p.add_argument('--val-masks-list', type=str, default=None, help='Optional: path to validation masks list file')
    p.add_argument('--net', type=str, default='BiReNet34', help='Model name')
    p.add_argument('--backbone', type=str, default='resnet34', choices=['resnet18', 'resnet34', 'resnet50', 'resnet101'], help='Backbone (for BiResUnetPlus; also scales LightSegNet)')
    p.add_argument('--epochs', type=int, default=120)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--lr', type=float, default=8e-4)
    p.add_argument('--img-size', type=int, nargs=2, default=[1024, 1024])
    p.add_argument('--loss', choices=['dice_bce', 'focal_tversky', 'binary_tversky', 'binary_focal_tversky', 'focal_tversky_edge', 'joint_seg_edge'], default='dice_bce')
    p.add_argument('--save-dir', type=str, default=os.path.join(FILE_DIR, 'weights'))
    p.add_argument('--name', type=str, default=None, help='Run name prefix (auto if omitted)')
    p.add_argument('--pretrained', action='store_true', help='Use ImageNet pretrained weights for supported backbones (e.g., BiResUnetPlus)')
    p.add_argument('--resume', type=str, default=None, help='Resume from a checkpoint (.pth/.th) path (supports full ckpt dict)')
    p.add_argument('--gpus', type=str, default=None, help='Comma-separated GPU ids (e.g., 0,1). If omitted, default CUDA_VISIBLE_DEVICES is used or CPU.')
    # Allow flexible device syntax: 'auto' | 'cpu' | 'cuda' | 'cuda:IDX' | numeric IDX
    p.add_argument('--device', type=str, default='auto', help="Compute device: auto|cpu|cuda or cuda:IDX / numeric IDX")
    p.add_argument('--limit-samples', type=int, default=None, help='Randomly subsample N pairs for quick tests')
    p.add_argument('--val-interval', type=int, default=1, help='Run validation every N epochs')
    p.add_argument('--ckpt-interval', type=int, default=5, help='Save rolling checkpoints every N epochs')
    p.add_argument('--threshold', type=float, default=0.5, help='Threshold for metric binarization')
    # Global seed for reproducibility
    p.add_argument('--global-seed', type=int, default=None, help='Global random seed for reproducibility (torch/np/random)')
    # Early-stop / LR-schedule / monitor controls
    p.add_argument('--early-stop-patience', type=int, default=10, help='Stop if no improvement for N checks (default: 10)')
    p.add_argument('--lr-reduce-patience', type=int, default=6, help='Reduce LR by factor if no improvement for N checks (default: 6)')
    p.add_argument('--lr-reduce-factor', type=float, default=2.5, help='LR reduction factor when patience exceeded (default: 2.5, i.e., lr/=2.5)')
    p.add_argument('--min-lr', type=float, default=2e-7, help='Minimum learning rate before stopping (default: 2e-7)')
    p.add_argument('--monitor-metric', choices=['val_loss','train_loss','dice'], default='dice', help='Metric to monitor for early-stop/LR schedule (default: dice)')
    # ALT / HSV differentiable builder parameters (BiResUnetPlusAdapter only)
    p.add_argument('--alt-build-mode', type=str, default='hsvgrad', choices=['rgbgrad','hsvgrad','hsv_cv'],
                   help='ALT 构造模式: rgbgrad|hsvgrad(可微)|hsv_cv(OpenCV旧版)')
    p.add_argument('--alt-hsv-stats', type=str, default=None,
                   help='HSV 自适应阈值统计文件路径 (.npz/.json)。仅在 hsvgrad 或 hsv_cv(adaptive) 时使用')
    p.add_argument('--alt-v-window-mode', type=str, default='mad', choices=['mad','legacy'],
                   help='Value 通道阈值窗口模式: mad 使用中位绝对偏差窗口, legacy 使用原始 v_low/v_high')
    p.add_argument('--alt-smooth-temp', type=float, default=4.0,
                   help='平滑阈值温度 (越大越平缓)。用于 hsvgrad sigmoid 区间与 S<thr mask')
    p.add_argument('--alt-channel-weights', type=str, default=None,
                   help='8 通道权重 (逗号分隔)。例如: 1,1,1,1,0.5,0.5,1,1 可抑制方向通道或某些 HSV')
    # EEM configuration passthrough (levels as comma-separated list, rgb/alt toggles, reduction)
    p.add_argument('--bires-eem-levels', type=str, default=None,
                   help='Comma-separated EEM levels to enable (choose from 1,2,3,4). Example: "1,2,3"')
    p.add_argument('--bires-eem-rgb', type=str, default=None, choices=['true','false'],
                   help='Whether to apply EEM to RGB encoder (true/false). If omitted, default true')
    p.add_argument('--bires-eem-alt', type=str, default=None, choices=['true','false'],
                   help='Whether to apply EEM to ALT encoder (true/false). If omitted, default true')
    p.add_argument('--bires-eem-reduction', type=int, default=None,
                   help='Reduction (squeeze) factor passed to EEM modules (default 2)')
    # ASPP selection: by default use Lite-ASPP; set --full-aspp to use FullASPP
    p.add_argument('--full-aspp', action='store_true', help='Use Full ASPP in BiResUnetPlus (otherwise LiteASPP)')
    return p.parse_args()


def main():
    args = parse_args()
    # Set global seed (if provided) for reproducibility
    if args.global_seed is not None:
        try:
            import random as _rnd
            import numpy as _np
            _rnd.seed(int(args.global_seed))
            _np.random.seed(int(args.global_seed))
            torch.manual_seed(int(args.global_seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(args.global_seed))
            # Deterministic behavior (slower but reproducible)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            os.environ['PYTHONHASHSEED'] = str(int(args.global_seed))
        except Exception:
            pass
    else:
        # Speed-optimized default when no seed is specified
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass
    # Resolve CUDA device visibility:
    # Precedence: explicit --gpus > device index in --device > default '0'
    if str(args.device).lower() != 'cpu':
        if args.gpus:
            os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
        else:
            dev_arg = str(args.device).lower() if args.device is not None else 'auto'
            explicit_gpu_idx = None
            if dev_arg.startswith('cuda:'):
                explicit_gpu_idx = dev_arg.split(':', 1)[1]
            else:
                try:
                    if dev_arg.isdigit():
                        explicit_gpu_idx = dev_arg
                except Exception:
                    explicit_gpu_idx = None
            if explicit_gpu_idx is not None:
                os.environ['CUDA_VISIBLE_DEVICES'] = str(explicit_gpu_idx)
            else:
                os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

    # Build train/val pairs
    if args.train_images_list and args.train_masks_list:
        train_pairs = make_pairs_for_lists(args.train_images_list, args.train_masks_list)
        if args.val_images_list and args.val_masks_list:
            val_pairs = make_pairs_for_lists(args.val_images_list, args.val_masks_list)
        else:
            # derive val from dataset split when not provided
            _, val_pairs = make_pairs_for_dataset_split(args.dataset, root=args.root)
    else:
        # use canonical split discovery
        train_pairs, val_pairs = make_pairs_for_dataset_split(args.dataset, root=args.root)

    # optional random subsample for quick tests
    if args.limit_samples is not None and args.limit_samples > 0:
        random.shuffle(train_pairs)
        train_pairs = train_pairs[:int(args.limit_samples)]
        random.shuffle(val_pairs)
        val_pairs = val_pairs[:max(1, int(args.limit_samples)//4)]

    img_size = (int(args.img_size[0]), int(args.img_size[1]))
    train_ds = SegPairDataset(train_pairs, img_size=img_size)
    val_ds = SegPairDataset(val_pairs, img_size=img_size)
    train_loader = DataLoader(train_ds, batch_size=max(1, int(args.batch_size)), shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=max(1, int(args.batch_size)), shuffle=False, num_workers=0, drop_last=False)

    # Build net and solver
    # Parse channel weights string into tuple[float] if provided
    ch_weights = None
    if args.alt_channel_weights:
        try:
            parts = [float(x) for x in args.alt_channel_weights.split(',') if x.strip()]
            if parts:
                ch_weights = tuple(parts)
        except Exception:
            ch_weights = None

    net_factory = build_net_factory(
        args.net,
        img_size=img_size,
        backbone=args.backbone,
        pretrained=bool(args.pretrained),
        alt_build_mode=args.alt_build_mode if args.net=='BiResUnetPlus' else None,
        hsv_stats_path=args.alt_hsv_stats if args.net=='BiResUnetPlus' else None,
        v_window_mode=args.alt_v_window_mode if args.net=='BiResUnetPlus' else None,
        smooth_temp=float(args.alt_smooth_temp) if args.net=='BiResUnetPlus' else 4.0,
        channel_weights=ch_weights if args.net=='BiResUnetPlus' else None,
        # EEM passthrough
        use_eem=bool(args.bires_eem) if args.net=='BiResUnetPlus' else None,
        eem_levels=(args.bires_eem_levels if args.net=='BiResUnetPlus' else None),
        eem_apply_rgb=(args.bires_eem_rgb if args.net=='BiResUnetPlus' else None),
        eem_apply_alt=(args.bires_eem_alt if args.net=='BiResUnetPlus' else None),
        eem_reduction=(args.bires_eem_reduction if args.net=='BiResUnetPlus' else None),
        # ASPP selection: pass use_lite_aspp (False if user requested full-aspp)
        use_lite_aspp=(not bool(args.full_aspp)) if args.net=='BiResUnetPlus' else None,
    )
    if args.loss == 'dice_bce':
        loss_cls = dice_bce_loss
    elif args.loss == 'focal_tversky':
        loss_cls = focal_tversky_loss  # Multi-class variant (rarely used for pure binary road extraction)
    elif args.loss == 'binary_tversky':
        loss_cls = binary_tversky_loss
    elif args.loss == 'binary_focal_tversky':
        loss_cls = binary_focal_tversky_loss
    elif args.loss == 'focal_tversky_edge':
        loss_cls = focal_tversky_edge_loss
    elif args.loss == 'joint_seg_edge':
        from experiments.loss import joint_seg_edge_loss
        loss_cls = joint_seg_edge_loss
    else:
        loss_cls = dice_bce_loss
    solver = MyFrame(net_factory, loss_cls, lr=float(args.lr), device=args.device)

    # Report pretrained status explicitly
    try:
        model_ref = solver.net
        # unwrap DataParallel if present
        if hasattr(model_ref, 'module'):
            model_ref = model_ref.module
        pretrained_msg = 'RandomInit'
        if args.net == 'BiResUnetPlus':
            eff = bool(getattr(model_ref, 'pretrained_effective', False))
            if eff:
                pretrained_msg = 'ImageNetPretrained'
            else:
                # requested pretrained but fell back
                pretrained_msg = 'RandomInit(fallback)'
        else:
            # For other nets, try to infer status from local backbone loader logs
            try:
                from networks.weights_helper import get_and_clear_status_logs  # type: ignore
                _logs = get_and_clear_status_logs()
                loaded_any = any((isinstance(x, dict) and x.get('event') == 'loaded') for x in _logs)
                fallback_any = any((isinstance(x, dict) and x.get('event') in ('not_found','load_failed')) for x in _logs)
                if loaded_any:
                    pretrained_msg = 'ImageNetPretrained(Local)'
                elif args.pretrained and fallback_any:
                    # requested but local weights missing or failed -> random init fallback
                    pretrained_msg = 'RandomInit(fallback)'
                elif args.pretrained:
                    # requested but model factory doesn't plumb this flag
                    pretrained_msg = 'RequestedButNotSupported'
                else:
                    pretrained_msg = 'RandomInit'
            except Exception:
                if args.pretrained:
                    pretrained_msg = 'RequestedButNotSupported'
                else:
                    pretrained_msg = 'RandomInit'
        print(f"[Init] net={args.net} backbone={args.backbone} pretrained={pretrained_msg} device={args.device}")
    except Exception as e:
        print(f"[Init] pretrained status check failed: {e}")

    # Name and log
    ds_norm = args.dataset.lower()
    if ds_norm in ('mas', 'massachusetts'):
        suffix = 'Mas'
    elif ds_norm in ('road', 'customer'):
        suffix = 'Road'
    else:
        suffix = ds_norm.capitalize()
    # include backbone in run name for disambiguation when applicable
    bb_tag = ''
    if args.net in ('BiResUnetPlus', 'LightSegNet') and args.backbone:
        bb_tag = f"-{args.backbone}"
    run_name = args.name or f"{suffix}-{args.net}{bb_tag}"
    os.makedirs(args.save_dir, exist_ok=True)
    log_dir = os.path.join(FILE_DIR, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{run_name}.log")
    mylog = open(log_path, 'w')

    # Monitoring state (configurable)
    no_optim = 0
    max_no_optim = int(args.early_stop_patience)
    reduce_lr_patience = int(args.lr_reduce_patience)
    monitor_metric = str(args.monitor_metric)
    monitor_higher_is_better = (monitor_metric == 'dice')
    best_score = -math.inf if monitor_higher_is_better else math.inf
    iters_per_epoch = max(1, len(train_ds) // max(1, int(args.batch_size)))
    tic = time()

    # Prepare CSV log
    csv_dir = os.path.join(FILE_DIR, 'logs')
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, f"{run_name}.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', encoding='utf-8') as fcsv:
            fcsv.write('epoch,train_loss,val_loss,acc,pre,rec,iou,dice,fpr,fnr,lr,time_s\n')

    # optional resume (support full checkpoint)
    start_epoch = 1
    best_dice = -1.0
    if args.resume and os.path.isfile(args.resume):
        try:
            state = torch.load(args.resume, map_location=solver.device)
            if isinstance(state, dict) and 'model_state_dict' in state:
                solver.net.load_state_dict(state['model_state_dict'], strict=False)
                if 'optimizer_state_dict' in state:
                    solver.optimizer.load_state_dict(state['optimizer_state_dict'])
                if 'epoch' in state:
                    start_epoch = int(state['epoch']) + 1
                if 'best_dice' in state:
                    best_dice = float(state['best_dice'])
                if 'old_lr' in state:
                    solver.old_lr = float(state['old_lr'])
                print(f"[Resume] Loaded full checkpoint from {args.resume} (start_epoch={start_epoch}, best_dice={best_dice:.4f})")
            else:
                solver.load(args.resume)
                print(f"[Resume] Loaded weights from {args.resume}")
        except Exception as e:
            print(f"[Resume] Failed to load {args.resume}: {e}")

    def evaluate_epoch() -> Tuple[float, List[float]]:
        solver.net.eval()
        val_loss_sum = 0.0
        n = 0
        agg = []
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(solver.device).float()
                masks = masks.to(solver.device).float()
                logits = solver.net(imgs)
                loss = solver.loss(masks, logits)
                val_loss_sum += float(loss.item())
                n += 1
                prob = proba_from_logits(logits)
                pred = (prob >= float(args.threshold)).float().squeeze(1).cpu().numpy()
                gt = masks.squeeze(1).cpu().numpy().astype(np.uint8)
                B = pred.shape[0]
                for b in range(B):
                    acc, pre, rec, iou, dice, fpr, fnr = metrics_from_binary(pred[b], gt[b])
                    agg.append([acc, pre, rec, iou, dice, fpr, fnr])
        val_loss = val_loss_sum / max(1, n)
        metrics = np.array(agg, dtype=np.float32).mean(axis=0).tolist() if agg else [0]*7
        solver.net.train()
        return val_loss, metrics

    for epoch in range(start_epoch, int(args.epochs) + 1):
        train_epoch_loss = 0.0
        pbar = tqdm(enumerate(train_loader), total=iters_per_epoch,
                    desc=f"Epoch [{epoch}/{args.epochs}] Loss: 0.0000 LR: {solver.old_lr:.6f}")
        for it, (img, mask) in pbar:
            solver.set_input(img, mask)
            loss, _ = solver.optimize()
            train_epoch_loss += float(loss)
            avg_loss = train_epoch_loss / (it + 1)
            pbar.set_description(f"Epoch [{epoch}/{args.epochs}] Loss: {avg_loss:.6f} LR: {solver.old_lr:.6f}")
            if it + 1 >= iters_per_epoch:
                break

        avg_loss = train_epoch_loss / iters_per_epoch
        print('********', file=mylog)
        print(f'epoch: {epoch}    time: {int(time() - tic)}', file=mylog)
        print(f'train_loss: {avg_loss}', file=mylog)
        print(f'SHAPE: {img_size}', file=mylog)
        print('********')
        mylog.flush()

        # Validation (every val-interval)
        do_val = (epoch % int(args.val_interval) == 0)
        val_loss, metrics = (0.0, [0, 0, 0, 0, 0, 0, 0])
        if do_val:
            val_loss, metrics = evaluate_epoch()
            acc, pre, rec, iou, dice, fpr, fnr = metrics
            print(f"[Val] epoch={epoch} val_loss={val_loss:.6f} acc={acc:.4f} pre={pre:.4f} rec={rec:.4f} iou={iou:.4f} dice={dice:.4f}")
            print(f"val_loss: {val_loss}", file=mylog)
            print(f"val_metrics: acc {acc:.4f} pre {pre:.4f} rec {rec:.4f} iou {iou:.4f} dice {dice:.4f} fpr {fpr:.4f} fnr {fnr:.4f}", file=mylog)
            mylog.flush()

        # CSV append
        with open(csv_path, 'a', encoding='utf-8') as fcsv:
            acc, pre, rec, iou, dice, fpr, fnr = metrics
            fcsv.write(f"{epoch},{avg_loss:.6f},{val_loss:.6f},{acc:.6f},{pre:.6f},{rec:.6f},{iou:.6f},{dice:.6f},{fpr:.6f},{fnr:.6f},{solver.old_lr:.8f},{int(time()-tic)}\n")

        # Checkpointing
        # Save last weights each epoch
        last_w = os.path.join(args.save_dir, f"{run_name}_last.pth")
        solver.save(last_w)
        # Save rolling checkpoint every ckpt-interval
        if (epoch % int(args.ckpt_interval)) == 0:
            ckpt = {
                'epoch': epoch,
                'best_dice': best_dice,
                'model_state_dict': solver.net.state_dict(),
                'optimizer_state_dict': solver.optimizer.state_dict(),
                'old_lr': solver.old_lr,
                'run_name': run_name,
            }
            torch.save(ckpt, os.path.join(args.save_dir, f"{run_name}_ckpt_ep{epoch}.pth"))

        # Early stopping & LR schedule based on selected monitor metric
        current_score = None
        if monitor_metric == 'val_loss':
            current_score = (val_loss if do_val else avg_loss)
        elif monitor_metric == 'train_loss':
            current_score = avg_loss
        elif monitor_metric == 'dice':
            # Only meaningful when validation ran
            if do_val:
                current_score = float(metrics[4])  # dice
            else:
                current_score = None

        if current_score is not None:
            improved = (current_score > best_score) if monitor_higher_is_better else (current_score < best_score)
            if improved:
                best_score = current_score
                no_optim = 0
            else:
                no_optim += 1
        else:
            # When monitoring dice but no val this epoch, count as no improvement
            no_optim += 1

        # Track best dice for model selection when validation ran
        if do_val:
            if metrics[4] > best_dice:
                best_dice = metrics[4]
                best_path = os.path.join(args.save_dir, f"{run_name}_best.pth")
                solver.save(best_path)
                # also refresh resume checkpoint for easy continuation
                ckpt_resume = {
                    'epoch': epoch,
                    'best_dice': best_dice,
                    'model_state_dict': solver.net.state_dict(),
                    'optimizer_state_dict': solver.optimizer.state_dict(),
                    'old_lr': solver.old_lr,
                    'run_name': run_name,
                }
                torch.save(ckpt_resume, os.path.join(args.save_dir, f"{run_name}_resume.th"))

        if no_optim > max_no_optim:
            print(f'Early stop at epoch {epoch}', file=mylog)
            break
        if no_optim > reduce_lr_patience:
            if solver.old_lr < float(args.min_lr):
                break
            # Reload best-loss weights if exist and reduce LR by factor 5
            best_guess = os.path.join(args.save_dir, f"{run_name}_best.pth")
            if os.path.exists(best_guess):
                solver.load(best_guess)
            solver.update_lr(float(args.lr_reduce_factor), mylog=mylog, factor=True)

    print('Finish!', file=mylog)
    mylog.close()


if __name__ == '__main__':
    main()
