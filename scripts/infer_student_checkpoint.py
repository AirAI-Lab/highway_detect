"""Infer using a LightSegNet student checkpoint and save overlays + preds.

Usage example:
  python scripts/infer_student_checkpoint.py --checkpoint models/light_distill_best_ep0_miou0.4761.pth \
      --img-list data/splits/val_road.txt --mask-list data/splits/val_road_masks.txt --out-dir models/infer_student_outputs --n 5

The script:
 - loads LightSegNet
 - loads checkpoint (handles raw state_dict or {'model': state_dict})
 - runs inference on up to N images from provided list
 - saves overlay_{i:06d}.png and pred_{i:06d}.npy in out-dir
 - if masks provided, computes per-image IoU/Dice and prints averages
"""
import argparse
import os
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from common.light_models import LightSegNet


def read_list(path):
    if path is None:
        return []
    p = Path(path)
    if not p.exists():
        raise RuntimeError(f'list file not found: {path}')
    lines = [l.strip().replace('\ufeff', '') for l in p.read_text(encoding='utf8').splitlines() if l.strip()]
    return lines


def make_green_overlay(orig_bgr, prob_map, thr=0.5, soft_alpha=True, alpha_max=0.6):
    # Mirror implementation from run_infer_random_val.py to avoid large temporary arrays
    h, w = orig_bgr.shape[:2]
    pm = prob_map
    if pm.shape != (h, w):
        pm = cv2.resize(pm.astype('float32'), (w, h), interpolation=cv2.INTER_LINEAR)
    pm = np.clip(pm, 0.0, 1.0).astype('float32')
    if soft_alpha:
        out = np.empty_like(orig_bgr, dtype='uint8')
        alpha_map = (pm * alpha_max).astype('float32')
        max_chunk_h = 256
        chunk_h = min(max_chunk_h, h)
        color_row = np.zeros((1, w, 3), dtype='uint8')
        color_row[0, :, :] = (0, 255, 0)
        for y0 in range(0, h, chunk_h):
            y1 = min(h, y0 + chunk_h)
            alpha_chunk = alpha_map[y0:y1, :][..., None]
            orig_chunk = orig_bgr[y0:y1, :, :].astype('float32')
            color_chunk = np.repeat(color_row, repeats=(y1 - y0), axis=0).astype('float32')
            blended_chunk = (orig_chunk * (1.0 - alpha_chunk) + color_chunk * alpha_chunk).astype('uint8')
            out[y0:y1, :, :] = blended_chunk
        return out
    else:
        mask = (pm > thr).astype('uint8')
        color = np.zeros_like(orig_bgr, dtype='uint8')
        color[:] = (0, 255, 0)
        overlay = orig_bgr.copy()
        overlay[mask.astype(bool)] = cv2.addWeighted(orig_bgr[mask.astype(bool)].astype('float32'), 1.0 - alpha_max, color[mask.astype(bool)].astype('float32'), alpha_max, 0.0).astype('uint8')
        return overlay


def load_checkpoint_to_model(model, ckpt_path, device):
    st = torch.load(str(ckpt_path), map_location=device)
    if isinstance(st, dict) and 'model' in st:
        sd = st['model']
    else:
        sd = st
    try:
        model.load_state_dict(sd)
    except Exception:
        # try non-strict
        model.load_state_dict(sd, strict=False)


def compute_metrics(pred_mask, gt_mask):
    # pred_mask, gt_mask: 2D arrays with 0/1
    pred = (pred_mask > 0.5).astype(np.uint8)
    gt = (gt_mask > 0.5).astype(np.uint8)
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    iou = float(inter) / (float(union) + 1e-9)
    p_sum = pred.sum()
    g_sum = gt.sum()
    dice = (2.0 * inter) / (p_sum + g_sum + 1e-9)
    return iou, dice


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--img-list', required=False, help='text file with image paths (one per line). Not required when mode=video')
    p.add_argument('--mask-list', default=None)
    p.add_argument('--out-dir', default='models/infer_student_outputs')
    p.add_argument('--mode', choices=['images','video'], default='images', help='operation mode: images (default) or video')
    p.add_argument('--video', default='data/test.mp4', help='video file to read when mode=video')
    p.add_argument('--video-fps', type=float, default=1.0, help='frames per second to extract from video when mode=video')
    p.add_argument('--direct-video', action='store_true', help='read frames directly from video and infer in memory (do not save frame images)')
    p.add_argument('--save-video', default=None, help='optional path to save overlay results as a video (e.g. models/out.mp4)')
    p.add_argument('--n', type=int, default=5)
    p.add_argument('--H', type=int, default=544)
    p.add_argument('--W', type=int, default=960)
    p.add_argument('--base-ch', type=int, default=16)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    # prepare image list depending on mode
    if args.mode == 'video':
        vid_path = Path(args.video)
        if not vid_path.exists():
            raise RuntimeError(f'video file not found: {vid_path}')
        # two video workflows:
        # - default: extract frames into out_dir/video_frames (keeps current behaviour)
        # - --direct-video: do not write individual frames, infer on frames in memory and
        #   optionally write a single combined overlay video via --save-video
        if args.direct_video:
            # we'll process frames in-memory later (after model is loaded)
            imgs = []
            masks = []
            video_direct = True
            video_path = vid_path
        else:
            # extract frames into out_dir/video_frames
            vid_out = Path(args.out_dir) / 'video_frames'
            vid_out.mkdir(parents=True, exist_ok=True)
            cap = cv2.VideoCapture(str(vid_path))
            if not cap.isOpened():
                raise RuntimeError(f'failed to open video {vid_path}')
            saved = 0
            t = 0.0
            idx = 0
            while True:
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
                ret, frame = cap.read()
                if not ret:
                    break
                idx += 1
                outp = vid_out / f'frame_{idx:06d}.png'
                cv2.imwrite(str(outp), frame)
                saved += 1
                t += 1.0 / args.video_fps
            cap.release()
            if saved == 0:
                raise RuntimeError('no frames extracted from video')
            print(f'Extracted {saved} frames from {vid_path} into {vid_out}')
            imgs = [str(p) for p in sorted(vid_out.glob('*.png'))]
            masks = []
            video_direct = False
    else:
        if not args.img_list:
            raise RuntimeError('--img-list required in images mode')
        imgs = read_list(args.img_list)
        masks = read_list(args.mask_list) if args.mask_list else []
        if len(imgs) == 0:
            raise RuntimeError('no images to run')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device', device)

    model = LightSegNet(base_ch=args.base_ch, input_channels=3)
    try:
        load_checkpoint_to_model(model, args.checkpoint, device)
        print('Loaded checkpoint', args.checkpoint)
    except Exception as e:
        print('Failed to load checkpoint:', e)
        raise
    model.to(device)
    model.eval()

    n = min(args.n, len(imgs))
    ious = []
    dices = []
    # if using direct video mode, process frames straight from the video capture
    if args.mode == 'video' and args.direct_video:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f'failed to open video {video_path}')
        video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = None
        if args.save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(args.save_video), fourcc, float(args.video_fps), (video_w, video_h))
            if not writer.isOpened():
                cap.release()
                raise RuntimeError(f'failed to open video writer {args.save_video}')
        saved = 0
        t = 0.0
        idx = 0
        while True:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ret, frame = cap.read()
            if not ret:
                break
            idx += 1
            orig = frame.copy()
            # convert to RGB, resize to W,H
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_rs = cv2.resize(img_rgb, (args.W, args.H), interpolation=cv2.INTER_LINEAR)
            inp = img_rs.astype('float32') / 255.0
            inp_t = torch.from_numpy(inp.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
            with torch.no_grad():
                logits = model(inp_t)
                probs = torch.sigmoid(logits).squeeze().cpu().numpy()
                if probs.ndim == 3:
                    prob_map = probs[0]
                else:
                    prob_map = probs
            name = f'frame_{idx:06d}'
            out_npy = Path(args.out_dir) / f'pred_{name}.npy'
            np.save(str(out_npy), prob_map.astype('float32'))
            overlay = make_green_overlay(orig, prob_map, thr=0.5, soft_alpha=True, alpha_max=0.6)
            out_png = Path(args.out_dir) / f'overlay_{name}.png'
            cv2.imwrite(str(out_png), overlay)
            pm_resized = cv2.resize(prob_map, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_LINEAR)
            mask_bin = (pm_resized > 0.5).astype('uint8') * 255
            mask_outp = Path(args.out_dir) / f'mask_{name}.png'
            cv2.imwrite(str(mask_outp), mask_bin)
            if writer is not None:
                writer.write(overlay)
            saved += 1
            t += 1.0 / args.video_fps
        cap.release()
        if writer is not None:
            writer.release()
        if saved == 0:
            raise RuntimeError('no frames processed from video')
        print(f'Processed {saved} frames from {video_path} and saved outputs to {args.out_dir}')
        print('Saved video to', args.save_video if args.save_video else 'None')
        return

    for i in range(n):
        img_path = imgs[i]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            print('skip missing image', img_path)
            continue
        orig = img.copy()
        # convert to RGB, resize to W,H
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rs = cv2.resize(img_rgb, (args.W, args.H), interpolation=cv2.INTER_LINEAR)
        inp = img_rs.astype('float32') / 255.0
        inp_t = torch.from_numpy(inp.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
        with torch.no_grad():
            logits = model(inp_t)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            if probs.ndim == 3:
                prob_map = probs[0]
            else:
                prob_map = probs
        # save prob (pred_{idx:06d}.npy) and overlays/masks using same naming as run_infer_random_val
        name = Path(img_path).stem
        # save prob using original filename stem
        out_npy = Path(args.out_dir) / f'pred_{name}.npy'
        np.save(str(out_npy), prob_map.astype('float32'))

        # create overlay using robust blending and save as overlay_{name}.png
        overlay = make_green_overlay(orig, prob_map, thr=0.5, soft_alpha=True, alpha_max=0.6)
        out_png = Path(args.out_dir) / f'overlay_{name}.png'
        cv2.imwrite(str(out_png), overlay)

        # save binary mask as mask_{name}.png (resized to orig resolution)
        pm_resized = cv2.resize(prob_map, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_LINEAR)
        mask_bin = (pm_resized > 0.5).astype('uint8') * 255
        mask_outp = Path(args.out_dir) / f'mask_{name}.png'
        cv2.imwrite(str(mask_outp), mask_bin)

        # if GT mask available, compute metrics
        if masks and i < len(masks):
            mpath = masks[i]
            m = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
            if m is None:
                print('warning: missing mask', mpath)
            else:
                m_rs = cv2.resize(m, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_NEAREST)
                m_bin = (m_rs > 127).astype(np.uint8)
                iou, dice = compute_metrics(pm_resized, m_bin)
                ious.append(iou)
                dices.append(dice)
                print(f'{name}: min={float(prob_map.min()):.4f} max={float(prob_map.max()):.4f} mean={float(prob_map.mean()):.4f} iou={iou:.4f} dice={dice:.4f}')
        else:
            print(f'{name}: min={float(prob_map.min()):.4f} max={float(prob_map.max()):.4f} mean={float(prob_map.mean()):.4f}')

    if ious:
        print('Avg IoU', float(np.mean(ious)), 'Avg Dice', float(np.mean(dices)))
    print('Saved outputs to', args.out_dir)


if __name__ == '__main__':
    main()
