import os
import sys
import argparse
import subprocess
from datetime import datetime
from typing import Optional

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(REPO_ROOT)  # up from scripts/


def build_run_name(dataset: str, net: str, backbone: Optional[str]) -> str:
    ds_norm = (dataset or '').lower()
    if ds_norm in ('mas', 'massachusetts'):
        suffix = 'Mas'
    elif ds_norm in ('road', 'customer'):
        suffix = 'Road'
    else:
        suffix = ds_norm.capitalize() if ds_norm else 'Run'
    bb_tag = f"-{backbone}" if net in ('BiResUnetPlus', 'LightSegNet') and backbone else ''
    return f"{suffix}-{net}{bb_tag}"


def parse_args():
    p = argparse.ArgumentParser(description='GPU server training launcher (wrapper around experiments/train_experiment1.py)')
    p.add_argument('--bires-eem', action='store_true', help='Enable EEM edge enhancement in BiResUnetPlus encoder (recommended for mas dataset)')
    p.add_argument('--dataset', required=True, help='mas|Massachusetts or road|Customer')
    p.add_argument('--root', type=str, default=None, help='Massachusetts root (for dataset=mas)')
    p.add_argument('--net', required=True, help='Model name (e.g., BiReNet34, DeepLabv3_plus, SSCNet, BiResUnetPlus)')
    p.add_argument('--backbone', type=str, default=None, help='Optional backbone for BiResUnetPlus/LightSegNet')
    p.add_argument('--pretrained', action='store_true', help='Enable ImageNet pretrained if supported (e.g., BiResUnetPlus)')
    p.add_argument('--epochs', type=int, default=120)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-3)
    # If not provided, we'll set a dataset-specific default in main():
    #   - mas/Massachusetts: 1024x1024 (square tiles, thin structures)
    #   - road/customer:     960x544   (16:9ish, both divisible by 32)
    p.add_argument('--img-size', type=int, nargs=2, default=None,
                   help='Input size as W H. If omitted, defaults to 1024x1024 for mas and 960x544 for road')
    p.add_argument('--val-interval', type=int, default=1)
    p.add_argument('--ckpt-interval', type=int, default=5)
    # Allow flexible device syntax: 'auto' | 'cpu' | 'cuda' | 'cuda:IDX' | numeric IDX
    p.add_argument('--device', type=str, default='cuda', help="Compute device: auto|cpu|cuda or cuda:IDX / numeric IDX")
    p.add_argument('--gpus', type=str, default=None, help='Comma-separated GPU ids, e.g., 0 or 0,1')
    p.add_argument('--limit-samples', type=int, default=None, help='Optional quick debug: subsample training pairs')
    p.add_argument('--resume', type=str, default=None, help='Explicit resume checkpoint path (.pth/.th)')
    p.add_argument('--resume-auto', action='store_true', help='Auto-detect experiments/weights/<RunName>_resume.th to resume')
    p.add_argument('--name', type=str, default=None, help='Optional override of run name')
    # Reproducibility
    p.add_argument('--global-seed', type=int, default=None, help='Global random seed for reproducibility (passed to experiments/train_experiment1.py)')
    # Optional BiResUnetPlus decoder options (wired via environment variables)
    p.add_argument('--bires-decoder-se', action='store_true', help='Enable SE attention in BiResUnetPlus decoder blocks')
    p.add_argument('--bires-bilinear-up', action='store_true', help='Use bilinear+conv upsampling in BiResUnetPlus decoder (instead of deconv)')
    p.add_argument('--bires-lite-aspp', action='store_true', help='Enable Lite-ASPP context in BiResUnetPlus bottleneck')
    p.add_argument('--bires-full-aspp', action='store_true', help='Use Full-ASPP in BiResUnetPlus (overrides lite-aspp)')
    p.add_argument('--bires-edge-aux', action='store_true', help='Enable edge auxiliary head in BiResUnetPlus decoder')
    p.add_argument('--edge-aux-weight', type=float, default=None, help='Loss weight for edge auxiliary head (default 0.25 via env)')
    # New architecture toggles (passed via env)
    p.add_argument('--bires-decoder-width-mult', type=float, default=None, help='Scale decoder channels by this factor (0.25~1.0)')
    p.add_argument('--bires-decoder-dw', action='store_true', help='Enable depthwise separable convs in decoder residual blocks')
    p.add_argument('--bires-strip-pool', action='store_true', help='Enable strip pooling branches inside Lite-ASPP (requires --bires-lite-aspp)')
    # Early-stop / LR-schedule / monitor controls (pass-through)
    p.add_argument('--early-stop-patience', type=int, default=None)
    p.add_argument('--lr-reduce-patience', type=int, default=None)
    p.add_argument('--lr-reduce-factor', type=float, default=None)
    p.add_argument('--min-lr', type=float, default=None)
    p.add_argument('--monitor-metric', type=str, default=None, choices=['val_loss','train_loss','dice'])
    # Pass-through loss selection to train_experiment1.py
    p.add_argument('--loss', type=str, default=None,
                   choices=['dice_bce', 'focal_tversky', 'binary_tversky', 'binary_focal_tversky', 'focal_tversky_edge', 'joint_seg_edge'],
                   help='Loss function to use (passed to experiments/train_experiment1.py)')
    # ALT / HSV 可微构造参数（透传到 experiments/train_experiment1.py）
    p.add_argument('--alt-build-mode', type=str, default='hsvgrad', choices=['rgbgrad','hsvgrad','hsv_cv'],
                   help='ALT 构造模式: rgbgrad|hsvgrad(可微)|hsv_cv(OpenCV旧版)')
    p.add_argument('--alt-hsv-stats', type=str, default=None,
                   help='HSV 自适应阈值统计文件(.npz/.json)，用于 hsvgrad 或 hsv_cv(adaptive)')
    p.add_argument('--alt-v-window-mode', type=str, default='mad', choices=['mad','legacy'],
                   help='Value 通道窗口策略: mad 使用中位绝对偏差窗口，legacy 使用固定 v_low/v_high')
    p.add_argument('--alt-smooth-temp', type=float, default=4.0,
                   help='平滑阈值温度 (sigmoid 温度)，越大掩码过渡越平缓 (默认4.0)')
    p.add_argument('--alt-channel-weights', type=str, default=None,
                   help='8 通道权重 (逗号分隔)，用于消融或下调某些通道影响，例如 1,1,1,1,0.5,0.5,1,1')
    # EEM configuration passthrough (levels as comma-separated list, rgb/alt toggles, reduction)
    p.add_argument('--bires-eem-levels', type=str, default=None,
                   help='Comma-separated EEM levels to enable (choose from 1,2,3,4). Example: "1,2,3"')
    p.add_argument('--bires-eem-rgb', type=str, default=None, choices=['true','false'],
                   help='Whether to apply EEM to RGB encoder (true/false). If omitted, default true')
    p.add_argument('--bires-eem-alt', type=str, default=None, choices=['true','false'],
                   help='Whether to apply EEM to ALT encoder (true/false). If omitted, default true')
    p.add_argument('--bires-eem-reduction', type=int, default=None,
                   help='Reduction (squeeze) factor passed to EEM modules (default 2)')
    return p.parse_args()


def main():
    args = parse_args()

    train_py = os.path.join(REPO_ROOT, 'experiments', 'train_experiment1.py')
    if not os.path.isfile(train_py):
        raise SystemExit(f"Missing training script: {train_py}")

    run_name = args.name or build_run_name(args.dataset, args.net, args.backbone)
    # 自动对 mas 数据集启用 EEM
    use_eem = args.bires_eem or (args.dataset.lower() in ('mas', 'massachusetts'))

    # Resolve device selection semantics: allow --device cpu | cuda | cuda:IDX | numeric IDX
    env = os.environ.copy()
    dev_arg = str(args.device).lower() if args.device is not None else 'auto'
    # If explicit GPU index provided via --device, prefer it over --gpus
    explicit_gpu_idx: Optional[str] = None
    if dev_arg.startswith('cuda:'):
        explicit_gpu_idx = dev_arg.split(':', 1)[1]
        args.device = 'cuda'
    else:
        # numeric device implies single GPU selection
        try:
            if dev_arg.isdigit():
                explicit_gpu_idx = dev_arg
                args.device = 'cuda'
        except Exception:
            pass

    # Resolve default img-size per dataset if not explicitly provided
    if args.img_size is None:
        ds_norm = (args.dataset or '').lower()
        if ds_norm in ('mas', 'massachusetts'):
            args.img_size = [1024, 1024]
            print('[train_gpu] Using default img-size for Massachusetts: 1024 1024')
        elif ds_norm in ('road', 'customer'):
            args.img_size = [960, 544]
            print('[train_gpu] Using default img-size for Road: 960 544')
        else:
            args.img_size = [960, 544]
            print('[train_gpu] Using fallback img-size: 960 544')

    cmd = [sys.executable, '-u', train_py,
           '--dataset', args.dataset,
           '--net', args.net,
           '--epochs', str(int(args.epochs)),
           '--batch-size', str(int(args.batch_size)),
           '--lr', str(float(args.lr)),
           '--img-size', str(int(args.img_size[0])), str(int(args.img_size[1])),
           '--val-interval', str(int(args.val_interval)),
           '--ckpt-interval', str(int(args.ckpt_interval)),
           '--device', args.device,
           '--name', run_name,
           ]
    if use_eem:
        cmd += ['--bires-eem']
    if args.global_seed is not None:
        cmd += ['--global-seed', str(int(args.global_seed))]
    if args.loss is not None:
        cmd += ['--loss', args.loss]
    if args.dataset.lower() in ('mas', 'massachusetts'):
        if not args.root:
            default_root = os.path.join(REPO_ROOT, 'data', 'Massachusetts')
            if os.path.isdir(default_root):
                args.root = default_root
                print(f"[train_gpu] Using default Massachusetts root: {args.root}")
            else:
                raise SystemExit('--root is required for dataset=mas')
        cmd += ['--root', args.root]
    if args.backbone:
        cmd += ['--backbone', args.backbone]
    if args.pretrained:
        cmd += ['--pretrained']
    if args.limit_samples is not None:
        cmd += ['--limit-samples', str(int(args.limit_samples))]
    if args.resume:
        cmd += ['--resume', args.resume]
    elif args.resume_auto:
        weights_dir = os.path.join(REPO_ROOT, 'experiments', 'weights')
        resume_guess = os.path.join(weights_dir, f'{run_name}_resume.th')
        if os.path.isfile(resume_guess):
            cmd += ['--resume', resume_guess]
            print(f"[train_gpu] Auto-resume from {resume_guess}")
        else:
            print(f"[train_gpu] No auto-resume file found: {resume_guess}")

    # Set CUDA_VISIBLE_DEVICES based on precedence: explicit index from --device > --gpus > leave as-is
    if args.device != 'cpu':
        if explicit_gpu_idx is not None:
            env['CUDA_VISIBLE_DEVICES'] = str(explicit_gpu_idx)
        elif args.gpus:
            env['CUDA_VISIBLE_DEVICES'] = args.gpus

        # Optional pass-through of training policy args
        if args.early_stop_patience is not None:
            cmd += ['--early-stop-patience', str(int(args.early_stop_patience))]
        if args.lr_reduce_patience is not None:
            cmd += ['--lr-reduce-patience', str(int(args.lr_reduce_patience))]
        if args.lr_reduce_factor is not None:
            cmd += ['--lr-reduce-factor', str(float(args.lr_reduce_factor))]
        if args.min_lr is not None:
            cmd += ['--min-lr', str(float(args.min_lr))]
        if args.monitor_metric is not None:
            cmd += ['--monitor-metric', str(args.monitor_metric)]

        # ALT / HSV 相关参数透传（独立于 monitor_metric）
        if args.alt_build_mode:
            cmd += ['--alt-build-mode', args.alt_build_mode]
        if args.alt_hsv_stats:
            cmd += ['--alt-hsv-stats', args.alt_hsv_stats]
        if args.alt_v_window_mode:
            cmd += ['--alt-v-window-mode', args.alt_v_window_mode]
        if args.alt_smooth_temp is not None:
            cmd += ['--alt-smooth-temp', str(float(args.alt_smooth_temp))]
        if args.alt_channel_weights:
            cmd += ['--alt-channel-weights', args.alt_channel_weights]
        # passthrough EEM settings
        if args.bires_eem_levels:
            cmd += ['--bires-eem-levels', args.bires_eem_levels]
        if args.bires_eem_rgb is not None:
            cmd += ['--bires-eem-rgb', args.bires_eem_rgb]
        if args.bires_eem_alt is not None:
            cmd += ['--bires-eem-alt', args.bires_eem_alt]
        if args.bires_eem_reduction is not None:
            cmd += ['--bires-eem-reduction', str(int(args.bires_eem_reduction))]

        # ASPP selection passthrough: prefer explicit full-aspp flag
        if args.bires_full_aspp:
            cmd += ['--full-aspp']

    # Wire BiResUnetPlus decoder options via env vars (adapter will read these)
    if args.bires_decoder_se:
        env['BIRES_DECODER_SE'] = '1'
    if args.bires_bilinear_up:
        env['BIRES_BILINEAR_UP'] = '1'
    # prefer explicit full-aspp flag to unset lite mode
    if args.bires_full_aspp:
        env['BIRES_LITE_ASPP'] = '0'
    elif args.bires_lite_aspp:
        env['BIRES_LITE_ASPP'] = '1'
    if args.bires_edge_aux:
        env['BIRES_EDGE_AUX'] = '1'
    if args.edge_aux_weight is not None:
        env['EDGE_AUX_WEIGHT'] = str(float(args.edge_aux_weight))
    if args.bires_decoder_width_mult is not None:
        env['BIRES_DECODER_WIDTH_MULT'] = str(float(args.bires_decoder_width_mult))
    if args.bires_decoder_dw:
        env['BIRES_DECODER_DW'] = '1'
    if args.bires_strip_pool:
        env['BIRES_STRIP_POOL'] = '1'

    # Log resolved device mapping for clarity
    if args.device != 'cpu':
        cvd = env.get('CUDA_VISIBLE_DEVICES', '')
        if cvd:
            print(f"[train_gpu] CUDA_VISIBLE_DEVICES={cvd}")
    print('[train_gpu] Launching:', ' '.join(cmd))
    ret = subprocess.run(cmd, env=env)
    if ret.returncode != 0:
        raise SystemExit(ret.returncode)

    # Show artifact hints
    weights_dir = os.path.join(REPO_ROOT, 'experiments', 'weights')
    logs_dir = os.path.join(REPO_ROOT, 'experiments', 'logs')
    print('[train_gpu] Done.')
    print(f"[train_gpu] Weights: {weights_dir}\n  - {run_name}_best.pth\n  - {run_name}_last.pth\n  - {run_name}_ckpt_ep*.pth\n  - {run_name}_resume.th")
    print(f"[train_gpu] Logs: {logs_dir}\n  - {run_name}.log\n  - {run_name}.csv")


if __name__ == '__main__':
    main()
