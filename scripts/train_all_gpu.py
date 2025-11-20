import argparse
import os
import sys
import subprocess
import csv
from typing import List, Dict, Tuple, Optional
import math
import signal
import time
import platform
from collections import deque

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(FILE_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


BACKBONE_AWARE = {"BiResUnetPlus", "LightSegNet"}
# Models that support local-only ImageNet pretrained when --regime imagenet is used
SUPPORTS_LOCAL_PRETRAINED = {
    'BiResUnetPlus',
    'RCFSNet',
    'DinkNet34',
    'NLinkNet34',
    'DeepLabv3_plus',
    'SSCNet',
    'LinkNet34',
    'DBRANet',
    'CARNet',
    'TransRoadNet',
}

DEFAULT_MODELS = [
    # Full list as requested
    'BiReNet34',
    'BiResUnetPlus',  # Adapter
    'RCFSNet',
    'DinkNet34',
    'NLinkNet34',
    'DeepLabv3_plus',
    'DBRANet',
    'MACUNet',
    'LinkNet34',
    'TransRoadNet',  # swin_s
    'U_Net',
    'CARNet',        # DAM_Net_5
    'MSMDFFNet',     # MSMDFF_Net_v3_plus
    'SSCNet',
    'LightSegNet',   # from common.light_models
]


def build_run_name(dataset: str, net: str, backbone: Optional[str]) -> str:
    ds_norm = (dataset or '').lower()
    if ds_norm in ('mas', 'massachusetts'):
        suffix = 'Mas'
    elif ds_norm in ('road', 'customer'):
        suffix = 'Road'
    else:
        suffix = ds_norm.capitalize() if ds_norm else 'Run'
    bb_tag = f"-{backbone}" if net in BACKBONE_AWARE and backbone else ''
    return f"{suffix}-{net}{bb_tag}"


def _digits_from_backbone(bb: Optional[str]) -> str:
    if not bb:
        return ''
    bb = bb.lower()
    if bb.startswith('resnet'):
        # extract trailing digits e.g., resnet34 -> 34
        s = ''.join(ch for ch in bb if ch.isdigit())
        return s
    return bb


def build_run_name_formal(dataset: str, net: str, backbone: Optional[str]) -> str:
    """Formal naming: <Mas|Road>-<Model><34?>-formal-patience

    - For backbone-aware models with resnet backbones, append digits to model name (e.g., BiResUnetPlus34)
    - For others, keep model as-is
    """
    ds_norm = (dataset or '').lower()
    if ds_norm in ('mas', 'massachusetts'):
        suffix = 'Mas'
    elif ds_norm in ('road', 'customer'):
        suffix = 'Road'
    else:
        suffix = ds_norm.capitalize() if ds_norm else 'Run'
    model_token = net
    if net in BACKBONE_AWARE and backbone:
        digit = _digits_from_backbone(backbone)
        if digit:
            model_token = f"{net}{digit}"
    return f"{suffix}-{model_token}-formal-patience"

# --- Global image size resolver (applies safety constraints across both formal & legacy paths) ---
def resolve_img_size(ds: str, net_name: str, iw: int, ih: int) -> Tuple[int,int]:
    """Resolve (iw, ih) applying dataset defaults and model-specific minimums.

    Rules:
      - Dataset defaults (when caller passes placeholder sizes): MAS=1024x1024, ROAD=960x544
      - RCFSNet: enforce >=1024 in both dims (large CDAM kernels)
      - TransRoadNet: enforce >=1024 and square (Swin + large strip/ASPP kernels expect square feature maps
        and sufficient spatial size to avoid kernel>feature or window assertion)
      - CARNet: inherits Swin ASPP path; same >=1024 square safety to satisfy L==H*W flatten & expand semantics
    If caller supplies larger sizes, only square enforcement (TransRoadNet/CARNet) will reshape using max(iw,ih).
    """
    ds_l = ds.lower()
    # If user provided generic smaller size (e.g., 960x544 for ROAD) keep unless model constraints escalate.
    if (iw, ih) == (0, 0):  # allow sentinel if future code uses 0,0
        if ds_l == 'mas':
            iw, ih = 1024, 1024
        else:
            iw, ih = 960, 544
    # Apply dataset default if training code passed original defaults
    if ds_l == 'mas' and (iw < 1024 or ih < 1024):
        # MAS always standardize to 1024 square for consistency
        iw, ih = 1024, 1024
    elif ds_l == 'road' and (iw, ih) == (1024, 1024):
        # Already square & large, fine
        pass
    # Model constraints
    net_l = str(net_name).lower()
    if net_l == 'rcfsnet':
        iw = max(iw, 1024); ih = max(ih, 1024)
    if net_l in ('transroadnet', 'carnet'):
        iw = max(iw, 1024); ih = max(ih, 1024)
        m = max(iw, ih); iw, ih = m, m
    # Stride divisibility (round up) and generic minimums
    def _ceil_to(x: int, base: int) -> int:
        return ((x + base - 1) // base) * base
    # Default stride requirement
    stride = 32
    # Models with typical 16x downsample pipelines (classic U-Net family)
    if net_l in ('u_net', 'macunet'):
        stride = 16
    # Generic min size for 32-stride families
    if stride == 32:
        iw = max(iw, 256); ih = max(ih, 256)
    else:
        iw = max(iw, 224); ih = max(ih, 224)
    iw = _ceil_to(iw, stride); ih = _ceil_to(ih, stride)
    return iw, ih


def run_subprocess(cmd: List[str], env=None, cwd=None, capture: bool = False) -> Tuple[int, str]:
    if capture:
        p = subprocess.run(cmd, env=env, cwd=cwd or REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        return p.returncode, p.stdout
    p = subprocess.run(cmd, env=env, cwd=cwd or REPO_ROOT)
    return p.returncode, ''


def parse_eval_output(out: str) -> Dict[str, str]:
    metrics: Dict[str, str] = {}
    avg_ms = ''
    fps = ''
    for line in out.splitlines():
        s = line.strip()
        if s.startswith('ACC PRE REC IOU DICE FPR FNR ='):
            parts = s.split('=')[-1].strip().split()
            if len(parts) >= 7:
                keys = ['ACC','PRE','REC','IOU','DICE','FPR','FNR']
                for k, v in zip(keys, parts[:7]):
                    metrics[k] = v
        if s.startswith('Speed:') and 'avg_ms_per_image' in s:
            try:
                segs = s.split(',')
                avg_part = segs[0]
                fps_part = segs[1]
                avg_ms = avg_part.split('=')[-1].replace('ms','').strip()
                fps = fps_part.split('=')[-1].strip()
            except Exception:
                pass
    if avg_ms:
        metrics['avg_ms_per_image'] = avg_ms
    if fps:
        metrics['fps'] = fps
    return metrics


def _safe_float(x: str) -> Optional[float]:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
        return None
    except Exception:
        return None


def monitor_training_log(run_name: str) -> Dict[str, str]:
    """Read experiments/logs/<run_name>.csv and produce a compact health summary.

    Returns keys:
      - train_log: OK | MISSING | NAN_INF | LOSS_EXPLODE | SHORT | NO_LEARNING
      - epochs_logged, last_epoch
      - train_last_loss, val_last_loss, train_last_dice, train_best_dice, last_lr
      - weights_best, weights_last (Y/N)
    """
    logs_dir = os.path.join(REPO_ROOT, 'experiments', 'logs')
    weights_dir = os.path.join(REPO_ROOT, 'experiments', 'weights')
    csv_path = os.path.join(logs_dir, f"{run_name}.csv")
    res: Dict[str, str] = {
        'train_log': 'MISSING',
        'epochs_logged': '0',
        'last_epoch': '',
        'train_last_loss': '',
        'val_last_loss': '',
        'train_last_dice': '',
        'train_best_dice': '',
        'last_lr': '',
        'weights_best': 'N',
        'weights_last': 'N',
    }

    # weights existence
    if os.path.exists(os.path.join(weights_dir, f"{run_name}_best.pth")):
        res['weights_best'] = 'Y'
    if os.path.exists(os.path.join(weights_dir, f"{run_name}_last.pth")):
        res['weights_last'] = 'Y'

    if not os.path.isfile(csv_path):
        return res

    try:
        rows: List[Dict[str, str]] = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)
        n = len(rows)
        res['epochs_logged'] = str(n)
        if n == 0:
            res['train_log'] = 'SHORT'
            return res
        last = rows[-1]
        # parse last values
        ep = last.get('epoch')
        res['last_epoch'] = ep or ''
        tr = _safe_float(last.get('train_loss', ''))
        va = _safe_float(last.get('val_loss', ''))
        di = _safe_float(last.get('dice', ''))
        lr = _safe_float(last.get('lr', ''))
        if tr is not None:
            res['train_last_loss'] = f"{tr:.6f}"
        if va is not None:
            res['val_last_loss'] = f"{va:.6f}"
        if di is not None:
            res['train_last_dice'] = f"{di:.6f}"
        if lr is not None:
            res['last_lr'] = f"{lr:.8f}"

        # compute best dice & epoch order
        best_d = None
        nan_inf = False
        epoch_sequence_ok = True
        prev_epoch = None
        for r in rows:
            d = _safe_float(r.get('dice', ''))
            tl = _safe_float(r.get('train_loss', ''))
            vl = _safe_float(r.get('val_loss', ''))
            # epoch ordering check (integer monotonic increasing)
            ep_raw = r.get('epoch')
            if ep_raw is not None and ep_raw != '':
                try:
                    ep_val = int(str(ep_raw).strip())
                    if prev_epoch is not None and ep_val <= prev_epoch:
                        epoch_sequence_ok = False
                    prev_epoch = ep_val
                except Exception:
                    epoch_sequence_ok = False
            if any(v is None for v in (d, tl, vl)):
                # treat any missing/NaN/Inf as anomaly
                nan_inf = True
            if d is not None:
                best_d = d if best_d is None else max(best_d, d)
        if best_d is not None:
            res['train_best_dice'] = f"{best_d:.6f}"

        res['epoch_order_ok'] = 'Y' if epoch_sequence_ok else 'N'

        # health classification
        status = 'OK'
        if n < 2:
            status = 'SHORT'
        if nan_inf:
            status = 'NAN_INF'
        # crude loss explode check on last values
        if tr is not None and tr > 1e3:
            status = 'LOSS_EXPLODE'
        if va is not None and va > 1e3:
            status = 'LOSS_EXPLODE'
        # no learning: very low dice after multiple epochs
        if best_d is not None and best_d < 0.01 and n >= 5:
            status = 'NO_LEARNING'
        if res.get('epoch_order_ok','Y') == 'N':
            status = 'EPOCH_ORDER'
        res['train_log'] = status
    except Exception:
        res['train_log'] = 'PARSE_ERROR'
    return res


def main():
    ap = argparse.ArgumentParser(description='Train & evaluate all models on GPU server with resume/logging/checkpoints.')
    ap.add_argument('--datasets', default='mas,road', help='Comma-separated: mas,road')
    ap.add_argument('--models', default=','.join(DEFAULT_MODELS))
    ap.add_argument('--backbones', default='resnet18,resnet34', help='For backbone-aware models (BiResUnetPlus, LightSegNet)')
    ap.add_argument('--regime', default='scratch', choices=['scratch','imagenet','both'], help='Training regime: scratch / imagenet / both')
    ap.add_argument('--epochs', type=int, default=120)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--img-size', type=int, nargs=2, default=[1024, 1024])
    ap.add_argument('--val-interval', type=int, default=1)
    ap.add_argument('--ckpt-interval', type=int, default=5)
    ap.add_argument('--gpus', type=str, default=None)
    # Allow flexible device syntax: 'auto' | 'cpu' | 'cuda' | 'cuda:IDX' | numeric IDX
    ap.add_argument('--device', default='cuda', help="Compute device: auto|cpu|cuda or cuda:IDX / numeric IDX")
    # Early-stop / LR-schedule / monitor controls (optional pass-through to training)
    ap.add_argument('--early-stop-patience', type=int, default=None, help='Pass-through: stop if no improvement for N checks')
    ap.add_argument('--lr-reduce-patience', type=int, default=None, help='Pass-through: reduce LR if no improvement for N checks')
    ap.add_argument('--lr-reduce-factor', type=float, default=None, help='Pass-through: LR reduction factor (lr/=factor)')
    ap.add_argument('--min-lr', type=float, default=None, help='Pass-through: minimum learning rate before stopping')
    ap.add_argument('--monitor-metric', type=str, default=None, choices=['val_loss','train_loss','dice'], help='Pass-through: monitor metric for early-stop/LR schedule')
    # Loss selection pass-through
    ap.add_argument('--loss', type=str, default=None,
                    choices=['dice_bce', 'focal_tversky', 'binary_tversky', 'binary_focal_tversky'],
                    help='Loss function passed to training script')
    # BiResUnetPlus architecture toggles (forwarded to training via CLI and to both train/eval via env)
    ap.add_argument('--bires-decoder-se', action='store_true')
    ap.add_argument('--bires-bilinear-up', action='store_true')
    ap.add_argument('--bires-lite-aspp', action='store_true')
    ap.add_argument('--bires-full-aspp', action='store_true', help='Use Full-ASPP in BiResUnetPlus (overrides lite-aspp)')
    ap.add_argument('--bires-strip-pool', action='store_true')
    ap.add_argument('--bires-decoder-dw', action='store_true')
    ap.add_argument('--bires-decoder-width-mult', type=float, default=None)
    ap.add_argument('--bires-edge-aux', action='store_true')
    ap.add_argument('--edge-aux-weight', type=float, default=None)
    ap.add_argument('--mas-root', default=os.path.join(REPO_ROOT, 'data', 'Massachusetts'))
    ap.add_argument('--resume-auto', action='store_true', help='Auto-resume from experiments/weights/<RunName>_resume.th when present')
    # ALT / HSV 可微构造参数（透传到 train_gpu.py 和 eval_gpu.py）
    ap.add_argument('--alt-build-mode', type=str, default='hsvgrad', choices=['rgbgrad','hsvgrad','hsv_cv'],
                    help='ALT 构造模式: rgbgrad|hsvgrad(可微)|hsv_cv(OpenCV旧版)')
    ap.add_argument('--alt-hsv-stats', type=str, default=None,
                    help='HSV 自适应阈值统计文件(.npz/.json)，用于 hsvgrad 或 hsv_cv(adaptive)')
    ap.add_argument('--alt-v-window-mode', type=str, default='mad', choices=['mad','legacy'],
                    help='Value 通道窗口策略: mad 使用中位绝对偏差窗口，legacy 使用固定 v_low/v_high')
    ap.add_argument('--alt-smooth-temp', type=float, default=4.0,
                    help='平滑阈值温度 (sigmoid 温度)，越大掩码过渡越平缓 (默认4.0)')
    ap.add_argument('--alt-channel-weights', type=str, default=None,
                    help='8 通道权重 (逗号分隔)，例如 1,1,1,1,0.5,0.5,1,1 降低方向或 H 通道影响')
    ap.add_argument('--save-report', default=os.path.join(REPO_ROOT, 'runs', 'train_eval_all_report.csv'))
    ap.add_argument('--dry-run', action='store_true')
    # Formal-patience automation across all models/datasets with 4-GPU scheduling
    ap.add_argument('--formal-patience-run', action='store_true', help='Run DEFAULT_MODELS across mas/road with fixed hyperparams and 4-GPU scheduler')
    ap.add_argument('--gpu-slots', default='0,1,2,3', help='GPU indices to use for scheduling (comma-separated)')
    ap.add_argument('--sleep-after-finish', type=int, default=60, help='Seconds to sleep after a job finishes before starting next')
    # Reproducibility
    ap.add_argument('--global-seed', type=int, default=1337, help='Global random seed for reproducibility (propagated to all train jobs)')
    # Image-size override for formal mode: if provided, applies to all jobs; otherwise use dataset-specific defaults
    ap.add_argument('--force-img-size', type=int, nargs=2, default=None,
                    help='Override image size (W H) for ALL jobs in formal mode. If omitted, MAS=1024x1024, ROAD=960x544')
    # Process management utilities
    ap.add_argument('--kill-all', action='store_true', help='Kill all running single-model training processes recorded in runs/pids/*.pid')
    ap.add_argument('--kill-stale', action='store_true', help='Remove stale pid files whose processes are no longer running')
    ap.add_argument('--pids-dir', default=os.path.join(REPO_ROOT, 'runs', 'pids'), help='Directory storing per-run pid files')
    args = ap.parse_args()
    print(f"[DEBUG] formal_patience_run={getattr(args,'formal_patience_run',None)} dry_run={args.dry_run}")

    # --- Process utilities: kill/list ---
    def _pid_alive(pid: int) -> bool:
        try:
            if platform.system() == 'Windows':
                # On Windows, os.kill(pid, 0) is available on Python 3.9+
                os.kill(pid, 0)
                return True
            else:
                # POSIX: signal 0 tests existence
                os.kill(pid, 0)
                return True
        except OSError:
            return False

    def _kill_pid(pid: int) -> str:
        try:
            if platform.system() == 'Windows':
                # Try graceful first
                try:
                    os.kill(pid, signal.SIGTERM)
                except Exception:
                    pass
                time.sleep(0.5)
                if _pid_alive(pid):
                    # Force kill including children
                    subprocess.run(['taskkill', '/PID', str(pid), '/F', '/T'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return 'KILLED'
            else:
                os.kill(pid, signal.SIGTERM)
                for _ in range(10):
                    if not _pid_alive(pid):
                        return 'TERMINATED'
                    time.sleep(0.3)
                os.kill(pid, signal.SIGKILL)
                return 'KILLED'
        except ProcessLookupError:
            return 'NOT_FOUND'
        except PermissionError:
            return 'NO_PERMISSION'
        except Exception:
            return 'ERROR'

    if args.kill_all or args.kill_stale:
        pids_dir = args.pids_dir
        os.makedirs(pids_dir, exist_ok=True)
        entries = [f for f in os.listdir(pids_dir) if f.endswith('.pid')]
        if not entries:
            print(f"[KILL] No pid files under {pids_dir}")
            return
        killed = 0
        removed = 0
        for fn in entries:
            path = os.path.join(pids_dir, fn)
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    raw = f.read().strip()
                pid = int(''.join(ch for ch in raw if ch.isdigit()))
            except Exception:
                print(f"[KILL] Skip unreadable pid file: {fn}")
                continue
            alive = _pid_alive(pid)
            if not alive:
                if args.kill_stale or args.kill_all:
                    try:
                        os.remove(path)
                        removed += 1
                        print(f"[KILL] Removed stale pid {pid} ({fn})")
                    except Exception:
                        pass
                continue
            if args.kill_all:
                status = _kill_pid(pid)
                print(f"[KILL] pid={pid} status={status} ({fn})")
                killed += 1
                # Best-effort remove pid file
                try:
                    os.remove(path)
                    removed += 1
                except Exception:
                    pass
        print(f"[KILL] Summary: killed={killed}, removed_pid_files={removed}, total_entries={len(entries)}")
        return

    datasets = [d.strip().lower() for d in args.datasets.split(',') if d.strip()]
    models = [m.strip() for m in args.models.split(',') if m.strip()]
    backbones = [b.strip().lower() for b in args.backbones.split(',') if b.strip()]
    backbones = [b for b in backbones if b in ('resnet18','resnet34','resnet50','resnet101')]
    regimes = ['scratch'] if args.regime == 'scratch' else (['imagenet'] if args.regime == 'imagenet' else ['scratch','imagenet'])

    os.makedirs(os.path.dirname(args.save_report), exist_ok=True)
    rows: List[Dict[str, str]] = []

    # Prepare a base environment with optional BiResUnetPlus toggles so both training and evaluation share the same architecture flags
    base_env = os.environ.copy()
    if args.bires_decoder_se:
        base_env['BIRES_DECODER_SE'] = '1'
    if args.bires_bilinear_up:
        base_env['BIRES_BILINEAR_UP'] = '1'
    # ASPP env: allow explicit full-aspp to unset lite
    if args.bires_full_aspp:
        base_env['BIRES_LITE_ASPP'] = '0'
    elif args.bires_lite_aspp:
        base_env['BIRES_LITE_ASPP'] = '1'
    if args.bires_strip_pool:
        base_env['BIRES_STRIP_POOL'] = '1'
    if args.bires_decoder_dw:
        base_env['BIRES_DECODER_DW'] = '1'
    if args.bires_decoder_width_mult is not None:
        base_env['BIRES_DECODER_WIDTH_MULT'] = str(float(args.bires_decoder_width_mult))
    if args.bires_edge_aux:
        base_env['BIRES_EDGE_AUX'] = '1'
    if args.edge_aux_weight is not None:
        base_env['EDGE_AUX_WEIGHT'] = str(float(args.edge_aux_weight))

    # Formal-patience automation path
    if args.formal_patience_run:  # type: ignore[attr-defined]
        # Fixed hyperparameters
        fixed_epochs = 120
        fixed_lr = 8e-4
        fixed_loss = 'dice_bce'
        fixed_monitor = 'dice'
        fixed_es_pat = 10
        fixed_lr_pat = 6
        fixed_min_lr = 2e-7
        # GPU scheduler
        gpu_slots = [g.strip() for g in str(args.gpu_slots).split(',') if g.strip()]
        available_slots = deque(gpu_slots)
        os.makedirs(os.path.join(REPO_ROOT, 'runs', 'logs'), exist_ok=True)
        os.makedirs(os.path.join(REPO_ROOT, 'runs', 'pids'), exist_ok=True)
        print(f"[SCHED] GPU slots active: {','.join(gpu_slots)}")

        # Build job queue: for each dataset (mas, road) and each model
        job_queue: List[Tuple[str, str, Optional[str]]] = []  # (ds, net, bb)
        # Backbone policy: backbone-aware -> resnet34, others None
        for ds in datasets:
            if ds not in ('mas','road'):
                print(f"[WARN] Unsupported dataset token '{ds}', skipping")
                continue
            for net in models:
                if net in BACKBONE_AWARE:
                    job_queue.append((ds, net, 'resnet34'))
                else:
                    job_queue.append((ds, net, None))

        active: List[Dict[str, object]] = []

        def get_bires_strategy(ds: str) -> dict:
            """
            返回 BiResUnetPlus 专属策略参数 dict，支持后续命令拼接。
            - road: alt_mode=hsvgrad, img_size=960x544, batch=6
            - mas:  alt_mode=rgbgrad, img_size=1024x1024, batch=2
            - 其它参数固定：lr=8e-4, min_lr=5e-7, epochs=120, seed=1337
            """
            if ds == 'mas':
                return {
                    'alt_build_mode': 'rgbgrad',
                    'img_size': (1024, 1024),
                    'batch_size': 2,
                    'lr': 8e-4,
                    'min_lr': 2e-7,
                    'epochs': 120,
                    'global_seed': 1337
                }
            else:
                return {
                    'alt_build_mode': 'hsvgrad',
                    'img_size': (960, 544),
                    'batch_size': 6,
                    'lr': 8e-4,
                    'min_lr': 2e-7,
                    'epochs': 120,
                    'global_seed': 1337
                }

        def _resolve_img_size(ds: str, net_name: str) -> Tuple[int,int]:
            # BiResUnetPlus 专属分辨率策略
            if str(net_name).lower() == 'biresunetplus':
                strat = get_bires_strategy(ds)
                return strat['img_size']
            # 其它模型保持原有逻辑
            if args.force_img_size is not None:
                iw, ih = int(args.force_img_size[0]), int(args.force_img_size[1])
            else:
                if ds == 'mas':
                    iw, ih = 1024, 1024
                else:
                    iw, ih = 960, 544
            net_l = str(net_name).lower()
            if net_l == 'rcfsnet':
                iw = max(iw, 1024); ih = max(ih, 1024)
            if net_l == 'transroadnet':
                iw = max(iw, 1024); ih = max(ih, 1024)
                m = max(iw, ih); iw, ih = m, m
            return iw, ih

        def _start_job(job: Tuple[str,str,Optional[str]], gpu_id: str):
            ds, net, bb = job
            net_l = str(net).lower()
            # BiResUnetPlus 专属策略
            if net_l == 'biresunetplus':
                strat = get_bires_strategy(ds)
                bs = strat['batch_size']
                iw, ih = strat['img_size']
                lr = strat['lr']
                min_lr = strat['min_lr']
                epochs = strat['epochs']
                seed = strat['global_seed']
                alt_mode = strat['alt_build_mode']
            else:
                bs = 4 if ds == 'mas' else 8
                iw, ih = _resolve_img_size(ds, net)
                lr = fixed_lr
                min_lr = fixed_min_lr
                epochs = fixed_epochs
                seed = int(args.global_seed)
                alt_mode = args.alt_build_mode
            run_name = build_run_name_formal(ds, net, bb)
            train_cmd = [sys.executable, os.path.join('scripts','train_gpu.py'),
                         '--dataset', ('mas' if ds=='mas' else 'road'),
                         '--net', net,
                         '--epochs', str(epochs),
                         '--batch-size', str(bs),
                         '--lr', str(lr),
                         '--img-size', str(iw), str(ih),
                         '--val-interval', '1',
                         '--ckpt-interval', '5',
                         '--device', 'cuda',
                         '--name', run_name,
                         '--global-seed', str(seed),
                         '--loss', fixed_loss,
                         '--monitor-metric', fixed_monitor,
                         '--early-stop-patience', str(fixed_es_pat),
                         '--lr-reduce-patience', str(fixed_lr_pat),
                         '--min-lr', str(min_lr),
                         ]
            if ds == 'mas':
                train_cmd += ['--root', args.mas_root]
            if bb is not None:
                train_cmd += ['--backbone', bb]
            if args.resume_auto:
                train_cmd += ['--resume-auto']
            # BiResUnetPlus 专属 alt_mode
            if net_l == 'biresunetplus':
                train_cmd += ['--alt-build-mode', alt_mode]
            elif alt_mode:
                train_cmd += ['--alt-build-mode', alt_mode]
            # 其它架构参数与原有逻辑一致
            if args.bires_decoder_se:
                train_cmd += ['--bires-decoder-se']
            if args.bires_bilinear_up:
                train_cmd += ['--bires-bilinear-up']
            if args.bires_full_aspp:
                train_cmd += ['--bires-full-aspp']
            elif args.bires_lite_aspp:
                train_cmd += ['--bires-lite-aspp']
            if args.bires_strip_pool:
                train_cmd += ['--bires-strip-pool']
            if args.bires_decoder_dw:
                train_cmd += ['--bires-decoder-dw']
            if args.bires_decoder_width_mult is not None:
                train_cmd += ['--bires-decoder-width-mult', str(float(args.bires_decoder_width_mult))]
            if args.bires_edge_aux:
                train_cmd += ['--bires-edge-aux']
            if args.edge_aux_weight is not None:
                train_cmd += ['--edge-aux-weight', str(float(args.edge_aux_weight))]
            if args.alt_hsv_stats:
                train_cmd += ['--alt-hsv-stats', args.alt_hsv_stats]
            if args.alt_v_window_mode:
                train_cmd += ['--alt-v-window-mode', args.alt_v_window_mode]
            if args.alt_smooth_temp is not None:
                train_cmd += ['--alt-smooth-temp', str(float(args.alt_smooth_temp))]
            if args.alt_channel_weights:
                train_cmd += ['--alt-channel-weights', args.alt_channel_weights]

            log_path = os.path.join(REPO_ROOT, 'runs', 'logs', f"{run_name}.out")
            pid_path = os.path.join(REPO_ROOT, 'runs', 'pids', f"{run_name}.pid")
            env = base_env.copy()
            env['CUDA_VISIBLE_DEVICES'] = gpu_id
            print('[SCHED][START][GPU{}]'.format(gpu_id), ' '.join(train_cmd))
            if args.dry_run:
                return {'proc': None, 'gpu': gpu_id, 'job': job, 'run_name': run_name, 'log': log_path, 'pid': pid_path}
            lf = open(log_path, 'w', encoding='utf-8')
            p = subprocess.Popen(train_cmd, env=env, cwd=REPO_ROOT, stdout=lf, stderr=subprocess.STDOUT, text=True)
            # record PID
            with open(pid_path, 'w', encoding='utf-8') as pf:
                pf.write(str(p.pid))
            return {'proc': p, 'gpu': gpu_id, 'job': job, 'run_name': run_name, 'log': log_path, 'pid': pid_path}

        if args.dry_run:
            # Simplified preview: just list all train commands that would be launched
            idx = 0
            for job in job_queue:
                gpu_id = gpu_slots[idx % len(gpu_slots)]
                ds, net, bb = job
                bs = 4 if ds == 'mas' else 8
                iw, ih = _resolve_img_size(ds, net)
                run_name = build_run_name_formal(ds, net, bb)
                train_cmd = [sys.executable, os.path.join('scripts','train_gpu.py'),
                             '--dataset', ('mas' if ds=='mas' else 'road'),
                             '--net', net,
                             '--epochs', str(fixed_epochs),
                             '--batch-size', str(bs),
                             '--lr', str(fixed_lr),
                             '--img-size', str(iw), str(ih),
                             '--val-interval', '1',
                             '--ckpt-interval', '5',
                             '--device', 'cuda',
                             '--name', run_name,
                             '--global-seed', str(int(args.global_seed)),
                             '--loss', fixed_loss,
                             '--monitor-metric', fixed_monitor,
                             '--early-stop-patience', str(fixed_es_pat),
                             '--lr-reduce-patience', str(fixed_lr_pat),
                             '--min-lr', str(fixed_min_lr),
                             ]
                if ds == 'mas':
                    train_cmd += ['--root', args.mas_root]
                if bb is not None:
                    train_cmd += ['--backbone', bb]
                print('[DRYRUN][GPU{}]'.format(gpu_id), ' '.join(train_cmd))
                row = {
                    'dataset': 'Mas' if ds=='mas' else 'Road',
                    'net': net,
                    'backbone': bb or '',
                    'regime': 'scratch',
                    'run_name': run_name,
                    'pretrained': 'RandomInit',
                    'status': 'DRYRUN',
                    'img_w': str(iw),  # resolved size (with model constraints)
                    'img_h': str(ih),
                    'early_stop_patience': str(fixed_es_pat),
                    'lr_reduce_patience': str(fixed_lr_pat),
                    'lr_reduce_factor': '',
                    'min_lr': str(fixed_min_lr),
                    'monitor_metric': fixed_monitor,
                    'train_log': 'DRYRUN',
                    'epochs_logged': '0',
                    'last_epoch': '',
                    'train_last_loss': '',
                    'val_last_loss': '',
                    'train_last_dice': '',
                    'train_best_dice': '',
                    'last_lr': '',
                    'weights_best': 'N',
                    'weights_last': 'N'
                }
                rows.append(row)
                idx += 1
        else:
            # Real scheduler loop
            qi = 0
            def _tail_log(path: str, max_bytes: int = 4096) -> str:
                try:
                    with open(path, 'rb') as f:
                        f.seek(0, os.SEEK_END)
                        size = f.tell()
                        f.seek(max(0, size - max_bytes), os.SEEK_SET)
                        data = f.read().decode('utf-8', errors='ignore')
                        return data
                except Exception:
                    return ''

            while qi < len(job_queue) or active:
                # fill available gpu slots using a queue of free slots
                while qi < len(job_queue) and available_slots:
                    gpu_id = available_slots.popleft()
                    ctx = _start_job(job_queue[qi], gpu_id)
                    active.append(ctx)
                    qi += 1
                # poll active
                new_active: List[Dict[str, object]] = []
                for ctx in active:
                    p = ctx['proc']
                    if p is not None and p.poll() is None:
                        new_active.append(ctx)
                        continue
                    # job finished
                    run_name = ctx['run_name']  # type: ignore[index]
                    ds, net, bb = ctx['job']  # type: ignore[index]
                    gpu_id = ctx['gpu']  # type: ignore[index]
                    log_path = ctx['log']  # type: ignore[index]
                    rc = None
                    if p is not None:
                        rc = p.returncode
                    print(f"[SCHED][DONE][GPU{gpu_id}] {run_name}")
                    if rc is not None and rc != 0:
                        print(f"[SCHED][WARN] Train exited with rc={rc} for {run_name}. Tail of log:")
                        tail = _tail_log(log_path)
                        if tail:
                            # Print only the last ~100 lines to avoid flooding
                            tail_lines = tail.strip().splitlines()[-100:]
                            for ln in tail_lines:
                                print('[LOGTAIL]', ln)
                        # Record failure row, skip eval
                        try:
                            train_log_info = monitor_training_log(run_name)
                            # Resolve image size for reporting (failure path had no local iw/ih)
                            iw, ih = _resolve_img_size(ds, net)
                            row = {
                                'dataset': 'Mas' if ds=='mas' else 'Road',
                                'net': net,
                                'backbone': bb or '',
                                'regime': 'scratch',
                                'run_name': run_name,
                                'pretrained': 'RandomInit',
                                'status': f'TRAIN_FAIL({rc})',
                                'img_w': str(iw),
                                'img_h': str(ih),
                                'early_stop_patience': str(fixed_es_pat),
                                'lr_reduce_patience': str(fixed_lr_pat),
                                'lr_reduce_factor': '',
                                'min_lr': str(fixed_min_lr),
                                'monitor_metric': fixed_monitor,
                            }
                            if isinstance(train_log_info, dict):
                                row.update(train_log_info)
                            rows.append(row)
                        finally:
                            # Ensure the GPU slot is always returned even if reporting fails
                            available_slots.append(gpu_id)
                            print(f"[SCHED][FREE][GPU{gpu_id}] (train fail) slots now: {list(available_slots)}")
                        continue
                    # Sleep after finish
                    if int(args.sleep_after_finish) > 0:
                        import time as _t
                        _t.sleep(int(args.sleep_after_finish))
                    # After training, run eval on same GPU
                    iw, ih = _resolve_img_size(ds, net)
                    eval_cmd = [sys.executable, os.path.join('scripts','eval_gpu.py'),
                                '--dataset', ('mas' if ds=='mas' else 'road'),
                                '--net', net,
                                '--img-size', str(iw), str(ih),
                                '--batch-size', str( max(1, (4 if ds=='mas' else 8)//2) ),
                                '--threshold', '0.5',
                                '--device', 'cuda',
                                '--measure-speed']
                    if ds == 'mas':
                        eval_cmd += ['--root', args.mas_root]
                    if bb is not None:
                        eval_cmd += ['--backbone', bb]
                    # ALT 透传到 eval_gpu.py
                    if args.alt_build_mode:
                        eval_cmd += ['--alt-build-mode', args.alt_build_mode]
                    if args.alt_hsv_stats:
                        eval_cmd += ['--alt-hsv-stats', args.alt_hsv_stats]
                    if args.alt_v_window_mode:
                        eval_cmd += ['--alt-v-window-mode', args.alt_v_window_mode]
                    if args.alt_smooth_temp is not None:
                        eval_cmd += ['--alt-smooth-temp', str(float(args.alt_smooth_temp))]
                    if args.alt_channel_weights:
                        eval_cmd += ['--alt-channel-weights', args.alt_channel_weights]
                    eval_env = base_env.copy()
                    eval_env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
                    print('[SCHED][EVAL][GPU{}]'.format(gpu_id), ' '.join(eval_cmd))
                    metrics: Dict[str, str] = {}
                    eval_rc = 0
                    eval_rc, out = run_subprocess(eval_cmd, env=eval_env, capture=True)
                    if eval_rc != 0:
                        print(f"[WARN] Eval failed rc={eval_rc} for {run_name}")
                    metrics = parse_eval_output(out)
                    # monitor training log
                    train_log_info = monitor_training_log(run_name)
                    # status summary
                    status = 'OK'
                    if eval_rc != 0:
                        status = f'EVAL_FAIL({eval_rc})'
                    row = {
                        'dataset': 'Mas' if ds=='mas' else 'Road',
                        'net': net,
                        'backbone': bb or '',
                        'regime': 'scratch',
                        'run_name': run_name,
                        'pretrained': 'RandomInit',
                        'status': status,
                        'img_w': str(iw),
                        'img_h': str(ih),
                        'early_stop_patience': str(fixed_es_pat),
                        'lr_reduce_patience': str(fixed_lr_pat),
                        'lr_reduce_factor': '',
                        'min_lr': str(fixed_min_lr),
                        'monitor_metric': fixed_monitor,
                    }
                    row.update(metrics)
                    if isinstance(train_log_info, dict):
                        row.update(train_log_info)
                    rows.append(row)
                    # After eval completes, return the GPU slot to the available pool
                    available_slots.append(gpu_id)
                    print(f"[SCHED][FREE][GPU{gpu_id}] (eval done) slots now: {list(available_slots)}")
                active = new_active
            # done scheduling
    else:
        # Legacy sequential path
        for ds in datasets:
            if ds not in ('mas','road'):
                print(f"[WARN] Unsupported dataset token '{ds}', skipping")
                continue
            for net in models:
                use_bbs = backbones if net in BACKBONE_AWARE else [None]
                for bb in use_bbs:
                    for regime in regimes:
                        use_pretrained = (regime == 'imagenet')
                        run_name = build_run_name(ds, net, bb)
                        net_l = str(net).lower()
                        # BiResUnetPlus 专属策略
                        if net_l == 'biresunetplus':
                            strat = get_bires_strategy(ds)
                            iw, ih = strat['img_size']
                            bs = strat['batch_size']
                            lr = strat['lr']
                            min_lr = strat['min_lr']
                            epochs = strat['epochs']
                            seed = strat['global_seed']
                            alt_mode = strat['alt_build_mode']
                        else:
                            try:
                                base_w, base_h = int(args.img_size[0]), int(args.img_size[1])
                            except Exception:
                                base_w, base_h = 1024, 1024 if ds == 'mas' else (960, 544)
                            iw, ih = resolve_img_size(ds, net, base_w, base_h)
                            bs = int(args.batch_size)
                            lr = float(args.lr)
                            min_lr = args.min_lr if args.min_lr is not None else ''
                            epochs = int(args.epochs)
                            seed = int(args.global_seed) if args.global_seed is not None else ''
                            alt_mode = args.alt_build_mode
                        train_cmd = [sys.executable, os.path.join('scripts','train_gpu.py'),
                                     '--dataset', ('mas' if ds=='mas' else 'road'),
                                     '--net', net,
                                     '--epochs', str(epochs),
                                     '--batch-size', str(bs),
                                     '--lr', str(lr),
                                     '--img-size', str(int(iw)), str(int(ih)),
                                     '--val-interval', str(int(args.val_interval)),
                                     '--ckpt-interval', str(int(args.ckpt_interval)),
                                     '--device', args.device,
                                     '--name', run_name,
                                     '--global-seed', str(seed) if seed != '' else ''
                                     ]
                        train_cmd = [t for t in train_cmd if t != '']
                        if ds == 'mas':
                            train_cmd += ['--root', args.mas_root]
                        if bb is not None:
                            train_cmd += ['--backbone', bb]
                        if use_pretrained:
                            train_cmd += ['--pretrained']
                        if args.gpus and args.device != 'cpu':
                            train_cmd += ['--gpus', args.gpus]
                        if args.resume_auto:
                            train_cmd += ['--resume-auto']
                        # BiResUnetPlus 专属 alt_mode
                        if net_l == 'biresunetplus':
                            train_cmd += ['--alt-build-mode', alt_mode]
                        elif alt_mode:
                            train_cmd += ['--alt-build-mode', alt_mode]
                        # Loss selection
                        if args.loss is not None:
                            train_cmd += ['--loss', str(args.loss)]
                        # Pass-through training policy args if provided
                        if args.early_stop_patience is not None:
                            train_cmd += ['--early-stop-patience', str(int(args.early_stop_patience))]
                        if args.lr_reduce_patience is not None:
                            train_cmd += ['--lr-reduce-patience', str(int(args.lr_reduce_patience))]
                        if args.lr_reduce_factor is not None:
                            train_cmd += ['--lr-reduce-factor', str(float(args.lr_reduce_factor))]
                        if min_lr != '':
                            train_cmd += ['--min-lr', str(min_lr)]
                        if args.monitor_metric is not None:
                            train_cmd += ['--monitor-metric', str(args.monitor_metric)]
                        # BiResUnetPlus CLI toggles (train_gpu will also export env for adapter)
                        if args.bires_decoder_se:
                            train_cmd += ['--bires-decoder-se']
                        if args.bires_bilinear_up:
                            train_cmd += ['--bires-bilinear-up']
                        if args.bires_full_aspp:
                            train_cmd += ['--bires-full-aspp']
                        elif args.bires_lite_aspp:
                            train_cmd += ['--bires-lite-aspp']
                        if args.bires_strip_pool:
                            train_cmd += ['--bires-strip-pool']
                        if args.bires_decoder_dw:
                            train_cmd += ['--bires-decoder-dw']
                        if args.bires_decoder_width_mult is not None:
                            train_cmd += ['--bires-decoder-width-mult', str(float(args.bires_decoder_width_mult))]
                        if args.bires_edge_aux:
                            train_cmd += ['--bires-edge-aux']
                        if args.edge_aux_weight is not None:
                            train_cmd += ['--edge-aux-weight', str(float(args.edge_aux_weight))]
                        if args.alt_hsv_stats:
                            train_cmd += ['--alt-hsv-stats', args.alt_hsv_stats]
                        if args.alt_v_window_mode:
                            train_cmd += ['--alt-v-window-mode', args.alt_v_window_mode]
                        if args.alt_smooth_temp is not None:
                            train_cmd += ['--alt-smooth-temp', str(float(args.alt_smooth_temp))]
                        if args.alt_channel_weights:
                            train_cmd += ['--alt-channel-weights', args.alt_channel_weights]

                        print('[ALL][TRAIN]', ' '.join(train_cmd))
                        train_rc = 0
                        if not args.dry_run:
                            train_rc, _ = run_subprocess(train_cmd, env=base_env)
                            if train_rc != 0:
                                print(f"[WARN] Training failed rc={train_rc} for {run_name}")

                        # Evaluation command via scripts/eval_gpu.py (use default <RunName>_best.pth)
                        eval_cmd = [sys.executable, os.path.join('scripts','eval_gpu.py'),
                                    '--dataset', ('mas' if ds=='mas' else 'road'),
                                    '--net', net,
                                    '--img-size', str(int(iw)), str(int(ih)),
                                    '--batch-size', str(int(max(1, bs//2))),
                                    '--threshold', '0.5',
                                    '--device', args.device,
                                    '--measure-speed']
                        if ds == 'mas':
                            eval_cmd += ['--root', args.mas_root]
                        if bb is not None:
                            eval_cmd += ['--backbone', bb]
                        if args.gpus and args.device != 'cpu':
                            eval_cmd += ['--gpus', args.gpus]
                        # BiResUnetPlus 专属 alt_mode
                        if net_l == 'biresunetplus':
                            eval_cmd += ['--alt-build-mode', alt_mode]
                        elif alt_mode:
                            eval_cmd += ['--alt-build-mode', alt_mode]
                        if args.alt_hsv_stats:
                            eval_cmd += ['--alt-hsv-stats', args.alt_hsv_stats]
                        if args.alt_v_window_mode:
                            eval_cmd += ['--alt-v-window-mode', args.alt_v_window_mode]
                        if args.alt_smooth_temp is not None:
                            eval_cmd += ['--alt-smooth-temp', str(float(args.alt_smooth_temp))]
                        if args.alt_channel_weights:
                            eval_cmd += ['--alt-channel-weights', args.alt_channel_weights]

                        print('[ALL][EVAL] ', ' '.join(eval_cmd))
                        metrics: Dict[str, str] = {}
                        eval_rc = 0
                        if not args.dry_run:
                            eval_rc, out = run_subprocess(eval_cmd, env=base_env, capture=True)
                            if eval_rc != 0:
                                print(f"[WARN] Eval failed rc={eval_rc} for {run_name}")
                            metrics = parse_eval_output(out)

                        # Training log monitoring (after training)
                        if args.dry_run:
                            train_log_info = {
                                'train_log': 'DRYRUN', 'epochs_logged': '0', 'last_epoch': '',
                                'train_last_loss': '', 'val_last_loss': '', 'train_last_dice': '', 'train_best_dice': '', 'last_lr': '',
                                'weights_best': 'N', 'weights_last': 'N'
                            }
                        else:
                            train_log_info = monitor_training_log(run_name)

                        # pretrained status summary (for report)
                        if use_pretrained:
                            if net in SUPPORTS_LOCAL_PRETRAINED:
                                pretrained_status = 'ImageNetPretrained(Local)'
                            else:
                                pretrained_status = 'RequestedButNotSupported'
                        else:
                            pretrained_status = 'RandomInit'

                        # status field
                        status = 'OK'
                        if not args.dry_run:
                            if train_rc != 0:
                                status = f'TRAIN_FAIL({train_rc})'
                            elif eval_rc != 0:
                                status = f'EVAL_FAIL({eval_rc})'

                        row = {
                            'dataset': 'Mas' if ds=='mas' else 'Road',
                            'net': net,
                            'backbone': bb or '',
                            'regime': regime,
                            'run_name': run_name,
                            'pretrained': pretrained_status,
                            'status': status,
                            'img_w': str(iw),
                            'img_h': str(ih),
                            # training policy snapshot
                            'early_stop_patience': '' if args.early_stop_patience is None else str(int(args.early_stop_patience)),
                            'lr_reduce_patience': '' if args.lr_reduce_patience is None else str(int(args.lr_reduce_patience)),
                            'lr_reduce_factor': '' if args.lr_reduce_factor is None else str(float(args.lr_reduce_factor)),
                            'min_lr': '' if min_lr == '' else str(min_lr),
                            'monitor_metric': '' if args.monitor_metric is None else str(args.monitor_metric),
                        }
                        row.update(metrics)
                        row.update(train_log_info)
                        rows.append(row)

    # Write report CSV
    fieldnames = [
        'dataset','net','backbone','regime','run_name','pretrained','status','img_w','img_h',
        'early_stop_patience','lr_reduce_patience','lr_reduce_factor','min_lr','monitor_metric',
        'avg_ms_per_image','fps','ACC','PRE','REC','IOU','DICE','FPR','FNR',
        # training log health
        'train_log','epochs_logged','last_epoch','train_last_loss','val_last_loss','train_last_dice','train_best_dice','last_lr','epoch_order_ok',
        'weights_best','weights_last'
    ]
    with open(args.save_report, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[ALL] Saved report: {args.save_report} ({len(rows)} rows)")


if __name__ == '__main__':
    main()
