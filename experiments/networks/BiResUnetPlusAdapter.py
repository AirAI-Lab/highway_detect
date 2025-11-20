"""Unified BiResUnetPlusAdapter with internal differentiable ALT build support.

ALT 构造三种模式 (alt_build_mode):
  - 'rgbgrad':  可微 8 通道 [R,G,B,gray,grad_mag,dir_cos,dir_sin,laplacian]
  - 'hsvgrad':  可微 8 通道 [S_mask,H_mask,V_mask,grad_mag,dir_cos,dir_sin,gray,laplacian]
  - 'hsv_cv':   OpenCV 路径 (旧版) [S_mask,H_mask,V_mask,edge,dt,dir_cos,dir_sin,gray]

当选择 rgbgrad/hsvgrad 时, 直接使用核心模型的 in-graph builder (auto_alt_mode) 保证训练=推理。
选择 hsv_cv 时保留历史 OpenCV 构造, 适合与旧权重或论文复现比较。
"""

import os
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.biresunet_plus import BiResUnetPlus
try:
    from common.adaptive_hsv import AdaptiveHSVExtractor  # type: ignore
except Exception:
    AdaptiveHSVExtractor = None  # type: ignore


def _edge_sobel(gray_: np.ndarray) -> np.ndarray:
    sx_ = cv2.Sobel(gray_, cv2.CV_32F, 1, 0, ksize=3)
    sy_ = cv2.Sobel(gray_, cv2.CV_32F, 0, 1, ksize=3)
    ed = np.sqrt(sx_ * sx_ + sy_ * sy_)
    return (ed - ed.min()) / (ed.max() - ed.min() + 1e-8)


def _build_alt_hsv_cv(img_bgr: np.ndarray, hsv_mode: str = 'adaptive', hsv_stats_path: str = None) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    if hsv_mode == 'adaptive' and AdaptiveHSVExtractor is not None:
        extractor = AdaptiveHSVExtractor(stats_path=hsv_stats_path)
        s_mask, h_mask, v_mask = extractor.build_masks(img_bgr)
    else:
        s_mask = cv2.GaussianBlur((s.astype(np.float32) < 50).astype(np.float32), (5, 5), 0)
        h_mask = ((h >= 65) & (h <= 133)).astype(np.float32)
        h_mask = cv2.GaussianBlur(h_mask, (5, 5), 0)
        v_blur = cv2.GaussianBlur(v, (5, 5), 0)
        hist = cv2.calcHist([v_blur], [0], None, [256], [0, 256]).flatten()
        peak = int(np.argmax(hist))
        peak_min = max(0, peak - 15)
        peak_max = min(255, peak + 15)
        v_mask = ((v_blur >= peak_min) & (v_blur <= peak_max)).astype(np.float32)
        v_mask = cv2.GaussianBlur(v_mask, (5, 5), 0)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # Always use Sobel for edge extraction (ALT_EDGE_MODE removed)
    edge = _edge_sobel(gray)
    try:
        s_bin = (s_mask > 0.5).astype(np.uint8)
        h_bin = (h_mask > 0.5).astype(np.uint8)
        v_bin = (v_mask > 0.5).astype(np.uint8)
        sm = (s_bin & h_bin & v_bin).astype(np.uint8)
        soft_mask = cv2.GaussianBlur(sm.astype(np.float32), (5, 5), 0)
    except Exception:
        soft_mask = cv2.GaussianBlur(((s < 50).astype(np.uint8) * ((h >= 65) & (h <= 133)).astype(np.uint8)), (5, 5), 0)
    dt = cv2.distanceTransform((soft_mask * 255).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
    dt = dt / (dt.max() + 1e-8)
    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    angle = np.arctan2(sy, sx + 1e-8)
    dir_cos = np.cos(angle).astype(np.float32)
    dir_sin = np.sin(angle).astype(np.float32)
    gray_n = (gray.astype(np.float32) - gray.min()) / (gray.max() - gray.min() + 1e-8)
    alt_channels = [s_mask, h_mask, v_mask, edge.astype(np.float32), dt.astype(np.float32), dir_cos, dir_sin, gray_n]
    ch_w_env = os.environ.get('ALT_CHANNEL_WEIGHTS', '').strip()
    if ch_w_env:
        try:
            w = [float(x) for x in ch_w_env.split(',')]
            if len(w) == len(alt_channels):
                for i in range(len(alt_channels)):
                    alt_channels[i] = (alt_channels[i].astype(np.float32) * w[i]).astype(np.float32)
        except Exception:
            pass
    return np.stack(alt_channels, axis=2).astype(np.float32)


class BiResUnetPlusAdapter(nn.Module):
    def __init__(self, out_ch: int = 1, backbone: str = 'resnet34', pretrained: bool = False,
                 alt_build_mode: str = None, hsv_stats_path: str = None, v_window_mode: str = 'mad',
                 smooth_temp: float = 4.0, channel_weights: str = None,
                 use_eem: bool = False, eem_levels: str = None, eem_apply_rgb: str = None,
                 eem_apply_alt: str = None, eem_reduction: int = None,
                 use_lite_aspp: object = None):
        super().__init__()
        bb = (backbone or 'resnet34').lower()
        if bb not in ('resnet18', 'resnet34', 'resnet50', 'resnet101'):
            bb = 'resnet34'
        # env fallbacks
        env_alt_mode = os.environ.get('ALT_BUILD_MODE', '').strip().lower()
        alt_mode = (alt_build_mode or env_alt_mode or 'rgbgrad').lower()
        hsv_stats_path = hsv_stats_path or os.environ.get('ALT_HSV_STATS', hsv_stats_path or os.path.join('data','RoadOnly','hsv_stats.npz'))
        v_window_mode = (v_window_mode or os.environ.get('ALT_V_WINDOW_MODE','mad')).lower()
        smooth_temp = float(os.environ.get('ALT_SMOOTH_TEMP', smooth_temp))
        # parse channel weights (comma-separated) -> tuple[float]
        cw_env = os.environ.get('ALT_CHANNEL_WEIGHTS', '').strip()
        cw_src = channel_weights or cw_env
        cw_tuple = None
        if cw_src:
            try:
                parts = [float(x) for x in cw_src.split(',') if x.strip()]
                if parts:
                    cw_tuple = tuple(parts)
            except Exception:
                cw_tuple = None
        # 默认: hsvgrad 模式未提供权重时下调 H 通道权重到 0.85 (顺序: S,H,V,grad_mag,dir_cos,dir_sin,gray,lap)
        if cw_tuple is None and alt_mode == 'hsvgrad':
            cw_tuple = (1.0, 0.85, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
        # base core model
        # allow explicit argument to override env variable
        if use_lite_aspp is None:
            use_lite_aspp = os.environ.get('BIRES_LITE_ASPP', '1') == '1'
        else:
            # normalize common string/bool representations
            if isinstance(use_lite_aspp, str):
                use_lite_aspp = use_lite_aspp.strip().lower() in ('1', 'true', 'yes', 'y')
            else:
                use_lite_aspp = bool(use_lite_aspp)
        decoder_use_se = os.environ.get('BIRES_DECODER_SE', '0') == '1'
        up_mode = 'bilinear' if os.environ.get('BIRES_BILINEAR_UP', '1') == '1' else 'deconv'
        # parse EEM args: allow passthrough via env or caller
        env_eem_levels = os.environ.get('BIRES_EEM_LEVELS', '').strip()
        env_eem_rgb = os.environ.get('BIRES_EEM_RGB', '').strip().lower()
        env_eem_alt = os.environ.get('BIRES_EEM_ALT', '').strip().lower()
        env_eem_reduction = os.environ.get('BIRES_EEM_REDUCTION', '').strip()
        # source values: explicit args take precedence, then env
        src_eem_levels = eem_levels if eem_levels is not None else (env_eem_levels or None)
        src_eem_rgb = (eem_apply_rgb if eem_apply_rgb is not None else (env_eem_rgb if env_eem_rgb else None))
        src_eem_alt = (eem_apply_alt if eem_apply_alt is not None else (env_eem_alt if env_eem_alt else None))
        src_eem_reduction = eem_reduction if eem_reduction is not None else (int(env_eem_reduction) if env_eem_reduction.isdigit() else None)

        # normalize boolean flags
        def _parse_bool_str(v):
            if v is None:
                return None
            if isinstance(v, bool):
                return v
            try:
                vs = str(v).strip().lower()
                if vs in ('1','true','yes','y'):
                    return True
                if vs in ('0','false','no','n'):
                    return False
            except Exception:
                pass
            return None

        src_eem_rgb_b = _parse_bool_str(src_eem_rgb)
        src_eem_alt_b = _parse_bool_str(src_eem_alt)

        # normalize levels string to tuple
        def _parse_levels(s):
            if s is None:
                return None
            if isinstance(s, (list,tuple)):
                try:
                    return tuple(int(x) for x in s)
                except Exception:
                    return None
            try:
                parts = [int(x) for x in str(s).split(',') if x.strip()]
                parts = tuple(x for x in parts if x in (1,2,3,4))
                return parts if parts else None
            except Exception:
                return None

        parsed_levels = _parse_levels(src_eem_levels)

        if alt_mode in ('rgbgrad','hsvgrad'):
            # in-graph differentiable builder
            self.core = BiResUnetPlus(out_ch=out_ch, backbone=bb, pretrained=bool(pretrained),
                                      decoder_use_se=decoder_use_se, upsample_mode=up_mode, use_lite_aspp=use_lite_aspp,
                                      auto_alt_mode=alt_mode, hsv_stats_path=(hsv_stats_path if alt_mode=='hsvgrad' else None),
                                      v_window_mode=v_window_mode, smooth_temp=smooth_temp,
                                      channel_weights=cw_tuple,
                                      use_eem=bool(use_eem),
                                      eem_levels=parsed_levels,
                                      eem_apply_rgb=(True if src_eem_rgb_b is None else src_eem_rgb_b),
                                      eem_apply_alt=(True if src_eem_alt_b is None else src_eem_alt_b),
                                      eem_reduction=(src_eem_reduction if src_eem_reduction is not None else 2))
        else:
            # legacy CV HSV path (alt_mode = 'hsv_cv' or others)
            self.core = BiResUnetPlus(out_ch=out_ch, backbone=bb, pretrained=bool(pretrained),
                                      decoder_use_se=decoder_use_se, upsample_mode=up_mode, use_lite_aspp=use_lite_aspp,
                                      use_eem=bool(use_eem),
                                      eem_levels=parsed_levels,
                                      eem_apply_rgb=(True if src_eem_rgb_b is None else src_eem_rgb_b),
                                      eem_apply_alt=(True if src_eem_alt_b is None else src_eem_alt_b),
                                      eem_reduction=(src_eem_reduction if src_eem_reduction is not None else 2))
        self.alt_mode = alt_mode
        self.hsv_stats_path = hsv_stats_path
        self.v_window_mode = v_window_mode
        self.smooth_temp = smooth_temp
        self.channel_weights = cw_tuple
        self.pretrained_effective = bool(getattr(self.core, 'pretrained_effective', False))

    @staticmethod
    def _denorm_to_bgr_uint8(x_chw: torch.Tensor) -> np.ndarray:
        x_np = x_chw.detach().cpu().numpy().transpose(1, 2, 0)
        bgr01 = (x_np + 1.6) / 3.2
        return np.clip(bgr01 * 255.0, 0, 255).astype(np.uint8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,3,H,W) in normalized space [-1.6,1.6]
        assert x.shape[1] == 3, 'Expect 3-channel input'
        if self.alt_mode in ('rgbgrad','hsvgrad'):
            # core builds ALT internally (alt=None)
            x01 = ((x + 1.6)/3.2).clamp(0.0,1.0)
            # 统一为 RGB 顺序以匹配可微 HSV 与 ResNet 预训练约定（输入来自 cv2 多为 BGR）
            x01 = x01[:, [2,1,0], :, :]
            logits = self.core(x01, alt=None)
            probs = torch.sigmoid(logits)
            # Upsample probs to input spatial size if needed
            if probs.shape[-2:] != x.shape[-2:]:
                probs = F.interpolate(probs, size=x.shape[-2:], mode='bilinear', align_corners=False)
            return probs
        # legacy CV path
        B = x.size(0)
        rgb_list = []
        alt_list = []
        stats_path = self.hsv_stats_path if (self.hsv_stats_path and os.path.isfile(self.hsv_stats_path)) else None
        hsv_mode = 'adaptive' if stats_path else 'fixed'
        for b in range(B):
            bgr = self._denorm_to_bgr_uint8(x[b])
            alt = _build_alt_hsv_cv(bgr, hsv_mode=hsv_mode, hsv_stats_path=stats_path)
            rgb01 = (bgr.astype(np.float32)/255.0).transpose(2,0,1)
            alt_list.append(alt.transpose(2,0,1))
            rgb_list.append(rgb01)
        device = x.device
        rgb_t = torch.from_numpy(np.stack(rgb_list,0)).to(device=device,dtype=torch.float32)
        alt_t = torch.from_numpy(np.stack(alt_list,0)).to(device=device,dtype=torch.float32)
        logits = self.core(rgb_t, alt_t)
        probs = torch.sigmoid(logits)
        if probs.shape[-2:] != x.shape[-2:]:
            probs = F.interpolate(probs, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return probs

__all__ = ['BiResUnetPlusAdapter']
