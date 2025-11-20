import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
import json
from torchvision import models as tv_models
try:
    from .adaptive_hsv import AdaptiveHSVExtractor  # relative import when used as package
except Exception:
    # fallback if run as a single script context
    try:
        from adaptive_hsv import AdaptiveHSVExtractor  # type: ignore
    except Exception:
        AdaptiveHSVExtractor = None  # type: ignore
try:
    # Torchvision >= 0.13: use weights enums instead of deprecated 'pretrained' flag
    from torchvision.models import (
        ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights,
    )
    _HAS_TV_WEIGHTS_ENUM = True
except Exception:
    _HAS_TV_WEIGHTS_ENUM = False
from typing import Tuple

# ---------------------------------------------------------------------------
# Differentiable ALT (aux) feature builder
# ---------------------------------------------------------------------------
class DifferentiableAltBuilder(nn.Module):
    """Build 8-channel auxiliary (ALT) features inside the graph so that
    training与推理完全一致, 支持两种模式:
      - 'rgbgrad':  R,G,B, gray, grad_mag, dir_cos, dir_sin, laplacian
      - 'hsvgrad':  S_mask, H_mask, V_mask, grad_mag, dir_cos, dir_sin, gray, laplacian

    可选自适应 HSV 阈值 (来自 stats npz/json):
      期望字段: s_low_thr, h_low, h_high, v_low_mad, v_high_mad 或 fallback v_low, v_high

    所有 mask 使用平滑 sigmoid 近似, 保持可微性。
    """
    def __init__(self, mode: str = 'hsvgrad', stats_path: str = None,
                 v_window_mode: str = 'mad', s_thr_mode: str = 'low', smooth_temp: float = 4.0,
                 channel_weights: Tuple[float, ...] = None):
        super().__init__()
        self.mode = (mode or 'hsvgrad').lower()
        self.stats_path = stats_path
        self.v_window_mode = (v_window_mode or 'mad').lower()  # 'mad' or 'legacy'
        self.s_thr_mode = (s_thr_mode or 'low').lower()  # 保留接口 (未来: median/quantile)
        self.smooth_temp = float(smooth_temp)
        self.register_buffer('stats_loaded', torch.zeros(1))
        # thresholds (缓存在 buffer 中, 便于多GPU broadcast)
        self.register_buffer('s_thr', torch.zeros(1))
        self.register_buffer('h_low', torch.zeros(1))
        self.register_buffer('h_high', torch.ones(1))
        self.register_buffer('v_low', torch.zeros(1))
        self.register_buffer('v_high', torch.ones(1))
        # 默认通道权重: 在 hsvgrad 模式下若未指定, 轻微下调 H 掩码权重到 0.85 以降低对 Hue 的过拟合风险
        # 通道顺序 hsvgrad: [S_mask(0), H_mask(1), V_mask(2), grad_mag(3), dir_cos(4), dir_sin(5), gray(6), laplacian(7)]
        if channel_weights is None and self.mode == 'hsvgrad':
            channel_weights = (1.0, 0.85, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
        self.channel_weights = channel_weights
        if stats_path:
            self._load_stats_into_buffers(stats_path)

    def _load_stats_into_buffers(self, path: str):
        data = None
        try:
            if path.endswith('.npz'):
                npz = np.load(path, allow_pickle=True)
                data = {k: npz[k].item() if npz[k].shape == () else npz[k] for k in npz.files}
            elif path.endswith('.json'):
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
        except Exception:
            data = None
        if not isinstance(data, dict):
            return
        # 解析阈值
        s_val = data.get('s_low_thr', data.get('s_thr', 50))
        h_low = data.get('h_low', 65)
        h_high = data.get('h_high', 133)
        if self.v_window_mode == 'mad':
            v_low = data.get('v_low_mad', data.get('v_low', 100))
            v_high = data.get('v_high_mad', data.get('v_high', 180))
        else:
            v_low = data.get('v_low', 100)
            v_high = data.get('v_high', 180)
        # 限制范围
        h_low = float(max(0, min(179, h_low)))
        h_high = float(max(0, min(179, h_high)))
        v_low = float(max(0, min(255, v_low)))
        v_high = float(max(0, min(255, v_high)))
        s_val = float(max(0, min(255, s_val)))
        # 写入 buffers (归一化到 [0,1])
        self.s_thr[:] = s_val / 255.0
        self.h_low[:] = h_low / 179.0
        self.h_high[:] = h_high / 179.0
        self.v_low[:] = v_low / 255.0
        self.v_high[:] = v_high / 255.0
        self.stats_loaded[:] = 1.0

    @staticmethod
    def _rgb_to_hsv(rgb: torch.Tensor):
        # rgb: (B,3,H,W) in [0,1]
        r, g, b = rgb[:,0], rgb[:,1], rgb[:,2]
        maxc, _ = torch.max(rgb, dim=1)
        minc, _ = torch.min(rgb, dim=1)
        diff = maxc - minc + 1e-6
        # Hue
        h = torch.zeros_like(maxc)
        mask_r = (maxc == r)
        mask_g = (maxc == g)
        mask_b = (maxc == b)
        h[mask_r] = ( (g - b)[mask_r] / diff[mask_r] ) % 6
        h[mask_g] = ( (b - r)[mask_g] / diff[mask_g] ) + 2
        h[mask_b] = ( (r - g)[mask_b] / diff[mask_b] ) + 4
        h = h / 6.0  # -> [0,1]
        # Saturation
        s = diff / (maxc + 1e-6)
        # Value
        v = maxc
        return h, s, v

    def _smooth_mask_range(self, x: torch.Tensor, low: torch.Tensor, high: torch.Tensor, temp: float) -> torch.Tensor:
        # sigmoid((x-low)/t) * sigmoid((high-x)/t)
        return torch.sigmoid((x - low)/temp) * torch.sigmoid((high - x)/temp)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        # rgb (B,3,H,W) normalized [0,1]
        if self.mode == 'rgbgrad':
            return self._build_rgbgrad(rgb)
        return self._build_hsvgrad(rgb)

    def _build_rgbgrad(self, rgb: torch.Tensor) -> torch.Tensor:
        r,g,b = rgb[:,0:1], rgb[:,1:2], rgb[:,2:3]
        gray = 0.299*r + 0.587*g + 0.114*b
        # Sobel
        gx = F.conv2d(gray, self._sobel_kernel_x().to(gray.device), padding=1)
        gy = F.conv2d(gray, self._sobel_kernel_y().to(gray.device), padding=1)
        grad_mag = torch.sqrt(gx*gx + gy*gy + 1e-8)
        angle = torch.atan2(gy, gx + 1e-8)
        dir_cos = torch.cos(angle)
        dir_sin = torch.sin(angle)
        lap = F.conv2d(gray, self._laplacian_kernel().to(gray.device), padding=1).abs()
        # 归一化各响应
        def _norm(t):
            return (t - t.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]) / (t.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0] + 1e-8)
        grad_mag = _norm(grad_mag)
        lap = _norm(lap)
        alt = torch.cat([r,g,b, gray, grad_mag, dir_cos, dir_sin, lap], dim=1)
        if self.channel_weights:
            w = torch.tensor(self.channel_weights, dtype=alt.dtype, device=alt.device).view(1,-1,1,1)
            if w.numel() == alt.shape[1]:
                alt = alt * w
        return alt

    def _build_hsvgrad(self, rgb: torch.Tensor) -> torch.Tensor:
        h,s,v = self._rgb_to_hsv(rgb)
        # 阈值 (stats 未加载则使用默认)
        s_thr = self.s_thr if bool(self.stats_loaded.item()) else torch.tensor(50/255.0, device=rgb.device)
        h_low = self.h_low if bool(self.stats_loaded.item()) else torch.tensor(65/179.0, device=rgb.device)
        h_high = self.h_high if bool(self.stats_loaded.item()) else torch.tensor(133/179.0, device=rgb.device)
        v_low = self.v_low if bool(self.stats_loaded.item()) else torch.tensor(100/255.0, device=rgb.device)
        v_high = self.v_high if bool(self.stats_loaded.item()) else torch.tensor(180/255.0, device=rgb.device)
        temp = self.smooth_temp
        # S 选低饱和度: mask = sigmoid((s_thr - s)/t)  (与原逻辑 s<阈值 相仿)
        s_mask = torch.sigmoid((s_thr - s)/temp).unsqueeze(1)
        # Hue wrap-around handling on [0,1): if h_low>h_high, take union of [h_low,1] and [0,h_high]
        if (h_low > h_high).item():
            one = torch.ones_like(h)
            zero = torch.zeros_like(h)
            h_mask_a = self._smooth_mask_range(h, h_low, one, temp)
            h_mask_b = self._smooth_mask_range(h, zero, h_high, temp)
            h_mask = torch.maximum(h_mask_a, h_mask_b).unsqueeze(1)
        else:
            h_mask = self._smooth_mask_range(h, h_low, h_high, temp).unsqueeze(1)
        v_mask = self._smooth_mask_range(v, v_low, v_high, temp).unsqueeze(1)
        # 灰度 (与 RGB 统一, 避免 Value 过敏)
        gray = 0.299*rgb[:,0:1] + 0.587*rgb[:,1:2] + 0.114*rgb[:,2:3]
        # Sobel 梯度与方向
        gx = F.conv2d(gray, self._sobel_kernel_x().to(gray.device), padding=1)
        gy = F.conv2d(gray, self._sobel_kernel_y().to(gray.device), padding=1)
        grad_mag = torch.sqrt(gx*gx + gy*gy + 1e-8)
        angle = torch.atan2(gy, gx + 1e-8)
        dir_cos = torch.cos(angle)
        dir_sin = torch.sin(angle)
        lap = F.conv2d(gray, self._laplacian_kernel().to(gray.device), padding=1).abs()
        def _norm(t):
            return (t - t.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]) / (t.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0] + 1e-8)
        grad_mag = _norm(grad_mag)
        lap = _norm(lap)
        alt = torch.cat([s_mask, h_mask, v_mask, grad_mag, dir_cos, dir_sin, gray, lap], dim=1)
        if self.channel_weights:
            w = torch.tensor(self.channel_weights, dtype=alt.dtype, device=alt.device).view(1,-1,1,1)
            if w.numel() == alt.shape[1]:
                alt = alt * w
        return alt

    # ---- Kernels ----
    @staticmethod
    def _sobel_kernel_x():
        k = torch.tensor([[[-1,0,1],[-2,0,2],[-1,0,1]]], dtype=torch.float32).unsqueeze(0)
        return k
    @staticmethod
    def _sobel_kernel_y():
        k = torch.tensor([[[-1,-2,-1],[0,0,0],[1,2,1]]], dtype=torch.float32).unsqueeze(0)
        return k
    @staticmethod
    def _laplacian_kernel():
        k = torch.tensor([[ [0,1,0],[1,-4,1],[0,1,0] ]], dtype=torch.float32).unsqueeze(0)
        return k


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, use_se=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        if in_ch != out_ch or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_ch)
            )
        else:
            self.downsample = None
        self.use_se = use_se
        if use_se:
            self.se = SqueezeExcite(out_ch)
        else:
            self.se = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        if self.se is not None:
            out = self.se(out)
        out = self.relu(out)
        return out


class SqueezeExcite(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1)

    def forward(self, x):
        w = F.adaptive_avg_pool2d(x, 1)
        w = F.relu(self.fc1(w), inplace=True)
        w = torch.sigmoid(self.fc2(w))
        return x * w


class ConditionalSE(nn.Module):
    """Conditional SE: produce channel weights for target features using a conditioning vector
    (for example, global pooled alt features).
    """
    def __init__(self, in_channels, cond_dim, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(cond_dim, max(8, in_channels // reduction))
        self.fc2 = nn.Linear(max(8, in_channels // reduction), in_channels)

    def forward(self, x, cond):
        # x: (B,C,H,W), cond: (B, cond_dim)
        w = F.relu(self.fc1(cond), inplace=True)
        w = torch.sigmoid(self.fc2(w)).unsqueeze(-1).unsqueeze(-1)
        return x * w


class Encoder(nn.Module):
    def __init__(self, in_ch, base=16, use_se=False):
        super().__init__()
        self.enc1 = ResidualBlock(in_ch, base, use_se=use_se)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ResidualBlock(base, base*2, use_se=use_se)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ResidualBlock(base*2, base*4, use_se=use_se)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ResidualBlock(base*4, base*8, use_se=use_se)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool1(x1))
        x3 = self.enc3(self.pool2(x2))
        x4 = self.enc4(self.pool3(x3))
        return x1, x2, x3, x4


class Decoder(nn.Module):
    def __init__(self, out_ch, base=16):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(base*16, base*8, kernel_size=2, stride=2)
        self.dec1 = ResidualBlock(base*16, base*8)
        self.up2 = nn.ConvTranspose2d(base*8, base*4, kernel_size=2, stride=2)
        self.dec2 = ResidualBlock(base*8, base*4)
        self.up3 = nn.ConvTranspose2d(base*4, base*2, kernel_size=2, stride=2)
        self.dec3 = ResidualBlock(base*4, base*2)
        self.head = nn.Conv2d(base*2, out_ch, kernel_size=1)

    def forward(self, b, r3h3, r2h2, r1h1):
        u1 = self.up1(b)
        if u1.shape[-2:] != r3h3.shape[-2:]:
            u1 = F.interpolate(u1, size=r3h3.shape[-2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([u1, r3h3], dim=1))
        u2 = self.up2(d1)
        if u2.shape[-2:] != r2h2.shape[-2:]:
            u2 = F.interpolate(u2, size=r2h2.shape[-2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([u2, r2h2], dim=1))
        u3 = self.up3(d2)
        if u3.shape[-2:] != r1h1.shape[-2:]:
            u3 = F.interpolate(u3, size=r1h1.shape[-2:], mode='bilinear', align_corners=False)
        d3 = self.dec3(torch.cat([u3, r1h1], dim=1))
        return self.head(d3)


class ResNetEncoder(nn.Module):
    """Wrap a torchvision ResNet (18/34/50/101) to provide encoder features (layer1..layer4).
    Returns feature maps (f1,f2,f3,f4) corresponding to ResNet's layer1..layer4 outputs.
    """
    def __init__(self, backbone='resnet18', pretrained=False):
        super().__init__()
        # support resnet18, resnet34, resnet50 and resnet101
        self.pretrained_loaded = False
        self.loaded_from_path = None
        m = None
        # 1) Prefer explicit local weights when pretrained=True to avoid downloads
        if pretrained:
            try:
                repo_root = os.path.dirname(os.path.abspath(__file__))
                weights_dir = os.path.join(repo_root, 'experiments', 'weights')
                # If common/ is at repo root, this resolves to <repo>/experiments/weights
                if not os.path.isdir(weights_dir):
                    # try sibling path when running from within experiments/
                    weights_dir = os.path.join(os.path.dirname(repo_root), 'experiments', 'weights')
                fname_map = {
                    'resnet18': 'resnet18.pth',
                    'resnet34': 'resnet34.pth',
                    'resnet50': 'resnet50.pth',
                    'resnet101': 'resnet101.pth',
                }
                wfile = os.path.join(weights_dir, fname_map.get(backbone, 'resnet34.pth'))
                # Build model without downloading weights
                if _HAS_TV_WEIGHTS_ENUM:
                    if backbone == 'resnet34':
                        m = tv_models.resnet34(weights=None)
                    elif backbone == 'resnet50':
                        m = tv_models.resnet50(weights=None)
                    elif backbone == 'resnet101':
                        m = tv_models.resnet101(weights=None)
                    else:
                        m = tv_models.resnet18(weights=None)
                else:
                    if backbone == 'resnet34':
                        m = tv_models.resnet34(pretrained=False)
                    elif backbone == 'resnet50':
                        m = tv_models.resnet50(pretrained=False)
                    elif backbone == 'resnet101':
                        m = tv_models.resnet101(pretrained=False)
                    else:
                        m = tv_models.resnet18(pretrained=False)
                if os.path.isfile(wfile):
                    state = torch.load(wfile, map_location='cpu')
                    try:
                        m.load_state_dict(state, strict=False)
                        self.pretrained_loaded = True
                        self.loaded_from_path = wfile
                    except Exception:
                        # try nested key formats
                        if isinstance(state, dict) and 'state_dict' in state:
                            m.load_state_dict(state['state_dict'], strict=False)
                            self.pretrained_loaded = True
                            self.loaded_from_path = wfile
                # If local file missing, we'll fall through to random init (no download)
            except Exception:
                # Any failure here should not break construction; proceed to next branch
                m = None

        # 2) If not using local or failed, avoid downloads by constructing random-initialized model
        if m is None:
            try:
                if _HAS_TV_WEIGHTS_ENUM:
                    if backbone == 'resnet34':
                        m = tv_models.resnet34(weights=None)
                    elif backbone == 'resnet50':
                        m = tv_models.resnet50(weights=None)
                    elif backbone == 'resnet101':
                        m = tv_models.resnet101(weights=None)
                    else:
                        m = tv_models.resnet18(weights=None)
                else:
                    if backbone == 'resnet34':
                        m = tv_models.resnet34(pretrained=False)
                    elif backbone == 'resnet50':
                        m = tv_models.resnet50(pretrained=False)
                    elif backbone == 'resnet101':
                        m = tv_models.resnet101(pretrained=False)
                    else:
                        m = tv_models.resnet18(pretrained=False)
            except Exception:
                # As last resort, pick resnet18 no-pretrain
                try:
                    if _HAS_TV_WEIGHTS_ENUM:
                        m = tv_models.resnet18(weights=None)
                    else:
                        m = tv_models.resnet18(pretrained=False)
                except Exception:
                    raise
        # reuse resnet's early layers
        self.conv1 = m.conv1
        self.bn1 = m.bn1
        self.relu = m.relu
        self.maxpool = m.maxpool
        self.layer1 = m.layer1
        self.layer2 = m.layer2
        self.layer3 = m.layer3
        self.layer4 = m.layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        f1 = self.layer1(x)  # typically 64 channels
        f2 = self.layer2(f1)  # typically 128
        f3 = self.layer3(f2)  # typically 256
        f4 = self.layer4(f3)  # typically 512
        return f1, f2, f3, f4


class AltProjection(nn.Module):
    """Project multi-channel alt input to 3 channels so it can be fed to a pretrained ResNet encoder.
    Simple 1x1 conv + BN + ReLU.
    """
    def __init__(self, in_ch, out_ch=3):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.proj(x)


class _UpBlock(nn.Module):
    """Upsampling block: either transposed-conv (deconv) or bilinear+conv.

    Default now uses bilinear upsampling for smoother wide-region reconstruction.
    """
    def __init__(self, in_ch: int, out_ch: int, mode: str = 'bilinear'):
        super().__init__()
        mode = (mode or 'bilinear').lower()
        if mode == 'bilinear':
            # bilinear upsample by 2 then refine with Conv-BN-ReLU
            self.op = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
        else:
            self.op = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)

    def forward(self, x):
        return self.op(x)


class DecoderResNet(nn.Module):
    """Decoder compatible with ResNet-style encoder where encoder outputs are
    channels = [64,128,256,512] and when dual-branch concatenation doubles them.
    """
    def __init__(self, out_ch: int, use_se: bool = False, up_mode: str = 'bilinear'):
        super().__init__()
        # after concatenation of two encoders: layer4 -> 512*2 = 1024
        self.up1 = _UpBlock(1024, 512, mode=up_mode)
        self.dec1 = ResidualBlock(1024, 512, use_se=use_se)
        self.up2 = _UpBlock(512, 256, mode=up_mode)
        self.dec2 = ResidualBlock(512, 256, use_se=use_se)
        self.up3 = _UpBlock(256, 128, mode=up_mode)
        self.dec3 = ResidualBlock(256, 128, use_se=use_se)
        self.head = nn.Conv2d(128, out_ch, kernel_size=1)

    def forward(self, b, r3h3, r2h2, r1h1):
        u1 = self.up1(b)
        if u1.shape[-2:] != r3h3.shape[-2:]:
            u1 = F.interpolate(u1, size=r3h3.shape[-2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([u1, r3h3], dim=1))
        u2 = self.up2(d1)
        if u2.shape[-2:] != r2h2.shape[-2:]:
            u2 = F.interpolate(u2, size=r2h2.shape[-2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([u2, r2h2], dim=1))
        u3 = self.up3(d2)
        if u3.shape[-2:] != r1h1.shape[-2:]:
            u3 = F.interpolate(u3, size=r1h1.shape[-2:], mode='bilinear', align_corners=False)
        d3 = self.dec3(torch.cat([u3, r1h1], dim=1))
        return self.head(d3)


class LiteASPP(nn.Module):
    """Lightweight ASPP-like context module (minimal integration, no strip pooling).
    Splits features into 4 branches (1x1, and depthwise 3x3 with dilation 1/2/3) producing
    equal-sized channel chunks which are concatenated and fused.
    """
    def __init__(self, in_ch: int, out_ch: int, inter_ratio: int = 4):
        super().__init__()
        c = max(32, out_ch // inter_ratio)
        self.b1 = nn.Sequential(
            nn.Conv2d(in_ch, c, kernel_size=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, c, kernel_size=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )
        self.b3 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=2, dilation=2, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, c, kernel_size=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )
        self.b4 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=3, dilation=3, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, c, kernel_size=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(c*4, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [self.b1(x), self.b2(x), self.b3(x), self.b4(x)]
        y = torch.cat(feats, dim=1)
        return self.fuse(y)


class FullASPP(nn.Module):
    """Full ASPP (DeepLabv3-like) with image-level pooling branch.
    in_ch -> several parallel branches -> concat -> 1x1 fuse -> out_ch
    """
    def __init__(self, in_ch: int, out_ch: int, inter_channels: int = 256, rates=(1, 6, 12, 18)):
        super().__init__()
        self.rates = tuple(rates)
        inter = int(inter_channels)
        # 1x1 branch
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_ch, inter, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter),
            nn.ReLU(inplace=True),
        )
        # atrous branches (3x3 with dilation) -- skip the first rate if it's 1 (we already have 1x1)
        self.branches = nn.ModuleList()
        for r in self.rates:
            if r == 1:
                # prefer the 1x1 branch for rate==1
                continue
            self.branches.append(nn.Sequential(
                nn.Conv2d(in_ch, inter, kernel_size=3, padding=r, dilation=r, bias=False),
                nn.BatchNorm2d(inter),
                nn.ReLU(inplace=True),
            ))
        # image pooling branch
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, inter, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter),
            nn.ReLU(inplace=True),
        )
        # fuse
        total = inter * (1 + len(self.branches) + 1)
        self.fuse = nn.Sequential(
            nn.Conv2d(total, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2], x.shape[-1]
        feats = [self.branch1(x)]
        for br in self.branches:
            feats.append(br(x))
        pooled = self.image_pool(x)
        pooled = F.interpolate(pooled, size=(h, w), mode='bilinear', align_corners=False)
        feats.append(pooled)
        y = torch.cat(feats, dim=1)
        return self.fuse(y)


class Decoder(nn.Module):
    def __init__(self, out_ch, base=16, up_mode: str = 'bilinear'):
        super().__init__()
        self.up1 = _UpBlock(base*16, base*8, mode=up_mode)
        self.dec1 = ResidualBlock(base*16, base*8)
        self.up2 = _UpBlock(base*8, base*4, mode=up_mode)
        self.dec2 = ResidualBlock(base*8, base*4)
        self.up3 = _UpBlock(base*4, base*2, mode=up_mode)
        self.dec3 = ResidualBlock(base*4, base*2)
        self.head = nn.Conv2d(base*2, out_ch, kernel_size=1)

    def forward(self, b, r3h3, r2h2, r1h1):
        u1 = self.up1(b)
        if u1.shape[-2:] != r3h3.shape[-2:]:
            u1 = F.interpolate(u1, size=r3h3.shape[-2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([u1, r3h3], dim=1))
        u2 = self.up2(d1)
        if u2.shape[-2:] != r2h2.shape[-2:]:
            u2 = F.interpolate(u2, size=r2h2.shape[-2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([u2, r2h2], dim=1))
        u3 = self.up3(d2)
        if u3.shape[-2:] != r1h1.shape[-2:]:
            u3 = F.interpolate(u3, size=r1h1.shape[-2:], mode='bilinear', align_corners=False)
        d3 = self.dec3(torch.cat([u3, r1h1], dim=1))
        return self.head(d3)


class BiResUnetPlus(nn.Module):
    """Dual-branch ResUNet++-like model. Two encoders (RGB & second modality),
    concatenate features at each level and decode.

    Parameters:
      out_ch: output channels (e.g., 1 or 2)
      backbone: None (use lightweight internal Encoder) or 'resnet18' or 'resnet34'
      pretrained: bool, whether to use pretrained weights for resnet backbone
      base/use_se: used only for the lightweight internal encoder (when backbone is None)

    Default: backbone='resnet18' (per request)
    """
    def __init__(self, out_ch=2, base=16, use_se=False, backbone='resnet18', pretrained=False,
                 decoder_use_se: bool = False, upsample_mode: str = 'bilinear', use_lite_aspp: bool = True,
                 auto_alt_mode: str = None, hsv_stats_path: str = None,
                 v_window_mode: str = 'mad', s_thr_mode: str = 'low', smooth_temp: float = 4.0,
                 channel_weights: Tuple[float, ...] = None,
                 use_eem: bool = False,
                 eem_levels: Tuple[int, ...] = None,
                 eem_apply_rgb: bool = True,
                 eem_apply_alt: bool = True,
                 eem_reduction: int = 2):
        super().__init__()
        self.backbone = backbone
        self.use_eem = use_eem
        # optional in-graph ALT builder (when alt is None at forward)
        self.auto_alt_mode = (auto_alt_mode or '').lower().strip() or None
        self.hsv_stats_path = hsv_stats_path
        self.v_window_mode = v_window_mode
        self.s_thr_mode = s_thr_mode
        self.smooth_temp = float(smooth_temp)
        self.channel_weights = channel_weights
        self.alt_builder: nn.Module = None
        if self.auto_alt_mode in ('hsvgrad', 'rgbgrad'):
            self.alt_builder = DifferentiableAltBuilder(
                mode=self.auto_alt_mode, stats_path=self.hsv_stats_path,
                v_window_mode=self.v_window_mode, s_thr_mode=self.s_thr_mode,
                smooth_temp=self.smooth_temp, channel_weights=self.channel_weights
            )
        if self.use_eem:
            from common.eem import EEM
            # 动态 kernel size 和 groups，可根据输入尺寸或配置参数调整
            kernel_size = 5 if hasattr(self, 'input_size') and max(self.input_size) > 800 else 3
            groups = 1
            # 默认层级：前3层
            if eem_levels is None:
                levels = (1, 2, 3)
            else:
                # sanitize levels
                try:
                    levels = tuple(sorted({int(x) for x in eem_levels if int(x) in (1, 2, 3, 4)}))
                except Exception:
                    levels = (1, 2, 3)
            self._eem_levels = tuple(levels)
            # modules stored in ModuleDict for flexible lookup; also set legacy attributes eem1.. to preserve compatibility
            self.eem_modules = nn.ModuleDict()
            self.alt_eem_modules = nn.ModuleDict()
            # channel mapping depends on backbone vs lightweight encoder
            if backbone in ('resnet18', 'resnet34', 'resnet50', 'resnet101'):
                ch_map = {1: 64, 2: 128, 3: 256, 4: 512}
            else:
                ch_map = {1: base, 2: base * 2, 3: base * 4, 4: base * 8}
            for lvl in self._eem_levels:
                ch = ch_map.get(lvl, base * (2 ** (lvl - 1)))
                if eem_apply_rgb:
                    m = EEM(ch_in=ch, ch_out=ch, kernel=kernel_size, groups=groups, reduction=eem_reduction)
                    self.eem_modules[f'r{lvl}'] = m
                    setattr(self, f'eem{lvl}', m)
                if eem_apply_alt:
                    ma = EEM(ch_in=ch, ch_out=ch, kernel=kernel_size, groups=groups, reduction=eem_reduction)
                    self.alt_eem_modules[f'a{lvl}'] = ma
                    setattr(self, f'alt_eem{lvl}', ma)
        if backbone in ('resnet18', 'resnet34', 'resnet50', 'resnet101'):
            # use ResNet-based encoder
            self.rgb_enc = ResNetEncoder(backbone=backbone, pretrained=pretrained)
            # alt may be multi-channel; we'll project to 3 channels before feeding into ResNet
            # alt_proj will be created by caller if needed; default assume 8-channel alt
            self.alt_proj_in: nn.Module = AltProjection(in_ch=8, out_ch=3)
            self.alt_enc = ResNetEncoder(backbone=backbone, pretrained=pretrained)
            # effective pretrained status (both encoders loaded pretrained weights)
            self.pretrained_effective = bool(getattr(self.rgb_enc, 'pretrained_loaded', False) and getattr(self.alt_enc, 'pretrained_loaded', False))
            # If we're using a larger resnet (resnet50/101) its layer outputs are larger
            # (e.g., [256,512,1024,2048]). To keep the existing decoder that expects
            # [64,128,256,512] we add 1x1 projections that reduce channels to the
            # expected sizes. For resnet18/34 these projections are identity.
            if backbone in ('resnet50', 'resnet101'):
                # projection convs for rgb features
                self.rgb_level_proj = nn.ModuleDict({
                    'r1': nn.Conv2d(256, 64, kernel_size=1),
                    'r2': nn.Conv2d(512, 128, kernel_size=1),
                    'r3': nn.Conv2d(1024, 256, kernel_size=1),
                    'r4': nn.Conv2d(2048, 512, kernel_size=1),
                })
                # projection convs for alt features
                self.alt_level_proj = nn.ModuleDict({
                    'a1': nn.Conv2d(256, 64, kernel_size=1),
                    'a2': nn.Conv2d(512, 128, kernel_size=1),
                    'a3': nn.Conv2d(1024, 256, kernel_size=1),
                    'a4': nn.Conv2d(2048, 512, kernel_size=1),
                })
            else:
                # identity projections (no-op convs) for uniform code path
                self.rgb_level_proj = None
                self.alt_level_proj = None
            # concatenated channel sizes for resnet: layer4 -> 512 -> doubled -> 1024
            self.bottleneck = ResidualBlock(512*2, 512*2, use_se=use_se)
            # choose lightweight or full ASPP according to flag
            if use_lite_aspp:
                self.aspp = LiteASPP(512*2, 512*2)
            else:
                # FullASPP uses a fixed intermediate channel count (256) which works well for ResNet path
                self.aspp = FullASPP(512*2, 512*2, inter_channels=256, rates=(1,6,12,18))
            self.cond_se = ConditionalSE(in_channels=512*2, cond_dim=512)  # cond_dim from pooled alt a4
            self.decoder = DecoderResNet(out_ch, use_se=bool(decoder_use_se), up_mode=str(upsample_mode))
        else:
            # fallback to lightweight custom encoder/decoder
            self.rgb_enc = Encoder(3, base=base, use_se=use_se)
            self.alt_enc = Encoder(3, base=base, use_se=use_se)
            self.bottleneck = ResidualBlock(base*16, base*16, use_se=use_se)
            if use_lite_aspp:
                self.aspp = LiteASPP(base*16, base*16)
            else:
                # scale intermediate channels with base for lightweight encoder path
                inter = max(64, (base * 16) // 4)
                self.aspp = FullASPP(base*16, base*16, inter_channels=inter, rates=(1,6,12,18))
            self.decoder = Decoder(out_ch, base=base, up_mode=str(upsample_mode))
            self.pretrained_effective = False

    def forward(self, rgb, alt=None):
        # rgb: (B,3,H,W), alt: (B,C_alt,H,W)
        if alt is None and getattr(self, 'alt_builder', None) is not None:
            # build ALT inside graph to guarantee train/infer consistency
            alt = self.alt_builder(rgb)
        r1, r2, r3, r4 = self.rgb_enc(rgb)
        if self.use_eem:
            # apply EEM to configured rgb levels if modules exist
            rmap = {1: r1, 2: r2, 3: r3, 4: r4}
            for lvl in getattr(self, '_eem_levels', (1, 2, 3)):
                key = f'r{lvl}'
                if hasattr(self, 'eem_modules') and key in self.eem_modules:
                    rmap[lvl] = self.eem_modules[key](rmap[lvl])
            r1, r2, r3, r4 = rmap[1], rmap[2], rmap[3], rmap[4]
        if self.backbone in ('resnet18', 'resnet34', 'resnet50', 'resnet101'):
            # ensure alt projected to 3 channels for input to the ResNet encoder
            if alt.shape[1] != 3:
                alt_p = self.alt_proj_in(alt)
            else:
                alt_p = alt
            a1, a2, a3, a4 = self.alt_enc(alt_p)
            if self.use_eem:
                amap = {1: a1, 2: a2, 3: a3, 4: a4}
                for lvl in getattr(self, '_eem_levels', (1, 2, 3)):
                    key = f'a{lvl}'
                    if hasattr(self, 'alt_eem_modules') and key in self.alt_eem_modules:
                        amap[lvl] = self.alt_eem_modules[key](amap[lvl])
                a1, a2, a3, a4 = amap[1], amap[2], amap[3], amap[4]
            # if we used resnet50/101 we need to project large-channel features down
            if self.backbone in ('resnet50', 'resnet101'):
                r1 = self.rgb_level_proj['r1'](r1)
                r2 = self.rgb_level_proj['r2'](r2)
                r3 = self.rgb_level_proj['r3'](r3)
                r4 = self.rgb_level_proj['r4'](r4)
                a1 = self.alt_level_proj['a1'](a1)
                a2 = self.alt_level_proj['a2'](a2)
                a3 = self.alt_level_proj['a3'](a3)
                a4 = self.alt_level_proj['a4'](a4)
            cat4 = torch.cat([r4, a4], dim=1)
            b = self.bottleneck(cat4)
            if getattr(self, 'aspp', None) is not None:
                b = self.aspp(b)
            # conditional SE using pooled alt_a4
            cond = F.adaptive_avg_pool2d(a4, 1).view(a4.size(0), -1)
            b = self.cond_se(b, cond)
        else:
            a1, a2, a3, a4 = self.alt_enc(alt)
            if self.use_eem:
                amap = {1: a1, 2: a2, 3: a3, 4: a4}
                for lvl in getattr(self, '_eem_levels', (1, 2, 3)):
                    key = f'a{lvl}'
                    if hasattr(self, 'alt_eem_modules') and key in self.alt_eem_modules:
                        amap[lvl] = self.alt_eem_modules[key](amap[lvl])
                a1, a2, a3, a4 = amap[1], amap[2], amap[3], amap[4]
            cat4 = torch.cat([r4, a4], dim=1)
            b = self.bottleneck(cat4)
            if getattr(self, 'aspp', None) is not None:
                b = self.aspp(b)
        merge3 = torch.cat([r3, a3], dim=1)
        merge2 = torch.cat([r2, a2], dim=1)
        merge1 = torch.cat([r1, a1], dim=1)
        out = self.decoder(b, merge3, merge2, merge1)
        # 保证输出尺寸与输入一致
        out = F.interpolate(out, size=rgb.shape[2:], mode='bilinear', align_corners=False)
        return out


def load_biresunet_checkpoint(model, checkpoint_path, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state = torch.load(checkpoint_path, map_location=device)
    try:
        model.load_state_dict(state)
    except Exception:
        model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


def infer_biresunet_on_image(model, image_path, downscale=None, device=None,
                             hsv_mode: str = None, hsv_stats_path: str = None,
                             return_intermediate: bool = False,
                             build_mode: str = None):
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError('failed to load '+image_path)
    h0,w0 = img.shape[:2]
    if downscale:
        maxd = max(h0,w0)
        if maxd > downscale:
            s = downscale/float(maxd)
            img = cv2.resize(img, (int(w0*s), int(h0*s)), interpolation=cv2.INTER_LINEAR)
    # OpenCV 读取为 BGR，这里统一转换为 RGB01 以匹配可微 HSV 与 ResNet 预训练
    rgb = img[:, :, ::-1].astype(np.float32)/255.0
    # If model has in-graph builder and user wants hsvgrad/rgbgrad, let model handle ALT.
    model_build_cap = getattr(model, 'alt_builder', None) is not None
    build_mode_env = os.environ.get('ALT_BUILD_MODE', '').strip().lower()
    build_mode = (build_mode or build_mode_env or ('hsvgrad' if model_build_cap else 'legacy_hsv_cv')).lower()
    hsv_mode_env = os.environ.get('ALT_HSV_MODE', '').strip().lower()
    hsv_mode = (hsv_mode or hsv_mode_env or 'fixed').lower()
    # Fast path: in-graph differentiable build -> alt=None
    if model_build_cap and build_mode in ('hsvgrad','rgbgrad'):
        device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        rgb_t = torch.from_numpy(rgb.transpose(2,0,1)).unsqueeze(0).float().to(device)
        model.to(device)
        with torch.no_grad():
            logits = model(rgb_t, alt=None)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
        if return_intermediate:
            return {
                'image': img,
                'probs': probs,
                'alt_mode': build_mode,
                'hsv_mode': hsv_mode,
            }
        return img, probs
    # Legacy OpenCV path (previous behavior) constructing HSV+geometry ALT
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    if hsv_mode == 'adaptive' and AdaptiveHSVExtractor is not None:
        extractor = AdaptiveHSVExtractor(stats_path=hsv_stats_path)
        s_mask, h_mask, v_mask = extractor.build_masks(img)
    else:
        # fixed heuristic (original)
        s_mask = cv2.GaussianBlur((s.astype(np.float32) < 50).astype(np.float32), (5,5), 0)
        h_mask = ((h >= 65) & (h <= 133)).astype(np.float32)
        h_mask = cv2.GaussianBlur(h_mask, (5,5), 0)
        v_blur = cv2.GaussianBlur(v, (5,5), 0)
        hist = cv2.calcHist([v_blur], [0], None, [256], [0,256]).flatten()
        peak = int(np.argmax(hist))
        peak_min = max(0, peak - 15)
        peak_max = min(255, peak + 15)
        v_mask = ((v_blur >= peak_min) & (v_blur <= peak_max)).astype(np.float32)
        v_mask = cv2.GaussianBlur(v_mask, (5,5), 0)
    # edge map construction: use Sobel-based gradient magnitude
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def _edge_sobel(gray_: np.ndarray) -> np.ndarray:
        sx_ = cv2.Sobel(gray_, cv2.CV_32F, 1, 0, ksize=3)
        sy_ = cv2.Sobel(gray_, cv2.CV_32F, 0, 1, ksize=3)
        ed = np.sqrt(sx_*sx_ + sy_*sy_)
        return (ed - ed.min()) / (ed.max() - ed.min() + 1e-8)

    # Always use Sobel for edge extraction (mcanny multi-scale Canny removed)
    edge = _edge_sobel(gray)
    # distance transform from HSV soft mask
    # In adaptive mode, derive soft_mask from adaptive S/H/V masks; else keep original heuristic
    if hsv_mode == 'adaptive' and AdaptiveHSVExtractor is not None:
        try:
            # ensure masks exist; combine via AND for a conservative prior
            s_bin = (s_mask > 0.5).astype(np.uint8)
            h_bin = (h_mask > 0.5).astype(np.uint8)
            v_bin = (v_mask > 0.5).astype(np.uint8)
            sm = (s_bin & h_bin & v_bin).astype(np.uint8)
            soft_mask = cv2.GaussianBlur(sm.astype(np.float32), (5,5), 0)
        except Exception:
            # fallback to fixed heuristic if anything goes wrong
            soft_mask = cv2.GaussianBlur(((s < 50).astype(np.uint8) * ((h >= 65) & (h <= 133)).astype(np.uint8)), (5,5), 0)
    else:
        soft_mask = cv2.GaussianBlur(((s < 50).astype(np.uint8) * ((h >= 65) & (h <= 133)).astype(np.uint8)), (5,5), 0)
    dt = cv2.distanceTransform((soft_mask*255).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
    dt = dt / (dt.max() + 1e-8)
    # direction: always derive from Sobel gradients (independent of chosen edge_mode)
    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gx = sx
    gy = sy
    angle = np.arctan2(gy, gx + 1e-8)
    dir_cos = np.cos(angle).astype(np.float32)
    dir_sin = np.sin(angle).astype(np.float32)
    # assemble alt channels (order: s_mask, h_mask, v_mask, edge, dt, dir_cos, dir_sin, gray_norm)
    gray_n = (gray.astype(np.float32) - gray.min()) / (gray.max() - gray.min() + 1e-8)
    alt_channels = [s_mask, h_mask, v_mask, edge.astype(np.float32), dt.astype(np.float32), dir_cos, dir_sin, gray_n]
    # Optional ablation/weighting: ALT_CHANNEL_WEIGHTS="w1,w2,...,w8"; 0 disables a channel
    ch_w_env = os.environ.get('ALT_CHANNEL_WEIGHTS', '').strip()
    if ch_w_env:
        try:
            w = [float(x) for x in ch_w_env.split(',')]
            if len(w) == len(alt_channels):
                for i in range(len(alt_channels)):
                    alt_channels[i] = (alt_channels[i].astype(np.float32) * w[i]).astype(np.float32)
        except Exception:
            pass
    else:
        # 默认未显式给权重且走 legacy hsv_cv 时也对 H 通道做 0.85 下调保持与可微路径一致
        try:
            alt_channels[1] = (alt_channels[1].astype(np.float32) * 0.85).astype(np.float32)
        except Exception:
            pass
    alt = np.stack(alt_channels, axis=2)
    # If multi-scale Canny was used, retain same channel order; downstream AltProjection expects 8 channels.
    rgb_t = torch.from_numpy(rgb.transpose(2,0,1)).unsqueeze(0).float()
    alt_t = torch.from_numpy(alt.transpose(2,0,1)).unsqueeze(0).float()
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    model.to(device)
    with torch.no_grad():
        rgb_t = rgb_t.to(device)
        alt_t = alt_t.to(device)
        logits = model(rgb_t, alt_t)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()
    if return_intermediate:
        return {
            'image': img,
            'probs': probs,
            'alt': alt,
            'hsv_mode': hsv_mode,
            'alt_mode': build_mode,
        }
    return img, probs


__all__ = ['BiResUnetPlus', 'load_biresunet_checkpoint', 'infer_biresunet_on_image', 'DifferentiableAltBuilder']
