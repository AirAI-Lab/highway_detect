import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
from torchvision import models as tv_models
from typing import Tuple

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
    """Wrap a torchvision ResNet (18/34) to provide encoder features (layer1..layer4).
    Returns feature maps (f1,f2,f3,f4) corresponding to ResNet's layer1..layer4 outputs.
    """
    def __init__(self, backbone='resnet18', pretrained=False):
        super().__init__()
        # support resnet18, resnet34 and resnet101
        if backbone == 'resnet34':
            m = tv_models.resnet34(pretrained=pretrained)
        elif backbone == 'resnet101':
            m = tv_models.resnet101(pretrained=pretrained)
        else:
            m = tv_models.resnet18(pretrained=pretrained)
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


class DecoderResNet(nn.Module):
    """Decoder compatible with ResNet-style encoder where encoder outputs are
    channels = [64,128,256,512] and when dual-branch concatenation doubles them.
    """
    def __init__(self, out_ch):
        super().__init__()
        # after concatenation of two encoders: layer4 -> 512*2 = 1024
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec1 = ResidualBlock(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec2 = ResidualBlock(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = ResidualBlock(256, 128)
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
    def __init__(self, out_ch=2, base=16, use_se=False, backbone='resnet18', pretrained=False):
        super().__init__()
        self.backbone = backbone
        if backbone in ('resnet18', 'resnet34', 'resnet101'):
            # use ResNet-based encoder
            self.rgb_enc = ResNetEncoder(backbone=backbone, pretrained=pretrained)
            # alt may be multi-channel; we'll project to 3 channels before feeding into ResNet
            # alt_proj will be created by caller if needed; default assume 8-channel alt
            self.alt_proj_in: nn.Module = AltProjection(in_ch=8, out_ch=3)
            self.alt_enc = ResNetEncoder(backbone=backbone, pretrained=pretrained)
            # If we're using a larger resnet (resnet101) its layer outputs are larger
            # (e.g., [256,512,1024,2048]). To keep the existing decoder that expects
            # [64,128,256,512] we add 1x1 projections that reduce channels to the
            # expected sizes. For resnet18/34 these projections are identity.
            if backbone == 'resnet101':
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
            self.cond_se = ConditionalSE(in_channels=512*2, cond_dim=512)  # cond_dim from pooled alt a4
            self.decoder = DecoderResNet(out_ch)
        else:
            # fallback to lightweight custom encoder/decoder
            self.rgb_enc = Encoder(3, base=base, use_se=use_se)
            self.alt_enc = Encoder(3, base=base, use_se=use_se)
            self.bottleneck = ResidualBlock(base*16, base*16, use_se=use_se)
            self.decoder = Decoder(out_ch, base=base)

    def forward(self, rgb, alt):
        # rgb: (B,3,H,W), alt: (B,C_alt,H,W)
        r1, r2, r3, r4 = self.rgb_enc(rgb)
        if self.backbone in ('resnet18', 'resnet34', 'resnet101'):
            # ensure alt projected to 3 channels for input to the ResNet encoder
            if alt.shape[1] != 3:
                alt_p = self.alt_proj_in(alt)
            else:
                alt_p = alt
            a1, a2, a3, a4 = self.alt_enc(alt_p)
            # if we used resnet101 we need to project large-channel features down
            if self.backbone == 'resnet101':
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
            # conditional SE using pooled alt_a4
            cond = F.adaptive_avg_pool2d(a4, 1).view(a4.size(0), -1)
            b = self.cond_se(b, cond)
        else:
            a1, a2, a3, a4 = self.alt_enc(alt)
            cat4 = torch.cat([r4, a4], dim=1)
            b = self.bottleneck(cat4)
        merge3 = torch.cat([r3, a3], dim=1)
        merge2 = torch.cat([r2, a2], dim=1)
        merge1 = torch.cat([r1, a1], dim=1)
        out = self.decoder(b, merge3, merge2, merge1)
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


def infer_biresunet_on_image(model, image_path, downscale=None, device=None):
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError('failed to load '+image_path)
    h0,w0 = img.shape[:2]
    if downscale:
        maxd = max(h0,w0)
        if maxd > downscale:
            s = downscale/float(maxd)
            img = cv2.resize(img, (int(w0*s), int(h0*s)), interpolation=cv2.INTER_LINEAR)
    rgb = img.astype(np.float32)/255.0
    # build multi-channel alt: soft HSV masks + edges + distance transform + direction
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    # soft s mask (low saturation -> likely road)
    s_mask = cv2.GaussianBlur((s.astype(np.float32) < 50).astype(np.float32), (5,5), 0)
    # H range mask
    h_mask = ((h >= 65) & (h <= 133)).astype(np.float32)
    h_mask = cv2.GaussianBlur(h_mask, (5,5), 0)
    # v peak masked (reuse original logic simplified)
    v_blur = cv2.GaussianBlur(v, (5,5), 0)
    hist = cv2.calcHist([v_blur], [0], None, [256], [0,256]).flatten()
    peak = int(np.argmax(hist))
    thr = max(1, int(hist[peak] * 0.1))
    # create peak mask around peak +-15
    peak_min = max(0, peak - 15)
    peak_max = min(255, peak + 15)
    v_mask = ((v_blur >= peak_min) & (v_blur <= peak_max)).astype(np.float32)
    v_mask = cv2.GaussianBlur(v_mask, (5,5), 0)
    # edge (Sobel magnitude)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.sqrt(sx*sx + sy*sy)
    edge = (edge - edge.min()) / (edge.max() - edge.min() + 1e-8)
    # distance transform from HSV soft mask
    soft_mask = cv2.GaussianBlur(((s < 50).astype(np.uint8) * ((h >= 65) & (h <= 133)).astype(np.uint8)), (5,5), 0)
    dt = cv2.distanceTransform((soft_mask*255).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
    dt = dt / (dt.max() + 1e-8)
    # direction: sobel based angle, encode as cos/sin
    gx = sx
    gy = sy
    angle = np.arctan2(gy, gx + 1e-8)
    dir_cos = np.cos(angle).astype(np.float32)
    dir_sin = np.sin(angle).astype(np.float32)
    # assemble alt channels (order: s_mask, h_mask, v_mask, edge, dt, dir_cos, dir_sin, gray_norm)
    gray_n = (gray.astype(np.float32) - gray.min()) / (gray.max() - gray.min() + 1e-8)
    alt = np.stack([s_mask, h_mask, v_mask, edge.astype(np.float32), dt.astype(np.float32), dir_cos, dir_sin, gray_n], axis=2)
    rgb_t = torch.from_numpy(rgb.transpose(2,0,1)).unsqueeze(0).float()
    alt_t = torch.from_numpy(alt.transpose(2,0,1)).unsqueeze(0).float()
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    model.to(device)
    with torch.no_grad():
        rgb_t = rgb_t.to(device)
        alt_t = alt_t.to(device)
        logits = model(rgb_t, alt_t)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()
    return img, probs


__all__ = ['BiResUnetPlus', 'load_biresunet_checkpoint', 'infer_biresunet_on_image']
