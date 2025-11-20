"""Lightweight segmentation model for StageA-light.

This file implements a small encoder using depthwise separable convolutions
and a compact decoder producing a single-channel logits output.

Design goals:
- Use standard nn.Conv2d / bn / relu and depthwise conv (groups==in_channels)
- Avoid custom ops so ONNX / TensorRT compatibility is straightforward
- Small parameter count and low FLOPs for Jetson Nano deployment

Usage:
  from common.light_models import LightSegNet
  model = LightSegNet(base_ch=16, input_channels=3)
  logits = model(x)  # logits shape (B,1,H,W)

This model is intended to be exported to ONNX then converted to TensorRT.
Keep input resolution fixed for best TensorRT performance. Recommended deployment shape
for StageA-light is 544 (H) x 960 (W) — both dimensions are divisible by 16 which
matches the network's total downsample factor and avoids internal padding issues.
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable conv: depthwise conv followed by pointwise conv."""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, stride=stride,
                            padding=padding, groups=in_ch, bias=bias)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.dw(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pw(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class SimpleDownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pool=True):
        super().__init__()
        self.conv = DepthwiseSeparableConv(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2, 2) if pool else None

    def forward(self, x):
        x = self.conv(x)
        if self.pool is not None:
            x = self.pool(x)
        return x


class SimpleUpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # use bilinear upsample + DWConv to keep ops simple
        self.conv = DepthwiseSeparableConv(in_ch, out_ch)

    def forward(self, x, skip: Optional[torch.Tensor] = None):
        x = F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=False)
        if skip is not None:
            # pad if shapes mismatch
            if skip.size(2) != x.size(2) or skip.size(3) != x.size(3):
                x = F.pad(x, [0, skip.size(3) - x.size(3), 0, skip.size(2) - x.size(2)])
            x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class LightSegNet(nn.Module):
    """Compact encoder-decoder producing a single binary segmentation logits map.

    Params:
      base_ch: base number of channels (will be scaled for deeper layers)
      input_channels: usually 3
    """
    def __init__(self, base_ch: int = 16, input_channels: int = 3):
        super().__init__()
        # Encoder: 4 stages, downsample by 2 each stage
        self.enc1 = SimpleDownBlock(input_channels, base_ch, pool=True)
        self.enc2 = SimpleDownBlock(base_ch, base_ch * 2, pool=True)
        self.enc3 = SimpleDownBlock(base_ch * 2, base_ch * 4, pool=True)
        self.enc4 = SimpleDownBlock(base_ch * 4, base_ch * 8, pool=True)

        # Bottleneck
        self.bottleneck = DepthwiseSeparableConv(base_ch * 8, base_ch * 8)

        # Decoder (skip connections)
        # note: after concatenation channel counts double for conv input
        self.up3 = SimpleUpBlock(base_ch * 8 + base_ch * 4, base_ch * 4)
        self.up2 = SimpleUpBlock(base_ch * 4 + base_ch * 2, base_ch * 2)
        self.up1 = SimpleUpBlock(base_ch * 2 + base_ch, base_ch)

        # final conv to 1 channel logits
        self.head = nn.Sequential(
            DepthwiseSeparableConv(base_ch, base_ch),
            nn.Conv2d(base_ch, 1, kernel_size=1)
        )

    def forward(self, x):
        # assume x in [0,1]
        e1 = self.enc1(x)   # 1/2
        e2 = self.enc2(e1)  # 1/4
        e3 = self.enc3(e2)  # 1/8
        e4 = self.enc4(e3)  # 1/16
        b = self.bottleneck(e4)
        u3 = self.up3(b, e3)
        u2 = self.up2(u3, e2)
        u1 = self.up1(u2, e1)
        out = self.head(u1)
        # return logits (B,1,H,W)
        return out


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # quick smoke test
    m = LightSegNet(base_ch=16, input_channels=3)
    x = torch.randn(1, 3, 512, 512)
    y = m(x)
    print('out', y.shape)
    print('params', count_params(m))
