import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from functools import partial

from .weights_helper import load_resnet34_backbone

# Reuse relu nonlinearity pattern used across other network files
nonlinearity = partial(F.relu, inplace=True)

############################################################
# Channel & Dual Attention Modules (Corrected)
############################################################
class CDAM2(nn.Module):
    """Dual channel + spatial attention for 320-channel feature (5*64).
    Original file had duplicated forward and incorrect reshape using input h/w.
    We fix by:
      - Using fixed pooling sizes self.h/self.w ONLY for conv1d preparation.
      - Reshaping with those fixed sizes (not runtime h,w) to avoid invalid shape errors.
      - Keeping logic consistent with CDAM3/4.
    """
    def __init__(self, k_size=9):
        super(CDAM2, self).__init__()
        self.h = 256
        self.w = 256
        self.relu1 = nn.ReLU()
        # Pool to fixed sizes for 1D conv channel mixing; do NOT reshape to runtime h/w.
        self.avg_pool_x = nn.AdaptiveAvgPool2d((self.h, 1))
        self.avg_pool_y = nn.AdaptiveAvgPool2d((1, self.w))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 1D conv over fixed spatial lengths
        self.conv1 = nn.Conv1d(self.h, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.conv2 = nn.Conv1d(self.w, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.conv11 = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.conv22 = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.convout = nn.Conv2d(64 * 5 * 4, 64 * 5, kernel_size=3, padding=1, bias=False)
        self.conv111 = nn.Conv2d(in_channels=64 * 5 * 2, out_channels=64 * 5 * 2, kernel_size=1, padding=0, stride=1)
        self.conv222 = nn.Conv2d(in_channels=64 * 5 * 2, out_channels=64 * 5 * 2, kernel_size=1, padding=0, stride=1)
        # Horizontal / vertical attention convs depend on fixed pooling spatial sizes
        self.conv1h = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(self.h, 1), padding=(0, 0), stride=1)
        self.conv1s = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(1, self.w), padding=(0, 0), stride=1)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Conv1d)):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.size()
        # Pool along height (width collapsed) -> (n,c,256,1) and reshape for 1D conv over length=self.h
        y1 = self.avg_pool_x(x).reshape(n, c, self.h)
        y1 = self.sigmoid(self.conv11(self.relu1(self.conv1(y1.transpose(1, 2)))).transpose(1, 2).reshape(n, c, 1, 1))
        # Pool along width (height collapsed) -> (n,c,1,256)
        y2 = self.avg_pool_y(x).reshape(n, c, self.w)
        y2 = self.sigmoid(self.conv22(self.relu1(self.conv2(y2.transpose(1, 2)))).transpose(1, 2).reshape(n, c, 1, 1))
        yac = self.conv111(torch.cat([x * y1.expand_as(x), x * y2.expand_as(x)], dim=1))
        avg_mean = torch.mean(x, dim=1, keepdim=True)
        avg_max, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.cat([avg_max, avg_mean], dim=1)
        y3 = self.sigmoid(self.conv1h(avg_out))
        y4 = self.sigmoid(self.conv1s(avg_out))
        yap = self.conv222(torch.cat([x * y3.expand_as(x), x * y4.expand_as(x)], dim=1))
        out = self.convout(torch.cat([yac, yap], dim=1))
        return out

class CDAM3(nn.Module):
    def __init__(self, k_size=7):
        super(CDAM3, self).__init__()
        self.h = 128
        self.w = 128
        self.relu1 = nn.ReLU()
        self.avg_pool_x = nn.AdaptiveAvgPool2d((self.h, 1))
        self.avg_pool_y = nn.AdaptiveAvgPool2d((1, self.w))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv1d(self.h, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.conv2 = nn.Conv1d(self.w, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.conv11 = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.conv22 = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.convout = nn.Conv2d(64 * 4 * 5, 64 * 5, kernel_size=3, padding=1, bias=False)
        self.conv111 = nn.Conv2d(in_channels=64 * 5 * 2, out_channels=64 * 5 * 2, kernel_size=1, padding=0, stride=1)
        self.conv222 = nn.Conv2d(in_channels=64 * 5 * 2, out_channels=64 * 5 * 2, kernel_size=1, padding=0, stride=1)
        self.conv1h = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(self.h, 1), padding=(0, 0), stride=1)
        self.conv1s = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(1, self.w), padding=(0, 0), stride=1)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Conv1d)):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        n, c, h, w = x.size()
        y1 = self.avg_pool_x(x).reshape(n, c, self.h)
        y1 = self.sigmoid(self.conv11(self.relu1(self.conv1(y1.transpose(1, 2)))).transpose(1, 2).reshape(n, c, 1, 1))
        y2 = self.avg_pool_y(x).reshape(n, c, self.w)
        y2 = self.sigmoid(self.conv22(self.relu1(self.conv2(y2.transpose(1, 2)))).transpose(1, 2).reshape(n, c, 1, 1))
        yac = self.conv111(torch.cat([x * y1.expand_as(x), x * y2.expand_as(x)], dim=1))
        avg_mean = torch.mean(x, dim=1, keepdim=True)
        avg_max, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.cat([avg_max, avg_mean], dim=1)
        y3 = self.sigmoid(self.conv1h(avg_out))
        y4 = self.sigmoid(self.conv1s(avg_out))
        yap = self.conv222(torch.cat([x * y3.expand_as(x), x * y4.expand_as(x)], dim=1))
        out = self.convout(torch.cat([yac, yap], dim=1))
        return out

class CDAM4(nn.Module):
    def __init__(self, k_size=5):
        super(CDAM4, self).__init__()
        self.h = 64
        self.w = 64
        self.avg_pool_x = nn.AdaptiveAvgPool2d((self.h, 1))
        self.avg_pool_y = nn.AdaptiveAvgPool2d((1, self.w))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv1d(self.h, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.conv2 = nn.Conv1d(self.w, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.conv11 = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.conv22 = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.convout = nn.Conv2d(64 * 4 * 5, 64 * 5, kernel_size=3, padding=1, bias=False)
        self.conv111 = nn.Conv2d(in_channels=64 * 5 * 2, out_channels=64 * 5 * 2, kernel_size=1, padding=0, stride=1)
        self.conv222 = nn.Conv2d(in_channels=64 * 5 * 2, out_channels=64 * 5 * 2, kernel_size=1, padding=0, stride=1)
        self.conv1h = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(self.h, 1), padding=(0, 0), stride=1)
        self.conv1s = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(1, self.w), padding=(0, 0), stride=1)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Conv1d)):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        n, c, h, w = x.size()
        y1 = self.avg_pool_x(x).reshape(n, c, self.h)
        y1 = self.sigmoid(self.conv11(self.relu1(self.conv1(y1.transpose(1, 2)))).transpose(1, 2).reshape(n, c, 1, 1))
        y2 = self.avg_pool_y(x).reshape(n, c, self.w)
        y2 = self.sigmoid(self.conv22(self.relu1(self.conv2(y2.transpose(1, 2)))).transpose(1, 2).reshape(n, c, 1, 1))
        yac = self.conv111(torch.cat([x * y1.expand_as(x), x * y2.expand_as(x)], dim=1))
        avg_mean = torch.mean(x, dim=1, keepdim=True)
        avg_max, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.cat([avg_max, avg_mean], dim=1)
        y3 = self.sigmoid(self.conv1h(avg_out))
        y4 = self.sigmoid(self.conv1s(avg_out))
        yap = self.conv222(torch.cat([x * y3.expand_as(x), x * y4.expand_as(x)], dim=1))
        out = self.convout(torch.cat([yac, yap], dim=1))
        return out

############################################################
# Utility conv helpers
############################################################
class ConvBnRelu(nn.Module):
    def __init__(self, in_planes=512, out_planes=512, ksize=3, stride=1, pad=1, dilation=1,
                 groups=1, has_bn=True, norm_layer=nn.BatchNorm2d,
                 has_relu=True, inplace=True, has_bias=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=ksize,
                              stride=stride, padding=pad,
                              dilation=dilation, groups=groups, bias=has_bias)
        self.has_bn = has_bn
        if self.has_bn:
            self.bn = nn.BatchNorm2d(out_planes)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_planes, out_planes, norm_layer=nn.BatchNorm2d, scale=2, relu=True, last=False):
        super(DecoderBlock, self).__init__()
        self.conv_3x3 = ConvBnRelu(in_planes, in_planes, 3, 1, 1,
                                   has_bn=True, norm_layer=norm_layer,
                                   has_relu=True, has_bias=False)
        self.conv_1x1 = ConvBnRelu(in_planes, out_planes, 1, 1, 0,
                                   has_bn=True, norm_layer=norm_layer,
                                   has_relu=True, has_bias=False)
        self.scale = scale
        self.last = last
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        if not self.last:
            x = self.conv_3x3(x)
        if self.scale > 1:
            x = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=True)
        x = self.conv_1x1(x)
        return x

class BaseNetHead(nn.Module):
    def __init__(self, in_planes, out_planes, scale, is_aux=False, norm_layer=nn.BatchNorm2d):
        super(BaseNetHead, self).__init__()
        if is_aux:
            self.conv_1x1_3x3 = nn.Sequential(
                ConvBnRelu(in_planes, 64, 1, 1, 0, has_bn=True, norm_layer=norm_layer, has_relu=True, has_bias=False),
                ConvBnRelu(64, 64, 3, 1, 1, has_bn=True, norm_layer=norm_layer, has_relu=True, has_bias=False))
        else:
            self.conv_1x1_3x3 = nn.Sequential(
                ConvBnRelu(in_planes, 32, 1, 1, 0, has_relu=True, has_bias=False),
                ConvBnRelu(32, 32, 3, 1, 1, has_bn=True, norm_layer=norm_layer, has_relu=True, has_bias=False))
        self.conv_1x1_2 = nn.Conv2d(64 if is_aux else 32, out_planes, kernel_size=1, stride=1, padding=0)
        self.scale = scale
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        if self.scale > 1:
            x = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=True)
        fm = self.conv_1x1_3x3(x)
        output = self.conv_1x1_2(fm)
        return output

############################################################
# Multi-Scale Context Enhancement (unchanged logic except style)
############################################################
class ASPPPoolingH(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPoolingH, self).__init__(
            nn.AdaptiveAvgPool2d((32, 1)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPPPoolingW(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPoolingW, self).__init__(
            nn.AdaptiveAvgPool2d((1, 32)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class MSCE(nn.Module):
    def __init__(self, channel):
        super(MSCE, self).__init__()
        self.dilate11 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate22 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate33 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate44 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=(3, 1), dilation=1, padding=(1, 0))
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=(3, 1), dilation=2, padding=(2, 0))
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=(3, 1), dilation=4, padding=(4, 0))
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=(3, 1), dilation=8, padding=(8, 0))
        self.dilate5 = nn.Conv2d(channel, channel, kernel_size=(1, 3), dilation=1, padding=(0, 1))
        self.dilate6 = nn.Conv2d(channel, channel, kernel_size=(1, 3), dilation=2, padding=(0, 2))
        self.dilate7 = nn.Conv2d(channel, channel, kernel_size=(1, 3), dilation=4, padding=(0, 4))
        self.dilate8 = nn.Conv2d(channel, channel, kernel_size=(1, 3), dilation=8, padding=(0, 8))
        self.dconv = nn.Conv2d(channel * 5, channel, kernel_size=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(channel, channel, kernel_size=1)
        self.conv2 = nn.Conv2d(channel, channel, kernel_size=1)
        self.conv3 = nn.Conv2d(channel, channel, kernel_size=1)
        self.conv4 = nn.Conv2d(channel, channel, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.ASPPH = ASPPPoolingH(in_channels=channel, out_channels=channel)
        self.ASPPW = ASPPPoolingW(in_channels=channel, out_channels=channel)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                if m.bias is not None:
                    m.bias.data.zero_()
    def forward(self, x):
        d11 = nonlinearity(self.dilate11(x))
        d22 = nonlinearity(self.dilate22(d11))
        d33 = nonlinearity(self.dilate33(d22))
        d44 = nonlinearity(self.dilate44(d33))
        d1_out = self.conv1(d11 + d22 + d33 + d44)
        e11 = nonlinearity(self.dilate1(x))
        e22 = nonlinearity(self.dilate2(e11))
        e33 = nonlinearity(self.dilate3(e22))
        e44 = nonlinearity(self.dilate4(e33))
        d2_out = self.conv2(e11 + e22 + e33 + e44)
        f11 = nonlinearity(self.dilate5(x))
        f22 = nonlinearity(self.dilate6(f11))
        f33 = nonlinearity(self.dilate7(f22))
        f44 = nonlinearity(self.dilate8(f33))
        d3_out = self.conv3(f11 + f22 + f33 + f44)
        dH = self.ASPPH(x)
        dW = self.ASPPW(x)
        outsum = torch.cat([d1_out, d2_out, d3_out, dH, dW], dim=1)
        out = self.dconv(outsum)
        out = self.gamma * out + x * (1 - self.gamma)
        return out

############################################################
# FSFF blocks (reuse CDAM modules)
############################################################
class FSFF_2(nn.Module):
    def __init__(self, in_channels, width=64, up_kwargs=None, norm_layer=nn.BatchNorm2d):
        super(FSFF_2, self).__init__()
        self.conv5 = nn.Sequential(nn.Conv2d(512, width, 3, padding=1, bias=False), nn.BatchNorm2d(width), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(256, width, 3, padding=1, bias=False), nn.BatchNorm2d(width), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(128, width, 3, padding=1, bias=False), nn.BatchNorm2d(width), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(64, width, 3, padding=1, bias=False), nn.BatchNorm2d(width), nn.ReLU(inplace=True))
        self.conv1 = nn.Sequential(nn.Conv2d(64, width, 3, padding=1, bias=False), nn.BatchNorm2d(width), nn.ReLU(inplace=True))
        self.conv_out = nn.Sequential(nn.Conv2d(width * 5, width, 1, bias=False), nn.BatchNorm2d(width))
        self.CDAM = CDAM2()
    def forward(self, *inputs):
        feats = [self.conv5(inputs[-1]), self.conv4(inputs[-2]), self.conv3(inputs[-3]), self.conv2(inputs[-4]), self.conv1(inputs[-5])]
        _, _, h, w = feats[-2].size()
        for i in range(5):
            feats[i] = F.interpolate(feats[i], (h, w))
        feat1 = torch.cat(feats, dim=1)
        feat2 = self.conv_out(self.CDAM(feat1))
        return feat2

class FSFF_3(nn.Module):
    def __init__(self, in_channels, width=64, up_kwargs=None, norm_layer=nn.BatchNorm2d):
        super(FSFF_3, self).__init__()
        self.conv5 = nn.Sequential(nn.Conv2d(512, width, 3, padding=1, bias=False), nn.BatchNorm2d(width), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(256, width, 3, padding=1, bias=False), nn.BatchNorm2d(width), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(128, width, 3, padding=1, bias=False), nn.BatchNorm2d(width), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(64, width, 3, padding=1, bias=False), nn.BatchNorm2d(width), nn.ReLU(inplace=True))
        self.conv1 = nn.Sequential(nn.Conv2d(64, width, 3, padding=1, bias=False), nn.BatchNorm2d(width), nn.ReLU(inplace=True))
        self.conv_out = nn.Sequential(nn.Conv2d(width * 5, 128, 1, bias=False), nn.BatchNorm2d(width * 2))
        self.CDAM = CDAM3()
    def forward(self, *inputs):
        feats = [self.conv5(inputs[-1]), self.conv4(inputs[-2]), self.conv3(inputs[-3]), self.conv2(inputs[-4]), self.conv1(inputs[-5])]
        _, _, h, w = feats[-3].size()
        for i in range(5):
            feats[i] = F.interpolate(feats[i], (h, w))
        feat1 = torch.cat(feats, dim=1)
        feat2 = self.conv_out(self.CDAM(feat1))
        return feat2

class FSFF_4(nn.Module):
    def __init__(self, in_channels, width=64, up_kwargs=None, norm_layer=nn.BatchNorm2d):
        super(FSFF_4, self).__init__()
        self.conv5 = nn.Sequential(nn.Conv2d(512, width, 3, padding=1, bias=False), nn.BatchNorm2d(width), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(256, width, 3, padding=1, bias=False), nn.BatchNorm2d(width), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(128, width, 3, padding=1, bias=False), nn.BatchNorm2d(width), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(width, width, 3, padding=1, bias=False), nn.BatchNorm2d(width), nn.ReLU(inplace=True))
        self.conv1 = nn.Sequential(nn.Conv2d(width, width, 3, padding=1, bias=False), nn.BatchNorm2d(width), nn.ReLU(inplace=True))
        self.conv_out = nn.Sequential(nn.Conv2d(width * 5, 256, 1, bias=False), nn.BatchNorm2d(width * 4))
        self.CDAM = CDAM4()
    def forward(self, *inputs):
        feats = [self.conv5(inputs[-1]), self.conv4(inputs[-2]), self.conv3(inputs[-3]), self.conv2(inputs[-4]), self.conv1(inputs[-5])]
        _, _, h, w = feats[-4].size()
        for i in range(5):
            feats[i] = F.interpolate(feats[i], (h, w))
        feat1 = torch.cat(feats, dim=1)
        feat2 = self.conv_out(self.CDAM(feat1))
        return feat2

############################################################
# Main RCFSNet
############################################################
class RCFSNet(nn.Module):
    def __init__(self, num_classes=1, ccm=True, norm_layer=nn.BatchNorm2d, is_training=True, expansion=2,
                 base_channel=32, pretrained: bool = False):
        super(RCFSNet, self).__init__()
        filters = [64, 64, 128, 256, 512]
        resnet = load_resnet34_backbone(pretrained=bool(pretrained), caller='RCFSNet')
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        self.up = nn.Upsample(scale_factor=2)
        self.ConvBnRelu = ConvBnRelu(in_planes=512)
        self.CDAM2 = CDAM2()
        self.CDAM3 = CDAM3()
        self.CDAM4 = CDAM4()
        self.hd5_d1 = nn.Upsample(scale_factor=16)
        self.hd4_d1 = nn.Upsample(scale_factor=8)
        self.hd3_d1 = nn.Upsample(scale_factor=4)
        self.hd2_d1 = nn.Upsample(scale_factor=2)
        self.MSCE = MSCE(channel=512)
        self.decoder5 = DecoderBlock(filters[-1], filters[-2], relu=False, last=True)
        self.decoder4 = DecoderBlock(filters[-2], filters[-3], relu=False)
        self.decoder3 = DecoderBlock(filters[-3], filters[-4], relu=False)
        self.decoder2 = DecoderBlock(filters[-4], filters[-4], relu=False)
        self.FSFF_2 = FSFF_2([filters[0], filters[1], filters[4]], width=filters[1], up_kwargs=2)
        self.FSFF_3 = FSFF_3([filters[1], filters[2], filters[4]], width=filters[1], up_kwargs=2)
        self.FSFF_4 = FSFF_4([filters[2], filters[3], filters[4]], width=filters[1], up_kwargs=2)
        self.main_head = BaseNetHead(filters[0], num_classes, 2, is_aux=False, norm_layer=norm_layer)
        self.conv5 = nn.Conv2d(in_channels=filters[-1], out_channels=filters[1], kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=filters[-2], out_channels=filters[1], kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=filters[-3], out_channels=filters[1], kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=filters[-4], out_channels=filters[1], kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.conv256 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv128 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv64_1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv64_2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)) and m.bias is not None:
                m.bias.data.zero_()
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        h1 = self.firstconv(inputs)
        h1 = self.firstbn(h1)
        h1 = self.firstrelu(h1)
        h2 = self.firstmaxpool(h1)
        h2 = self.encoder1(h2)
        h3 = self.encoder2(h2)
        h4 = self.encoder3(h3)
        h5 = self.encoder4(h4)
        hd5 = self.MSCE(h5)
        m2 = self.FSFF_2(h1, h2, h3, h4, h5)
        m3 = self.FSFF_3(h1, h2, h3, h4, h5)
        m4 = self.FSFF_4(h1, h2, h3, h4, h5)
        d4 = self.relu(self.conv256(torch.cat([self.decoder5(hd5), m4], dim=1)))
        d3 = self.relu(self.conv128(torch.cat([self.decoder4(d4), m3], dim=1)))
        d2 = self.relu(self.conv64_1(torch.cat([self.decoder3(d3), m2], dim=1)))
        d1 = self.relu(self.conv64_2(torch.cat([self.decoder2(d2), h1], dim=1)))
        main_out = torch.sigmoid(self.main_head(d1 + self.conv5(self.hd5_d1(hd5)) + self.conv4(self.hd4_d1(d4)) + self.conv3(self.hd3_d1(d3)) + self.conv2(self.hd2_d1(d2))))
        return main_out

############################################################
# Factory
############################################################
def lunwen3():
    return RCFSNet()

if __name__ == "__main__":
    a = torch.randn(1, 3, 1024, 1024)
    model = RCFSNet()
    out = model(a)
    print(out.shape)
