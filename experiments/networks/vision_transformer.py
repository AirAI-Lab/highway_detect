"""
This module has been removed.

SwinUnet and related vision_transformer models are no longer supported in this repository.
If you need the previous implementation, please retrieve it from the git history.
"""

raise ImportError(
    "experiments.networks.vision_transformer has been removed. SwinUnet is no longer supported."
)

# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging

import torch
import torch.nn as nn

from functools import partial
import torch.nn.functional as F

logger = logging.getLogger(__name__)

import torchvision
from timm.models.layers import trunc_normal_
from timm.models.layers import DropPath, to_2tuple
import torch.utils.checkpoint as checkpoint
from .weights_helper import load_resnet34_backbone

from torch.nn import Module, Conv2d, Parameter, Softmax

nonlinearity = partial(F.relu, inplace=True)


class SwinUnet(nn.Module):
    def __init__(self,img_size=256, num_classes=1, zero_head=False, patch_size=4,
                 patch_norm=True, depths=[2], ape=False, mlp_ratio=4.  ):
        super(SwinUnet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.patch_norm = patch_norm
        self.ape = ape
        self.num_layers = len(depths)
        self.mlp_ratio = mlp_ratio
        # encoder stage1和stage2
        # Prefer local ResNet34 weights; fall back if unavailable
        resnet = load_resnet34_backbone(pretrained=True, caller='vision_transformer.SwinUnet')
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.pool = resnet.maxpool
        self.encoder_stage1 = resnet.layer1
        self.encoder_stage2 = resnet.layer2
        self.encoder_stage3 = resnet.layer3
        self.encoder_stage4 = resnet.layer4


        # 全局上下文信息
        self.dac = DAC(channel=512)
        self.spp = SPPblock(512)

        self.da_net_haed = DANetHead(in_channels=512, out_channels=512, norm_layer=nn.BatchNorm2d)

        # 跳跃连接中的swin_transformer
        self.swin1 = Jump_swin(img_size=img_size//16, patch_chans=256, patch_size=patch_size)
        self.swin2 = Jump_swin(img_size=img_size//32, patch_chans=512, patch_size=patch_size)
        # 跳跃连接中的注意力机制
        self.atten1 = cbam(in_channel=64)
        self.atten2 = cbam(in_channel=64)
        self.atten3 = cbam(in_channel=128)
        #
        # self.atten4 = cbam(in_channel=256)
        # self.atten5 = cbam(in_channel=512)


        # 解码器
        self.decoder_stage1 = DecoderBlock(64, 1)
        self.decoder_stage2 = DecoderBlock(64, 64)
        self.decoder_stage3 = DecoderBlock(128, 64)
        self.decoder_stage4 = DecoderBlock(256, 128)
        self.decoder_stage5 = DecoderBlock(512, 256)

    def forward(self, x):
        jump_x = []
        # 编码器
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        jump_x.append(x)
        x = self.pool(x)
        x = self.encoder_stage1(x)
        jump_x.append(x)
        x = self.encoder_stage2(x)
        jump_x.append(x)
        x = self.encoder_stage3(x)
        jump_x.append(x)
        x = self.encoder_stage4(x)
        jump_x.append(x)

        # 全局上下文信息模块
        dac_x = self.dac(x)
        dac_x = self.spp(dac_x)

        x = self.da_net_haed(x)
        x = x + dac_x

        # 跳跃连接和注意力机制
        jump_x[0] = self.atten1(jump_x[0])
        jump_x[1] = self.atten2(jump_x[1])
        jump_x[2] = self.atten3(jump_x[2])
        jump_x[3] = self.swin1(jump_x[3])
        jump_x[4] = self.swin2(jump_x[4])
        # jump_x[3] = self.atten4(jump_x[3])
        # jump_x[4] = self.atten5(jump_x[4])

        # 解码器
        x = jump_x[4] + x
        x = self.decoder_stage5(x)
        x = jump_x[3] + x
        x = self.decoder_stage4(x)
        x = jump_x[2] + x
        x = self.decoder_stage3(x)
        x = jump_x[1] + x
        x = self.decoder_stage2(x)
        x = jump_x[0] + x
        x = self.decoder_stage1(x)

        x = F.sigmoid(x)

        return x


torch_ver = torch.__version__[:3]

__all__ = ['PAM_Module', 'CAM_Module']


class PAM_Module(Module):
    """ Position attention module"""

    # Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        # 先经过3个卷积层生成3个新特征图B C D （尺寸不变）
        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))  # α尺度系数初始化为0，并逐渐地学习分配到更大的权重

        self.softmax = Softmax(dim=-1)  # 对每一行进行softmax

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B × C × H × W)
            returns :
                out : attention value + input feature
                attention: B × (H×W) × (H×W)
        """
        m_batchsize, C, height, width = x.size()
        # B -> (N,C,HW) -> (N,HW,C)
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        # C -> (N,C,HW)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        # BC，空间注意图 -> (N,HW,HW)
        energy = torch.bmm(proj_query, proj_key)
        # S = softmax(BC) -> (N,HW,HW)
        attention = self.softmax(energy)
        # D -> (N,C,HW)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)
        # DS -> (N,C,HW)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # torch.bmm表示批次矩阵乘法
        # output -> (N,C,H,W)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


class CAM_Module(Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.gamma = Parameter(torch.zeros(1))  # β尺度系数初始化为0，并逐渐地学习分配到更大的权重
        self.softmax = Softmax(dim=-1)  # 对每一行进行softmax

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B × C × H × W)
            returns :
                out : attention value + input feature
                attention: B × C × C
        """
        m_batchsize, C, height, width = x.size()
        # A -> (N,C,HW)
        proj_query = x.view(m_batchsize, C, -1)
        # A -> (N,HW,C)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        # 矩阵乘积，通道注意图：X -> (N,C,C)
        energy = torch.bmm(proj_query, proj_key)
        # 这里实现了softmax用最后一维的最大值减去了原始数据，获得了一个不是太大的值
        # 沿着最后一维的C选择最大值，keepdim保证输出和输入形状一致，除了指定的dim维度大小为1
        # expand_as表示以复制的形式扩展到energy的尺寸
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy

        attention = self.softmax(energy_new)
        # A -> (N,C,HW)
        proj_value = x.view(m_batchsize, C, -1)
        # XA -> （N,C,HW）
        out = torch.bmm(attention, proj_value)
        # output -> (N,C,H,W)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


class DANetHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer):
        super(DANetHead, self).__init__()
        inter_channels = in_channels // 4  # in_channels=2018，通道数缩减为512

        self.conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels), nn.ReLU())
        self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels), nn.ReLU())

        self.sa = PAM_Module(inter_channels)  # 空间注意力模块
        self.sc = CAM_Module(inter_channels)  # 通道注意力模块

        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels), nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels), nn.ReLU())

        # nn.Dropout2d(p,inplace)：p表示将元素置0的概率；inplace若设置为True，会在原地执行操作。
        self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))  # 输出通道数为类别的数目
        self.conv7 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))
        self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x):
        # 经过一个1×1卷积降维后，再送入空间注意力模块
        feat1 = self.conv5a(x)
        sa_feat = self.sa(feat1)
        # 先经过一个卷积后，再使用有dropout的1×1卷积输出指定的通道数
        sa_conv = self.conv51(sa_feat)
        sa_output = self.conv6(sa_conv)

        # 经过一个1×1卷积降维后，再送入通道注意力模块
        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)
        # 先经过一个卷积后，再使用有dropout的1×1卷积输出指定的通道数
        sc_conv = self.conv52(sc_feat)
        sc_output = self.conv7(sc_conv)

        feat_sum = sa_conv + sc_conv  # 两个注意力模块结果相加
        sasc_output = self.conv8(feat_sum)  # 最后再送入1个有dropout的1×1卷积中



        return sasc_output  # 输出模块融合后的结果，以及两个模块各自的结果


# ---------------------------------------------------- #
# CBAM注意力机制
# 结合了通道注意力机制和空间注意力机制
# ---------------------------------------------------- #

# （1）通道注意力机制
class channel_attention(nn.Module):
    # 初始化, in_channel代表输入特征图的通道数, ratio代表第一个全连接的通道下降倍数
    def __init__(self, in_channel, ratio=4):
        # 继承父类初始化方法
        super(channel_attention, self).__init__()

        # 全局最大池化 [b,c,h,w]==>[b,c,1,1]
        self.max_pool = nn.AdaptiveMaxPool2d(output_size=1)
        # 全局平均池化 [b,c,h,w]==>[b,c,1,1]
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)

        # 第一个全连接层, 通道数下降4倍
        self.fc1 = nn.Linear(in_features=in_channel, out_features=in_channel // ratio, bias=False)
        # 第二个全连接层, 恢复通道数
        self.fc2 = nn.Linear(in_features=in_channel // ratio, out_features=in_channel, bias=False)

        # relu激活函数
        self.relu = nn.ReLU()
        # sigmoid激活函数
        self.sigmoid = nn.Sigmoid()

    # 前向传播
    def forward(self, inputs):
        # 获取输入特征图的shape
        b, c, h, w = inputs.shape

        # 输入图像做全局最大池化 [b,c,h,w]==>[b,c,1,1]
        max_pool = self.max_pool(inputs)
        # 输入图像的全局平均池化 [b,c,h,w]==>[b,c,1,1]
        avg_pool = self.avg_pool(inputs)

        # 调整池化结果的维度 [b,c,1,1]==>[b,c]
        max_pool = max_pool.view([b, c])
        avg_pool = avg_pool.view([b, c])

        # 第一个全连接层下降通道数 [b,c]==>[b,c//4]
        x_maxpool = self.fc1(max_pool)
        x_avgpool = self.fc1(avg_pool)

        # 激活函数
        x_maxpool = self.relu(x_maxpool)
        x_avgpool = self.relu(x_avgpool)

        # 第二个全连接层恢复通道数 [b,c//4]==>[b,c]
        x_maxpool = self.fc2(x_maxpool)
        x_avgpool = self.fc2(x_avgpool)

        # 将这两种池化结果相加 [b,c]==>[b,c]
        x = x_maxpool + x_avgpool
        # sigmoid函数权值归一化
        x = self.sigmoid(x)
        # 调整维度 [b,c]==>[b,c,1,1]
        x = x.view([b, c, 1, 1])
        # 输入特征图和通道权重相乘 [b,c,h,w]
        outputs = inputs * x

        return outputs


# ---------------------------------------------------- #
# （2）空间注意力机制
class spatial_attention(nn.Module):
    # 初始化，卷积核大小为7*7
    def __init__(self, kernel_size=7):
        # 继承父类初始化方法
        super(spatial_attention, self).__init__()

        # 为了保持卷积前后的特征图shape相同，卷积时需要padding
        padding = kernel_size // 2
        # 7*7卷积融合通道信息 [b,2,h,w]==>[b,1,h,w]
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size,
                              padding=padding, bias=False)
        # sigmoid函数
        self.sigmoid = nn.Sigmoid()

    # 前向传播
    def forward(self, inputs):
        # 在通道维度上最大池化 [b,1,h,w]  keepdim保留原有深度
        # 返回值是在某维度的最大值和对应的索引
        x_maxpool, _ = torch.max(inputs, dim=1, keepdim=True)

        # 在通道维度上平均池化 [b,1,h,w]
        x_avgpool = torch.mean(inputs, dim=1, keepdim=True)
        # 池化后的结果在通道维度上堆叠 [b,2,h,w]
        x = torch.cat([x_maxpool, x_avgpool], dim=1)

        # 卷积融合通道信息 [b,2,h,w]==>[b,1,h,w]
        x = self.conv(x)
        # 空间权重归一化
        x = self.sigmoid(x)
        # 输入特征图和空间权重相乘
        outputs = inputs * x

        return outputs


# ---------------------------------------------------- #
# （3）CBAM注意力机制
class cbam(nn.Module):
    # 初始化，in_channel和ratio=4代表通道注意力机制的输入通道数和第一个全连接下降的通道数
    # kernel_size代表空间注意力机制的卷积核大小
    def __init__(self, in_channel, ratio=4, kernel_size=7):
        # 继承父类初始化方法
        super(cbam, self).__init__()

        # 实例化通道注意力机制
        self.channel_attention = channel_attention(in_channel=in_channel, ratio=ratio)
        # 实例化空间注意力机制
        self.spatial_attention = spatial_attention(kernel_size=kernel_size)

    # 前向传播
    def forward(self, inputs):
        # 先将输入图像经过通道注意力机制
        x = self.channel_attention(inputs)
        # 然后经过空间注意力机制
        x = self.spatial_attention(x)

        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class DAC(nn.Module):
    def __init__(self, channel):
        super(DAC, self).__init__()
        # self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        # self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        # self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        # self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)


        self.relu = nn.ReLU()
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = self.dilate1(x)
        dilate1_out = self.relu(dilate1_out)
        dilate2_out = self.conv1x1(self.dilate2(x))
        dilate2_out = self.relu(dilate2_out)
        dilate3_out = self.conv1x1(self.dilate2(self.dilate1(x)))
        dilate3_out = self.relu(dilate3_out)
        dilate4_out = self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x))))
        dilate4_out = self.relu(dilate4_out)
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out

        return out

class SPPblock(nn.Module):
    def __init__(self, in_channels):
        super(SPPblock, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=[3, 3], stride=3)
        self.pool3 = nn.MaxPool2d(kernel_size=[5, 5], stride=5)
        self.pool4 = nn.MaxPool2d(kernel_size=[6, 6], stride=6)

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=516, out_channels=512, kernel_size=1, padding=0)
    def forward(self, x):
        self.in_channels, h, w = x.size(1), x.size(2), x.size(3)
        self.layer1 = F.upsample(self.conv(self.pool1(x)), size=(h, w), mode='bilinear')
        self.layer2 = F.upsample(self.conv(self.pool2(x)), size=(h, w), mode='bilinear')
        self.layer3 = F.upsample(self.conv(self.pool3(x)), size=(h, w), mode='bilinear')
        self.layer4 = F.upsample(self.conv(self.pool4(x)), size=(h, w), mode='bilinear')
        out = torch.cat([self.layer1, self.layer2, self.layer3, self.layer4, x], 1)
        out = self.conv2(out)
        return out


class Jump_swin(nn.Module):
    def __init__(self, img_size=16, zero_head=False, patch_size=4,
                 patch_chans=512, norm_layer=nn.LayerNorm, patch_norm=True, drop_rate=0.,
                 drop_path_rate=0.1, depths=[1], ape=False, num_heads=[4], window_size=8,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, attn_drop_rate=0., use_checkpoint=False):
        super(Jump_swin, self).__init__()

        self.zero_head = zero_head
        self.patch_norm = patch_norm
        embed_dim = patch_size ** 2 * patch_chans // 4
        self.ape = ape
        self.num_layers = len(depths)
        self.mlp_ratio = mlp_ratio

        self.conv1 = nn.Conv2d(patch_chans, patch_chans // 4, kernel_size=1)
        self.norm1 = nn.BatchNorm2d(patch_chans // 4)
        self.relu1 = nonlinearity

        self.conv3 = nn.Conv2d(patch_chans // 4, patch_chans, 1)
        self.norm3 = nn.BatchNorm2d(patch_chans)
        self.relu3 = nonlinearity

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=patch_chans // 4,
            embed_dim=embed_dim, norm_layer=norm_layer if self.patch_norm else None)

        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution

        self.patches_resolution = patches_resolution
        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            # 参数初始化为正态分布
            trunc_normal_(self.absolute_pos_embed, std=.02)
        self.pos_drop = nn.Dropout(p=drop_rate)
        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        # encoder stage3和stage4
        self.jump_swin = nn.ModuleList()

        for i in range(self.num_layers):
            swin = BasicLayer(dim=int(embed_dim * 2 ** i),
                              input_resolution=(patches_resolution[0] // (2 ** i),
                                                patches_resolution[1] // (2 ** i)),
                              depth=depths[i],
                              num_heads=num_heads[i],
                              window_size=window_size,
                              mlp_ratio=self.mlp_ratio,
                              qkv_bias=qkv_bias, qk_scale=qk_scale,
                              drop=drop_rate, attn_drop=attn_drop_rate,
                              drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                              norm_layer=norm_layer,
                              downsample=None,
                              use_checkpoint=use_checkpoint)
            self.jump_swin.append(swin)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        B, C, H, W = x.shape

        x = self.patch_embed(x)

        for i in self.jump_swin:
            x = i(x)

        x = x.view(B, C, H, W)

        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)

        return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

        # Local Perception Unit
        # self.lpu = nn.Sequential(
        #     nn.Conv2d(in_channels=self.dim, out_channels=self.dim, kernel_size=3, stride=1, padding=1, groups=self.dim),
        #     nn.GELU(),
        #     nn.BatchNorm2d(dim)
        # )

    def forward(self, x):

        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)  # (1,96, 128, 128)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size,
                                                            C)  # （B*H*W//window_size//window_size, window_size,window_size, C）
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """

        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=256, patch_size=4, in_chans=128, embed_dim=2048, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)  # (4, 4)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]  # [61, 61]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]  # 3721

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_channels=self.in_chans, out_channels=embed_dim, kernel_size=patch_size,
                              stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # flatten:[B, C, H, W]->[B, C, HW]
        # transpose:[B, C, H, W]->[B, HW, C]
        y = self.proj(x)
        x = self.proj(x).flatten(2).transpose(1,
                                              2)  # (1,16384,96)  B Ph*Pw C  前面提及的尺寸中通道X都是在第3维度，而pytorch中原始输入的图像通道位于第一维度，一定要注意，否则fl（2）这个操作就很难理解了
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


if __name__ == '__main__':
    from torchinfo import summary

    net = SwinUnet(1024)
    summary(net, input_size=(1, 3, 1024, 1024))
