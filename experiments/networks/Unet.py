import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
from torch import nn, einsum
from einops import rearrange, repeat

"""
    构造上采样模块--左边特征提取基础模块    
"""


class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            # 在卷积神经网络的卷积层之后总会添加BatchNorm2d进行数据的归一化处理，这使得数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


"""
    构造下采样模块--右边特征融合基础模块    
"""


class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


"""
    模型主架构
"""


class LightASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LightASPP, self).__init__()

        # ASPP模块中的1×1卷积
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv_out = nn.Conv2d(in_channels*5, out_channels, kernel_size=1)

        # ASPP模块中的3×3卷积
        self.conv3x3_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv3x3_2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=2, dilation=2)
        self.conv3x3_3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3, dilation=3)

        # ASPP模块中的全局池化层
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1x1_pool = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # ASPP模块中的BN层和ReLU激活函数
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 进行ASPP模块中的各个分支操作
        x1 = self.conv1x1(x)
        x2 = self.conv3x3_1(x)
        x3 = self.conv3x3_2(x)
        x4 = self.conv3x3_3(x)
        x5 = self.conv1x1_pool(self.global_pool(x))
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        # 将各个分支的结果拼接在一起，并进行BN和ReLU操作
        out = torch.cat((x1, x2, x3, x4, x5), dim=1)
        out = self.conv_out(out)
        out = self.bn(out)
        out = self.relu(out)

        return out


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def divisible_by(numer, denom):
    return (numer % denom) == 0


# tensor helpers

def create_grid_like(t, dim=0):
    h, w, device = *t.shape[-2:], t.device

    grid = torch.stack(torch.meshgrid(
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing='ij'), dim=dim)

    grid.requires_grad = False
    grid = grid.type_as(t)
    return grid


def normalize_grid(grid, dim=1, out_dim=-1):
    # normalizes a grid to range from -1 to 1
    h, w = grid.shape[-2:]
    grid_h, grid_w = grid.unbind(dim=dim)

    grid_h = 2.0 * grid_h / max(h - 1, 1) - 1.0
    grid_w = 2.0 * grid_w / max(w - 1, 1) - 1.0

    return torch.stack((grid_h, grid_w), dim=out_dim)


class Scale(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale


# continuous positional bias from SwinV2

class CPB(nn.Module):
    """ https://arxiv.org/abs/2111.09883v1 """

    def __init__(self, dim, *, heads, offset_groups, depth):
        super().__init__()
        self.heads = heads
        self.offset_groups = offset_groups

        self.mlp = nn.ModuleList([])

        self.mlp.append(nn.Sequential(
            nn.Linear(2, dim),
            nn.ReLU()
        ))

        for _ in range(depth - 1):
            self.mlp.append(nn.Sequential(
                nn.Linear(dim, dim),
                nn.ReLU()
            ))

        self.mlp.append(nn.Linear(dim, heads // offset_groups))

    def forward(self, grid_q, grid_kv):
        device, dtype = grid_q.device, grid_kv.dtype

        grid_q = rearrange(grid_q, 'h w c -> 1 (h w) c')
        grid_kv = rearrange(grid_kv, 'b h w c -> b (h w) c')

        pos = rearrange(grid_q, 'b i c -> b i 1 c') - rearrange(grid_kv, 'b j c -> b 1 j c')
        bias = torch.sign(pos) * torch.log(pos.abs() + 1)  # log of distance is sign(rel_pos) * log(abs(rel_pos) + 1)

        for layer in self.mlp:
            bias = layer(bias)

        bias = rearrange(bias, '(b g) i j o -> b (g o) i j', g=self.offset_groups)

        return bias


class DeformableAttention2D(nn.Module):
    def __init__(
            self,
            *,
            dim,
            dim_head=64,
            heads=8,
            dropout=0.,
            downsample_factor=4,
            offset_scale=None,
            offset_groups=None,
            offset_kernel_size=6,
            group_queries=True,
            group_key_values=True
    ):
        super().__init__()
        offset_scale = default(offset_scale, downsample_factor)
        assert offset_kernel_size >= downsample_factor, 'offset kernel size must be greater than or equal to the downsample factor'
        assert divisible_by(offset_kernel_size - downsample_factor, 2)

        offset_groups = default(offset_groups, heads)
        assert divisible_by(heads, offset_groups)

        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.offset_groups = offset_groups

        offset_dims = inner_dim // offset_groups

        self.downsample_factor = downsample_factor

        self.to_offsets = nn.Sequential(
            nn.Conv2d(offset_dims, offset_dims, offset_kernel_size, groups=offset_dims, stride=downsample_factor,
                      padding=(offset_kernel_size - downsample_factor) // 2),
            nn.GELU(),
            nn.Conv2d(offset_dims, 2, 1, bias=False),
            nn.Tanh(),
            Scale(offset_scale)
        )

        self.rel_pos_bias = CPB(dim // 4, offset_groups=offset_groups, heads=heads, depth=2)

        self.dropout = nn.Dropout(dropout)
        self.to_q = nn.Conv2d(dim, inner_dim, 1, groups=offset_groups if group_queries else 1, bias=False)
        self.to_k = nn.Conv2d(dim, inner_dim, 1, groups=offset_groups if group_key_values else 1, bias=False)
        self.to_v = nn.Conv2d(dim, inner_dim, 1, groups=offset_groups if group_key_values else 1, bias=False)
        self.to_out = nn.Conv2d(inner_dim, dim, 1)

    def forward(self, x, return_vgrid=False):
        """
        b - batch
        h - heads
        x - height
        y - width
        d - dimension
        g - offset groups
        """

        heads, b, h, w, downsample_factor, device = self.heads, x.shape[0], *x.shape[-2:], self.downsample_factor, x.device

        # queries

        q = self.to_q(x)

        # calculate offsets - offset MLP shared across all groups

        group = lambda t: rearrange(t, 'b (g d) ... -> (b g) d ...', g=self.offset_groups)

        grouped_queries = group(q)
        offsets = self.to_offsets(grouped_queries)

        # calculate grid + offsets

        grid = create_grid_like(offsets)
        vgrid = grid + offsets

        vgrid_scaled = normalize_grid(vgrid)

        kv_feats = F.grid_sample(
            group(x),
            vgrid_scaled,
            mode='bilinear', padding_mode='zeros', align_corners=False)

        kv_feats = rearrange(kv_feats, '(b g) d ... -> b (g d) ...', b=b)

        # derive key / values

        k, v = self.to_k(kv_feats), self.to_v(kv_feats)

        # scale queries

        q = q * self.scale

        # split out heads

        q, k, v = map(lambda t: rearrange(t, 'b (h d) ... -> b h (...) d', h=heads), (q, k, v))

        # query / key similarity

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # relative positional bias

        grid = create_grid_like(x)
        grid_scaled = normalize_grid(grid, dim=0)
        rel_pos_bias = self.rel_pos_bias(grid_scaled, vgrid_scaled)
        sim = sim + rel_pos_bias

        # numerical stability

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()

        # attention

        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        # aggregate and combine heads

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        out = self.to_out(out)

        if return_vgrid:
            return out, vgrid

        return out


class U_Net(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """

    # 输入是3个通道的RGB图，输出是0或1——因为我的任务是2分类任务
    def __init__(self, in_ch=3, out_ch=2):
        super(U_Net, self).__init__()

        # 卷积参数设置
        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        # 最大池化层
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 左边特征提取卷积层
        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])


        #注意力模块
        # self.DA_block1 = DeformableAttention2D(dim=128)
        # self.DA_block2 = DeformableAttention2D(dim=256)
        # self.DA_block3 = DeformableAttention2D(dim=512)
        # self.DA_block4 = DeformableAttention2D(dim=1024)
        #Light_ASPP
        # self.aspp = LightASPP(filters[4], filters[4])
        #

        # 右边特征融合反卷积层
        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

    # 前向计算，输出一张与原图相同尺寸的图片矩阵
    def forward(self, x):
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        # e4 = self.DA_block3(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        # e5 = self.DA_block4(e5)

        # e5 = self.aspp(e5)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)  # 将e4特征图与d5特征图横向拼接

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)  # 将e3特征图与d4特征图横向拼接
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)  # 将e2特征图与d3特征图横向拼接
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)  # 将e1特征图与d1特征图横向拼接
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        return out


if __name__ == '__main__':
    from torchinfo import summary

    net = U_Net(3, 2)
    # net = DeformableAttention(64, 64)
    summary(net, input_size=(1, 3, 1024, 1024))