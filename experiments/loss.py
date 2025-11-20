# 兼容性补充：空的 focal_tversky_edge_loss 占位类，避免导入错误
import torch.nn as nn
class focal_tversky_edge_loss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, y_true, y_pred):
        raise NotImplementedError("focal_tversky_edge_loss 已废弃，仅用于兼容导入。")
import torch
import torch.nn as nn
from torch.autograd import Variable as V

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import cv2
import os
import numpy as np


class weighted_cross_entropy(nn.Module):
    def __init__(self, num_classes=12, batch=True):
        super(weighted_cross_entropy, self).__init__()
        self.batch = batch
        self.weight = torch.Tensor([52.] * num_classes).cuda()
        self.ce_loss = nn.CrossEntropyLoss(weight=self.weight)

    def __call__(self, y_true, y_pred):
        y_ce_true = y_true.squeeze(dim=1).long()

        a = self.ce_loss(y_pred, y_ce_true)

        return a


class dice_loss(nn.Module):
    def __init__(self, batch=True):
        super(dice_loss, self).__init__()
        self.batch = batch

    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 0.0  # may change
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        # score = (intersection + smooth) / (i + j - intersection + smooth)#iou
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss

    def __call__(self, y_true, y_pred):

        b = self.soft_dice_loss(y_true, y_pred)
        return b


def test_weight_cross_entropy():
    N = 4
    C = 12
    H, W = 128, 128

    inputs = torch.rand(N, C, H, W)
    targets = torch.LongTensor(N, H, W).random_(C)
    inputs_fl = Variable(inputs.clone(), requires_grad=True)
    targets_fl = Variable(targets.clone())
    print(weighted_cross_entropy()(targets_fl, inputs_fl))


class dice_bce_loss(nn.Module):
    def __init__(self, batch=True):
        super(dice_bce_loss, self).__init__()
        self.batch = batch
        self.bce_loss = nn.BCELoss()
        # Optional weighting via environment variables (strings acceptable)
        try:
            import os
            self.w_bce = float(os.environ.get('BCE_LOSS_WEIGHT', os.environ.get('LOSS_BCE_WEIGHT', '1.0')))
            self.w_dice = float(os.environ.get('DICE_LOSS_WEIGHT', os.environ.get('LOSS_DICE_WEIGHT', '1.0')))
        except Exception:
            self.w_bce = 1.0
            self.w_dice = 1.0

    def soft_dice_coeff(self, y_true, y_pred):

        smooth = 0.0  # may change
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        # score = (intersection + smooth) / (i + j - intersection + smooth)#iou
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss

    def __call__(self, y_true, y_pred):
        # Support composite outputs: [main, edge_aux]
        if isinstance(y_pred, (list, tuple)):
            main = y_pred[0]
            # main segmentation loss
            seg_bce = self.bce_loss(main, y_true)
            seg_dice = self.soft_dice_loss(y_true, main)
            loss = self.w_bce * seg_bce + self.w_dice * seg_dice
            # optional edge auxiliary supervision
            if len(y_pred) > 1:
                edge_map = y_pred[1]
                # build edge label from y_true via morph ops (dilation - erosion) normalized to {0,1}
                with torch.no_grad():
                    # y_true: (B,1,H,W)
                    kernel = torch.ones((1,1,3,3), device=y_true.device)
                    dilated = F.conv2d(y_true, kernel, padding=1)
                    eroded = -F.conv2d(-y_true, kernel, padding=1)
                    edge = (dilated - eroded).clamp(min=0.0)
                    edge = (edge > 0.0).float()
                edge_bce = self.bce_loss(edge_map, edge)
                # treat edge dice separately for sparsity robustness
                edge_dice = self.soft_dice_loss(edge, edge_map)
                # weight auxiliary modestly
                aux_w = float(os.environ.get('EDGE_AUX_WEIGHT', '0.25'))
                loss = loss + aux_w * (edge_bce + edge_dice)
            return loss
        else:
            a = self.bce_loss(y_pred, y_true)
            b = self.soft_dice_loss(y_true, y_pred)
            return self.w_bce * a + self.w_dice * b


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, y_pred, y_true):
        smooth = 0.0  # may change
        i = torch.sum(y_true)
        j = torch.sum(y_pred)
        intersection = torch.sum(y_true * y_pred)

        score = (2. * intersection + smooth) / (i + j + smooth)
        # score = (intersection + smooth) / (i + j - intersection + smooth)#iou
        return 1 - score.mean()


class MulticlassDiceLoss(nn.Module):
    """
    requires one hot encoded target. Applies DiceLoss on each class iteratively.
    requires input.shape[0:1] and target.shape[0:1] to be (N, C) where N is
      batch size and C is number of classes
    """

    def __init__(self):
        super(MulticlassDiceLoss, self).__init__()

    def forward(self, input, target, weights=None):

        C = target.shape[1]

        # if weights is None:
        # 	weights = torch.ones(C) #uniform weights for all classes

        dice = DiceLoss()
        totalLoss = 0

        for i in range(C):
            diceLoss = dice(input[:, i, :, :], target[:, i, :, :])
            if weights is not None:
                diceLoss *= weights[i]
            totalLoss += diceLoss

        return totalLoss


class focal_tversky_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.focal = FocalLoss
        self.tversky = TverskyLoss

    def forward(self, y_true, y_pred):
        a = self.focal(y_pred, y_true)
        b = self.tversky(y_pred, y_true)

        return a + b


def FocalLoss(y_pred, y_true, alpha=0.25, gamma=2, reduction='mean'):
    ce_loss = F.cross_entropy(y_pred, y_true.squeeze(dim=1).long(), reduction='none')
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss

    if reduction == 'mean':
        return focal_loss.mean()
    elif reduction == 'sum':
        return focal_loss.sum()
    elif reduction == 'none':
        return focal_loss


def TverskyLoss(y_pred, y_true, alpha=0.7, beta=0.3, smooth=0):
    # Apply softmax to the predictions
    y_pred = F.softmax(y_pred, dim=1)

    # Calculate the number of channels
    num_channels = y_pred.size(1)

    total_loss = 0
    for channel in range(num_channels):
        pre_channel = y_pred[:, channel, :, :]
        true_channel = (y_true == channel).float()

        TP = torch.sum(pre_channel * true_channel)
        FN = torch.sum((1 - pre_channel) * true_channel)
        FP = torch.sum(pre_channel * (1 - true_channel))

        score = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
        total_loss += 1 - score
        # score = (intersection + smooth) / (i + j - intersection + smooth)#iou
    return total_loss / num_channels


class binary_tversky_loss(nn.Module):
    """Binary Tversky loss operating on probability maps (after Sigmoid).
    Favors recall when alpha>beta (defaults alpha=0.7, beta=0.3).
    """
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, smooth: float = 0.0):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.smooth = float(smooth)

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        # Accept composite outputs (list/tuple); use primary segmentation map
        if isinstance(y_pred, (list, tuple)):
            y_pred = y_pred[0]
        # Ensure probabilities
        y_pred = y_pred.clamp(0.0, 1.0)
        y_true = y_true.clamp(0.0, 1.0)
        # Flatten over batch and spatial dims
        TP = torch.sum(y_true * y_pred)
        FP = torch.sum((1.0 - y_true) * y_pred)
        FN = torch.sum(y_true * (1.0 - y_pred))
        num = TP + self.smooth
        den = TP + self.alpha * FP + self.beta * FN + self.smooth
        tversky = num / den
        return 1.0 - tversky


class binary_focal_tversky_loss(nn.Module):
    """Binary Focal Tversky loss: (1 - Tversky)^gamma.
    Recommended for highly imbalanced segmentation to emphasize FN reductions.
    """
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, gamma: float = 2.0, smooth: float = 0.0):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.smooth = float(smooth)

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        # Accept composite outputs (list/tuple); use primary segmentation map
        if isinstance(y_pred, (list, tuple)):
            y_pred = y_pred[0]
        y_pred = y_pred.clamp(0.0, 1.0)
        y_true = y_true.clamp(0.0, 1.0)
        TP = torch.sum(y_true * y_pred)
        FP = torch.sum((1.0 - y_true) * y_pred)
        FN = torch.sum(y_true * (1.0 - y_pred))
        num = TP + self.smooth
        den = TP + self.alpha * FP + self.beta * FN + self.smooth
        t = num / den
        return torch.pow(1.0 - t, self.gamma)



# 新增：联合分割+边缘损失函数
class joint_seg_edge_loss(nn.Module):
    """
    联合分割+边缘损失：主分割分支采用 Dice + BCE，边缘辅助分支采用 BCE + Dice，权重可调。
    支持输入为 [main, edge] 或单分支。
    环境变量 EDGE_AUX_WEIGHT 控制边缘分支权重，默认 0.25。
    """
    def __init__(self, seg_bce_weight=1.0, seg_dice_weight=1.0, edge_aux_weight=None):
        super().__init__()
        self.bce = nn.BCELoss()
        self.seg_bce_weight = seg_bce_weight
        self.seg_dice_weight = seg_dice_weight
        # 支持通过参数或环境变量设置边缘辅助权重
        if edge_aux_weight is not None:
            self.edge_aux_weight = float(edge_aux_weight)
        else:
            try:
                self.edge_aux_weight = float(os.environ.get('EDGE_AUX_WEIGHT', '0.25'))
            except Exception:
                self.edge_aux_weight = 0.25

    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 0.0
        i = torch.sum(y_true)
        j = torch.sum(y_pred)
        inter = torch.sum(y_true * y_pred)
        return (2.0 * inter + smooth) / (i + j + smooth)

    def soft_dice_loss(self, y_true, y_pred):
        return 1.0 - self.soft_dice_coeff(y_true, y_pred)

    def forward(self, y_true, y_pred):
        main = y_pred
        edge_map = None
        if isinstance(y_pred, (list, tuple)) and len(y_pred) > 0:
            main = y_pred[0]
            if len(y_pred) > 1:
                edge_map = y_pred[1]

        # 主分割损失（Dice + BCE）
        main = main.clamp(0.0, 1.0)
        y_true = y_true.clamp(0.0, 1.0)
        seg_bce = self.bce(main, y_true)
        seg_dice = self.soft_dice_loss(y_true, main)
        loss = self.seg_bce_weight * seg_bce + self.seg_dice_weight * seg_dice

        # 边缘辅助损失（BCE + Dice）
        if edge_map is not None:
            edge_map = edge_map.clamp(0.0, 1.0)
            with torch.no_grad():
                kernel = torch.ones((1, 1, 3, 3), device=y_true.device)
                dilated = F.conv2d(y_true, kernel, padding=1)
                eroded = -F.conv2d(-y_true, kernel, padding=1)
                edge = (dilated - eroded).clamp(min=0.0)
                edge = (edge > 0.0).float()
            edge_bce = self.bce(edge_map, edge)
            edge_dice = self.soft_dice_loss(edge, edge_map)
            loss = loss + self.edge_aux_weight * (edge_bce + edge_dice)

        return loss
