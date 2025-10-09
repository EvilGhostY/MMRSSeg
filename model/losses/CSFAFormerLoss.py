import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
from .soft_ce import SoftCrossEntropyLoss
from .joint_loss import JointLoss
from .dice import DiceLoss
from torch.nn.modules.loss import _Loss
from .functional import focal_loss_with_logits
from functools import partial
from .lovasz import _lovasz_softmax
import cv2
from pytorch_msssim import SSIM
from PIL import Image
import torch
import matplotlib.pyplot as plt
def show_first_batch_label(labels: torch.Tensor):
    """
    显示第一个batch的标签图像

    参数:
    - labels: 形状为 (B, H, W) 的 Tensor，其中 B 为 batch size，H 和 W 为图像的高和宽。
    """
    # 确保 labels 是一个三维 Tensor (B, H, W)，且 B 大于 0

    # 选取第一个 batch 图像
    first_image = labels[0].cpu().detach().numpy()  # 形状为 (H, W)  # 形状为 (H, W)

    # 将 tensor 转换为 PIL Image，首先需要将 tensor 转换为 uint8 类型
    first_image_pil = Image.fromarray(first_image.astype('uint8'))
    plt.figure(figsize=(12, 6))
    plt.imshow(first_image_pil)
    plt.colorbar()
    plt.title(f'first_image_pil')
    plt.show()

def visualize_entropy_and_diff(prob_map):
    """
    计算并展示信息熵和前两个最大概率差值图像

    :param prob_map: Tensor，大小为 [B, K, H, W]，其中 B 是 batch 大小，K 是类别数，H, W 是图像大小
    """
    # 获取第一个 batch 的结果进行展示
    entropy_map = prob_map[0].detach().cpu().numpy()  # 获取第一个图像的归一化熵

    # 获取类别数 K
    num_classes = entropy_map.shape[0]

    # 显示图像
    plt.figure(figsize=(12, 6))

    for i in range(num_classes):
        plt.subplot(1, num_classes, i + 1)
        plt.imshow(entropy_map[i], cmap='jet')
        plt.colorbar()
        plt.title(f'Class {i + 1} Entropy')

    plt.tight_layout()
    plt.show()

class CrossEntropyLoss(nn.Module):

    def __init__(self, ignore_index=255):
        super().__init__()
        self.main_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, logits, labels,dsm):
        loss = self.main_loss(logits, labels)

        return loss

class UNetLoss(nn.Module):
    def __init__(self, ignore_index=255, n_classes=6, radius=1, alpha=0.01):
        super().__init__()
       # self.main_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.main_loss = JointLoss(SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
                                   DiceLoss(smooth=0.05, ignore_index=ignore_index), 1.0, 1.0)


    def forward(self, logits, labels, dsm):

        loss = self.main_loss(logits, labels)

        return loss

class FocalLoss(_Loss):
    def __init__(self, alpha=0.5, gamma=2, ignore_index=None, reduction="mean", normalized=False,
                 reduced_threshold=None):
        """
        Focal loss for multi-class problem.

        :param alpha:
        :param gamma:
        :param ignore_index: If not None, targets with given index are ignored
        :param reduced_threshold: A threshold factor for computing reduced focal loss
        """
        super().__init__()
        self.ignore_index = ignore_index
        self.focal_loss_fn = partial(
            focal_loss_with_logits,
            alpha=alpha,
            gamma=gamma,
            reduced_threshold=reduced_threshold,
            reduction=reduction,
            normalized=normalized,
        )

    def forward(self, label_input, label_target):
        num_classes = label_input.size(1)
        loss = 0

        # Filter anchors with -1 label from loss computation
        if self.ignore_index is not None:
            not_ignored = label_target != self.ignore_index

        for cls in range(num_classes):
            cls_label_target = (label_target == cls).long()
            cls_label_input = label_input[:, cls, ...]

            if self.ignore_index is not None:
                cls_label_target = cls_label_target[not_ignored]
                cls_label_input = cls_label_input[not_ignored]

            loss += self.focal_loss_fn(cls_label_input, cls_label_target)
        return loss

class EdgeLoss(nn.Module):
    def __init__(self, n_classes=19, radius=1, alpha=0.01):
        super(EdgeLoss, self).__init__()
        self.n_classes = n_classes
        self.radius = radius
        self.alpha = alpha

    def forward(self, logits, label):
        prediction = F.softmax(logits, dim=1)
        ks = 2 * self.radius + 1
        filt1 = torch.ones(1, 1, ks, ks)
        filt1[:, :, self.radius:2 * self.radius, self.radius:2 * self.radius] = -8
        filt1.requires_grad = False
        filt1 = filt1.cuda()
        label = label.unsqueeze(1)
        lbedge = F.conv2d(label.float(), filt1, bias=None, stride=1, padding=self.radius)
        lbedge = 1 - torch.eq(lbedge, 0).float()

        filt2 = torch.ones(self.n_classes, 1, ks, ks)
        filt2[:, :, self.radius:2 * self.radius, self.radius:2 * self.radius] = -8
        filt2.requires_grad = False
        filt2 = filt2.cuda()
        prededge = F.conv2d(prediction.float(), filt2, bias=None,
                            stride=1, padding=self.radius, groups=self.n_classes)

        norm = torch.sum(torch.pow(prededge, 2), 1).unsqueeze(1)
        prededge = norm / (norm + self.alpha)

        return BinaryDiceLoss()(prededge.float(), lbedge.float())


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """

    def __init__(self, smooth=1, p=2):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = 2 * torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den
        return loss.sum()

class BoundaryLoss(nn.Module):
    def __init__(self, n_classes=1, radius=1, alpha=0.01):
        super(BoundaryLoss, self).__init__()
        self.n_classes = n_classes
        self.radius = radius
        self.alpha = alpha

    def forward(self, logits, label):
        ks = 2 * self.radius + 1
        filt1 = torch.ones(1, 1, ks, ks)
        filt1[:, :, self.radius:2 * self.radius, self.radius:2 * self.radius] = -8
        filt1.requires_grad = False
        filt1 = filt1.cuda()
        label = label.unsqueeze(1)
        lbedge = F.conv2d(label.float(), filt1, bias=None, stride=1, padding=self.radius)
        lbedge = 1 - torch.eq(lbedge, 0).float()

        prededge = logits

        return BinaryDiceLoss()(prededge.float(), lbedge.float())

class HuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super(HuberLoss, self).__init__()
        self.delta = delta

    def forward(self, preds, targets):
        error = preds - targets
        abs_error = torch.abs(error)
        quadratic = torch.where(abs_error < self.delta, 0.5 * error ** 2, self.delta * abs_error - 0.5 * self.delta ** 2)
        return torch.mean(quadratic)

class LogCoshLoss(nn.Module):
    def __init__(self):
        super(LogCoshLoss, self).__init__()

    def forward(self, preds, targets):
        error = preds - targets
        return torch.mean(torch.log(torch.cosh(error)))

class CSFAFormerLoss(nn.Module):
    def __init__(self, ignore_index=255, edge_factor=10.0, n_classes=6, radius=1, alpha=0.01):
        super(CSFAFormerLoss, self).__init__()
        self.main_loss = JointLoss(SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
                                   DiceLoss(smooth=0.05, ignore_index=ignore_index), 1.0, 1.0)
        self.aux_loss = SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index)

    def forward(self, logits, labels, dsm):
        if self.training and len(logits) == 7:
            weight1 = 1.0   #0.6
            weight2 = 1.0   #0.3
            weight3 = 1.0   #0.2
            weight_rgb = 0.0
            weight_dsm = 0.0
            out, RGB_Pre1, DSM_Pre1, RGB_Pre2, DSM_Pre2, RGB_Pre3, DSM_Pre3= logits
            loss_main = self.main_loss(out, labels)
            loss_rgb1 =( self.main_loss(RGB_Pre1, labels))* weight1
            loss_rgb2 =( self.main_loss(RGB_Pre2, labels))* weight2
            loss_rgb3 = (self.main_loss(RGB_Pre3, labels))* weight3
            loss_rgb = (loss_rgb1+loss_rgb2+loss_rgb3)*weight_rgb

            loss_dsm1 = (self.main_loss(DSM_Pre1, labels)) * weight1
            loss_dsm2 = (self.main_loss(DSM_Pre2, labels)) * weight2
            loss_dsm3 = (self.main_loss(DSM_Pre3, labels)) * weight3
            loss_dsm = (loss_dsm1 + loss_dsm2 + loss_dsm3) * weight_dsm


            loss = loss_main  + loss_rgb  + loss_dsm

        else:
            loss = self.main_loss(logits, labels)

        return loss
