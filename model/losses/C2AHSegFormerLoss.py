# coding=utf-8
from abc import ABC
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
from .soft_ce import SoftCrossEntropyLoss
from .joint_loss import JointLoss
from .dice import DiceLoss
import torch

class PixelContrastLoss(nn.Module, ABC):
    def __init__(self, base_temperature=0.07, temperature=0.1, ignore_label=6,
                 max_samples=1200, max_views=200):
        super(PixelContrastLoss, self).__init__()

        # 温度参数，用于缩放对比学习中的logits
        self.temperature = temperature
        self.base_temperature = base_temperature
        # 忽略标签值（例如：-1表示不参与loss计算）
        self.ignore_label = ignore_label
        # 每个批次最多采样多少个像素
        self.max_samples = max_samples
        # 每个类别最多采样多少个像素
        self.max_views = max_views

    # 难易样本均衡采样（Hard Anchor Sampling）
    def _hard_anchor_sampling(self, X, y_hat, y):
        batch_size, feat_dim = X.shape[0], X.shape[-1]

        classes = []
        total_classes = 0
        for ii in range(batch_size):
            this_y = y_hat[ii]
            # 找出预测中出现的所有类别
            this_classes = torch.unique(this_y)
            # 过滤掉 ignore_label
            this_classes = [x for x in this_classes if x != self.ignore_label]
            # 只保留预测像素数足够多（>max_views）的类别
            this_classes = [x for x in this_classes if (this_y == x).nonzero().shape[0] > self.max_views]

            classes.append(this_classes)
            total_classes += len(this_classes)

        # 如果所有样本都没有满足条件的类别，返回None
        if total_classes == 0:
            return None, None

        # 每个类别采样多少个像素
        n_view = self.max_samples // total_classes
        n_view = min(n_view, self.max_views)

        # 初始化存放采样后的特征和标签
        X_ = torch.zeros((total_classes, n_view, feat_dim), dtype=torch.float).cuda()
        y_ = torch.zeros(total_classes, dtype=torch.float).cuda()

        X_ptr = 0  # 指向当前存放的位置

        for ii in range(batch_size):
            this_y_hat = y_hat[ii]
            this_y = y[ii]
            this_classes = classes[ii]

            for cls_id in this_classes:
                # 获取难样本（预测错了的）
                hard_indices = ((this_y_hat == cls_id) & (this_y != cls_id)).nonzero()
                # 获取易样本（预测对了的）
                easy_indices = ((this_y_hat == cls_id) & (this_y == cls_id)).nonzero()

                num_hard = hard_indices.shape[0]
                num_easy = easy_indices.shape[0]

                # 动态决定采样多少难样本和易样本
                if num_hard >= n_view / 2 and num_easy >= n_view / 2:
                    num_hard_keep = n_view // 2
                    num_easy_keep = n_view - num_hard_keep
                elif num_hard >= n_view / 2:
                    num_easy_keep = num_easy
                    num_hard_keep = n_view - num_easy_keep
                elif num_easy >= n_view / 2:
                    num_hard_keep = num_hard
                    num_easy_keep = n_view - num_hard_keep
                else:
                    # 理论上不会到这里，属于异常情况
                    print('this should be never touched! {} {} {}'.format(num_hard, num_easy, n_view))
                    raise Exception

                # 随机打乱后选取指定数量的样本
                perm = torch.randperm(num_hard)
                hard_indices = hard_indices[perm[:num_hard_keep]]
                perm = torch.randperm(num_easy)
                easy_indices = easy_indices[perm[:num_easy_keep]]

                # 合并难样本和易样本
                indices = torch.cat((hard_indices, easy_indices), dim=0)

                # 把采样到的特征保存到X_中
                X_[X_ptr, :, :] = X[ii, indices, :].squeeze(1)
                y_[X_ptr] = cls_id  # 保存对应的类别标签
                X_ptr += 1

        return X_, y_

    # 对比学习损失计算
    def _contrastive(self, feats_, labels_):
        '''
        feats_: 采样到的特征 (total_classes, n_view, feature_dim)
        labels_: 每个类别对应的标签 (total_classes,)
        '''
        anchor_num, n_view = feats_.shape[0], feats_.shape[1]

        # 生成标签的相似性矩阵（相同类别为1，否则为0）
        labels_ = labels_.contiguous().view(-1, 1)
        mask = torch.eq(labels_, torch.transpose(labels_, 0, 1)).float().cuda()

        # 将每个view展平
        contrast_count = n_view
        contrast_feature = torch.cat(torch.unbind(feats_, dim=1), dim=0)

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        # 论文中深度对比损失的公式实现
        # 计算相似度 (点积除以温度)
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, torch.transpose(contrast_feature, 0, 1)),
                                        self.temperature)
        # 为了数值稳定性，减去每行的最大值
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()  # sim(fi,fj)

        # 扩展mask
        mask = mask.repeat(anchor_count, contrast_count)
        neg_mask = 1 - mask  # 负样本掩码

        # 自己与自己不要算入正样本
        logits_mask = torch.ones_like(mask).scatter_(1,
                                                     torch.arange(anchor_num * anchor_count).view(-1, 1).cuda(),
                                                     0)
        mask = mask * logits_mask

        # 负样本得分
        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True)

        # 正负样本得分
        exp_logits = torch.exp(logits)

        # 计算每个anchor的对数概率
        log_prob = logits - torch.log(exp_logits + neg_logits + 1e-8)  # 论文的公式9

        # 只保留正样本，计算平均对数概率
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)  # 论文的公式10

        # 计算最终loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss

    # 前向传播
    def forward(self, feats, labels=None, predict=None):
        '''
        feats: 特征图 (B, C, H, W)
        labels: 真值标签图 (B, 1, H', W')
        predict: 网络预测的标签 (B, K, H', W')
        '''
        # 将label resize到feature的大小
        labels = torch.nn.functional.interpolate(labels.unsqueeze(1).float(),
                                                 (feats.shape[2], feats.shape[3]), mode='nearest')

        labels = labels.squeeze(1).long()
        assert labels.shape[-1] == feats.shape[-1], '{} {}'.format(labels.shape, feats.shape)
        predict = torch.nn.functional.interpolate(predict.float(),
                                                 (feats.shape[2], feats.shape[3]), mode='nearest')
        batch_size = feats.shape[0]

        # 展平label和预测
        labels = labels.contiguous().view(batch_size, -1)
        predict = torch.argmax(predict, dim=1)  # 输出维度为 BxHxW
        predict = predict.contiguous().view(batch_size, -1)

        feats = F.normalize(feats, dim=1)
        # 将特征从 (B, C, H, W) 转成 (B, H*W, C)
        feats = feats.permute(0, 2, 3, 1)
        feats = feats.contiguous().view(feats.shape[0], -1, feats.shape[-1])

        # 难易样本采样
        feats_, labels_ = self._hard_anchor_sampling(feats, labels, predict)

        # 计算对比损失
        loss = self._contrastive(feats_, labels_)
        return loss

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

class EdgeLoss(nn.Module):
    def __init__(self, n_classes=19, radius=1, alpha=0.01):
        super(EdgeLoss, self).__init__()
        self.n_classes = n_classes
        self.radius = radius
        self.alpha = alpha

    def forward(self, logits, label):
    #    prediction = F.softmax(logits, dim=1)
        ks = 2 * self.radius + 1
        filt1 = torch.ones(1, 1, ks, ks)
        filt1[:, :, self.radius:2 * self.radius, self.radius:2 * self.radius] = -8
        filt1.requires_grad = False
        filt1 = filt1.cuda()
        label = label.unsqueeze(1)
        lbedge = F.conv2d(label.float(), filt1, bias=None, stride=1, padding=self.radius)
        lbedge = 1 - torch.eq(lbedge, 0).float()

    #    filt2 = torch.ones(self.n_classes, 1, ks, ks)
    #    filt2[:, :, self.radius:2 * self.radius, self.radius:2 * self.radius] = -8
    #    filt2.requires_grad = False
    #    filt2 = filt2.cuda()
        prededge = logits

        norm = torch.sum(torch.pow(prededge, 2), 1).unsqueeze(1)
        prededge = norm / (norm + self.alpha)

        return BinaryDiceLoss()(prededge.float(), lbedge.float())

def KD_KLDivLoss(Stu_output, Tea_output, temperature):
    T = temperature
    KD_loss = nn.KLDivLoss()(F.log_softmax(Stu_output / T, dim=1), F.softmax(Tea_output / T, dim=1))
    KD_loss = KD_loss * T * T
    return KD_loss

class AFD_semantic(nn.Module):
    '''
    Pay Attention to Features, Transfer Learn Faster CNNs
    https://openreview.net/pdf?id=ryxyCeHtPB
    '''

    def __init__(self, in_channels, att_f):
        super(AFD_semantic, self).__init__()
        mid_channels = int(in_channels * att_f)

        self.attention = nn.Sequential(*[
            nn.Conv2d(in_channels, mid_channels, 3, 1, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, 3, 1, 1, bias=True)
        ])
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, fm_s, fm_t, eps=1e-6):

        fm_t_pooled = self.avg_pool(fm_t)
        rho = self.attention(fm_t_pooled)
        rho = torch.sigmoid(rho.squeeze())
        rho = rho / torch.sum(rho, dim=1, keepdim=True)

        fm_s_norm = torch.norm(fm_s, dim=(2, 3), keepdim=True)
        fm_s = torch.div(fm_s, fm_s_norm + eps)
        fm_t_norm = torch.norm(fm_t, dim=(2, 3), keepdim=True)
        fm_t = torch.div(fm_t, fm_t_norm + eps)

        dets_vec = fm_s.detach().cpu().numpy().flatten()
        gts_vec = fm_t.detach().cpu().numpy().flatten()
        dot_product = np.dot(dets_vec, gts_vec)
        norm_dets = np.linalg.norm(dets_vec)
        norm_gts = np.linalg.norm(gts_vec)
        eps = 1e-8
        norm_product = norm_dets * norm_gts + eps
        cosine_similarity = dot_product / norm_product
        beta = (cosine_similarity + 1) / 2
        LIPU_loss = KD_KLDivLoss(fm_s, fm_t.detach(), temperature=10)

        loss = beta * LIPU_loss
        #     loss = LIPU_loss
        #     loss = loss.sum(loss).mean(0)

        #      loss = rho * torch.pow(fm_s - fm_t, 2).mean(dim=(2, 3))
        #       loss = loss.sum(1).mean(0)

        return loss

class AFD_spatial(nn.Module):
    '''
    Pay Attention to Features, Transfer Learn Faster CNNs
    https://openreview.net/pdf?id=ryxyCeHtPB
    '''

    def __init__(self, in_channels):
        super(AFD_spatial, self).__init__()

        self.attention = nn.Sequential(*[
            nn.Conv2d(in_channels, 1, 3, 1, 1)
        ])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, fm_s, fm_t, eps=1e-6):

     #   rho = self.attention(fm_t)
     #   rho = torch.sigmoid(rho)
     #   rho = rho / torch.sum(rho, dim=(2, 3), keepdim=True)

        fm_s_norm = torch.norm(fm_s, dim=1, keepdim=True)
        fm_s = torch.div(fm_s, fm_s_norm + eps)
        fm_t_norm = torch.norm(fm_t, dim=1, keepdim=True)
        fm_t = torch.div(fm_t, fm_t_norm + eps)



        dets_vec = fm_s.detach().cpu().numpy().flatten()
        gts_vec = fm_t.detach().cpu().numpy().flatten()
        dot_product = np.dot(dets_vec, gts_vec)
        norm_dets = np.linalg.norm(dets_vec)
        norm_gts = np.linalg.norm(gts_vec)
        eps = 1e-8
        norm_product = norm_dets * norm_gts + eps
        cosine_similarity = dot_product / norm_product
        beta = (cosine_similarity + 1) / 2

        LIPU_loss = KD_KLDivLoss(fm_s, fm_t.detach(), temperature=10)

        loss = beta * LIPU_loss
        #    loss =  LIPU_loss
        #      loss = loss.sum(1).mean(0)
        #     loss = rho * torch.pow(fm_s - fm_t, 2).mean(dim=1, keepdim=True)
        #      loss =torch.sum(loss,dim=(2,3)).mean(0)
        return loss

class CrossModalKD(nn.Module):
    def __init__(self, T=10.0, eps=1e-8):
        """
        Cross-Modal Knowledge Distillation Loss Module

        Args:
            T (float): Temperature for softmax smoothing
            eps (float): Small constant to avoid divide-by-zero
        """
        super(CrossModalKD, self).__init__()
        self.T = T
        self.eps = eps

    def compute_entropy(self, prob_map):
        """
        计算每个像素点的类别概率分布的信息熵，并进行最大值归一化

        :param prob_map: Tensor，大小为 [B, K, H, W]，其中 B 是 batch 大小，K 是类别数，H, W 是图像大小
        :return: Tensor，大小为 [B, H, W]，每个像素的信息熵，归一化到 [0, 1]
        """
        # 避免 log(0) 的情况，加入一个小的常数 epsilon
        epsilon = 1e-8
        # 对概率图进行 softmax 归一化，确保每个类别的概率之和为 1
        prob_map = F.softmax(prob_map, dim=1)  # 在类别维度上计算 softmax
        # 计算信息熵：-sum(p_k * log(p_k))
        entropy_map = -torch.sum(prob_map * torch.log(prob_map + epsilon), dim=1)  # 沿类别维度求和
        # 最大信息熵为 log(K)
        K = prob_map.shape[1]  # 获取类别数
        max_entropy = torch.log(torch.tensor(K, dtype=torch.float32))  # 计算最大信息熵
        # 对信息熵进行最大值归一化
        normalized_entropy_map = entropy_map / max_entropy  # 归一化到 [0, 1]

        return normalized_entropy_map  # 返回大小为 [B, H, W] 的归一化信息熵图

    def cal_cos(self, fm_s, fm_t, eps=1e-8):

        fm_s_norm = torch.norm(fm_s, dim=1, keepdim=True)
        fm_s = torch.div(fm_s, fm_s_norm + eps)
        fm_t_norm = torch.norm(fm_t, dim=1, keepdim=True)
        fm_t = torch.div(fm_t, fm_t_norm + eps)
        dets_vec = fm_s.detach()
        gts_vec = fm_t.detach()
        dot_product = torch.sum(dets_vec * gts_vec, dim=1, keepdim=True)
        dot_product = (dot_product + 1.0) /2
        return dot_product


    def forward(self, pre_rgb, pre_dsm, mask=None):
        """
        Args:
            pre_rgb: Tensor (B, C, H, W) - logits from RGB branch
            pre_dsm: Tensor (B, C, H, W) - logits from DSM branch
            mask: Optional (B, 1, H, W) binary mask to exclude some pixels

        Returns:
            final_loss: Scalar weighted symmetric KL divergence loss
        """
        T = self.T
        eps = self.eps

        # Temperature-scaled probabilities
        dsm_prob = F.softmax(pre_dsm / T, dim=1)
        rgb_log_prob = F.log_softmax(pre_rgb / T, dim=1)

        # KL divergence in both directions
        kl_rgb_to_dsm = F.kl_div(rgb_log_prob, dsm_prob.detach(), reduction='none').sum(dim=1)  # (B, H, W)
       # kl_dsm_to_rgb = F.kl_div(dsm_log_prob, rgb_prob, reduction='none').sum(dim=1)  # (B, H, W)
        # Symmetric KL
        kl_loss = kl_rgb_to_dsm  # (B, H, W)
        weight_map = self.cal_cos(pre_rgb, pre_dsm).unsqueeze(1)  # (B, 1, H, W)

        # Weighted mean KL
        weighted_kl = kl_loss * weight_map.squeeze(1)
        final_loss = (T * T) * torch.mean(weighted_kl)

        return final_loss

class C2AHSegFormerLoss(nn.Module):
    def __init__(self, ignore_index=255, max_samples=1024, max_views=100):
        super(C2AHSegFormerLoss, self).__init__()
        self.main_loss = JointLoss(SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
                                   DiceLoss(smooth=0.05, ignore_index=ignore_index), 1.0, 1.0)
        self.ConsistencyLoss = CrossModalKD(T=10.0, eps=1e-8)

     #   self.ConsistencyLoss = ConsistencyLoss(use_attention=True)  # False

    def forward(self, logits, labels, dsm):

        if self.training and len(logits) == 7:
            out, RGB_Pre1, DSM_Pre1, RGB_Pre2, DSM_Pre2, RGB_Pre3, DSM_Pre3 = logits
            #out, RGB_Pre1, DSM_Pre1, RGB_Pre2, DSM_Pre2 = logits
            weight1 = 1.0  # 0.6
            weight2 = 1.0  # 0.3
            weight3 = 1.0  # 0.2
            weight_rgb = 1.0
            weight_dsm = 1.0
            loss_main = self.main_loss(out, labels)

            loss_rgb1 = (self.main_loss(RGB_Pre1, labels)) * weight1
            loss_rgb2 = (self.main_loss(RGB_Pre2, labels)) * weight2
            loss_rgb3 = (self.main_loss(RGB_Pre3, labels)) * weight3
            loss_rgb = (loss_rgb1 + loss_rgb2+loss_rgb3) * weight_rgb

            loss_dsm1 = (self.main_loss(DSM_Pre1, labels)) * weight1
            loss_dsm2 = (self.main_loss(DSM_Pre2, labels)) * weight2
            loss_dsm3 = (self.main_loss(DSM_Pre3, labels)) * weight3

            loss_dsm = (loss_dsm1 + loss_dsm2+loss_dsm3) * weight_dsm

            loss_cons1 = self.ConsistencyLoss(RGB_Pre1, DSM_Pre1)
            loss_cons2 = self.ConsistencyLoss(RGB_Pre2, DSM_Pre2)
            loss_cons3 = self.ConsistencyLoss(RGB_Pre3, DSM_Pre3)
            loss_rgb2dsm = (loss_cons1 + loss_cons2+loss_cons3) * 1.0
           # loss_cons11 = self.ConsistencyLoss(DSM_Pre1, RGB_Pre1)
          #  loss_cons22 = self.ConsistencyLoss(DSM_Pre2, RGB_Pre2)
           # loss_cons33 = self.ConsistencyLoss(DSM_Pre3, RGB_Pre3)
          #  loss_dsm2rgb = (loss_cons11 + loss_cons22+ loss_cons33)*1

            #  loss = loss_main + loss_rgb + loss_dsm  + loss_dsm2rgb
            loss = loss_main + loss_rgb + loss_dsm + loss_rgb2dsm


        else:
            loss = self.main_loss(logits, labels)

        return loss

if __name__ == "__main__":
    B, C, H, W = 2, 192, 64, 64
    NUM_CLASSES = 6

    # 模拟语义分割输出 logits 和 features
    logits = torch.randn(B, NUM_CLASSES, H, W).cuda()
    features = torch.randn(B, C, 64, 64).cuda()
    labels = torch.randint(0, NUM_CLASSES, (B, H, W)).cuda()
    features = F.normalize(features, dim=1).cuda()  # 在通道维上归一化，保持每个向量单位长度

    # 实例化 In2NeCTLoss
    pcl = PixelContrastLoss().cuda()

    # 总损失（用于反向传播）
    l_con= pcl( features, labels, logits)
    print("l_con:", l_con.item())
