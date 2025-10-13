import timm
import torch.nn.functional as F
import torch
from torch import nn
#from functools import partial
#from torch.autograd import Variable
#from einops import rearrange
#from timm.models.layers import DropPath
#import cv2
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from .SwinTransformer import *

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1,
                 norm_layer=nn.BatchNorm2d, linearity=nn.ReLU6, groups=1, bias=False, mode="square"):
        super().__init__()

        if mode == "vertical":
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), bias=bias,
                                  dilation=(dilation, dilation), stride=(stride, stride),
                                  padding=(((stride - 1) + dilation * (kernel_size - 1)) // 2, 0), groups=groups
                                  )
        elif mode == "horizontal":
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size), bias=bias,
                                  dilation=(dilation, dilation), stride=(stride, stride),
                                  padding=(0, ((stride - 1) + dilation * (kernel_size - 1)) // 2), groups=groups
                                  )
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                                  dilation=dilation, stride=stride,
                                  padding=((stride - 1) + dilation * (kernel_size - 1)) // 2, groups=groups)


        # If norm_layer is provided, initialize it, otherwise None
        self.with_batchnorm = norm_layer is not None
        if self.with_batchnorm:
            self.bn = norm_layer(out_channels)

        # If linearity is provided, initialize it, otherwise None
        self.with_nonlinearity = linearity is not None
        if self.with_nonlinearity:
            self.relu = linearity()

    def forward(self, x):
        x = self.conv(x)
        if self.with_batchnorm:
            x = self.bn(x)
        if self.with_nonlinearity:
            x = self.relu(x)
        return x

class SeparableConvBlock(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size=3, dilation=1, stride=1,
                 norm_layer=nn.BatchNorm2d, linearity=nn.ReLU6, bias=False, mode="square"):
        super().__init__()

        if mode == "vertical":
            self.dwc = nn.Conv2d(in_channels, in_channels, kernel_size=(kernel_size, 1), bias=bias,
                                  dilation=(dilation, dilation), stride=(stride, stride),
                                  padding=(((stride - 1) + dilation * (kernel_size - 1)) // 2, 0), groups=in_channels
                                  )
        elif mode == "horizontal":
            self.dwc = nn.Conv2d(in_channels, in_channels, kernel_size=(1, kernel_size), bias=bias,
                                  dilation=(dilation, dilation), stride=(stride, stride),
                                  padding=(0, ((stride - 1) + dilation * (kernel_size - 1)) // 2), groups=in_channels
                                  )
        else:
            self.dwc = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, bias=bias,
                                  dilation=dilation, stride=stride,
                                  padding=((stride - 1) + dilation * (kernel_size - 1)) // 2, groups=in_channels)


        self.pwc = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        # If norm_layer is provided, initialize it, otherwise None
        self.with_batchnorm = norm_layer is not None
        if self.with_batchnorm:
            self.bn = norm_layer(in_channels)

        # If linearity is provided, initialize it, otherwise None
        self.with_nonlinearity = linearity is not None
        if self.with_nonlinearity:
            self.relu = linearity()


    def forward(self, x):
        x = self.dwc(x)
        if self.with_batchnorm:
            x = self.bn(x)
        x = self.pwc(x)
        if self.with_nonlinearity:
            x = self.relu(x)
        return x

class GroupedLinear(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias=False):
        super(GroupedLinear, self).__init__()
        self.num_groups = num_groups
        self.in_features_per_group = in_features // num_groups
        self.out_features_per_group = out_features // num_groups
        assert in_features % num_groups == 0, "in_features must be divisible by num_groups"
        assert out_features % num_groups == 0, "out_features must be divisible by num_groups"

        # 定义每个组的线性变换
        self.linears = nn.ModuleList([
            nn.Linear(self.in_features_per_group, self.out_features_per_group, bias=bias)
            for _ in range(num_groups)
        ])

    def forward(self, x):
        b, N, c_in = x.size()
        # 直接reshape，而不是split
        x_reshaped = x.view(b, N, self.num_groups,
                            self.in_features_per_group)  # reshape to (b, N, num_groups, in_features_per_group)
        x_reshaped = x_reshaped.permute(0, 2, 1, 3)  # Change shape to (b, num_groups, N, in_features_per_group)

        # 每个组独立进行线性变换
        out_split = [self.linears[i](x_reshaped[:, i, :, :]) for i in range(self.num_groups)]

        # 将各组的输出合并
        return torch.cat(out_split, dim=-1)

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)

class OCM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()

        self.Recv = ConvBlock(in_channels, out_channels, kernel_size=kernel_size, stride=stride, mode="vertical")
        self.Rech = ConvBlock(in_channels, out_channels, kernel_size=kernel_size, stride=stride, mode="horizontal")
        self.conv = ConvBlock(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=1, mode="square")

    def forward(self, x):

        feats = self.Recv(x) + self.Rech(x) + self.conv(x)

        return feats

class OACM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,dilation=[1, 2, 4, 8]):
        super().__init__()

        self.preconv = ConvBlock(in_channels, out_channels, kernel_size=1, stride=1)

        self.Recv = ConvBlock(out_channels, out_channels//4, kernel_size=kernel_size, mode="vertical")
        self.Recv2 = ConvBlock(out_channels, out_channels//4, kernel_size=kernel_size, dilation=dilation[1], mode="vertical")
        self.Recv4 = ConvBlock(out_channels, out_channels//4, kernel_size=kernel_size, dilation=dilation[2], mode="vertical")
        self.Recv8 = ConvBlock(out_channels, out_channels//4, kernel_size=kernel_size, dilation=dilation[3], mode="vertical")

        self.Rech = ConvBlock(out_channels, out_channels//4, kernel_size=kernel_size, mode="horizontal")
        self.Rech2 = ConvBlock(out_channels, out_channels//4, kernel_size=kernel_size, dilation=dilation[1], mode="horizontal")
        self.Rech4 = ConvBlock(out_channels, out_channels//4, kernel_size=kernel_size, dilation=dilation[2], mode="horizontal")
        self.Rech8 = ConvBlock(out_channels, out_channels//4, kernel_size=kernel_size, dilation=dilation[3], mode="horizontal")

        self.conv = ConvBlock(out_channels, out_channels//4, kernel_size=kernel_size, stride=1, dilation=1)
        self.conv2 = ConvBlock(out_channels, out_channels//4, kernel_size=kernel_size, stride=1, dilation=dilation[1])
        self.conv4 = ConvBlock(out_channels, out_channels//4, kernel_size=kernel_size, stride=1, dilation=dilation[2])
        self.conv8 = ConvBlock(out_channels, out_channels//4, kernel_size=kernel_size, stride=1, dilation=dilation[3])

        self.convxout = ConvBlock(out_channels, out_channels, stride=1)


    def forward(self, x):

        x = self.preconv(x)

        featsv = torch.cat((self.Recv(x), self.Recv2(x), self.Recv4(x), self.Recv8(x)),dim=1)
        featsh = torch.cat((self.Rech(x), self.Rech2(x), self.Rech4(x), self.Rech8(x)),dim=1)
        feats = torch.cat((self.conv(x), self.conv2(x), self.conv4(x), self.conv8(x)),dim=1)
        out = featsv + featsh + feats

        out = self.convxout(out)

        return out

class MultiHeadAttention(nn.Module):
    '''
    input:
        query --- [N, T_q, query_dim]
        key --- [N, T_k, key_dim]
        mask --- [N, T_k]
    output:
        out --- [N, T_q, num_units]
        scores -- [h, N, T_q, T_k]
    '''

    def __init__(self, query_dim, key_dim, num_units, num_heads, Mode = 'conv', Group_Linear = False):
        super().__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.mode =Mode

        if Group_Linear ==True:
            if self.mode != 'no_conv':
                self.W_query = GroupedLinear(in_features=query_dim, out_features=query_dim, num_groups = num_heads)
            self.W_key = GroupedLinear(in_features=key_dim, out_features=key_dim, num_groups = num_heads)
            self.W_value = GroupedLinear(in_features=num_units, out_features=num_units, num_groups=num_heads)
        else:
            if self.mode != 'no_conv':
                self.W_query = nn.Linear(in_features=query_dim, out_features=query_dim, bias=False)
            self.W_key = nn.Linear(in_features=key_dim, out_features=key_dim, bias=False)
            self.W_value = nn.Linear(in_features=num_units, out_features=num_units, bias=False)

        self.out = nn.Linear(in_features=num_units, out_features=num_units, bias=False)

    def forward(self, query, value, mask=None):

        if self.mode=='no_conv':
            querys = query  # [N, T_q, num_units]
            keys = self.W_key(query) + query
        else:
            querys = self.W_query(query)  # [N, T_q, num_units]
            keys = self.W_key(query)  # [N, T_k, num_units]
        values = self.W_value(value)

        split_size_qk = self.key_dim // self.num_heads
        split_size = self.num_units // self.num_heads
        querys = torch.stack(torch.split(querys, split_size_qk, dim=2), dim=0)  # [h, N, T_q, num_units/h]
        keys = torch.stack(torch.split(keys, split_size_qk, dim=2), dim=0)  # [h, N, T_k, num_units/h]
        values = torch.stack(torch.split(values, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]

        scores = torch.matmul(querys, keys.transpose(2, 3))  # [h, N, T_q, T_k]
        scores = scores / (split_size_qk ** 0.5)
        scores = F.softmax(scores, dim=3)

        out = torch.matmul(scores, values)  # [h, N, T_q, num_units/h]
        out = torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, num_units]
        out = self.out(out)

        return out, scores

class Dual_MultiHeadAttention(nn.Module):
    '''
    input:
        query --- [N, T_q, query_dim]
        key --- [N, T_k, key_dim]
        mask --- [N, T_k]
    output:
        out --- [N, T_q, num_units]
        scores -- [h, N, T_q, T_k]
    '''

    def __init__(self, query_dim, key_dim, rgbv_dim, dsmv_dim, num_heads, Mode = 'conv', Group_Linear = False):
        super().__init__()

        self.query_dim = query_dim
        self.key_dim = key_dim
        self.rgbv_dim = rgbv_dim
        self.dsmv_dim = dsmv_dim
        self.num_heads = num_heads
        self.mode =Mode

        if Group_Linear ==True:
            if self.mode != 'no_conv':
                self.W_query = GroupedLinear(in_features=query_dim, out_features=query_dim, num_groups = num_heads)
            self.W_key = GroupedLinear(in_features=key_dim, out_features=key_dim, num_groups = num_heads)
            self.W_rgbvalue = GroupedLinear(in_features=rgbv_dim, out_features=rgbv_dim, num_groups=num_heads)
            self.W_dsmvalue = GroupedLinear(in_features=dsmv_dim, out_features=dsmv_dim, num_groups=num_heads)
        else:
            if self.mode != 'no_conv':
                self.W_query = nn.Linear(in_features=query_dim, out_features=query_dim, bias=False)
            self.W_key = nn.Linear(in_features=key_dim, out_features=key_dim, bias=False)
            self.W_rgbvalue = nn.Linear(in_features=rgbv_dim, out_features=rgbv_dim, bias=False)
            self.W_dsmvalue = nn.Linear(in_features=dsmv_dim, out_features=dsmv_dim, bias=False)

        self.rgb_out = nn.Linear(in_features=rgbv_dim, out_features=rgbv_dim, bias=False)
        self.dsm_out = nn.Linear(in_features=dsmv_dim, out_features=dsmv_dim, bias=False)

    def forward(self, query, rgb_value, dsm_value):

        if self.mode=='no_conv':
            querys = query  # [N, T_q, num_units]
            keys = self.W_key(query) + query
        else:
            querys = self.W_query(query)  # [N, T_q, num_units]
            keys = self.W_key(query)  # [N, T_k, num_units]
        rgb_values = self.W_rgbvalue(rgb_value)
        dsm_values = self.W_dsmvalue(dsm_value)

        split_size_qk = self.key_dim // self.num_heads
        split_size_rgb = self.rgbv_dim // self.num_heads
        split_size_dsm = self.dsmv_dim // self.num_heads
        querys = torch.stack(torch.split(querys, split_size_qk, dim=2), dim=0)  # [h, N, T_q, num_units/h]
        keys = torch.stack(torch.split(keys, split_size_qk, dim=2), dim=0)  # [h, N, T_k, num_units/h]
        rgb_values = torch.stack(torch.split(rgb_values, split_size_rgb, dim=2), dim=0)  # [h, N, T_k, num_units/h]
        dsm_values = torch.stack(torch.split(dsm_values, split_size_dsm, dim=2), dim=0)  # [h, N, T_k, num_units/h]

        scores = torch.matmul(querys, keys.transpose(2, 3))  # [h, N, T_q, T_k]
        scores = scores / (split_size_qk ** 0.5)
        scores = F.softmax(scores, dim=3)

        ## RGB特征聚合
        rgb_out = torch.matmul(scores, rgb_values)  # [h, N, T_q, num_units/h]
        rgb_out = torch.cat(torch.split(rgb_out, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, num_units]
        rgb_out = self.rgb_out(rgb_out)

        ## DSM特征聚合
        dsm_out = torch.matmul(scores, dsm_values)  # [h, N, T_q, num_units/h]
        dsm_out = torch.cat(torch.split(dsm_out, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, num_units]
        dsm_out = self.dsm_out(dsm_out)

        return rgb_out, dsm_out, scores

class Channel_Selection(nn.Module):
    def __init__(self, channels, ratio=8):
        super(Channel_Selection, self).__init__()

        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.max_pooling = nn.AdaptiveMaxPool2d(1)

        self.fc_layers = nn.Sequential(
            ConvBlock(channels, channels // ratio, kernel_size=1, norm_layer=None, linearity=None),
            nn.ReLU(),
            ConvBlock(channels // ratio, channels, kernel_size=1, norm_layer=None, linearity=None)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape
        avg_x = self.avg_pooling(x).view(b, c, 1, 1)
        max_x = self.max_pooling(x).view(b, c, 1, 1)
        v = self.fc_layers(avg_x) + self.fc_layers(max_x)
        v = self.sigmoid(v).view(b, c, 1, 1)

        return v

class AdaptiveLocalFeatureExtraction(nn.Module):
    def __init__(self, dim, ratio=8, mode='v'):
        super(AdaptiveLocalFeatureExtraction, self).__init__()

        self.preconv = ConvBlock(in_channels=dim, out_channels=dim, kernel_size=3, linearity=None)

        self.Channel_Selection = Channel_Selection(channels=dim, ratio=ratio)

        if mode == 'v':
            self.convbase = ConvBlock(in_channels=dim, out_channels=dim, kernel_size=3, linearity=None, mode="vertical")
            self.convlarge = ConvBlock(in_channels=dim, out_channels=dim, kernel_size=5, linearity=None, mode="vertical")
        elif mode == 'h':
            self.convbase = ConvBlock(in_channels=dim, out_channels=dim, kernel_size=3, linearity=None, mode="horizontal")
            self.convlarge = ConvBlock(in_channels=dim, out_channels=dim, kernel_size=5, linearity=None, mode="horizontal")
        else:
            self.convbase = ConvBlock(in_channels=dim, out_channels=dim, kernel_size=3, linearity=None)
            self.convlarge = ConvBlock(in_channels=dim, out_channels=dim, kernel_size=5, linearity=None)

        self.post_conv = SeparableConvBlock(in_channels=dim, out_channels=dim, kernel_size=3)

    def forward(self, x):

        s = self.Channel_Selection(self.preconv(x))
        x = self.post_conv(s * self.convbase(x) + (1 - s) * self.convlarge(x))

        return x

class GLTM(nn.Module):
    def __init__(self, dim=512, num_heads=6,  mlp_ratio=4,drop=0.,
                 drop_path=0., act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.norm1 = norm_layer(dim)

        self.msa_v = MultiHeadAttention(dim, dim, dim, num_heads)
        self.local_v = AdaptiveLocalFeatureExtraction(dim, ratio=8,mode='v')
        self.conv_v = ConvBlock(in_channels=dim, out_channels=dim, kernel_size=3, stride=1)

        self.msa_h = MultiHeadAttention(dim, dim, dim, num_heads)
        self.local_h = AdaptiveLocalFeatureExtraction(dim, ratio=8, mode='h')
        self.conv_h = ConvBlock(in_channels=dim, out_channels=dim, kernel_size=3, stride=1)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim // mlp_ratio)

        self.mlp = Mlp_decoder(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer,
                               drop=drop)
        self.norm2 = norm_layer(dim)

    def forward(self, x):

        b, c, h, w = x.size(0), x.size(1), x.size(2), x.size(3)


        vf = (x).clone()  # [:, :, :, :].contiguous()  # b,c,h,w
        vqk_view = x.clone().permute(0, 3, 2, 1).reshape(b * w, h, -1).contiguous()
        x, vscores = self.msa_v(vqk_view, vqk_view)  #
        x = x.reshape(b, w, h, c).permute(0, 3, 2, 1).contiguous()  # + vf

        x = x + self.local_v(vf)
        x = self.conv_v(x)

        hf = x.clone()  # [:, :, :, :].contiguous()  # b,c,h,w
        hqk_view = x.clone().permute(0, 2, 3, 1).reshape(b * h, w, -1).contiguous()  # b,h,w,c
        x, hscores = self.msa_h(hqk_view, hqk_view)
        x = x.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()  # + hf
        x = x + self.local_h(hf)
        x = self.conv_h(x)

        x = x + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x, vscores, hscores

class Mlp_decoder(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU6, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True)
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class DualChannelSplitConcat(nn.Module):
    def __init__(self, rgb_channels, dsm_channels, num_splits):
        super(DualChannelSplitConcat, self).__init__()
        assert rgb_channels % num_splits == 0, 
        assert dsm_channels % num_splits == 0, 

        self.num_splits = num_splits
        self.split_size1 = rgb_channels // num_splits
        self.split_size2 = dsm_channels // num_splits

    def forward(self, x1, x2):
        # 
        x1_split = torch.chunk(x1, self.num_splits, dim=1)  # (num_splits, B, split_size1, H, W)
        x2_split = torch.chunk(x2, self.num_splits, dim=1)  # (num_splits, B, split_size2, H, W)

        # 
        output_splits = []
        for i in range(self.num_splits):
            output_splits.append(x1_split[i])  #
            output_splits.append(x2_split[i])  # 

        # 
        output = torch.cat(output_splits, dim=1)

        return output

class Feature_Aux_Predictor(nn.Module):
    def __init__(self, fg_feats_dim = 192, bg_feats_dim = 192, num_splits = 6,
                Fea_mode='spilt', Aux_Fusion = True, aux_Fusion_ratio=1):  #Mode = 'all' 所有通道
        super(Feature_Aux_Predictor, self).__init__()

        assert fg_feats_dim % num_splits == 0, "The number of channels in the first tensor must be divisible by the number of splits."
        assert bg_feats_dim % num_splits == 0, "The number of channels in the first tensor must be divisible by the number of splits."

        self.num_splits = num_splits
        self.split_fg_size = fg_feats_dim // num_splits
        self.split_bg_size = bg_feats_dim // num_splits

        self.Fea_mode = Fea_mode
        self.Aux_Fusion = Aux_Fusion
        self.Aux_Fusion_ratio = Aux_Fusion

        if self.Fea_mode=='all':
            self.fg_prehead = nn.ModuleList(
                [nn.Sequential(
                    ConvBlock(fg_feats_dim, self.split_fg_size, kernel_size=1),
                    nn.Dropout(0.1),
                    ConvBlock(self.split_fg_size, 1, kernel_size=1,norm_layer=None, linearity=None),
                ) for _ in range(num_splits)]
            )
            self.bg_prehead = nn.ModuleList(
                [nn.Sequential(
                    ConvBlock(bg_feats_dim, self.split_bg_size, kernel_size=1),
                    nn.Dropout(0.1),
                    ConvBlock(self.split_bg_size, 1, kernel_size=1,norm_layer=None, linearity=None),
                ) for _ in range(num_splits)]
            )
        else:
            self.fg_prehead = nn.ModuleList(
                [nn.Sequential(
                    ConvBlock(self.split_fg_size, self.split_fg_size, kernel_size=1),
                    nn.Dropout(0.1),
                    ConvBlock(self.split_fg_size, 1, kernel_size=1, norm_layer=None, linearity=None),
                ) for _ in range(num_splits)]
            )

            self.bg_prehead = nn.ModuleList(
                [nn.Sequential(
                    ConvBlock(self.split_bg_size, self.split_bg_size, kernel_size=1),
                    nn.Dropout(0.1),
                    ConvBlock(self.split_bg_size, 1, kernel_size=1, norm_layer=None, linearity=None),
                ) for _ in range(num_splits)]
            )

        if self.Aux_Fusion_ratio == True:
            self.adcsc = DualChannelSplitConcat(num_classes * 2, num_classes * 2 * aux_Fusion_ratio, num_classes)
            self.fuse_out = nn.Sequential(ConvBlock(num_classes * 2 * (aux_Fusion_ratio + 1), num_classes * 2, groups=num_classes * 2),
                                          nn.Dropout(0.1),
                                          ConvBlock(num_classes * 2, num_classes * 2, kernel_size=1, norm_layer=None, linearity=None))

    def forward(self, fg_feats, bg_feats = None, aux_pre=None):

        h1, w1 = fg_feats.size(2), fg_feats.size(3)

        if bg_feats == None:
            bg_feats = fg_feats

        if self.Fea_mode =='all':
            fg_pre = [conv(fg_feats) for conv in self.fg_prehead]
            bg_pre = [conv(bg_feats) for conv in self.bg_prehead]
        else:
            # Split the first tensor into K parts
            fg_feats_split = torch.split(fg_feats, self.split_fg_size, dim=1)
            bg_feats_split = torch.split(bg_feats, self.split_fg_size, dim=1)
            # Predict for each split part
            fg_pre = [conv(t1) for conv, t1 in zip(self.fg_prehead, fg_feats_split)]
            bg_pre = [conv(t2) for conv, t2 in zip(self.bg_prehead, bg_feats_split)]

        combined_predictions = [F.softmax(torch.cat((fpred, bpred), dim=1), dim=1) for fpred, bpred in
                                zip(fg_pre, bg_pre)]

        combined_compute_entropy = [(compute_entropy(singleprd)* top2_prob_difference(singleprd)).unsqueeze(1) for singleprd in combined_predictions]

        # Concatenate all the softmax outputs along the channel dimension
        single_class_Pre = torch.cat(combined_predictions, dim=1)
        single_Pre_diff = torch.cat(combined_compute_entropy, dim=1)

        if (self.Aux_Fusion) & (aux_pre != None):
            aux = F.interpolate(aux_pre, size=(h1, w1), mode='bilinear', align_corners=False)
            single_class_Pre = self.fuse_out(self.adcsc(single_class_Pre, aux))

        return single_class_Pre, single_Pre_diff

class MutiClass_SegHead(nn.Module):
    def __init__(self, in_channels=64, num_classes=8):
        super().__init__()
        self.conv = ConvBlock(in_channels, in_channels)
        self.drop = nn.Dropout(0.1)
        self.conv_out = ConvBlock(in_channels, num_classes, kernel_size=1,norm_layer = None, linearity= None)
    def forward(self, x):
        feat = self.conv(x)
        feat = self.drop(feat)
        feat = self.conv_out(feat)
        return feat

def top2_prob_difference1(prob_map):
    """
    :param prob_map: Tensor，[B, K, H, W]，
    :return: Tensor，[B, H, W]，
    """
    # 
    prob_map = F.softmax(prob_map, dim=1)
    top2_probs, _ = torch.topk(prob_map, 2, dim=1)  # top2_probs: [B, 2, H, W]
    # 
    prob_diff = top2_probs[:, 0] - top2_probs[:, 1]  # 
    # 
    prob_diff = torch.clamp(prob_diff, min=0)  # 
    # 
    # 
    prob_diff_min = prob_diff.view(prob_diff.shape[0], -1).min(dim=1, keepdim=True)[0].view(prob_diff.shape[0], 1,
                                                                                            1)  # 
    prob_diff_max = prob_diff.view(prob_diff.shape[0], -1).max(dim=1, keepdim=True)[0].view(prob_diff.shape[0], 1,
                                                                                            1)  # 

    # 
    prob_diff_normalized = (prob_diff - prob_diff_min) / (prob_diff_max - prob_diff_min + 1e-8)  # 防止除以 0

    return prob_diff_normalized  # 

def top2_prob_difference(prob_map):
    """
    """
    #
    prob_map = F.softmax(prob_map, dim=1)
    top2_probs, _ = torch.topk(prob_map, 2, dim=1)  # top2_probs: [B, 2, H, W]
    # 
    prob_diff = torch.abs(top2_probs[:, 0] - top2_probs[:, 1])  #
    prob_diff_normalized = prob_diff  # 

    return prob_diff_normalized  # 

def compute_entropy(prob_map):
    """
    """
    # 
    epsilon = 1e-8
    # 
    prob_map = F.softmax(prob_map, dim=1)  # 
    # -sum(p_k * log(p_k))
    entropy_map = -torch.sum(prob_map * torch.log(prob_map + epsilon), dim=1)  # 
    # 
    K = prob_map.shape[1]  # 
    max_entropy = torch.log(torch.tensor(K, dtype=torch.float32))  # 

    # 
    normalized_entropy_map = entropy_map / max_entropy  # 

    return 1.0 - normalized_entropy_map  # 

class Dual_Aux_Enhancement(nn.Module):
    def __init__(self, num_classes=6, mode='mutli', eps=1e-8, aux=False):
        super().__init__()

        self.num_classes = num_classes

        self.eps = eps
        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)

        if aux:
            self.aux_weights = nn.Parameter(torch.ones(1, dtype=torch.float32), requires_grad=True)

    def forward(self, rgb_pre, dsm_pre, aux_pre=None):

        rgb_weight = (top2_prob_difference(rgb_pre) * compute_entropy(rgb_pre)).unsqueeze(dim=1)  # B 1 H W
        dsm_weight = (top2_prob_difference(dsm_pre) * compute_entropy(dsm_pre)).unsqueeze(dim=1) # B 1 H W

        weights = nn.ReLU6()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        fusion_pre = fuse_weights[0] * rgb_weight *  rgb_pre + fuse_weights[1] * dsm_weight * dsm_pre

        if aux_pre is not None:
            aux_pre = F.interpolate(aux_pre, size=(rgb_pre.size(2), rgb_pre.size(3)), mode='bilinear', align_corners=False)
            aux_weight = (top2_prob_difference(aux_pre) * compute_entropy(aux_pre)).unsqueeze(dim=1)  # B 1 H W
            aux_weights = nn.ReLU6()(self.aux_weights)
            fusion_pre = fusion_pre + aux_weights * aux_weight * aux_pre


        return fusion_pre

class GLSTB(nn.Module):
    def __init__(self, query_dim=6, key_dim=6, rgbv_dim=192, dsmv_dim=96, num_classes=6, nums_heads=6, mlp_ratio=8,
                 weight_ratio=1.0):
        super(GLSTB, self).__init__()

        self.query_dim = query_dim
        self.key_dim = key_dim
        self.rgbv_dim = rgbv_dim
        self.dsmv_dim = dsmv_dim
        self.nums_heads = nums_heads
        self.num_classes = num_classes
        self.weight_ratio = weight_ratio
        self.mlp_ratio = mlp_ratio
        self.rgb_split_channels = rgbv_dim // num_classes
        self.dsm_split_channels = dsmv_dim // num_classes

        # 
        self.msa_v, self.rgb_local_v, self.rgb_conv_v, self.dsm_local_v, self.dsm_conv_v  \
            = self._init_heads(query_dim, key_dim, rgbv_dim, dsmv_dim, num_classes, nums_heads,mlp_ratio, 'v')
        self.msa_h, self.rgb_local_h, self.rgb_conv_h, self.dsm_local_h, self.dsm_conv_h \
            = self._init_heads(query_dim, key_dim, rgbv_dim, dsmv_dim, num_classes, nums_heads,mlp_ratio, 'h')

    def _init_heads(self, query_dim, key_dim, rgbv_dim, dsmv_dim, num_classes, nums_heads, mlp_ratio, mode):
        # 
        per_class_query_dim = query_dim // num_classes
        per_class_key_dim = key_dim // num_classes
        per_class_rgbv_dim = rgbv_dim // num_classes
        per_class_dsmv_dim = dsmv_dim // num_classes

        # 
        if per_class_query_dim < nums_heads:
            repeat_factor = nums_heads // per_class_query_dim
            remainder = nums_heads % per_class_query_dim
            new_query_dim = per_class_query_dim * repeat_factor + remainder
        else:
            new_query_dim = per_class_query_dim

        msa = nn.ModuleList(
            [Dual_MultiHeadAttention(new_query_dim, new_query_dim, per_class_rgbv_dim, per_class_dsmv_dim,
                                     nums_heads, Mode='no_conv', Group_Linear = True) for _ in
             range(num_classes)])

        rgb_local = nn.ModuleList(
            [AdaptiveLocalFeatureExtraction(per_class_rgbv_dim, ratio=mlp_ratio, mode=mode) for _ in range(num_classes)])
        rgb_conv = nn.ModuleList([ConvBlock(in_channels= per_class_rgbv_dim, out_channels= per_class_rgbv_dim, kernel_size=3, stride=1) for _ in
                              range(num_classes)])

        dsm_local = nn.ModuleList(
            [AdaptiveLocalFeatureExtraction(per_class_dsmv_dim, ratio=mlp_ratio, mode=mode) for _ in
             range(num_classes)])
        dsm_conv = nn.ModuleList(
            [ConvBlock(in_channels=per_class_dsmv_dim, out_channels=per_class_dsmv_dim, kernel_size=3, stride=1) for _
             in
             range(num_classes)])

        return msa, rgb_local, rgb_conv, dsm_local, dsm_conv

    def _process_heads(self, qk, rgb, dsm, msa_heads, rgb_local_heads, rgb_conv_heads,
                       dsm_local_heads, dsm_conv_heads, is_vertical=True):
        b, c_rgb, h, w = rgb.size()
        c_dsm = dsm.size(1)
        k = qk.size(1)

        # 
        if k // self.num_classes < self.nums_heads:
            repeat_factor = self.nums_heads // (k // self.num_classes)
            remainder = self.nums_heads % (k // self.num_classes)
            expanded_qk = [qk[:, i:i + 1, :, :].repeat(1, repeat_factor, 1, 1) for i in range(k)]
            qk = torch.cat(expanded_qk, dim=1)
            if remainder > 0:
                qk = torch.cat([qk, qk[:, :remainder, :, :]], dim=1)

        # 
        if is_vertical:
            qk_view = qk.permute(0, 3, 2, 1).reshape(b * w, h, -1).contiguous()
            rgb_view = rgb.permute(0, 3, 2, 1).reshape(b * w, h, c_rgb).contiguous()
            dsm_view = dsm.permute(0, 3, 2, 1).reshape(b * w, h, c_dsm).contiguous()
        else:
            qk_view = qk.permute(0, 2, 3, 1).reshape(b * h, w, -1).contiguous()
            rgb_view = rgb.permute(0, 2, 3, 1).reshape(b * h, w, c_rgb).contiguous()
            dsm_view = dsm.permute(0, 2, 3, 1).reshape(b * h, w, c_dsm).contiguous()

        # 
        qk_chunks = torch.chunk(qk_view, self.num_classes, dim=-1)
        rgb_chunks = torch.chunk(rgb_view, self.num_classes, dim=-1)
        dsm_chunks = torch.chunk(dsm_view, self.num_classes, dim=-1)

        rgb_outputs, dsm_outputs, scores_o = [], [], []
        for i in range(self.num_classes):
            rgb_attention_out, dsm_attention_out, scores = msa_heads[i](qk_chunks[i], rgb_chunks[i], dsm_chunks[i])
            rgb_attention_out = rgb_attention_out.reshape(b, w if is_vertical else h, h if is_vertical else w, self.rgb_split_channels)
            rgb_attention_out = rgb_attention_out.permute(0, 3, 2, 1).contiguous() if is_vertical else rgb_attention_out.permute(0,3,1,2).contiguous()
            dsm_attention_out = dsm_attention_out.reshape(b, w if is_vertical else h, h if is_vertical else w,
                                                  self.dsm_split_channels)
            dsm_attention_out = dsm_attention_out.permute(0, 3, 2, 1).contiguous() if is_vertical else dsm_attention_out.permute(0,
                                                                                                                     3,
                                                                                                                     1,
                                                                                                                     2).contiguous()


            rgb_attention_out = rgb_attention_out + rgb_local_heads[i](rgb[:, i * self.rgb_split_channels:(i + 1) * self.rgb_split_channels, :, :])
            dsm_attention_out = dsm_attention_out + dsm_local_heads[i](dsm[:, i * self.dsm_split_channels:(i + 1) * self.dsm_split_channels, :, :])

            rgb_outputs.append(rgb_conv_heads[i](rgb_attention_out))
            dsm_outputs.append(dsm_conv_heads[i](dsm_attention_out))
            scores_o.append(scores)

        return torch.cat(rgb_outputs, dim=1),torch.cat(dsm_outputs, dim=1), torch.cat(scores_o, dim=0)

    def forward(self, qk, x, dsm):
        qk = qk * self.weight_ratio

        # 
        rgb_v, dsm_v, vscores = self._process_heads(qk, x.clone(), dsm.clone(), self.msa_v, self.rgb_local_v,
                                          self.rgb_conv_v, self.dsm_local_v, self.dsm_conv_v, is_vertical=True)

        # 
        rgb_h, dsm_h, hscores = self._process_heads(qk, rgb_v.clone(), dsm_v.clone(), self.msa_h, self.rgb_local_h,
                                                    self.rgb_conv_h, self.dsm_local_h, self.dsm_conv_h, is_vertical=False)

        return rgb_h, dsm_h, vscores, hscores

class GLSTM(nn.Module):
    def __init__(self, query_dim=6, key_dim=6, rgb_dim=192, dsm_dim=96, num_classes=6, nums_heads=6, mlp_ratio=4,
                 weight_ratio=1.0, drop=0., drop_path=0., act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.rgb_norm1 = norm_layer(rgb_dim)
        self.dsm_norm1 = norm_layer(dsm_dim)
        self.attn = GLSTB(query_dim=query_dim, key_dim=key_dim, rgbv_dim=rgb_dim, dsmv_dim=dsm_dim,
                          num_classes=num_classes, nums_heads=nums_heads, mlp_ratio=mlp_ratio,weight_ratio=weight_ratio)

        self.rgb_drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.rgb_mlp = Mlp_decoder(in_features=rgb_dim, hidden_features=int(rgb_dim // mlp_ratio), out_features=rgb_dim, act_layer=act_layer,
                               drop=drop)
        self.rgb_norm2 = norm_layer(rgb_dim)

        self.dsm_drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.dsm_mlp = Mlp_decoder(in_features=dsm_dim, hidden_features=int(dsm_dim // mlp_ratio), out_features=dsm_dim,
                                   act_layer=act_layer,
                                   drop=drop)
        self.dsm_norm2 = norm_layer(dsm_dim)


    def forward(self, aux, rgb, dsm):

        rgb, dsm, vscores, hscores = self.attn(aux, rgb, dsm)
        rgb = rgb + self.rgb_drop_path(rgb)
        rgb = rgb + self.rgb_drop_path(self.rgb_mlp(self.rgb_norm2(rgb)))

        dsm = dsm + self.dsm_drop_path(dsm)
        dsm = dsm + self.dsm_drop_path(self.dsm_mlp(self.dsm_norm2(dsm)))

        return rgb, dsm, vscores, hscores

class Dual_Branch_CGSelfattention(nn.Module):
    def __init__(self, rgb_dim = 192, dsm_dim=96, num_classes=6, mlp_ratio=4,weight_ratio=1.0,
                 drop=0., drop_path=0., act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d, aux=False):
        super(Dual_Branch_CGSelfattention, self).__init__()


        self.RGB_MutiClass_SegHead = MutiClass_SegHead(in_channels=rgb_dim, num_classes=num_classes)
        self.DSM_MutiClass_SegHead = MutiClass_SegHead(in_channels=dsm_dim, num_classes=num_classes)
        self.aux_pre_enhancement = Dual_Aux_Enhancement(num_classes=num_classes, mode='mutli', eps=1e-8, aux=aux)

        self.mutli_glstm = GLSTM(query_dim=num_classes, key_dim= num_classes, rgb_dim=rgb_dim, dsm_dim=dsm_dim, num_classes=num_classes,
                                  nums_heads=num_classes//num_classes, mlp_ratio=mlp_ratio, weight_ratio = weight_ratio,
                                  drop=drop, drop_path=drop_path, act_layer=act_layer, norm_layer=norm_layer)

        self.rgb_outconv = SeparableConvBlock(in_channels=rgb_dim ,out_channels=rgb_dim ,kernel_size=3)
        self.dsm_outconv = SeparableConvBlock(in_channels=dsm_dim, out_channels=dsm_dim, kernel_size=3)

    def forward(self, rgb, dsm, aux=None):

        RGB_Pre = self.RGB_MutiClass_SegHead(rgb)
        DSM_Pre = self.DSM_MutiClass_SegHead(dsm)
        if aux is not None:
            aux_Pre = self.aux_pre_enhancement(RGB_Pre, DSM_Pre, aux)
        else:
            aux_Pre = self.aux_pre_enhancement(RGB_Pre, DSM_Pre)
        rgb, dsm, vscores, hscores = self.mutli_glstm(aux_Pre, rgb, dsm)
        rgb = self.rgb_outconv(rgb)
        dsm = self.dsm_outconv(dsm)

        return rgb, dsm, RGB_Pre, DSM_Pre, aux_Pre

class Fusion(nn.Module):
    def __init__(self, in_channsel=64,out_channels=64, eps=1e-8):
        super(Fusion, self).__init__()


        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.Preconv = ConvBlock(in_channels=in_channsel,out_channels=out_channels,kernel_size=1,norm_layer = None, linearity= None)
        self.post_conv = SeparableConvBlock(in_channels=out_channels, out_channels=out_channels, kernel_size=5)


    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        weights = nn.ReLU6()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * res + fuse_weights[1] *self.Preconv(x)
        x = self.post_conv(x)
        return x

class FRSH(nn.Module):
    def __init__(self, dim, fc_ratio, dilation=[1, 2, 4, 8], dropout=0., num_classes=6):
        super(FRSH, self).__init__()

        self.oacm = OACM(in_channels=dim, out_channels=dim, kernel_size=3, dilation=dilation)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(dim, dim//fc_ratio, 1, 1),
            nn.ReLU6(),
            nn.Conv2d(dim//fc_ratio, dim, 1, 1),
            nn.Sigmoid()
        )

        self.s_conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=5, padding=2)
        self.sigmoid = nn.Sigmoid()

        self.head = nn.Sequential(SeparableConvBlock(dim, dim, kernel_size=3),
                                  nn.Dropout2d(p=dropout, inplace=True),
                                  ConvBlock(dim, num_classes, kernel_size=1,norm_layer = None, linearity= None))

    def forward(self, x):
        u = x.clone()

        attn = self.oacm(x)
        attn = attn * u

        c_attn = self.avg_pool(x)
        c_attn = self.fc(c_attn)
        c_attn = u * c_attn

        s_max_out, _ = torch.max(x, dim=1, keepdim=True)
        s_avg_out = torch.mean(x, dim=1, keepdim=True)
        s_attn = torch.cat((s_avg_out, s_max_out), dim=1)
        s_attn = self.s_conv(s_attn)
        s_attn = self.sigmoid(s_attn)
        s_attn = u * s_attn

        out = self.head(attn + c_attn + s_attn)

        return out

class Decoder(nn.Module):
    def __init__(self,
                 rgb_encode_channels=[256, 512, 1024, 2048],
                 rgb_decode_channels=[256, 512, 1024, 2048],
                 dsm_encode_channels=[256, 512, 1024, 2048],
                 dsm_decode_channels=[256, 512, 1024, 2048],
                 dilation = [[1, 2, 4, 8], [1, 2, 4, 8], [1, 2, 4, 8], [1, 2, 4, 8]],
                 fc_ratio=4,
                 dropout=0.1,
                 num_classes=6, single_glstm_nums_heads=4,
                 weight_ratio = 1.0):
        super(Decoder, self).__init__()

        self.rgb_Conv1 = ConvBlock(rgb_encode_channels[-1], rgb_decode_channels[-1], 1)
        self.rgb_Conv2 = ConvBlock(rgb_encode_channels[-2], rgb_decode_channels[-2], 1)
        self.rgb_Conv3 = ConvBlock(rgb_encode_channels[-3], rgb_decode_channels[-3], 1)
        self.rgb_Conv4 = ConvBlock(rgb_encode_channels[-4], rgb_decode_channels[-4], 1)

        self.dsm_Conv1 = ConvBlock(dsm_encode_channels[-1], dsm_decode_channels[-1], 1)
        self.dsm_Conv2 = ConvBlock(dsm_encode_channels[-2], dsm_decode_channels[-2], 1)
        self.dsm_Conv3 = ConvBlock(dsm_encode_channels[-3], dsm_decode_channels[-3], 1)
        self.dsm_Conv4 = ConvBlock(dsm_encode_channels[-4], dsm_decode_channels[-4], 1)


        self.rgb_b4 = GLTM(dim=rgb_decode_channels[-1], num_heads=num_classes, mlp_ratio=fc_ratio)
        self.dsm_b4 = GLTM(dim=dsm_decode_channels[-1], num_heads=num_classes, mlp_ratio=fc_ratio)

        self.rgb_p3 = Fusion(rgb_decode_channels[-1], rgb_decode_channels[-2])
        self.dsm_p3 = Fusion(dsm_decode_channels[-1], dsm_decode_channels[-2])
        self.b3 = Dual_Branch_CGSelfattention(rgb_dim = rgb_decode_channels[-2], dsm_dim= dsm_decode_channels[-2], num_classes=num_classes,
                                              mlp_ratio=fc_ratio,weight_ratio=weight_ratio,aux=False)

        self.rgb_p2 = Fusion(rgb_decode_channels[-2], rgb_decode_channels[-3])
        self.dsm_p2 = Fusion(dsm_decode_channels[-2], dsm_decode_channels[-3])
        self.b2 = Dual_Branch_CGSelfattention(rgb_dim=rgb_decode_channels[-3], dsm_dim=dsm_decode_channels[-3],
                                              num_classes=num_classes,
                                              mlp_ratio=fc_ratio, weight_ratio=weight_ratio, aux=True)

        self.rgb_p1 = Fusion(rgb_decode_channels[-3], rgb_decode_channels[-4])
        self.dsm_p1 = Fusion(dsm_decode_channels[-3], dsm_decode_channels[-4])
        self.b1 = Dual_Branch_CGSelfattention(rgb_dim=rgb_decode_channels[-4], dsm_dim=dsm_decode_channels[-4],
                                              num_classes=num_classes,
                                              mlp_ratio=fc_ratio, weight_ratio=weight_ratio, aux=True)

        self.DualChannelSplitConcat = DualChannelSplitConcat(rgb_decode_channels[-4],dsm_decode_channels[-4],num_splits=num_classes)
        self.Conv5 = ConvBlock(rgb_decode_channels[-4]+dsm_decode_channels[-4], rgb_encode_channels[-5], 1)
        self.p = Fusion(rgb_encode_channels[-5])
        self.seg_head = FRSH(rgb_encode_channels[-5], fc_ratio=fc_ratio, dilation=dilation[3], dropout=dropout, num_classes=num_classes)

        #FeatureRefinementHead(encoder_channels[-4], decode_channels)

        ##
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.up3 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.init_weight()

    def forward(self, res, res1, res2, res3, res4, dsm1, dsm2, dsm3, dsm4, h, w):

        res4 = self.rgb_Conv1(res4)
        res3 = self.rgb_Conv2(res3)
        res2 = self.rgb_Conv3(res2)
        res1 = self.rgb_Conv4(res1)

        dsm4 = self.dsm_Conv1(dsm4)
        dsm3 = self.dsm_Conv2(dsm3)
        dsm2 = self.dsm_Conv3(dsm2)
        dsm1 = self.dsm_Conv4(dsm1)

        dsm, _, _ = self.dsm_b4(dsm4)
        x, _, _ = self.rgb_b4(res4)

        x = self.rgb_p3(x, res3)
        dsm = self.dsm_p3(dsm, dsm3)
        x, dsm, RGB_Pre3, DSM_Pre3, aux_Pre3 = self.b3(x, dsm)

        x = self.rgb_p2(x, res2)
        dsm = self.dsm_p2(dsm, dsm2)
        x, dsm, RGB_Pre2, DSM_Pre2, aux_Pre2 = self.b2(x, dsm, aux_Pre3)

        x = self.rgb_p1(x, res1)
        dsm = self.dsm_p1(dsm, dsm1)
        x, dsm, RGB_Pre1, DSM_Pre1, aux_Pre1 = self.b1(x, dsm, aux_Pre2)

        out = self.DualChannelSplitConcat(x, dsm)
        out = self.Conv5(out)
        out = self.p(out, res)
        out = self.seg_head(out)

        out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)

        RGB_Pre1 = F.interpolate(RGB_Pre1, size=(h, w), mode='bilinear', align_corners=False)
        DSM_Pre1 = F.interpolate(DSM_Pre1, size=(h, w), mode='bilinear', align_corners=False)
        RGB_Pre2 = F.interpolate(RGB_Pre2, size=(h, w), mode='bilinear', align_corners=False)
        DSM_Pre2 = F.interpolate(DSM_Pre2, size=(h, w), mode='bilinear', align_corners=False)
        RGB_Pre3 = F.interpolate(RGB_Pre3, size=(h, w), mode='bilinear', align_corners=False)
        DSM_Pre3 = F.interpolate(DSM_Pre3, size=(h, w), mode='bilinear', align_corners=False)


     #   visualize_entropy_and_diff(single_Pre_diff3)
       # visualize_entropy_and_diff(single_Pre_diff2)
      #  visualize_entropy_and_diff(single_Pre_diff1)

        return out, RGB_Pre1, DSM_Pre1, RGB_Pre2, DSM_Pre2, RGB_Pre3, DSM_Pre3

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

def visualize_entropy_and_diff(prob_map):
    """
    """
    # 
    entropy_map = prob_map[0].detach().cpu().numpy()  # 

    # 
    num_classes = entropy_map.shape[0]

    # 
    plt.figure(figsize=(12, 6))

    for i in range(num_classes):
        plt.subplot(1, num_classes, i + 1)
        plt.imshow(entropy_map[i], cmap='jet')
        plt.colorbar()
        plt.title(f'Class {i + 1} Entropy')

    plt.tight_layout()
    plt.show()

class DSM_Preprocessing(nn.Module):

    def __init__(self, k=3, mu=0, sigma=1):
            super().__init__()

            self.k = k
            self.mu = mu
            self.sigma=sigma

    def get_gaussian_kernel(self, k=3, mu=0, sigma=1, normalize=True):
            # compute 1 dimension gaussian
            gaussian_1D = np.linspace(-1, 1, k)
            # compute a grid distance from center
            x, y = np.meshgrid(gaussian_1D, gaussian_1D)
            distance = (x ** 2 + y ** 2) ** 0.5

            # compute the 2 dimension gaussian
            gaussian_2D = np.exp(-(distance - mu) ** 2 / (2 * sigma ** 2))
            gaussian_2D = gaussian_2D / (2 * np.pi * sigma ** 2)

            # normalize part (mathematically)
            if normalize:
                gaussian_2D = gaussian_2D / np.sum(gaussian_2D)
            return gaussian_2D

    def get_sobel_kernel(self, k):
            # get range
            range = np.linspace(-(k // 2), k // 2, k)
            # compute a grid the numerator and the axis-distances
            x, y = np.meshgrid(range, range)
            sobel_2D_numerator = x
            sobel_2D_denominator = (x ** 2 + y ** 2)
            sobel_2D_denominator[:, k // 2] = 1  # avoid division by zero
            sobel_2D = sobel_2D_numerator / sobel_2D_denominator
            return sobel_2D

    def get_boundary(self, x, k_size, gaussian=True):
            gaussian_2D = self.get_gaussian_kernel(k_size, mu=0, sigma=1)
            gaussian_filter = nn.Conv2d(in_channels=1,
                                        out_channels=1,
                                        kernel_size=k_size,
                                        padding=k_size // 2,
                                        bias=False).cuda()
            # sobel
            sobel_2D = self.get_sobel_kernel(k_size)
            sobel_filter_x = nn.Conv2d(in_channels=1,
                                       out_channels=1,
                                       kernel_size=k_size,
                                       padding=k_size // 2,
                                       bias=False).cuda()

            sobel_filter_y = nn.Conv2d(in_channels=1,
                                       out_channels=1,
                                       kernel_size=k_size,
                                       padding=k_size // 2,
                                       bias=False).cuda()

            with torch.no_grad():
                gaussian_filter.weight[:] = torch.from_numpy(gaussian_2D).float().requires_grad_(False)
                sobel_filter_x.weight[:] = torch.from_numpy(sobel_2D).float().requires_grad_(False)
                sobel_filter_y.weight[:] = torch.from_numpy(sobel_2D.T).float().requires_grad_(False)

            x = x.float()
            if gaussian:
                x = gaussian_filter(x)
            grad_x = torch.abs(sobel_filter_x(x))
            grad_y = torch.abs(sobel_filter_y(x))
            grad_magnitude = (grad_x ** 2 + grad_y ** 2) ** 0.5

            return grad_x, grad_y, grad_magnitude

    def forward(self, dsm):

        grad_x, grad_y, grad_magnitude = self.get_boundary(dsm,self.k)

        return grad_x, grad_y, grad_magnitude

class CSFAFormer(nn.Module):
    def __init__(self,num_classes=6,
                 dropout=0.1,vis_channels =2,
                 fc_ratio=4,
                 decode_channels=32,
                 dsmencoder_channels=(96, 192, 384, 768),
                 embed_dim=128,
                 depths=(2, 2, 18, 2),
                 num_heads=(4, 8, 16, 32),
                 frozen_stages=2):
        super(CSFAFormer, self).__init__()

        self.backbone = timm.create_model('swsl_resnet50', in_chans=vis_channels+1, features_only=True, output_stride=32,
                                          out_indices=(1, 2, 3, 4), pretrained=True)

        rgb_encoder_channels = [info['num_chs'] for info in self.backbone.feature_info]


        self.cnn = nn.Sequential(self.backbone.conv1,
                                 self.backbone.bn1,
                                 self.backbone.act1
                                 )

        self.cnn1 = nn.Sequential(self.backbone.maxpool,self.backbone.layer1)
        self.cnn2 = self.backbone.layer2
        self.cnn3 = self.backbone.layer3
        self.cnn4 = self.backbone.layer4

        self.dsmbackbone = timm.create_model('swsl_resnet50', features_only=True,
                                             output_stride=32,
                                             out_indices=(1, 2, 3, 4), pretrained=True)

        dsm_channels = [info['num_chs'] for info in self.dsmbackbone.feature_info]

        ## 
        #       self.dsmbackbone = SwinTransformer(embed_dim=embed_dim, depths=depths, num_heads=num_heads,
        #                                       frozen_stages=frozen_stages)
        #        dsm_channels = dsmencoder_channels

        self.dsmp = DSM_Preprocessing(k=3, mu=0, sigma=1)


        rgb_decode_channels = [decode_channels * num_classes,decode_channels * num_classes,
                               decode_channels * num_classes,decode_channels * num_classes]
        dsm_decode_channels = [decode_channels * num_classes // 2, decode_channels * num_classes // 2,
                               decode_channels * num_classes // 2, decode_channels * num_classes // 2]

        ##
        self.decoder = Decoder(rgb_encode_channels=rgb_encoder_channels,
                 rgb_decode_channels=rgb_decode_channels,
                 dsm_encode_channels=dsm_channels,
                 dsm_decode_channels=dsm_decode_channels,
                 num_classes=num_classes,dropout=dropout,
                 weight_ratio = 1.0)

    def forward(self, vis, ir, dsm):

        h, w = vis.size()[-2:]
        dsm = dsm.float() / 255.0
        grad_x, grad_y, grad_magnitude = self.dsmp(dsm)
      #  x = torch.cat((vis, ir), dim=1)
        dsm = torch.cat((grad_x, grad_y, dsm), dim=1)

        # Encoder ResNet50
        x_pre = self.cnn(vis)    ##H/2
        res1 = self.cnn1(x_pre)##H/4
        res2 = self.cnn2(res1) ##H/8
        res3 = self.cnn3(res2) ##H/16
        res4 = self.cnn4(res3) ##H/32

        #Encoder DSM
        dsm1, dsm2, dsm3, dsm4 = self.dsmbackbone(dsm)

        ##
        out, RGB_Pre1, DSM_Pre1, RGB_Pre2, DSM_Pre2, RGB_Pre3, DSM_Pre3 = self.decoder(x_pre, res1, res2, res3, res4,dsm1, dsm2, dsm3, dsm4,h, w)
      #  visualize_entropy_and_diff(out)

        if self.training:

            return out, RGB_Pre1, DSM_Pre1, RGB_Pre2, DSM_Pre2, RGB_Pre3, DSM_Pre3
        else:

            return out

def CSFAFormer_base(pretrained=True, num_classes=6, weight_path='./MMRSSeg/pretrain_weights/stseg_base.pth'):

    # pretrained weights are load from official repo of Swin Transformer
    model = CSFAFormer(dsmencoder_channels=(128, 256, 512, 1024),
                    num_classes=num_classes,
                    embed_dim=128,
                    depths=(2, 2, 18, 2),
                    num_heads=(4, 8, 16, 32),
                    frozen_stages=2)
    if pretrained and weight_path is not None:
        old_dict = torch.load(weight_path)['state_dict']
        model_dict = model.state_dict()
        old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}
        model_dict.update(old_dict)
        model.load_state_dict(model_dict)
    return model


def CSFAFormer_small(pretrained=True, num_classes=4,
                  weight_path='./MMRSSeg/pretrain_weights/stseg_small.pth'):
    model = CSFAFormer(dsmencoder_channels=(96, 192, 384, 768),
                    num_classes=num_classes,
                    embed_dim=96,
                    depths=(2, 2, 18, 2),
                    num_heads=(3, 6, 12, 24),
                    frozen_stages=2)
    if pretrained and weight_path is not None:
        old_dict = torch.load(weight_path)['state_dict']
        model_dict = model.state_dict()
        old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}
        model_dict.update(old_dict)
        model.load_state_dict(model_dict)
    return model


def CSFAFormer_tiny(pretrained=True, vis_channels=2, num_classes=4,
                 weight_path='./MMRSSeg/pretrain_weights/stseg_tiny.pth'):
    model = CSFAFormer(dsmencoder_channels=(96, 192, 384, 768),
                    num_classes=num_classes, vis_channels=vis_channels,
                    embed_dim=96,
                    depths=(2, 2, 6, 2),
                    num_heads=(3, 6, 12, 24),
                    frozen_stages=2)
    if pretrained and weight_path is not None:
        old_dict = torch.load(weight_path)['state_dict']
        model_dict = model.state_dict()
        old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}
        model_dict.update(old_dict)
        model.load_state_dict(model_dict)
    return model


if __name__ == '__main__':

    num_classes = 6
    in_batch, inchannel, in_h, in_w = 4, 3, 512, 512
    x = torch.randn(in_batch, 2, in_h, in_w).cuda()
    ir = torch.randn(in_batch, 1, in_h, in_w).cuda()
    dsm = torch.randn(in_batch, 1, in_h, in_w).cuda()
    net = CSFAFormer(num_classes).cuda()
    out, RGB_Pre1, DSM_Pre1, RGB_Pre2, DSM_Pre2, RGB_Pre3, DSM_Pre3 = net(x,ir,dsm)
    print(out.shape)
    print(RGB_Pre1.shape)
    print(DSM_Pre1.shape)
    print(RGB_Pre2.shape)
    print(DSM_Pre2.shape)
    print(RGB_Pre3.shape)
    print(DSM_Pre3.shape)

    total = sum([param.nelement() for param in net.parameters()])

    print("Number of parameter: %.2fM" % (total / 1e6))
