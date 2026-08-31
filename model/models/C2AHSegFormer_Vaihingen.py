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
from einops import rearrange, einsum

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

class MultiHeadAttention(nn.Module):
    def __init__(self, rgb_q_dim=96, rgb_k_dim=96, rgb_v_dim=96,
                 num_classes = 6, num_heads = 6,direction='v',Group_Linear = False):
        super().__init__()

        self.rgb_q_dim = rgb_q_dim
        self.rgb_k_dim = rgb_k_dim
        self.rgb_v_dim = rgb_v_dim
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.direction = direction

        if Group_Linear ==True:
            self.w_rgb_q = GroupedLinear(in_features=self.rgb_q_dim, out_features=self.rgb_q_dim, num_groups = num_heads)
            self.w_rgb_k = GroupedLinear(in_features=self.rgb_k_dim, out_features=self.rgb_k_dim, num_groups = num_heads)
            self.w_rgb_v = GroupedLinear(in_features=self.rgb_v_dim, out_features=self.rgb_v_dim, num_groups = num_heads)
        else:
            self.w_rgb_q = nn.Linear(in_features=self.rgb_q_dim, out_features=self.rgb_q_dim, bias=False)
            self.w_rgb_k = nn.Linear(in_features=self.rgb_k_dim, out_features=self.rgb_k_dim, bias=False)
            self.w_rgb_v = nn.Linear(in_features=self.rgb_v_dim, out_features=self.rgb_v_dim, bias=False)

        self.rgb_out = nn.Linear(in_features=self.rgb_v_dim, out_features=self.rgb_v_dim, bias=False)

    def tensor_reshape(self, x, b, h, w):

        if self.direction == 'v':
            x = x.permute(0, 3, 2, 1).reshape(b * w, h, -1).contiguous()
        else:
            x = x.permute(0, 2, 3, 1).reshape(b * h, w, -1).contiguous()

        return x

    def tensor_restore(self, x, b, h, w):

        if self.direction == 'v':
            x = x.reshape(b, w, h, -1).permute(0, 3, 2, 1).contiguous()
        else:
            x = x.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()

        return x

    def muti_head_split(self, q, k, v, q_dim, k_dim, v_dim):

        split_size_q = q_dim // self.num_heads
        split_size_k = k_dim // self.num_heads
        split_size_v = v_dim // self.num_heads
        q = torch.stack(torch.split(q, split_size_q, dim=2), dim=0)  # [h, N, T_q, num_units/h]
        k = torch.stack(torch.split(k, split_size_k, dim=2), dim=0)  # [h, N, T_q, num_units/h]
        v = torch.stack(torch.split(v, split_size_v, dim=2), dim=0)  # [h, N, T_q, num_units/h]

        return q, k, v

    def attn_cal(self, q, k, v, k_dim):

        ## Self Attention rgb: Qx, Kx, Vx
        scores = torch.matmul(q, k.transpose(2, 3))  # [h, N, T_q, T_k]
        scores = scores / ((k_dim// self.num_heads) ** 0.5)
        scores = F.softmax(scores, dim=3)

        ## RGB特征聚合
        out = torch.matmul(scores, v)  # [h, N, T_q, num_units/h]
        out = torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, num_units]

        return out

    def forward(self, q_rgb, k_rgb, v_rgb):

        b, _, h, w = v_rgb.size(0), v_rgb.size(1), v_rgb.size(2), v_rgb.size(3)

        ## 预处理
        q_rgb, k_rgb, v_rgb = self.tensor_reshape(q_rgb, b, h, w), self.tensor_reshape(k_rgb, b, h, w),self.tensor_reshape(v_rgb, b, h, w)

        q_rgb = self.w_rgb_q(q_rgb)
        k_rgb = self.w_rgb_k(k_rgb)                 # [N, T_k, num_units]
        v_rgb = self.w_rgb_v(v_rgb)

        ## channel_split
        q_rgb, k_rgb, v_rgb = self.muti_head_split(q_rgb, k_rgb, v_rgb, self.rgb_q_dim, self.rgb_k_dim, self.rgb_v_dim)

        ## Self Attention rgb: Qx, Kx, Vx
        rgb_out = self.attn_cal(q_rgb, k_rgb, v_rgb, self.rgb_k_dim)
        rgb_out = self.rgb_out(rgb_out)

        return self.tensor_restore(rgb_out, b, h, w)

class GLTM(nn.Module):
    def __init__(self, rgb_dim=512, dsm_dim = 512, num_heads=6,  mlp_ratio=4,
                 drop_path=0., act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d):
        super().__init__()


        self.msa_rgb_v = MultiHeadAttention(rgb_q_dim=rgb_dim, rgb_k_dim=rgb_dim, rgb_v_dim=rgb_dim,
                                              num_classes = num_heads, num_heads = num_heads,
                                              direction='v',Group_Linear = False)
        self.msa_dsm_v = MultiHeadAttention(rgb_q_dim=dsm_dim, rgb_k_dim=dsm_dim, rgb_v_dim=dsm_dim,
                                            num_classes=num_heads, num_heads=num_heads,
                                            direction='v', Group_Linear=False)
        self.local_rgb_v = AdaptiveLocalFeatureExtraction(rgb_dim, ratio=8,mode='v')
        self.local_dsm_v = AdaptiveLocalFeatureExtraction(dsm_dim, ratio=8, mode='v')
        self.conv_rgb_v = ConvBlock(in_channels=rgb_dim, out_channels=rgb_dim, kernel_size=3, stride=1)
        self.conv_dsm_v = ConvBlock(in_channels=dsm_dim, out_channels=dsm_dim, kernel_size=3, stride=1)


        self.msa_rgb_h = MultiHeadAttention(rgb_q_dim=rgb_dim, rgb_k_dim=rgb_dim, rgb_v_dim=rgb_dim,
                                             num_classes=num_heads, num_heads=num_heads,
                                             direction='h', Group_Linear=False)
        self.msa_dsm_h = MultiHeadAttention(rgb_q_dim=dsm_dim, rgb_k_dim=dsm_dim, rgb_v_dim=dsm_dim,
                                            num_classes=num_heads, num_heads=num_heads,
                                            direction='h', Group_Linear=False)
        self.local_rgb_h = AdaptiveLocalFeatureExtraction(rgb_dim, ratio=8, mode='h')
        self.local_dsm_h = AdaptiveLocalFeatureExtraction(dsm_dim, ratio=8, mode='h')
        self.conv_rgb_h = ConvBlock(in_channels=rgb_dim, out_channels=rgb_dim, kernel_size=3, stride=1)
        self.conv_dsm_h = ConvBlock(in_channels=dsm_dim, out_channels=dsm_dim, kernel_size=3, stride=1)


        self.rgb_drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.rgb_mlp = Mlp_decoder(in_features=rgb_dim, hidden_features=int(rgb_dim // mlp_ratio), out_features=rgb_dim, act_layer=act_layer,
                               drop=drop_path)
        self.rgb_norm = norm_layer(rgb_dim)

        self.dsm_drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.dsm_mlp = Mlp_decoder(in_features=dsm_dim, hidden_features=int(dsm_dim // mlp_ratio), out_features=dsm_dim,
                                   act_layer=act_layer,
                                   drop=drop_path)
        self.dsm_norm = norm_layer(dsm_dim)

    def forward(self, x, y):

        v_rgb,  v_dsm= x.clone(), y.clone()

        vg_rgb, vg_dsm = self.msa_rgb_v(v_rgb,v_rgb,v_rgb), self.msa_dsm_v(v_dsm,v_dsm,v_dsm)
        vl_rgb, vl_dsm = self.local_rgb_v(v_rgb), self.local_dsm_v(v_dsm)
        rgb, dsm = self.conv_rgb_v(vg_rgb+vl_rgb), self.conv_dsm_v(vg_dsm+vl_dsm)

        h_rgb, h_dsm = rgb.clone(), dsm.clone()
        hg_rgb, hg_dsm = self.msa_rgb_h(h_rgb, h_rgb, h_rgb), self.msa_dsm_h(h_dsm, h_dsm, h_dsm)
        hl_rgb, hl_dsm = self.local_rgb_h(h_rgb), self.local_dsm_h(h_dsm)
        rgb, dsm = self.conv_rgb_h(hg_rgb + hl_rgb), self.conv_dsm_h(vg_dsm + vl_dsm)

        ## 后处理
        rgb = rgb + self.rgb_drop_path(rgb)
        rgb = rgb + self.rgb_drop_path(self.rgb_mlp(self.rgb_norm(rgb)))

        dsm = dsm + self.dsm_drop_path(dsm)
        dsm = dsm + self.dsm_drop_path(self.dsm_mlp(self.dsm_norm(dsm)))

        return rgb, dsm

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
        assert rgb_channels % num_splits == 0, "第一个张量的通道数量必须能够被分割数整除。"
        assert dsm_channels % num_splits == 0, "第二个张量的通道数量必须能够被分割数整除。"

        self.num_splits = num_splits
        self.split_size1 = rgb_channels // num_splits
        self.split_size2 = dsm_channels // num_splits

    def forward(self, x1, x2):
        # 使用 torch.chunk 拆分通道
        x1_split = torch.chunk(x1, self.num_splits, dim=1)  # (num_splits, B, split_size1, H, W)
        x2_split = torch.chunk(x2, self.num_splits, dim=1)  # (num_splits, B, split_size2, H, W)

        # 交叉拼接x1和x2的拆分
        output_splits = []
        for i in range(self.num_splits):
            output_splits.append(x1_split[i])  # 添加x1的拆分
            output_splits.append(x2_split[i])  # 添加x2的拆分

        # 将所有拆分拼接在一起 (B, C1 + C2, H, W)
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

def top2_prob_difference(prob_map):
    """
    计算每个像素点的前两个最大概率之间的绝对差值，并将其归一化到 [0, 1] 范围内。

    :param prob_map: Tensor，大小为 [B, K, H, W]，其中 B 是 batch 大小，K 是类别数，H, W 是图像大小
    :return: Tensor，大小为 [B, H, W]，每个像素的前两个最大概率的绝对差值，归一化到 [0, 1] 范围
    """
    # 获取每个像素点的前两个最大概率及其类别
    prob_map = F.softmax(prob_map, dim=1)
    top2_probs, _ = torch.topk(prob_map, 2, dim=1)  # top2_probs: [B, 2, H, W]
    # 计算前两个最大概率的差值的绝对值
    prob_diff = torch.abs(top2_probs[:, 0] - top2_probs[:, 1])  # 绝对值计算
    prob_diff_normalized = prob_diff  # 归一化到 [0, 1]

    return prob_diff_normalized  # 返回归一化后的差值图

def compute_entropy(prob_map):
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

    return 1.0 - normalized_entropy_map  # 返回大小为 [B, H, W] 的归一化信息熵图

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

class ClassGuidedAttention(nn.Module):
    def __init__(self,
                 query_dim = 6,
                 key_dim = 6,
                 value_dim=256,
                 num_heads=6,
                 num_classes=6,
                 qkv_bias=False,
                 window_size=8,
                 use_memory = True,
                 momentum=0.5,
                 relative_pos_embedding=True
                 ):
        super().__init__()

        self.query_dim = query_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.num_heads = num_heads
        self.num_classes=num_classes
        self.momentum = momentum
        self.eps = 1e-8

        self.num_heads = num_heads
        self.head_qdim = self.query_dim // self.num_heads
        self.head_kdim = self.key_dim // self.num_heads
        self.head_vdim = self.value_dim // self.num_heads
        self.scale = self.head_kdim ** -0.5
        self.use_memory = use_memory
        self.ws = window_size

  #      self.k = ConvBlock(self.key_dim, self.key_dim, kernel_size=1, bias=qkv_bias,
  #                         norm_layer = None, linearity = None)
        self.v = ConvBlock(self.value_dim, self.value_dim, kernel_size=1, bias=qkv_bias,
                           norm_layer = None, linearity = None)

        self.proj = SeparableConvBlock(self.value_dim, self.value_dim,
                                       kernel_size=window_size, linearity = None)

        self.relative_pos_embedding = relative_pos_embedding

        if self.relative_pos_embedding:
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.ws)
            coords_w = torch.arange(self.ws)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.ws - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.ws - 1
            relative_coords[:, :, 0] *= 2 * self.ws - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)

            trunc_normal_(self.relative_position_bias_table, std=.02)

        # 初始化 memory bank
        if self.use_memory:
            self.register_buffer("memory_bank", torch.zeros(self.num_classes, value_dim))

    @torch.no_grad()
    def update_memory(self, v_center, momentum=0.9):
        """向量化更新memory bank"""
        v_center_mean = v_center.mean(dim=0).transpose(0, 1)  # [num_classes, value_dim]

        # 向量化更新
        mask = (self.memory_bank.abs().sum(dim=1) == 0).float().unsqueeze(1)
        self.memory_bank = (
                mask * v_center_mean +
                (1 - mask) * (momentum * self.memory_bank + (1 - momentum) * v_center_mean)
        )

    @torch.no_grad()
    def mask_to_onehot(self, mask, num_classes, ignore_index=None):
        """
        将整型标签掩码转换为 one-hot 编码，并忽略特定类别（如背景）

        Args:
            mask: [B, H, W] 整型标签（值范围 0 ~ K，其中 K 是忽略类别）
            num_classes: int, 有效类别数（实际统计的类别数，不包括忽略类别）
            ignore_index: int, 要忽略的类别索引（如 K）

        Returns:
            onehot: [B, num_classes, H, W] 的 one-hot 张量（忽略 ignore_index 的位置）
        """
        # 确保输入是长整型（torch.long）
        mask = mask.long()

        # 创建 one-hot 张量（仅统计有效类别）
        onehot = torch.zeros((mask.size(0), num_classes, mask.size(1), mask.size(2)),
                             device=mask.device)

        # 如果 ignore_index 不为 None，则过滤掉该类别
        if ignore_index is not None:
            # 生成有效像素的掩码（非 ignore_index 的位置）
            valid_mask = (mask != ignore_index)
            # 将 ignore_index 映射为 0（避免 scatter_ 越界）
            masked = torch.where(valid_mask, mask, torch.zeros_like(mask))
            # 填充 one-hot（仅填充有效位置）
            onehot.scatter_(1, masked.unsqueeze(1), 1)
            # 将无效位置置零（确保 ignore_index 不影响统计）
            onehot = onehot * valid_mask.unsqueeze(1)
        else:
            # 无忽略类别，直接填充
            onehot.scatter_(1, mask.unsqueeze(1), 1)

        return onehot

    def class_center(self, prob_map, features):
        # Step 1: Gumbel-Softmax 采样（直接调用 PyTorch）
        prob_map = F.gumbel_softmax(prob_map, tau=0.1, hard=True, dim=1)  # [B, K, H, W]
        # Step 2: 计算类别原型特征 [B, K, C]
        numerator = torch.einsum('bkhw,bchw->bkc', prob_map, features)  # ∑(A*F)
        denominator = (prob_map.sum(dim=(2, 3), keepdim=False) + self.eps).unsqueeze(2)  # ∑A
        out = (numerator / denominator).permute(0, 2, 1)  # [B, C, K]

        return out

    def pad(self, x, ps):
        _, _, H, W = x.size()
        if W % ps != 0:
            x = F.pad(x, (0, ps - W % ps), mode='reflect')
        if H % ps != 0:
            x = F.pad(x, (0, 0, 0, ps - H % ps), mode='reflect')
        return x

    def pad_out(self, x):
        x = F.pad(x, pad=(0, 1, 0, 1), mode='reflect')
        return x

    def buffer_predict(self, v, memory_bank):
        """
        基于相似度的内存预测
        参数:
            v: 输入特征 [B, C, H, W]
            memory_bank: 内存库 [K, C] K是类别数

        返回:
            预测结果 [B, K, H, W]
        """
        B, C, H, W = v.shape
        K = memory_bank.shape[0]
        # 将输入特征重塑为 [B, H*W, C]
        v_flat = v.permute(0, 2, 3, 1).reshape(B, H * W, C)
        # 计算相似度 (使用点积相似度)
        similarity = torch.matmul(v_flat, memory_bank.T)  # [B, H*W, K]
        # 重塑为 [B, K, H, W]
        similarity = similarity.permute(0, 2, 1).reshape(B, K, H, W)
        # 应用softmax得到预测概率
        prediction = F.softmax(similarity, dim=1)

        return prediction

    def cross_win_attention(self, q, k, v):
        B, K, H, W = q.shape
        C = v.shape[1]

        q = rearrange(q, 'b (h d) (H) (W) -> b h (H W) d',
                      b=B, h=self.num_heads, d=self.query_dim // self.num_heads,
                      H=H, W=W)

        # prob = F.gumbel_softmax(prob, tau=tau, hard=False, dim=2)
        # Initial rearranges (as you have them)
        prob = rearrange(k, 'b k (hh ws1) (ww ws2) -> (b hh ww) k (ws1 ws2)',
                         b=B, k=self.num_heads,
                         hh=H // self.ws, ww=W // self.ws, ws1=self.ws, ws2=self.ws)

        features = rearrange(v, 'b d (hh ws1) (ww ws2) -> (b hh ww) d (ws1 ws2)',
                             b=B, d=C,
                             hh=H // self.ws, ww=W // self.ws, ws1=self.ws, ws2=self.ws)

        # 1. prob @ prob.transpose to get (b hh ww) k k
        prob_kk = (prob @ prob.transpose(-2, -1))
        prob_reshaped = rearrange(prob_kk, '(b hh ww) k1 k2 -> b k1 (k2 hh ww)',
                                  b=B, hh=H // self.ws, ww=W // self.ws)
 #       prob_reshaped = F.softmax(prob_reshaped,dim=1)
        k = rearrange(prob_reshaped, 'b (h k1) (k2 hh ww) -> b h (k2 hh ww) k1',
                      b=B, h=self.num_heads, k1=self.num_heads// self.num_heads, k2=self.num_heads,
                       hh=H // self.ws, ww=W // self.ws)
        # 2. features @ prob.transpose to get (b hh ww) d k
        features_dk = (features @ prob.transpose(-2, -1))
        features_reshaped = rearrange(features_dk, '(b hh ww) d k -> b d (k hh ww)',
                                      b=B, hh=H // self.ws, ww=W // self.ws)
        v = rearrange(features_reshaped, 'b (h d) (k hh ww) -> b h (k hh ww) d',
                      b=B, h=self.num_heads, d=C // self.num_heads, k=self.num_heads,hh=H // self.ws, ww=W // self.ws)

        dots = (q @ k.transpose(-2, -1)) * self.scale
        attn = dots.softmax(dim=-1)
        attn = attn @ v

        attn = rearrange(attn, 'b h (H W) d -> b (h d) H W', b=B, h=self.num_heads, d=C // self.num_heads,H=H, W=W)

        return attn

    def forward(self, q, k, v, mask=None):
        B, C, H, W = v.shape
        K = q.size(1)
        v = self.pad(v, self.ws)
        q = self.pad(q, self.ws)
        B, C, Hp, Wp = v.shape


        q = q                  # [N, T_q, num_units]
        k = k    # [N, T_k, num_units]
        v = self.v(v)

        q_win = rearrange(q, 'b (h d) (hh ws1) (ww ws2) -> (b hh ww) h (ws1 ws2) d',
                      b=B, h=self.num_heads, d=self.query_dim // self.num_heads,
                      hh=Hp // self.ws, ww=Wp // self.ws, ws1=self.ws, ws2=self.ws)
        k_win = rearrange(k, 'b (h d) (hh ws1) (ww ws2) -> (b hh ww) h (ws1 ws2) d',
                      b=B,h=self.num_heads, d=self.key_dim // self.num_heads,
                      hh=Hp // self.ws, ww=Wp // self.ws, ws1=self.ws, ws2=self.ws)
        v_win = rearrange(v, 'b (h d) (hh ws1) (ww ws2) -> (b hh ww) h (ws1 ws2) d',
                      b=B,h=self.num_heads, d=self.value_dim // self.num_heads,
                      hh=Hp // self.ws, ww=Wp // self.ws, ws1=self.ws, ws2=self.ws)

        dots = (q_win @ k_win.transpose(-2, -1)) * self.scale

        if self.relative_pos_embedding:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.ws * self.ws, self.ws * self.ws, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            dots += relative_position_bias.unsqueeze(0)

        attn = dots.softmax(dim=-1)
        attn = attn @ v_win

        attn = rearrange(attn, '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)', h=self.num_heads,
                         d=C//self.num_heads, hh=Hp//self.ws, ww=Wp//self.ws, ws1=self.ws, ws2=self.ws)

        attn = attn[:, :, :H, :W]
        ##特征中心获取
        v_center = self.class_center(k, v)
        ## memory bank
        if self.train and self.use_memory:
            self.update_memory(v_center, self.momentum)
    #    if self.use_memory:
     #       v_center = v_center + self.memory_bank.unsqueeze(0).repeat(B, 1, 1).permute(0, 2, 1)
        # 将类别概率与融合后的center相乘得到每像素特征
        out = torch.einsum('bck,bkhw->bchw', v_center, q)  # 输出 [B, C, H, W]
        out = self.pad_out(out + attn)
        out = self.proj(out)

        return out

class Dual_Class_GuideAttention(nn.Module):
    def __init__(self, rgb_q_dim=6, rgb_k_dim=6, rgb_v_dim=96,
                        dsm_q_dim=6, dsm_k_dim=6, dsm_v_dim=96, window_size=8,
                        num_classes = 6, num_heads = 6, momentum=0.5, use_memory = True):
        super().__init__()

        self.rgb_q_dim = rgb_q_dim
        self.rgb_k_dim = rgb_k_dim
        self.rgb_v_dim = rgb_v_dim
        self.dsm_q_dim = dsm_q_dim
        self.dsm_k_dim = dsm_k_dim
        self.dsm_v_dim = dsm_v_dim
        self.num_classes = num_classes
        self.num_heads = num_heads

        self.RGB_self_ClassGuidedAttention = ClassGuidedAttention(query_dim = self.rgb_q_dim,key_dim = self.rgb_k_dim,value_dim=self.rgb_v_dim,
                 num_heads=num_heads,num_classes=num_classes,window_size=window_size,use_memory = use_memory,momentum=momentum)
        self.DSM_self_ClassGuidedAttention = ClassGuidedAttention(query_dim = self.dsm_q_dim,key_dim = self.dsm_k_dim,value_dim=self.dsm_v_dim,
                 num_heads=num_heads,num_classes=num_classes,window_size=window_size,use_memory = use_memory,momentum=momentum)


    def forward(self, aux, rgb, dsm, mask=None):

        if mask is not None:

            rgb_out = self.RGB_self_ClassGuidedAttention(aux, aux, rgb, mask)
            dsm_out = self.DSM_self_ClassGuidedAttention(aux, aux, dsm, mask)
        else:
            rgb_out = self.RGB_self_ClassGuidedAttention(aux, aux, rgb)
            dsm_out = self.DSM_self_ClassGuidedAttention(aux, aux, dsm)

        return rgb_out, dsm_out

class GLSTM(nn.Module):
    def __init__(self, rgb_q_dim=6, rgb_k_dim=6, rgb_v_dim=96,
                 dsm_q_dim=6, dsm_k_dim=6, dsm_v_dim=96, drop_path=0.,  mlp_ratio=4,
                 num_classes = 6, num_heads = 6, weight_ratio=1.0, window_size=8,
                 act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d, aux_enhance = False, momentum=0.5, use_memory = True):
        super(GLSTM, self).__init__()

        self.weight_ratio = weight_ratio
        self.aux_enhance = aux_enhance

        self.aux_rgb_head = MutiClass_SegHead(in_channels=rgb_v_dim, num_classes=num_classes)
        self.aux_dsm_head = MutiClass_SegHead(in_channels=dsm_v_dim, num_classes=num_classes)
        self.aux_pre_enhancement = Dual_Aux_Enhancement(num_classes=num_classes, mode='mutli', eps=1e-8, aux=aux_enhance)

        self.ClassGuidedAttention = Dual_Class_GuideAttention( rgb_q_dim=rgb_q_dim, rgb_k_dim=rgb_k_dim, rgb_v_dim=rgb_v_dim,
                                                                dsm_q_dim=dsm_q_dim, dsm_k_dim=dsm_k_dim, dsm_v_dim=dsm_v_dim,
                                                                window_size=window_size,num_classes=num_classes, num_heads=num_heads,
                                                                momentum=momentum, use_memory=use_memory)

        self.local_rgb_v = AdaptiveLocalFeatureExtraction(rgb_v_dim, ratio=8, mode='v')
        self.local_rgb_h = AdaptiveLocalFeatureExtraction(rgb_v_dim, ratio=8, mode='h')
        self.local_dsm_v = AdaptiveLocalFeatureExtraction(dsm_v_dim, ratio=8, mode='v')
        self.local_dsm_h = AdaptiveLocalFeatureExtraction(dsm_v_dim, ratio=8, mode='h')

        self.rgb_out = ConvBlock(in_channels=rgb_v_dim, out_channels=rgb_v_dim, kernel_size=3, stride=1)
        self.dsm_out = ConvBlock(in_channels=dsm_v_dim, out_channels=dsm_v_dim, kernel_size=3, stride=1)

        self.rgb_drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.rgb_mlp = Mlp_decoder(in_features=rgb_v_dim, hidden_features=int(rgb_v_dim // mlp_ratio), out_features=rgb_v_dim,
                                   act_layer=act_layer,
                                   drop=drop_path)
        self.rgb_norm = norm_layer(rgb_v_dim)

        self.dsm_drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.dsm_mlp = Mlp_decoder(in_features=dsm_v_dim, hidden_features=int(dsm_v_dim // mlp_ratio), out_features=dsm_v_dim,
                                   act_layer=act_layer,
                                   drop=drop_path)
        self.dsm_norm = norm_layer(dsm_v_dim)

    def top2_prob_difference(self, prob_map):
        """
        计算每个像素点的前两个最大概率之间的绝对差值，并将其归一化到 [0, 1] 范围内。

        :param prob_map: Tensor，大小为 [B, K, H, W]，其中 B 是 batch 大小，K 是类别数，H, W 是图像大小
        :return: Tensor，大小为 [B, H, W]，每个像素的前两个最大概率的绝对差值，归一化到 [0, 1] 范围
        """
        # 获取每个像素点的前两个最大概率及其类别
        prob_map = F.softmax(prob_map, dim=1)
        top2_probs, _ = torch.topk(prob_map, 2, dim=1)  # top2_probs: [B, 2, H, W]
        # 计算前两个最大概率的差值的绝对值
        prob_diff = torch.abs(top2_probs[:, 0] - top2_probs[:, 1])  # 绝对值计算
        prob_diff_normalized = prob_diff  # 归一化到 [0, 1]

        return prob_diff_normalized  # 返回归一化后的差值图

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

        return 1.0 - normalized_entropy_map  # 返回大小为 [B, H, W] 的归一化信息熵图

    def aux_enhancement(self, prob_map):
        weight_en = self.compute_entropy(prob_map).unsqueeze(1)
        weight_prob_difference = self.top2_prob_difference(prob_map).unsqueeze(1)
        prob_map = weight_en * weight_prob_difference * prob_map
        prob_map = F.softmax(prob_map, dim=1)
        return prob_map

    def forward(self, rgb, dsm, mask=None, aux=None):

        pre_rgb = self.aux_rgb_head(rgb)
        pre_dsm = self.aux_dsm_head(dsm)

        ## 概率分布优化
        aux_rgb = F.softmax(pre_rgb * self.weight_ratio, dim=1)
        aux_dsm = F.softmax(pre_dsm * self.weight_ratio, dim=1)
        if aux is not None:
            aux_Pre = self.aux_pre_enhancement(aux_rgb, aux_dsm, aux)
        else:
            aux_Pre = self.aux_pre_enhancement(aux_rgb, aux_dsm)
 #       if self.aux_enhance == True:
 #           aux_rgb = self.aux_enhancement(aux_rgb)
 #           aux_dsm = self.aux_enhancement(aux_dsm)
        ## global
        if mask is not None:
            rgb_g, dsm_g = self.ClassGuidedAttention(aux_Pre, rgb, dsm, mask)
        else:
            rgb_g, dsm_g = self.ClassGuidedAttention(aux_Pre, rgb, dsm)
        ## local
        rgb_l, dsm_l = self.local_rgb_h(self.local_rgb_v(rgb)), self.local_dsm_h(self.local_dsm_v(dsm))

        ## local_branch
        rgb, dsm = self.rgb_out(rgb_g + rgb_l), self.dsm_out(dsm_g + dsm_l)

        ## 后处理
        rgb = rgb + self.rgb_drop_path(rgb)
        rgb = rgb + self.rgb_drop_path(self.rgb_mlp(self.rgb_norm(rgb)))

        dsm = dsm + self.dsm_drop_path(dsm)
        dsm = dsm + self.dsm_drop_path(self.dsm_mlp(self.dsm_norm(dsm)))

        return rgb, dsm, pre_rgb, pre_dsm, aux_Pre

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

        out = attn + c_attn + s_attn

        return out

class Decoder(nn.Module):
    def __init__(self,
                 rgb_encode_channels=[256, 512, 1024, 2048],
                 rgb_decode_channels=[256, 512, 1024, 2048],
                 dsm_encode_channels=[256, 512, 1024, 2048],
                 dsm_decode_channels=[256, 512, 1024, 2048],
                 dilation = [[1, 2, 4, 8], [1, 2, 4, 8], [1, 2, 4, 8], [1, 2, 4, 8]],
                 fc_ratio=4,
                 dropout=0.1, window_size=8,
                 num_classes=6, momentum=0.5, use_memory = True,
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


        self.b4 = GLTM(rgb_dim=rgb_decode_channels[-1], dsm_dim = dsm_decode_channels[-1],
                       num_heads=num_classes,  mlp_ratio=fc_ratio,drop_path=dropout)


        self.rgb_p3 = Fusion(rgb_decode_channels[-1], rgb_decode_channels[-2])
        self.dsm_p3 = Fusion(dsm_decode_channels[-1], dsm_decode_channels[-2])
        self.b3 = GLSTM(rgb_q_dim=num_classes, rgb_k_dim=num_classes, rgb_v_dim=rgb_decode_channels[-2],
                 dsm_q_dim=num_classes, dsm_k_dim=num_classes, dsm_v_dim=dsm_decode_channels[-2], drop_path=dropout,  mlp_ratio=fc_ratio,
                 num_classes = num_classes, num_heads = num_classes, weight_ratio=weight_ratio, window_size= window_size,
                 act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d, aux_enhance = True,momentum=momentum, use_memory = use_memory)

        self.rgb_p2 = Fusion(rgb_decode_channels[-2], rgb_decode_channels[-3])
        self.dsm_p2 = Fusion(dsm_decode_channels[-2], dsm_decode_channels[-3])
        self.b2 = GLSTM(rgb_q_dim=num_classes, rgb_k_dim=num_classes, rgb_v_dim=rgb_decode_channels[-3],
                 dsm_q_dim=num_classes, dsm_k_dim=num_classes, dsm_v_dim=dsm_decode_channels[-3], drop_path=dropout,  mlp_ratio=fc_ratio,
                 num_classes = num_classes, num_heads = num_classes, weight_ratio=weight_ratio, window_size= window_size,
                 act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d, aux_enhance = True,momentum=momentum, use_memory = use_memory)

        self.rgb_p1 = Fusion(rgb_decode_channels[-3], rgb_decode_channels[-4])
        self.dsm_p1 = Fusion(dsm_decode_channels[-3], dsm_decode_channels[-4])
        self.b1 = GLSTM(rgb_q_dim=num_classes, rgb_k_dim=num_classes, rgb_v_dim=rgb_decode_channels[-4],
                 dsm_q_dim=num_classes, dsm_k_dim=num_classes, dsm_v_dim=dsm_decode_channels[-4], drop_path=dropout,  mlp_ratio=fc_ratio,
                 num_classes = num_classes, num_heads = num_classes, weight_ratio=weight_ratio, window_size= window_size,
                 act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d, aux_enhance = True,momentum=momentum, use_memory = use_memory)


        self.DualChannelSplitConcat = DualChannelSplitConcat(rgb_decode_channels[-4],dsm_decode_channels[-4],num_splits=num_classes)
        self.Conv5 = ConvBlock(rgb_decode_channels[-4]+dsm_decode_channels[-4], rgb_encode_channels[-5], 1)
        self.p = Fusion(rgb_encode_channels[-5])
        self.frsh = FRSH(rgb_encode_channels[-5], fc_ratio=fc_ratio, dilation=dilation[3], dropout=dropout, num_classes=num_classes)
        self.seg_head = nn.Sequential(SeparableConvBlock(rgb_encode_channels[-5], rgb_encode_channels[-5], kernel_size=3),
                                  nn.Dropout2d(p=dropout, inplace=True),
                                  ConvBlock(rgb_encode_channels[-5], num_classes, kernel_size=1, norm_layer=None, linearity=None))
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

        x, dsm = self.b4(res4, dsm4)

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
        out = self.frsh(out)
        features = out
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

        return out, features, RGB_Pre1, DSM_Pre1, aux_Pre1, RGB_Pre2, DSM_Pre2, aux_Pre2, RGB_Pre3, DSM_Pre3, aux_Pre3

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

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


class C2AHSegformer(nn.Module):
    def __init__(self,num_classes=7,
                 dropout=0.1,vis_channels =2,
                 fc_ratio=4,
                 decode_channels=32,
                 dsmencoder_channels=(96, 192, 384, 768),
                 embed_dim=128,
                 depths=(2, 2, 18, 2),
                 num_heads=(4, 8, 16, 32),
                 frozen_stages=2):
        super(C2AHSegformer, self).__init__()

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

        ## DSM特征提取
   #     self.dsmbackbone = SwinTransformer(embed_dim=embed_dim, depths=depths, num_heads=num_heads,
   #                                            frozen_stages=frozen_stages)
   #     dsm_channels = dsmencoder_channels

        self.dsmp = DSM_Preprocessing(k=3, mu=0, sigma=1)


        rgb_decode_channels = [decode_channels * num_classes,decode_channels * num_classes,
                               decode_channels * num_classes,decode_channels * num_classes]
        dsm_decode_channels = [decode_channels * num_classes // 2, decode_channels * num_classes// 2,
                               decode_channels * num_classes// 2, decode_channels * num_classes// 2]

        ##
        self.decoder = Decoder(rgb_encode_channels=rgb_encoder_channels,
                 rgb_decode_channels=rgb_decode_channels,
                 dsm_encode_channels=dsm_channels,
                 dsm_decode_channels=dsm_decode_channels,
                 num_classes=num_classes,dropout=dropout,
                 weight_ratio = 1.0, momentum=0.5, use_memory = False)

    def forward(self, vis, ir, dsm, mask=None):

        h, w = vis.size()[-2:]
        grad_x, grad_y, grad_magnitude = self.dsmp(dsm)
        x = torch.cat((vis, ir), dim=1)
        dsm = torch.cat((dsm, grad_x, grad_y), dim=1)

        # Encoder ResNet50
        x_pre = self.cnn(x)    ##H/2
        res1 = self.cnn1(x_pre)##H/4
        res2 = self.cnn2(res1) ##H/8
        res3 = self.cnn3(res2) ##H/16
        res4 = self.cnn4(res3) ##H/32

        #Encoder DSM
        dsm1, dsm2, dsm3, dsm4 = self.dsmbackbone(dsm)

        ##
        #  visualize_entropy_and_diff(out)

        if self.training:
            out, features, RGB_Pre1, DSM_Pre1, aux_Pre1, RGB_Pre2, DSM_Pre2, aux_Pre2, RGB_Pre3, DSM_Pre3, aux_Pre3 \
                = self.decoder(x_pre, res1, res2, res3, res4, dsm1, dsm2, dsm3, dsm4, h, w)
            return out, RGB_Pre1, DSM_Pre1, RGB_Pre2, DSM_Pre2, RGB_Pre3, DSM_Pre3

        else:
            out, features, RGB_Pre1, DSM_Pre1, aux_Pre1, RGB_Pre2, DSM_Pre2, aux_Pre2, RGB_Pre3, DSM_Pre3, aux_Pre3 \
                = self.decoder(x_pre, res1, res2, res3, res4, dsm1, dsm2, dsm3, dsm4, h, w)
            return out


'''

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
def CGGLNet_base(pretrained=True, num_classes=6, weight_path='/home/ny/Ni/MMRSSeg/pretrain_weights/stseg_base.pth'):

    # pretrained weights are load from official repo of Swin Transformer
    model = CGGLNet(dsmencoder_channels=(128, 256, 512, 1024),
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

def CGGLNet_small(pretrained=True, num_classes=4,
                  weight_path='/home/ny/Ni/MMRSSeg/pretrain_weights/stseg_small.pth'):
    model = CGGLNet(dsmencoder_channels=(96, 192, 384, 768),
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

def CGGLNet_tiny(pretrained=True, vis_channels=2, num_classes=4,
                 weight_path='/home/ny/Ni/MMRSSeg/pretrain_weights/stseg_tiny.pth'):
    model = CGGLNet(dsmencoder_channels=(96, 192, 384, 768),
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
'''
from fvcore.nn import FlopCountAnalysis
if __name__ == '__main__':

    num_classes = 6
    in_batch, inchannel, in_h, in_w = 1, 3, 1024, 1024
    x = torch.randn(in_batch, 2, in_h, in_w).cuda()
    ir = torch.randn(in_batch, 1, in_h, in_w).cuda()
    dsm = torch.randn(in_batch, 1, in_h, in_w).cuda()
    labels = torch.randint(0, num_classes+1, (in_batch, in_h, in_w)).cuda()
    net = C2AHSegformer(num_classes).cuda()
    net.eval()

    # 参数量
    params = sum(p.numel() for p in net.parameters())
    print("Params: %.2f M" % (params / 1e6))

    # FLOPs / GFLOPs
    flops = FlopCountAnalysis(net, (x, ir, dsm))
    print("FLOPs: %.2f G" % (flops.total() / 1e9))

    import time
    import torch

    # 清空显存统计
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # 预热，避免第一次运行不稳定
    warmup = 50
    repeat = 300

    with torch.no_grad():
        for _ in range(warmup):
            _ = net(x, ir, dsm)

        torch.cuda.synchronize()

        start = time.time()
        for _ in range(repeat):
            _ = net(x, ir, dsm)

        torch.cuda.synchronize()
        end = time.time()

    total_time = end - start
    fps = repeat / total_time

    memory_allocated = torch.cuda.max_memory_allocated() / 1024 / 1024
    memory_reserved = torch.cuda.max_memory_reserved() / 1024 / 1024

    print("FPS: %.2f" % fps)
    print("Inference time: %.2f ms/image" % (1000 / fps))
    print("Max memory allocated: %.2f MB" % memory_allocated)
    print("Max memory reserved: %.2f MB" % memory_reserved)