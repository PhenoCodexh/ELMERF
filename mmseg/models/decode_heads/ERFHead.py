from typing import List, Optional, Sequence, Tuple
from mmengine.logging import MMLogger
logger = MMLogger.get_current_instance()
from mmcv.cnn import ConvModule
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS
from mmseg.models.utils import resize
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1),
                          torch.mean(x, 1).unsqueeze(1)), dim=1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1,
                                 padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)  # [B,1,H,W]
        return x * scale


class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16,
                 pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not self.no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        # Channel Attention
        x_out = self.ChannelGate(x)

        # Spatial Attention
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)

        return x_out
class DropPathModule(nn.Module):
    """
   Randomised dropout path module
    """

    def __init__(self, drop_prob=0.0):
        super(DropPathModule, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x
class RelationUnit(nn.Module):
    def __init__(self, num_state, num_node, bias=False):
        super(RelationUnit, self).__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, bias=bias)

    def forward(self, x):
        h = self.conv1(x.permute(0, 2, 1)).permute(0, 2, 1)
        h = h - x
        h = self.relu(self.conv2(h))
        return h
class AnchorRelationEnhance(nn.Module):
    def __init__(self, feat_dim=768, gcn_dim=64, mids=4):
        super().__init__()
        self.num_s = gcn_dim
        self.num_n = mids * mids
        self.priors = nn.AdaptiveAvgPool2d(output_size=(mids + 2, mids + 2))
        self.conv_state = nn.Conv2d(feat_dim, self.num_s, 1)
        self.conv_proj = nn.Conv2d(feat_dim, self.num_s, 1)
        self.conv_extend = nn.Conv2d(self.num_s, feat_dim, 1)
        self.gcn = RelationUnit(self.num_s, self.num_n)

    def forward(self, x_feat, attn):
        n, _, h, w = x_feat.size()
        x_state = self.conv_state(x_feat).view(n, self.num_s, -1)
        x_proj = self.conv_proj(x_feat)
        x_mask = x_proj * attn
        x_anchor = self.priors(x_mask)[:, :, 1:-1, 1:-1].reshape(n, self.num_s, -1)
        x_proj_map = torch.matmul(x_anchor.permute(0, 2, 1), x_proj.view(n, self.num_s, -1))
        x_proj_map = torch.nn.functional.softmax(x_proj_map, dim=1)
        x_rel = torch.matmul(x_state, x_proj_map.permute(0, 2, 1)) * (1.0 / x_state.size(2))
        x_rel = self.gcn(x_rel)
        x_decoded = torch.matmul(x_rel, x_proj_map).view(n, self.num_s, h, w)
        return x_feat + self.conv_extend(x_decoded)



class ECA(nn.Module):
    """
    Efficient Channel Attention
    """

    def __init__(self, channels, kernel_size=3):
        super(ECA, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.pool(x).squeeze(-1).transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y


def get_norm(norm_type='bn_2d'):
    if norm_type == 'bn_2d':
        return lambda channels: nn.BatchNorm2d(channels)
    elif norm_type == 'ln':
        return lambda channels: nn.LayerNorm(channels)
    elif norm_type == 'identity':
        return lambda channels: nn.Identity()
    else:
        raise NotImplementedError("Normalization {} is not implemented.".format(norm_type))


def get_act(act_type='relu'):
    if act_type == 'relu':
        return nn.ReLU
    elif act_type == 'silu':
        return nn.SiLU
    elif act_type == 'gelu':
        return nn.GELU
    else:
        return nn.Identity


class ConvNormActBlock(nn.Sequential):
    """
    Convolution-Normalisation-Activation
    """

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None, dilation=1,
                 groups=1, bias=False, norm_layer='bn_2d', act_layer='relu'):
        if padding is None:
            padding = (kernel_size - 1) // 2
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)]
        if norm_layer != 'none':
            layers.append(get_norm(norm_layer)(out_channels))
        if act_layer != 'none':
            layers.append(get_act(act_layer)())
        super(ConvNormActBlock, self).__init__(*layers)


class EdgeAwareAttentionFusion(nn.Module):
    """
    GuidedWindowCrossAttention：Performs cross-attention computation within a window, further fused with local convolution and ECA.
    """

    def __init__(self, dim_in, dim_out, norm_in=True, has_skip=True, exp_ratio=1.0, norm_layer='bn_2d',
                 act_layer='relu', dw_ks=3, stride=1, dilation=1, eca_ratio=0.0, dim_head=64,
                 window_size=7, attn_drop=0.1, drop=0.0, drop_path=0.0, v_group=False):
        super(EdgeAwareAttentionFusion, self).__init__()
        self.norm = get_norm(norm_layer)(dim_in) if norm_in else nn.Identity()
        self.has_skip = (dim_in == dim_out and stride == 1) and has_skip
        dim_mid = int(dim_in * exp_ratio)
        self.window_size = window_size

        # The relationship between the number of channels and the number of heads
        assert dim_in % dim_head == 0, "The number of input channels must be divisible by dim_head."
        self.dim_head = dim_head
        self.num_head = dim_in // dim_head
        self.scale = dim_head ** -0.5

        self.q_proj = ConvNormActBlock(dim_in, dim_in, kernel_size=1, norm_layer='none', act_layer='none')
        self.k_proj = ConvNormActBlock(dim_in, dim_in, kernel_size=1, norm_layer='none', act_layer='none')
        self.v_proj = ConvNormActBlock(dim_in, dim_mid, kernel_size=1, norm_layer='none',
                                       act_layer=act_layer,
                                       groups=self.num_head if v_group else 1)

        self.attn_drop = nn.Dropout(attn_drop)

        self.conv_local = ConvNormActBlock(dim_mid, dim_mid, kernel_size=dw_ks, stride=stride, dilation=dilation,
                                           groups=dim_mid, norm_layer=norm_layer, act_layer='silu')
        self.eca = ECA(dim_mid, kernel_size=3) if eca_ratio > 0 else nn.Identity()
        self.proj = ConvNormActBlock(dim_mid, dim_out, kernel_size=1, norm_layer='none', act_layer='none')
        self.proj_drop = nn.Dropout(drop)
        self.drop_path = DropPathModule(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x_edge, x_vit):
        # Early fusion for V-branches.
        x_fused = self.norm(x_vit + x_edge)

        B, C, H, W = x_fused.shape
        ws = self.window_size

        pad_r = (ws - W % ws) % ws
        pad_b = (ws - H % ws) % ws

        x_vit_pad = F.pad(x_vit, (0, pad_r, 0, pad_b), value=0)
        x_edge_pad = F.pad(x_edge, (0, pad_r, 0, pad_b), value=0)
        x_fused_pad = F.pad(x_fused, (0, pad_r, 0, pad_b), value=0)

        Hp, Wp = x_fused_pad.shape[2], x_fused_pad.shape[3]
        nh, nw = Hp // ws, Wp // ws

        def window_unfold(x):
            x = x.reshape(B, C, nh, ws, nw, ws)
            x = x.permute(0, 2, 4, 1, 3, 5)
            return x.reshape(B * nh * nw, C, ws, ws)

        q_windows = window_unfold(x_edge_pad)
        k_windows = window_unfold(x_vit_pad)
        v_windows = window_unfold(x_fused_pad)

        q = self.q_proj(q_windows).view(-1, self.num_head, self.dim_head, ws * ws).permute(0, 1, 3, 2)
        k = self.k_proj(k_windows).view(-1, self.num_head, self.dim_head, ws * ws).permute(0, 1, 3, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        v = self.v_proj(v_windows).view(-1, self.num_head, self.dim_head, ws * ws).permute(0, 1, 3, 2)
        x_spatial = attn @ v
        x_spatial = x_spatial.permute(0, 1, 3, 2).contiguous().view(-1, C, ws, ws)

        # Importance-weighted fusion: Integrating attention with original local features.
        importance = attn.abs().mean(dim=(-1, -2)).mean(dim=1, keepdim=True)
        importance = importance / (importance.max() + 1e-6)
        x_spatial = importance.view(-1, 1, 1, 1) * x_spatial + (1 - importance.view(-1, 1, 1, 1)) * v_windows


        # Window collapses back to original image.
        x = x_spatial.view(B, nh, nw, C, ws, ws).permute(0, 3, 1, 4, 2, 5).reshape(B, C, Hp, Wp)
        x = x[:, :, :H, :W]

        y = self.eca(self.conv_local(x))
        y = self.proj_drop(self.proj(y))
        return x + self.drop_path(y) if self.has_skip else y
@MODELS.register_module()
class ERFHead(BaseDecodeHead):

    def __init__(self, feature_strides, decoder_params, **kwargs):
        super(ERFHead, self).__init__(
            input_transform='multiple_select',
            **kwargs
        )

        assert len(feature_strides) == len(self.in_channels), "Feature strides must match input channels"
        self.feature_strides = feature_strides

        M1_in_channels, M2_in_channels, M3_in_channels, M4_in_channels,E1_in_channels,E2_in_channels = self.in_channels
        embed_dim = decoder_params.get('embed_dim', 256)

        # Guided Window Cross Attention
        self.EAAF1 = EdgeAwareAttentionFusion(
            dim_in=embed_dim,
            dim_out=embed_dim,
            dim_head=32,
            window_size=7,
            attn_drop=0.1,
        )
        self.EAAF2 = EdgeAwareAttentionFusion(
            dim_in=embed_dim,
            dim_out=embed_dim,
            dim_head=64,
            window_size=7,
            attn_drop=0.1,
        )
        self.fuse1 = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim, 1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU()
        )
        self.fuse2 = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim, 1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU()
        )

        self.cbam_M1 = CBAM(gate_channels=embed_dim, reduction_ratio=8,
                            pool_types=['avg', 'max'])
        self.cbam_M2 = CBAM(gate_channels=embed_dim, reduction_ratio=8,
                            pool_types=['avg', 'max'])
        self.cbam_M4 = CBAM(gate_channels=embed_dim, reduction_ratio=4,
                            pool_types=['avg', 'max'])

        # Feature transformation: Mapping input features of various scales to a unified embedding dimension.

        self.conv_E1 = nn.Conv2d(E1_in_channels, embed_dim, kernel_size=1)
        self.conv_E2 = nn.Conv2d(E2_in_channels, embed_dim, kernel_size=1)

        self.linear_M4 = MLP(input_dim=M4_in_channels, embed_dim=embed_dim)
        self.linear_M3 = MLP(input_dim=M3_in_channels, embed_dim=embed_dim)
        self.linear_M2 = MLP(input_dim=M2_in_channels, embed_dim=embed_dim)
        self.linear_M1 = MLP(input_dim=M1_in_channels, embed_dim=embed_dim)

        self.AREM1 = AnchorRelationEnhance(feat_dim=embed_dim)
        self.AREM2 = AnchorRelationEnhance(feat_dim=embed_dim)
        self.AREM3 = AnchorRelationEnhance(feat_dim=embed_dim)
        self.AREM4 = AnchorRelationEnhance(feat_dim=embed_dim)

        self.linear_fuse = ConvModule(
            in_channels=embed_dim * 4,
            out_channels=embed_dim,
            kernel_size=1,
            norm_cfg=dict(type='BN', requires_grad=True)
        )

        self.linear_pred = nn.Conv2d(embed_dim, self.num_classes, kernel_size=1)
        self.dropout = nn.Dropout(0.1)

    def _forward_with_aux(self, inputs):
        x = self._transform_inputs(inputs)
        M1, M2, M3, M4,E1,E2 = x
        n = M1.size(0)

        # MLP projection
        S4 = self.linear_M4(M4).permute(0, 2, 1).reshape(n, -1, M4.shape[2], M4.shape[3])
        S3 = self.linear_M3(M3).permute(0, 2, 1).reshape(n, -1, M3.shape[2], M3.shape[3])
        S2 = self.linear_M2(M2).permute(0, 2, 1).reshape(n, -1, M2.shape[2], M2.shape[3])
        S1 = self.linear_M1(M1).permute(0, 2, 1).reshape(n, -1, M1.shape[2], M1.shape[3])
        R2 = self.conv_E2(E2)
        R1 = self.conv_E1(E1)

        # Compute auxiliary branch logits (attention mask portion) for auxiliary supervision.
        tmp_fuse = self.linear_fuse(torch.cat([
            resize(self.cbam_M4(S4), size=S1.shape[2:]),
            resize(S3, size=S1.shape[2:]),
            resize(S2, size=S1.shape[2:]),
            S1], dim=1))

        aux_logits = self.linear_pred(self.dropout(tmp_fuse))  # [B, num_classes, H, W]

        # Use auxiliary logits to calculate the attn_mask, which has a shape of [B, 1, H, W].
        MASK = F.softmax(aux_logits, dim=1)[:, 1:self.num_classes, :, :].sum(dim=1, keepdim=True)
        cnn_1 = self.fuse1(torch.cat([R1, S1], dim=1))
        C1 = self.cbam_M1(cnn_1)
        cnn_2 = self.fuse2(torch.cat([R2, S2], dim=1))
        C2 = self.cbam_M2(cnn_2)
        F2 = self.EAAF2(C2, S2)
        F1 = self.EAFF1(C1, S1)
        L1 = self.AREM1(F1, resize(MASK, size=S1.shape[2:]))
        L2 = self.AREM2(F2, resize(MASK, size=S2.shape[2:]))
        L3 = self.AREM3(S3, resize(MASK, size=S3.shape[2:]))
        L4 = self.AREM4(S4, resize(MASK, size=S4.shape[2:]))

        # Resize to same scale
        er_c4 = resize(L4, size=S1.shape[2:], mode='bilinear', align_corners=False)
        er_c3 = resize(L3, size=S1.shape[2:], mode='bilinear', align_corners=False)
        er_c2 = resize(L2, size=S1.shape[2:], mode='bilinear', align_corners=False)

        # Fuse and predict
        _c = self.linear_fuse(torch.cat([er_c4, er_c3, er_c2, L1], dim=1))

        final_logits = self.linear_pred(self.dropout(_c))
        return final_logits, aux_logits

    def forward(self, inputs):
        final_logits, _ = self._forward_with_aux(inputs)
        return final_logits

    def loss(self, inputs, data_samples, train_cfg=None):
        final_logits, aux_logits = self._forward_with_aux(inputs)
        main_loss_dict = super().loss_by_feat(final_logits, data_samples)
        # Assemble GT (shape [B, H, W]) for use by the auxiliary branch.
        gt_list = []
        for ds in data_samples:
            seg = ds.gt_sem_seg.data
            if seg.dim() == 3 and seg.size(0) == 1:
                seg = seg.squeeze(0)
            gt_list.append(seg.long())
        gt = torch.stack(gt_list, dim=0).to(aux_logits.device)  # [B,H,W]

        # Calculate auxiliary loss (upsampled to GT size)
        aux_up = F.interpolate(aux_logits, size=gt.shape[-2:], mode='bilinear', align_corners=False)
        cw = aux_up.new_tensor([1.0,5.0,5.5,5.4])
        loss_aux = F.cross_entropy(aux_up, gt, weight=cw,ignore_index=self.ignore_index)

        # Weighted and merged directly into a returned dictionary.
        main_loss_dict['loss_aux'] =0.4 * loss_aux

        return main_loss_dict
