# file: mmseg/models/decode_heads/pixel_decoder_gnn.py
# Copyright (c) OpenMMLab.
# Detectron2 PixelDecoder + GNN（Meijering节点提取）移植版

from __future__ import annotations
import copy
from typing import Dict, List, Tuple, Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmcv.cnn import Conv2d, ConvModule
from mmcv.cnn.bricks.transformer import MultiScaleDeformableAttention
from mmengine.model import (BaseModule, ModuleList, caffe2_xavier_init,
                            normal_init, xavier_init)
from mmengine.logging import MMLogger

from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptMultiConfig
from mmdet.models.task_modules.prior_generators import MlvlPointGenerator
from mmdet.models.layers.positional_encoding import SinePositionalEncoding

# ---------------------------------------------------------------------
# 工具函数：clone module / 激活函数（Detectron2里的实现）
# ---------------------------------------------------------------------
def _get_clones(module: nn.Module, N: int) -> ModuleList:
    return ModuleList([copy.deepcopy(module) for _ in range(N)])

def _get_activation_fn(activation: str):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == "glu":
        return F.glu
    else:
        raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")

# =============================================================================
# 1. 可微分 Meijering 滤波器
# =============================================================================
class DifferentiableMeijering(nn.Module):
    def __init__(self, sigmas: List[float]):
        super().__init__()
        self.scales = nn.ModuleList([self._build_single_scale(s) for s in sigmas])

    @staticmethod
    def _gaussian_1d(size: int, sigma: float):
        coords = torch.arange(size) - (size - 1) / 2
        g = torch.exp(-(coords ** 2) / (2.0 * sigma * sigma))
        return g / g.sum()

    @staticmethod
    def _second_derivative_1d(size: int, sigma: float):
        coords = torch.arange(size) - (size - 1) / 2
        factor = (coords ** 2 - sigma ** 2) / (sigma ** 4)
        g = torch.exp(-(coords ** 2) / (2.0 * sigma * sigma))
        g = g / g.sum()
        d2 = factor * g
        return d2 - d2.mean()

    def _build_single_scale(self, sigma: float) -> nn.Module:
        radius = int(3 * sigma + 0.5)
        size = radius * 2 + 1
        g = self._gaussian_1d(size, sigma).view(1, 1, 1, size)
        d2 = self._second_derivative_1d(size, sigma).view(1, 1, 1, size)

        gauss_x = nn.Conv2d(1, 1, (1, size), padding=(0, radius), bias=False)
        gauss_y = nn.Conv2d(1, 1, (size, 1), padding=(radius, 0), bias=False)
        d2x = nn.Conv2d(1, 1, (1, size), padding=(0, radius), bias=False)
        d2y = nn.Conv2d(1, 1, (size, 1), padding=(radius, 0), bias=False)
        with torch.no_grad():
            gauss_x.weight.copy_(g)
            gauss_y.weight.copy_(g.transpose(-1, -2))
            d2x.weight.copy_(d2)
            d2y.weight.copy_(d2.transpose(-1, -2))
        for m in (gauss_x, gauss_y, d2x, d2y):
            m.weight.requires_grad = False
        return nn.ModuleDict(dict(gauss_x=gauss_x, gauss_y=gauss_y, d2x=d2x, d2y=d2y))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        responses = []
        for m in self.scales:
            smoothed = m['gauss_y'](m['gauss_x'](x))
            rx = m['d2x'](smoothed).abs()
            ry = m['d2y'](smoothed).abs()
            resp = torch.max(rx, ry)
            responses.append(resp)
        return torch.max(torch.stack(responses, dim=0), dim=0)[0]


# =============================================================================
# 2. 节点提取器：MeijeringNodeExtractor
# =============================================================================
class MeijeringNodeExtractor(nn.Module):
    def __init__(self, in_channels: int, num_nodes: int, sigmas: List[float]):
        super().__init__()
        self.num_nodes = num_nodes
        self.projection = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.meijering_filter = DifferentiableMeijering(sigmas)
        self.nms_kernel_size = 5

    def forward(self, features: torch.Tensor):
        """
        Args:
            features: [B, C, H, W]
        Returns:
            node_features: [B, K, C]
            node_coords_pixel: [B, K, 2]  (x, y) 像素坐标
        """
        projected_view = self.projection(features)  # [B,1,H,W]
        heatmap = self.meijering_filter(projected_view)

        B, C, H, W = features.shape
        padding = self.nms_kernel_size // 2
        local_max = F.max_pool2d(heatmap, self.nms_kernel_size, stride=1, padding=padding)
        is_peak = (heatmap == local_max).float()
        heatmap = heatmap * is_peak

        heatmap_flat = heatmap.view(B, -1)
        _, topk_indices = torch.topk(heatmap_flat, k=self.num_nodes, dim=1)

        node_y = torch.div(topk_indices, W, rounding_mode='floor').unsqueeze(-1)
        node_x = (topk_indices % W).unsqueeze(-1)
        node_coords_pixel = torch.cat([node_x, node_y], dim=-1)  # [B,K,2]

        # 归一化到 [-1,1]，用于 grid_sample
        divisor = torch.tensor([W - 1, H - 1], device=features.device, dtype=torch.float32)
        node_coords_norm = (node_coords_pixel.float() / divisor) * 2.0 - 1.0  # [B,K,2]

        sampled_features = F.grid_sample(
            features,
            node_coords_norm.unsqueeze(1),  # [B,1,K,2]
            mode='bilinear',
            align_corners=True
        )  # [B,C,1,K]

        node_features = sampled_features.squeeze(-2).permute(0, 2, 1)  # [B,K,C]
        return node_features, node_coords_pixel


# =============================================================================
# 3. MSDeformAttn Encoder（Detectron2版本的简移植）
# =============================================================================
class MSDeformAttnTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        self.self_attn = MultiScaleDeformableAttention(
            embed_dims=d_model, num_levels=n_levels,
            num_heads=n_heads, num_points=n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # deformable self attention
        src2 = self.self_attn(
            query=self.with_pos_embed(src, pos),
            key=None,
            value=src,
            identity=None,
            query_pos=None,
            key_padding_mask=padding_mask,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reference_points=reference_points
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src = self.forward_ffn(src)
        return src


class MSDeformAttnTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer: nn.Module, num_layers: int):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device),
                indexing='ij'
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)
        return output


class MSDeformAttnTransformerEncoderOnly(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", num_feature_levels=4, enc_n_points=4):
        super().__init__()

        encoder_layer = MSDeformAttnTransformerEncoderLayer(
            d_model=d_model,
            d_ffn=dim_feedforward,
            dropout=dropout,
            activation=activation,
            n_levels=num_feature_levels,
            n_heads=nhead,
            n_points=enc_n_points
        )
        self.encoder = MSDeformAttnTransformerEncoder(encoder_layer, num_encoder_layers)
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        normal_init(self.level_embed, mean=0, std=1)
        # MultiScaleDeformableAttention 内部也有 init_weights，已在其 __init__ 中完成

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs: List[Tensor], pos_embeds: List[Tensor]):
        # 这里假定无 padding，直接 zeros mask
        masks = [torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool) for x in srcs]

        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []

        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shapes.append((h, w))

            src = src.flatten(2).transpose(1, 2)           # [B,HW,C]
            mask = mask.flatten(1)                          # [B,HW]
            pos_embed = pos_embed.flatten(2).transpose(1, 2)# [B,HW,C]
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)

            src_flatten.append(src)
            mask_flatten.append(mask)
            lvl_pos_embed_flatten.append(lvl_pos_embed)

        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((
            spatial_shapes.new_zeros((1,)),
            spatial_shapes.prod(1).cumsum(0)[:-1]
        ))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        memory = self.encoder(
            src_flatten, spatial_shapes, level_start_index,
            valid_ratios, lvl_pos_embed_flatten, mask_flatten
        )  # [B, HW_total, C]
        return memory, spatial_shapes, level_start_index


# =============================================================================
# 4. PixelDecoder（MMSeg风格）
# =============================================================================
@MODELS.register_module()
class GNNMSDeformPixelDecoder(BaseModule):
    """带 GNN 节点提取的 PixelDecoder（Detectron2 -> MMSeg 移植）

    Args:
        in_channels (list[int]): backbone 各层通道
        strides (list[int]): backbone 各层 stride
        feat_channels (int): 内部特征维度 (= conv_dim)
        out_channels (int): mask_feature 维度 (= mask_dim)
        num_outs (int): 输出特征层数（供 head 使用）
        encoder (dict): deformable encoder 配置（如果你想直接用 mmdet 的 Mask2FormerTransformerEncoder，可改这里）
        positional_encoding (dict): 位置编码配置
        common_stride (int): 与原 pixel decoder 一致（决定 FPN 层数）
        gnn_num_nodes (int), gnn_sigmas (List[float]), hidden_dim (int): GNN相关
        gnn_in_feature (str): 用不到具体通道（已统一 conv_dim），仅保留兼容字段
    """

    def __init__(self,
                 in_channels: Union[List[int], Tuple[int]],
                 strides: Union[List[int], Tuple[int]],
                 feat_channels: int = 256,
                 out_channels: int = 256,
                 num_outs: int = 3,
                 norm_cfg: ConfigType = dict(type='GN', num_groups=32),
                 act_cfg: ConfigType = dict(type='ReLU'),
                 # deformable encoder args (与 Detectron2 对应)
                 transformer_dropout: float = 0.1,
                 transformer_nheads: int = 8,
                 transformer_dim_feedforward: int = 1024,
                 transformer_enc_layers: int = 6,
                 transformer_in_level: int = 4,   # 使用多少层输入进 encoder（倒序取）
                 common_stride: int = 4,
                 # gnn args
                 gnn_in_feature: str = '',
                 gnn_num_nodes: int = 128,
                 gnn_sigmas: List[float] = [1.0, 2.0, 3.0],
                 hidden_dim: int = 256,
                 init_cfg: OptMultiConfig = None,
                 ) -> None:
        super().__init__(init_cfg=init_cfg)
        logger = MMLogger.get_current_instance()

        assert len(in_channels) == len(strides)
        self.strides = strides
        self.num_input_levels = len(in_channels)

        # === encoder 相关 ===
        self.transformer_num_feature_levels = transformer_in_level
        assert self.transformer_num_feature_levels >= 1
        # 从最高 stride 往下取 transformer_in_level 个
        input_conv_list = []
        for i in range(self.num_input_levels - 1,
                       self.num_input_levels - self.transformer_num_feature_levels - 1,
                       -1):
            input_conv_list.append(
                ConvModule(in_channels[i], feat_channels, 1,
                           norm_cfg=norm_cfg, act_cfg=None, bias=True)
            )
        self.input_convs = ModuleList(input_conv_list)

        self.encoder = MSDeformAttnTransformerEncoderOnly(
            d_model=feat_channels,
            dropout=transformer_dropout,
            nhead=transformer_nheads,
            dim_feedforward=transformer_dim_feedforward,
            num_encoder_layers=transformer_enc_layers,
            num_feature_levels=self.transformer_num_feature_levels,
        )
        self.positional_encoding = SinePositionalEncoding(num_feats=feat_channels // 2, normalize=True)
        self.level_encoding = nn.Embedding(self.transformer_num_feature_levels, feat_channels)

        # === FPN-like 结构 ===
        self.use_bias = norm_cfg is None
        self.lateral_convs = ModuleList()
        self.output_convs = ModuleList()

        # 剩下没进 encoder 的低层特征
        for i in range(self.num_input_levels - self.transformer_num_feature_levels - 1, -1, -1):
            self.lateral_convs.append(
                ConvModule(in_channels[i], feat_channels, 1,
                           bias=self.use_bias, norm_cfg=norm_cfg, act_cfg=None)
            )
            self.output_convs.append(
                ConvModule(feat_channels, feat_channels, 3, padding=1,
                           bias=self.use_bias, norm_cfg=norm_cfg, act_cfg=act_cfg)
            )

        # 最终 mask feature
        self.mask_feature = Conv2d(feat_channels, out_channels, 1)

        self.num_outs = num_outs
        self.point_generator = MlvlPointGenerator(strides)

        # ==== GNN 节点提取 ====
        self.node_extractor = MeijeringNodeExtractor(
            in_channels=feat_channels,
            num_nodes=gnn_num_nodes,
            sigmas=gnn_sigmas
        )
        self.gnn_feature_proj = nn.Linear(feat_channels, hidden_dim)

        self.maskformer_num_feature_levels = 4
        self.common_stride = common_stride

        # 用于传出额外信息
        self.extra_outputs: Optional[Dict[str, object]] = None

        logger.info(f'[GNNMSDeformPixelDecoder] transformer_in_levels={self.transformer_num_feature_levels}, '
                    f'num_input_levels={self.num_input_levels}, num_outs={num_outs}')

    # -----------------------------------------------------------------
    # 权重初始化
    # -----------------------------------------------------------------
    def init_weights(self) -> None:
        for i in range(self.transformer_num_feature_levels):
            xavier_init(self.input_convs[i].conv, gain=1, bias=0, distribution='uniform')

        for i in range(self.num_input_levels - self.transformer_num_feature_levels):
            caffe2_xavier_init(self.lateral_convs[i].conv, bias=0)
            caffe2_xavier_init(self.output_convs[i].conv, bias=0)

        caffe2_xavier_init(self.mask_feature, bias=0)
        normal_init(self.level_encoding, mean=0, std=1)

    # -----------------------------------------------------------------
    # forward
    # -----------------------------------------------------------------
    def forward(self, feats: List[Tensor]) -> Tuple[Tensor, List[Tensor]]:
        """
        Args:
            feats: list[Tensor], 每层 (B, C, H, W)，顺序与 in_channels 对应
        Returns:
            mask_feature: (B, out_channels, H, W)
            multi_scale_features: list[Tensor] 供后续 decoder 使用
        同时把额外 GNN 信息放到 self.extra_outputs 上
        """
        batch_size = feats[0].shape[0]
        encoder_input_list, padding_mask_list, level_pos_list = [], [], []
        spatial_shapes, reference_points_list = [], []

        # ----- encoder 部分，从高到低分辨率 -----
        for i in range(self.transformer_num_feature_levels):
            level_idx = self.num_input_levels - i - 1
            feat = feats[level_idx]                               # (B, C, H, W)
            feat_proj = self.input_convs[i](feat)
            feat_hw = torch._shape_as_tensor(feat)[2:].to(feat.device)

            padding_mask = feat.new_zeros((batch_size,) + feat.shape[-2:], dtype=torch.bool)
            pos_embed = self.positional_encoding(padding_mask)
            level_embed = self.level_encoding.weight[i].view(1, -1, 1, 1)
            level_pos = level_embed + pos_embed                  # (B, C, H, W)

            ref_points = self.point_generator.single_level_grid_priors(
                feat.shape[-2:], level_idx, device=feat.device
            )
            feat_wh = feat_hw.unsqueeze(0).flip(0, 1)
            ref_points = ref_points / (feat_wh * self.strides[level_idx])

            # flatten
            encoder_input_list.append(feat_proj.flatten(2).permute(0, 2, 1))
            level_pos_list.append(level_pos.flatten(2).permute(0, 2, 1))
            padding_mask_list.append(padding_mask.flatten(1))
            spatial_shapes.append(feat_hw)
            reference_points_list.append(ref_points)

        padding_masks = torch.cat(padding_mask_list, dim=1)
        encoder_inputs = torch.cat(encoder_input_list, dim=1)
        level_positional_encodings = torch.cat(level_pos_list, dim=1)
        spatial_shapes = torch.cat(spatial_shapes).view(-1, 2)
        num_queries_per_level = [h * w for h, w in spatial_shapes.tolist()]
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )
        reference_points = torch.cat(reference_points_list, dim=0)[None, :, None].repeat(
            batch_size, 1, self.transformer_num_feature_levels, 1
        )
        valid_ratios = reference_points.new_ones((batch_size, self.transformer_num_feature_levels, 2))

        memory = self.encoder.encoder(
            encoder_inputs,
            spatial_shapes,
            level_start_index,
            valid_ratios,
            level_positional_encodings,
            padding_masks
        )  # (B, total_queries, C)
        memory = memory.permute(0, 2, 1)  # (B, C, total_queries)

        outs = torch.split(memory, num_queries_per_level, dim=-1)
        outs = [
            x.reshape(batch_size, -1, spatial_shapes[i][0].item(), spatial_shapes[i][1].item())
            for i, x in enumerate(outs)
        ]  # 从低分辨率到高分辨率（因为 spatial_shapes 是按 encoder 遍历顺序）

        # ----- FPN 补齐剩下低层特征 -----
        for i in range(self.num_input_levels - self.transformer_num_feature_levels - 1, -1, -1):
            cur_feat = self.lateral_convs[i](feats[i])
            y = cur_feat + F.interpolate(outs[-1], size=cur_feat.shape[-2:], mode='bilinear', align_corners=False)
            y = self.output_convs[i](y)
            outs.append(y)

        # 选前 num_outs 个作为 multi_scale_features
        multi_scale_features = outs[:self.num_outs]
        mask_feature = self.mask_feature(outs[-1])

        # ----- 额外：GNN 节点提取 -----
        node_features_list, node_coords_list, feature_map_sizes_list = [], [], []
        # 注意：这里和你 D2 代码里一样，从 multi_scale_features 中提取（不含最高分辨率可自行调）
        for fmap in multi_scale_features:
            fmap_f = fmap.float()
            feature_map_sizes_list.append(fmap_f.shape[-2:])
            nf_conv_dim, ncoords = self.node_extractor(fmap_f)
            nf_hidden = self.gnn_feature_proj(nf_conv_dim)
            node_features_list.append(nf_hidden)
            node_coords_list.append(ncoords)

        # 缓存额外结果
        self.extra_outputs = dict(
            transformer_encoder_output=outs[0],  # 最低分辨率
            node_features_list=node_features_list,
            node_coords_list=node_coords_list,
            feature_map_sizes_list=feature_map_sizes_list
        )

        return mask_feature, multi_scale_features
