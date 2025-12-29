# Copyright (c) OpenMMLab. All rights reserved.
import math
import warnings
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import Conv2d, build_activation_layer, build_norm_layer
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn.bricks.transformer import MultiheadAttention
from mmengine.model import BaseModule, ModuleList, Sequential
from mmengine.model.weight_init import (constant_init, normal_init,
                                        trunc_normal_init)
from typing import List, Optional, Sequence, Tuple, Dict, Any
from timm.models.layers import DropPath
from mmseg.registry import MODELS
from ..utils import PatchEmbed, nchw_to_nlc, nlc_to_nchw

class LayerNorm(nn.Module):
    """
    A general LayerNorm implementation that supports both channels_first and channels_last data formats

    Args:
        normalized_shape (int): The feature dimension to be normalised
        eps (float): A small value to prevent division by zero, default 1e-6
        data_format (str): Data format, 'channels_last' or 'channels_first'
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError(f"Unsupported data format: {data_format}")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ConvEncoder(nn.Module):
    """
    Encoder block based on depthwise separable convolution.

    Args:
        dim (int): Number of input and output channels.
        drop_path (float): DropPath probability.
        layer_scale_init_value (float): Layer scale initialisation value.
        expan_ratio (int): FFN expansion ratio.
        kernel_size (int): Kernel size for the depthwise convolution.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6,
                 expan_ratio=4, kernel_size=7):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size,
                                padding=kernel_size // 2, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, expan_ratio * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(expan_ratio * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = input + self.drop_path(x)
        return x

class DifferentiableLoG(nn.Module):
    """
    Differentiable LoG
    """
    def __init__(self, sigmas: List[float], normalize: bool = True):
        super().__init__()
        if not isinstance(sigmas, (list, tuple)) or len(sigmas) == 0:
            raise ValueError('sigmas must be a non-empty list/tuple')
        self.normalize = normalize
        self.scales = nn.ModuleList([self._build_kernel(s) for s in sigmas])

    @staticmethod
    def _build_kernel(sigma: float) -> nn.Conv2d:
        # Compute LoG convolution kernel (2D)
        radius = max(1, int(3 * sigma + 0.5))
        size = radius * 2 + 1
        y, x = torch.meshgrid(torch.arange(size), torch.arange(size))
        y = y - radius
        x = x - radius
        norm2 = (x ** 2 + y ** 2).float()
        sigma2 = sigma ** 2
        # Classic LoG discrete formula
        kernel = (norm2 - 2 * sigma2) / (sigma2 ** 2) * torch.exp(-norm2 / (2 * sigma2))
        kernel = kernel - kernel.mean()
        kernel = kernel / kernel.abs().sum()
        conv = nn.Conv2d(1, 1, size, padding=radius, bias=False)
        with torch.no_grad():
            conv.weight.copy_(kernel.view(1, 1, size, size))
        conv.weight.requires_grad = False
        return conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) != 1:           # Convert to grayscale
            x = x.mean(dim=1, keepdim=True)
        responses = [scale(x).abs() for scale in self.scales]
        resp = torch.max(torch.stack(responses, dim=0), dim=0)[0]  # Take the maximum at multiple scales
        if self.normalize:
            maxv = resp.amax(dim=(2, 3), keepdim=True)
            minv = resp.amin(dim=(2, 3), keepdim=True)
            resp = (resp - minv) / (maxv - minv + 1e-12)
        return resp


class LoGModel(nn.Module):
    def __init__(self, sigmas: List[float] = [1.0], normalize: bool = True):
        super().__init__()
        self.log = DifferentiableLoG(sigmas, normalize=normalize)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.log(x)


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation

    Args:
        channels (int): Number of input channels
        reduction (int): Reduction ratio

    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.avg(x)
        w = self.fc(w)
        return x * w


class DenoiseConvBlock(nn.Module):
    """
    Denoising convolutional module for feature upsampling and noise suppression.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        norm_groups (int): Number of groups for GroupNorm
        se_reduction (int): Reduction ratio for the SE module
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 256,
                 norm_groups: int = 32, se_reduction: int = 16):
        super().__init__()

        # Ensure that `norm_groups` is a divisor of `out_channels`.
        if out_channels % norm_groups != 0:
            norm_groups = min(norm_groups, out_channels)
            while out_channels % norm_groups != 0:
                norm_groups -= 1

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(norm_groups, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(norm_groups, out_channels),
            nn.GELU(),
        )
        self.se = SEBlock(out_channels, se_reduction)

    def forward(self, x):
        """
        Args:
            x: input features (B, in_channels, H, W)
        Returns:
            Denoised features (B, out_channels, H, W)
        """
        y = self.conv(x)
        y = self.se(y)
        return y


class ConvBlock(nn.Module):
    """
    Convolutional block, containing ConvEncoder.

    Args:
        dim (int): The feature dimension.
        drop_path (float): The DropPath probability.
        conv_cfg (dict): The configuration for the ConvEncoder.
    """

    def __init__(self, dim, drop_path=0., conv_cfg=None):
        super().__init__()
        conv_cfg = conv_cfg or {}
        self.conv_enc = ConvEncoder(dim, drop_path=drop_path, **conv_cfg)

    def forward(self, x):
        return self.conv_enc(x)


class Edge_guided_LoG_CNN(nn.Module):
    """
    LoG-based CNN backbone

   Args:
        in_channels (int): Number of input channels
        denoise_cfg (dict): Configuration for the denoising module
        dims (Tuple[int, int]): Number of channels in each stage
        depths (Tuple[int, int]): Number of layers in each stage
        drop_path (float): DropPath probability
        conv_cfg (dict): Configuration for the ConvEncoder
        se_reduction (int): Reduction ratio for the SE (Squeeze-and-Excitation) module
    """

    def __init__(self,
                 in_channels: int = 1,
                 denoise_cfg: dict = None,
                 dims: Tuple[int, int] = (256, 320),
                 depths: Tuple[int, int] = (2, 2),
                 drop_path: float = 0.1,
                 conv_cfg: dict = None,
                 se_reduction: int = 16):
        super().__init__()

        assert len(dims) == 2, "`dims must contain two values: (C4x, C8x)"
        assert len(depths) == 2, "depths must contain two values corresponding to the two stages"

        # Noise Reduction Module Configuration
        denoise_cfg = denoise_cfg or {}
        self.Denoise = DenoiseConvBlock(in_channels, **denoise_cfg)

        # DropPath rate decay
        dpr = torch.linspace(0, drop_path, sum(depths)).tolist()
        # Stem: Two downsampling operations, with a total stride of 4
        self.stem = nn.Sequential(
            nn.Conv2d(denoise_cfg.get('out_channels', 256), dims[0], 3, stride=2, padding=1, bias=False),
            LayerNorm(dims[0], data_format="channels_first"),
            nn.GELU(),
            nn.Conv2d(dims[0], dims[0], 3, stride=2, padding=1, bias=False),
            LayerNorm(dims[0], data_format="channels_first"),
            nn.GELU(),
        )

        # Stage-0 (H/4)
        self.stage0 = nn.Sequential(*[
            ConvBlock(dims[0], drop_path=dpr[i], conv_cfg=conv_cfg)
            for i in range(depths[0])
        ])

        # DownsamplingH/8
        self.down_48 = nn.Sequential(
            LayerNorm(dims[0], data_format="channels_first"),
            nn.Conv2d(dims[0], dims[1], 3, stride=2, padding=1, bias=False),
        )

        # Stage-1 (H/8)
        self.stage1 = nn.Sequential(*[
            ConvBlock(dims[1], drop_path=dpr[depths[0] + i], conv_cfg=conv_cfg)
            for i in range(depths[1])
        ])

        # SE
        self.se0 = SEBlock(dims[0], se_reduction)
        self.se1 = SEBlock(dims[1], se_reduction)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Input features of shape (B, in_channels, H, W)
        Returns:
            feat_4x: Tensor of shape (B, dims[0], H/4, W/4)
            feat_8x: Tensor of shape (B, dims[1], H/8, W/8)

        """
        x = self.Denoise(x)
        x = self.stem(x)
        feat_4x = self.se0(self.stage0(x))

        x = self.down_48(feat_4x)
        feat_8x = self.se1(self.stage1(x))

        return feat_4x, feat_8x


class MixFFN(BaseModule):
    """An implementation of MixFFN of Segformer.

    The differences between MixFFN & FFN:
        1. Use 1X1 Conv to replace Linear layer.
        2. Introduce 3X3 Conv to encode positional information.
    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`. Defaults: 256.
        feedforward_channels (int): The hidden dimension of FFNs.
            Defaults: 1024.
        act_cfg (dict, optional): The activation config for FFNs.
            Default: dict(type='ReLU')
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 act_cfg=dict(type='GELU'),
                 ffn_drop=0.,
                 dropout_layer=None,
                 init_cfg=None):
        super().__init__(init_cfg)

        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.act_cfg = act_cfg
        self.activate = build_activation_layer(act_cfg)

        in_channels = embed_dims
        fc1 = Conv2d(
            in_channels=in_channels,
            out_channels=feedforward_channels,
            kernel_size=1,
            stride=1,
            bias=True)
        # 3x3 depth wise conv to provide positional encode information
        pe_conv = Conv2d(
            in_channels=feedforward_channels,
            out_channels=feedforward_channels,
            kernel_size=3,
            stride=1,
            padding=(3 - 1) // 2,
            bias=True,
            groups=feedforward_channels)
        fc2 = Conv2d(
            in_channels=feedforward_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            bias=True)
        drop = nn.Dropout(ffn_drop)
        layers = [fc1, pe_conv, self.activate, drop, fc2, drop]
        self.layers = Sequential(*layers)
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else torch.nn.Identity()

    def forward(self, x, hw_shape, identity=None):
        out = nlc_to_nchw(x, hw_shape)
        out = self.layers(out)
        out = nchw_to_nlc(out)
        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)


class EfficientMultiheadAttention(MultiheadAttention):
    """An implementation of Efficient Multi-head Attention of Segformer.

    This module is modified from MultiheadAttention which is a module from
    mmcv.cnn.bricks.transformer.
    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        attn_drop (float): A Dropout layer on attn_output_weights.
            Default: 0.0.
        proj_drop (float): A Dropout layer after `nn.MultiheadAttention`.
            Default: 0.0.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut. Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default: False.
        qkv_bias (bool): enable bias for qkv if True. Default True.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        sr_ratio (int): The ratio of spatial reduction of Efficient Multi-head
            Attention of Segformer. Default: 1.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=None,
                 init_cfg=None,
                 batch_first=True,
                 qkv_bias=False,
                 norm_cfg=dict(type='LN'),
                 sr_ratio=1):
        super().__init__(
            embed_dims,
            num_heads,
            attn_drop,
            proj_drop,
            dropout_layer=dropout_layer,
            init_cfg=init_cfg,
            batch_first=batch_first,
            bias=qkv_bias)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = Conv2d(
                in_channels=embed_dims,
                out_channels=embed_dims,
                kernel_size=sr_ratio,
                stride=sr_ratio)
            # The ret[0] of build_norm_layer is norm name.
            self.norm = build_norm_layer(norm_cfg, embed_dims)[1]

        # handle the BC-breaking from https://github.com/open-mmlab/mmcv/pull/1418 # noqa
        from mmseg import digit_version, mmcv_version
        if mmcv_version < digit_version('1.3.17'):
            warnings.warn('The legacy version of forward function in'
                          'EfficientMultiheadAttention is deprecated in'
                          'mmcv>=1.3.17 and will no longer support in the'
                          'future. Please upgrade your mmcv.')
            self.forward = self.legacy_forward

    def forward(self, x, hw_shape, identity=None):

        x_q = x
        if self.sr_ratio > 1:
            x_kv = nlc_to_nchw(x, hw_shape)
            x_kv = self.sr(x_kv)
            x_kv = nchw_to_nlc(x_kv)
            x_kv = self.norm(x_kv)
        else:
            x_kv = x

        if identity is None:
            identity = x_q

        # Because the dataflow('key', 'query', 'value') of
        # ``torch.nn.MultiheadAttention`` is (num_query, batch,
        # embed_dims), We should adjust the shape of dataflow from
        # batch_first (batch, num_query, embed_dims) to num_query_first
        # (num_query ,batch, embed_dims), and recover ``attn_output``
        # from num_query_first to batch_first.
        if self.batch_first:
            x_q = x_q.transpose(0, 1)
            x_kv = x_kv.transpose(0, 1)

        out = self.attn(query=x_q, key=x_kv, value=x_kv)[0]

        if self.batch_first:
            out = out.transpose(0, 1)

        return identity + self.dropout_layer(self.proj_drop(out))

    def legacy_forward(self, x, hw_shape, identity=None):
        """multi head attention forward in mmcv version < 1.3.17."""

        x_q = x
        if self.sr_ratio > 1:
            x_kv = nlc_to_nchw(x, hw_shape)
            x_kv = self.sr(x_kv)
            x_kv = nchw_to_nlc(x_kv)
            x_kv = self.norm(x_kv)
        else:
            x_kv = x

        if identity is None:
            identity = x_q

        # `need_weights=True` will let nn.MultiHeadAttention
        # `return attn_output, attn_output_weights.sum(dim=1) / num_heads`
        # The `attn_output_weights.sum(dim=1)` may cause cuda error. So, we set
        # `need_weights=False` to ignore `attn_output_weights.sum(dim=1)`.
        # This issue - `https://github.com/pytorch/pytorch/issues/37583` report
        # the error that large scale tensor sum operation may cause cuda error.
        out = self.attn(query=x_q, key=x_kv, value=x_kv, need_weights=False)[0]

        return identity + self.dropout_layer(self.proj_drop(out))


class TransformerEncoderLayer(BaseModule):
    """Implements one encoder layer in Segformer.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        drop_rate (float): Probability of an element to be zeroed.
            after the feed forward layer. Default 0.0.
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.0.
        drop_path_rate (float): stochastic depth rate. Default 0.0.
        qkv_bias (bool): enable bias for qkv if True.
            Default: True.
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default: False.
        init_cfg (dict, optional): Initialization config dict.
            Default:None.
        sr_ratio (int): The ratio of spatial reduction of Efficient Multi-head
            Attention of Segformer. Default: 1.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save
            some memory while slowing down the training speed. Default: False.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 qkv_bias=True,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 batch_first=True,
                 sr_ratio=1,
                 with_cp=False):
        super().__init__()

        # The ret[0] of build_norm_layer is norm name.
        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]

        self.attn = EfficientMultiheadAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            batch_first=batch_first,
            qkv_bias=qkv_bias,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratio)

        # The ret[0] of build_norm_layer is norm name.
        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]

        self.ffn = MixFFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg)

        self.with_cp = with_cp

    def forward(self, x, hw_shape):

        def _inner_forward(x):
            x = self.attn(self.norm1(x), hw_shape, identity=x)
            x = self.ffn(self.norm2(x), hw_shape, identity=x)
            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


@MODELS.register_module()
class Edge_LoG_MixVisionTransformer_cfg(BaseModule):
    """Backbone network integrating edge detection and CNN encoder

    Args:
        # Transformer related parameters
        in_channels (int): Number of input channels
        embed_dims (int): Embedding dimensions
        num_stages (int): Number of Transformer stages
        num_layers (Sequence[int]): Number of layers in each stage
        num_heads (Sequence[int]): Number of attention heads in each stage
        patch_sizes (Sequence[int]): Patch size for each stage
        strides (Sequence[int]): Stride for each stage
        sr_ratios (Sequence[int]): Spatial reduction ratio for each stage
        out_indices (Sequence[int]): Stage indices for output
        mlp_ratio (int): MLP hidden layer expansion ratio

        # Edge detection related parameters
        edge_sigmas (List[float]): Scale parameters for LoG filter
        cnn_cfg (dict): CNN encoder configuration
    """

    def __init__(self,
                 in_channels=3,
                 embed_dims=64,
                 num_stages=4,
                 num_layers=[3, 4, 6, 3],
                 num_heads=[1, 2, 4, 8],
                 patch_sizes=[7, 3, 3, 3],
                 strides=[4, 2, 2, 2],
                 sr_ratios=[8, 4, 2, 1],
                 out_indices=(0, 1, 2, 3),
                 mlp_ratio=4,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN', eps=1e-6),
                 pretrained=None,
                 init_cfg=None,
                 with_cp=False,

                 cnn_cfg=None):
        super().__init__(init_cfg=init_cfg)

        # Parameter validation
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')


        self.embed_dims = embed_dims
        self.num_stages = num_stages
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.patch_sizes = patch_sizes
        self.strides = strides
        self.sr_ratios = sr_ratios
        self.with_cp = with_cp
        self.out_indices = out_indices

        # Verify parameter consistency
        assert num_stages == len(num_layers) == len(num_heads) \
               == len(patch_sizes) == len(strides) == len(sr_ratios)
        assert max(out_indices) < self.num_stages

        # Edge Detection (LoG)
        self.lie = LoGModel(
            sigmas=[0.8, 1.2, 1.6],
            normalize=True
        )

        # CNN
        cnn_cfg = cnn_cfg or {
            'in_channels': 1,
            'denoise_cfg': {'out_channels': 64},
            'dims': (64, 128),
            'depths': (2, 4),
            'drop_path': 0.1,
            'se_reduction': 16
        }
        self.CNN_Encoder = Edge_guided_LoG_CNN(**cnn_cfg)
        self.cnn_dims = cnn_cfg['dims']

        # Transformer
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(num_layers))]

        cur = 0
        self.layers = ModuleList()
        for i, num_layer in enumerate(num_layers):
            embed_dims_i = embed_dims * num_heads[i]
            patch_embed = PatchEmbed(
                in_channels=in_channels,
                embed_dims=embed_dims_i,
                kernel_size=patch_sizes[i],
                stride=strides[i],
                padding=patch_sizes[i] // 2,
                norm_cfg=norm_cfg)

            layer = ModuleList([
                TransformerEncoderLayer(
                    embed_dims=embed_dims_i,
                    num_heads=num_heads[i],
                    feedforward_channels=mlp_ratio * embed_dims_i,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=dpr[cur + idx],
                    qkv_bias=qkv_bias,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    with_cp=with_cp,
                    sr_ratio=sr_ratios[i]) for idx in range(num_layer)
            ])
            in_channels = embed_dims_i
            norm = build_norm_layer(norm_cfg, embed_dims_i)[1]
            self.layers.append(ModuleList([patch_embed, layer, norm]))
            cur += num_layer

    def init_weights(self):
        #Weight initialisation
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)
        else:
            super().init_weights()

    def forward(self, x):
        """
        Args:
            x: Input image (B, C, H, W).
        Returns:
            A list of multi-scale feature maps.
        """
        #CNN Branch
        edge_map = self.lie(x)
        cnn_feat4, cnn_feat8 = self.CNN_Encoder(edge_map)

        # Transformer Branch
        outs = []
        for i, layer in enumerate(self.layers):
            x, hw_shape = layer[0](x)
            for block in layer[1]:
                x = block(x, hw_shape)
            x = layer[2](x)
            x = nlc_to_nchw(x, hw_shape)
            if i in self.out_indices:
                outs.append(x)

        #Combining CNN and Transformer Features
        outs.append(cnn_feat4)
        outs.append(cnn_feat8)

        return outs