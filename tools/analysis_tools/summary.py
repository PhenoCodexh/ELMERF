#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用 torchinfo.summary 统计 mmseg 模型的参数量和 MACs/GFLOPs（可选乘2）
- 支持从 mmengine/mmseg 配置构建模型
- 优先走 forward_dummy；否则走 extract_feat + decode_head.forward 的主路径
- 仅统计主干+主头（关闭 auxiliary_head）
"""
import argparse
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from mmengine import Config, DictAction
from mmengine.registry import init_default_scope
from mmengine.model import revert_sync_batchnorm

from mmseg.registry import MODELS
from mmseg.models import BaseSegmentor

try:
    from torchinfo import summary
except ImportError as e:
    raise SystemExit("请先安装 torchinfo： pip install torchinfo") from e


class SegInferWrapper(nn.Module):
    """把 mmseg Segmentor 包装成一个标准 nn.Module，避免数据预处理分支干扰。"""
    def __init__(self, segmentor: BaseSegmentor):
        super().__init__()
        self.segmentor = segmentor
        # 只统计主干+主头
        if hasattr(self.segmentor, "auxiliary_head"):
            self.segmentor.auxiliary_head = None

    def forward(self, x: torch.Tensor):
        seg = self.segmentor
        # 优先使用 forward_dummy（如果实现了）
        if hasattr(seg, "forward_dummy"):
            return seg.forward_dummy(x)
        # 其次走常规主路径
        feats = seg.extract_feat(x)
        out = seg.decode_head.forward(feats)
        if isinstance(out, (list, tuple)):
            out = out[0]
        return out


def parse_args():
    p = argparse.ArgumentParser("mmseg + torchinfo 统计脚本")
    p.add_argument("config", help="mmseg 训练配置文件路径")
    p.add_argument("--shape", type=int, nargs="+", default=[512, 512],
                   help="输入尺寸，支持 H W 或 S（方形）")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                   choices=["cuda", "cpu"], help="运行设备")
    p.add_argument("--count-adds-as-flops", action="store_true",
                   help="把一次乘+一次加计为 2 FLOPs（有些论文口径）")
    p.add_argument("--cfg-options", nargs="+", action=DictAction, help="覆盖配置：key=val")
    return p.parse_args()


def fmt_num(n):
    # 友好打印
    for unit, div in [("G", 1e9), ("M", 1e6), ("K", 1e3)]:
        if n >= div:
            return f"{n / div:.3f}{unit}"
    return str(n)


def main():
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)

    cfg = Config.fromfile(cfg_path)
    cfg.work_dir = tempfile.TemporaryDirectory().name
    init_default_scope(cfg.get("scope", "mmseg"))

    # 解析输入形状
    if len(args.shape) == 1:
        H = W = int(args.shape[0])
    elif len(args.shape) == 2:
        H, W = map(int, args.shape)
    else:
        raise ValueError("--shape 只接受 1 或 2 个数字")

    # 构建模型
    model: BaseSegmentor = MODELS.build(cfg.model)
    model = revert_sync_batchnorm(model)
    model.eval()
    wrapper = SegInferWrapper(model).to(args.device)

    # torchinfo 统计：注意 input_size 里的 batch 维
    # col_names 里开启 mult_adds 才会统计 MACs
    s = summary(
        wrapper,
        input_size=(1, 3, H, W),
        device=args.device,
        verbose=0,
        col_names=("input_size", "output_size", "num_params", "mult_adds"),
        row_settings=("var_names",),
        mode="eval",
    )

    total_params = int(s.total_params)
    total_macs = int(getattr(s, "total_mult_adds", 0))  # MACs（Multiply-Adds）

    # 有些模块（注意力、插值等）torchinfo不一定会计入 mult_adds
    # 这里按你的口径输出两种：
    gmacs = total_macs / 1e9
    if args.count_adds_as_flops:
        gflops = 2.0 * gmacs
        flops_label = "GFLOPs (乘加各算1次)"
        flops_value = f"{gflops:.3f}"
    else:
        flops_label = "GMacs（常见论文口径）"
        flops_value = f"{gmacs:.3f}"

    print("=" * 34)
    print(f"Input size      : {(H, W)}")
    print(f"Params          : {fmt_num(total_params)}")
    print(f"{flops_label} : {flops_value}")
    print("=" * 34)
    # 如需看逐层表，可取消注释：
    # print(s)


if __name__ == "__main__":
    main()
