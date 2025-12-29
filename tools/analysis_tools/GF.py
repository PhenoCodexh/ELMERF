# -*- coding: utf-8 -*-
# Strict FLOPs & Params counter for mmseg models
import argparse
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
from mmengine import Config, DictAction
from mmengine.logging import MMLogger
from mmengine.model import revert_sync_batchnorm
from mmengine.registry import init_default_scope
from mmseg.registry import MODELS
from mmseg.models import BaseSegmentor
from mmseg.structures import SegDataSample

from fvcore.nn import FlopCountAnalysis

# ---- 替换你脚本顶部的这些 helper ----
# JIT Value -> numel（若存在动态维度就返回 0，避免崩）
def _value_numel(v):
    try:
        t = v.type()
        if hasattr(t, "sizes") and t.sizes() is not None:
            num = 1
            for s in t.sizes():
                if isinstance(s, int) and s > 0:
                    num *= s
                else:
                    return 0  # 动态维度，没法精确，先返回 0
            return num
    except Exception:
        pass
    return 0

def _numel(values):
    if not isinstance(values, (list, tuple)):
        values = [values]
    return sum(_value_numel(v) for v in values)

def _per_elem_handle(factor=1):
    # JIT op handle: (inputs, outputs) -> flops(int)
    def h(inputs, outputs):
        return _numel(outputs) * factor
    return h

def _softmax_handle(inputs, outputs):
    # 粗略：exp + sum + div ≈ 5 次/元素
    return _numel(outputs) * 5

def _upsample_bilinear_handle(inputs, outputs):
    # 粗略：双线性插值 ~ 8 ops/像素（4乘+4加）
    return _numel(outputs) * 8

def _register_default_jit_handles(fa, include_activations: bool):
    # 归一化/Pool/插值
    fa.set_op_handle("aten::layer_norm", _per_elem_handle(2))
    fa.set_op_handle("aten::batch_norm", _per_elem_handle(2))
    fa.set_op_handle("aten::group_norm", _per_elem_handle(2))
    fa.set_op_handle("aten::adaptive_avg_pool2d", _per_elem_handle(1))
    fa.set_op_handle("aten::avg_pool2d", _per_elem_handle(1))
    fa.set_op_handle("aten::max_pool2d", _per_elem_handle(1))
    fa.set_op_handle("aten::upsample_nearest2d", _per_elem_handle(1))
    fa.set_op_handle("aten::upsample_bilinear2d", _upsample_bilinear_handle)

    # 激活/逐元素
    factor = 1 if include_activations else 0
    fa.set_op_handle("aten::gelu", _per_elem_handle(6 * factor or 0))
    fa.set_op_handle("aten::silu", _per_elem_handle(4 * factor or 0))
    fa.set_op_handle("aten::sigmoid", _per_elem_handle(4 * factor or 0))
    fa.set_op_handle("aten::softmax",
                     _softmax_handle if include_activations else _per_elem_handle(0))

    for op in [
        "aten::add", "aten::sub", "aten::mul", "aten::div", "aten::pow",
        "aten::sqrt", "aten::abs", "aten::rsub", "aten::sum", "aten::mean",
        "aten::amin", "aten::amax", "aten::expand_as", "aten::pad",
        "aten::unflatten", "aten::reciprocal"
    ]:
        fa.set_op_handle(op, _per_elem_handle(factor))


def _force_infer_path(model: BaseSegmentor):
    """尽量让复杂度走到 decode 主路径。优先 forward_dummy。"""
    if hasattr(model, "forward_dummy"):
        def _f(x):
            return model.forward_dummy(x)
        return _f

    # 通用兜底：尽量对齐 predict/encode_decode
    if hasattr(model, "encode_decode"):
        def _f(x):
            feats = model.extract_feat(x)
            out = model.decode_head.forward(feats)
            if isinstance(out, (list, tuple)):
                out = out[0]
            return out
        return _f

    # 最后兜底：直接调用 forward（mmseg 会走 predict 分支）
    def _f(x):
        return model(x)
    return _f

def parse_args():
    p = argparse.ArgumentParser(description="Strict FLOPs/Params for mmseg models")
    p.add_argument("config", help="train config file path")
    p.add_argument("--shape", type=int, nargs="+", default=[2048, 1024], help="input image size, e.g., 512 512")
    p.add_argument("--include-activations", action="store_true",
                   help="把 LN/GELU/Softmax/逐元素/插值等也计入 FLOPs（更严格，数值更大）")
    p.add_argument(
        "--cfg-options", nargs="+", action=DictAction,
        help="override settings in xxx=yyy format"
    )
    return p.parse_args()

def main():
    args = parse_args()
    logger = MMLogger.get_instance(name="MMLogger")

    config_name = Path(args.config)
    if not config_name.exists():
        raise FileNotFoundError(f"Config file {config_name} does not exist.")

    cfg = Config.fromfile(config_name)
    cfg.work_dir = tempfile.TemporaryDirectory().name
    cfg.log_level = "WARN"
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    init_default_scope(cfg.get("scope", "mmseg"))

    # 解析输入形状
    if len(args.shape) == 1:
        input_shape = (3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (3,) + tuple(args.shape)
    else:
        raise ValueError("invalid --shape, expect one or two ints")

    # 构建模型
    model: BaseSegmentor = MODELS.build(cfg.model)
    if hasattr(model, "auxiliary_head"):
        model.auxiliary_head = None  # 只统计主干 + 主头
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    model = revert_sync_batchnorm(model)

    # 构造一条真实的预处理路径（保证尺寸/Pad 一致）
    meta = {"ori_shape": input_shape[-2:], "pad_shape": input_shape[-2:]}
    data_batch = {
        "inputs": [torch.rand(input_shape)],
        "data_samples": [SegDataSample(metainfo=meta)]
    }
    data = model.data_preprocessor(data_batch)
    x = data["inputs"]  # Tensor[B,C,H,W]
    if torch.cuda.is_available():
        x = x.cuda()

    # 强制走主路径
    forward_fn = _force_infer_path(model)

    # fvcore FLOPs
    fa = FlopCountAnalysis(model, (x,))
    # 用我们自己的前向函数避免跑到奇怪分支
    fa._cache = {}  # 清缓存
    orig_forward = model.forward
    try:
        model.forward = forward_fn  # 临时替换
        _register_default_jit_handles(fa, include_activations=args.include_activations)
        flops = int(fa.total())
    finally:
        model.forward = orig_forward

    # 参数量
    params = sum(p.numel() for p in model.parameters())

    # 输出
    def _fmt(v):
        # 和 mmengine._format_size 类似
        kb = 1024.0
        mb = kb * 1024
        gb = mb * 1024
        if v >= gb:
            return f"{v / gb:.3f}G"
        if v >= mb:
            return f"{v / mb:.3f}M"
        if v >= kb:
            return f"{v / kb:.3f}K"
        return str(v)

    split = "=" * 30
    print(f"{split}\nCompute type: fvcore + custom aten handles"
          f"\nInput shape: {input_shape[-2:]}"
          f"\nInclude activations: {args.include_activations}"
          f"\nFlops: {_fmt(flops)}"
          f"\nParams: {_fmt(params)}\n{split}")
    print("注：如需进一步逼近注意力 FLOPs，请为自定义 Attention 模块单独写 handle（见下说明）。")

if __name__ == "__main__":
    main()
