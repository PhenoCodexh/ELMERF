# -*- coding: utf-8 -*-
"""
Batch Grad-CAM for MMSeg semantic segmentation results
- 兼容 VOC2012 的 ImageSets/Segmentation/test.txt
- 兼容单张图片/目录批处理
- 兼容新旧 pytorch-grad-cam（自动处理是否带 use_cuda 参数）
- 支持多层 target（逗号分隔）
- 若指定 decode_head.pre_logits 而模型无此层，则回退到 decode_head.linear_fuse

依赖：
  pip install -U pillow numpy
  pip install -U pytorch-grad-cam   # 新版已无 use_cuda 参数，本脚本自动兼容
  mim install mmengine mmcv mmsegmentation  # 或按你的环境安装
"""

from argparse import ArgumentParser
from pathlib import Path
from inspect import signature
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from mmengine import Config
from mmengine.model import revert_sync_batchnorm
from mmseg.apis import init_model, inference_model, show_result_pyplot
from mmseg.utils import register_all_modules

from pytorch_grad_cam import GradCAM          # 如需换法：XGradCAM/EigenCAM/GradCAMPlusPlus
from pytorch_grad_cam.utils.image import preprocess_image, show_cam_on_image


# ---------------------- Grad-CAM 目标封装 ----------------------
class SemanticSegmentationTarget:
    """将 CAM 目标限制在某个类别的预测区域内。"""
    def __init__(self, category, mask, size):
        self.category = category
        self.mask = torch.from_numpy(mask).float()
        self.size = size
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()

    def __call__(self, model_output):
        # model_output: [C, H', W']（logits/map）
        model_output = torch.unsqueeze(model_output, dim=0)  # [1,C,H',W']
        model_output = F.interpolate(model_output, size=self.size, mode='bilinear', align_corners=False)
        model_output = torch.squeeze(model_output, dim=0)    # [C,H,W]
        return (model_output[self.category, :, :] * self.mask).sum()


# ---------------------- 工具函数 ----------------------
def list_images(root: Path, recursive: bool, exts):
    if root.is_file():
        return [root]
    pattern = "**/*" if recursive else "*"
    return [p for p in root.glob(pattern) if p.suffix.lower() in exts and p.is_file()]


def load_voc_ids(list_file: Path):
    ids = []
    with open(list_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                ids.append(line)
    return ids


def resolve_voc_image_path(voc_root: Path, img_id: str, exts):
    jpeg_dir = voc_root / "JPEGImages"
    for ext in exts:
        p = jpeg_dir / f"{img_id}{ext}"
        if p.exists():
            return p
    return None


def pick_decode_head_target_layer(model):
    """自动挑选 decode_head 中最后一个非分类卷积；若没有则回退到分类卷积。"""
    import torch.nn as nn
    last_conv, last_name = None, None
    cls_conv, cls_name = None, None
    for name, m in model.decode_head.named_modules():
        if isinstance(m, nn.Conv2d):
            if name.endswith('conv_seg') or name.endswith('linear_pred'):
                cls_conv, cls_name = m, f"decode_head.{name}"
            else:
                last_conv, last_name = m, f"decode_head.{name}"
    if last_conv is not None:
        print(f"[AUTO] Use {last_name}")
        return [last_conv]
    if cls_conv is not None:
        print(f"[AUTO] Fallback to {cls_name}")
        return [cls_conv]
    raise RuntimeError("decode_head 中未找到 Conv2d，可手动指定 --target-layers")


def parse_target_layers(model, target_layers_str: str):
    """支持逗号分隔的多层；支持 'auto'；对 pre_logits 做回退。"""
    if target_layers_str.strip().lower() == "auto":
        return pick_decode_head_target_layer(model)

    layers = []
    for token in target_layers_str.split(","):
        token = token.strip()
        if not token:
            continue
        # 对 pre_logits 做友好回退
        if token == "decode_head.pre_logits":
            try:
                lyr = eval(f"model.{token}")
            except Exception:
                print("[WARN] 未找到 decode_head.pre_logits，回退到 decode_head.linear_fuse")
                lyr = eval("model.decode_head.linear_fuse")
            layers.append(lyr)
            continue

        try:
            lyr = eval(f"model.{token}")
            layers.append(lyr)
        except Exception as e:
            raise RuntimeError(f"无法解析 target layer: {token} | 错误: {e}")
    if not layers:
        raise RuntimeError("解析 target layers 为空，请检查 --target-layers")
    return layers


# ---------------------- 核心处理 ----------------------
def process_one(img_path: Path, model, target_layers, category, cfg,
                pred_out_dir: Path, cam_out_dir: Path,
                overwrite: bool = True):
    # 1) 分割可视化
    pred_fname = pred_out_dir / f"{img_path.stem}_pred.png"
    if overwrite or not pred_fname.exists():
        result = inference_model(model, str(img_path))
        show_result_pyplot(model, str(img_path), result,
                           draw_gt=False, show=False, out_file=str(pred_fname))
    else:
        result = inference_model(model, str(img_path))  # 仍需结果用于 CAM

    # 2) 构造当前类别掩码
    pred = result.pred_sem_seg.data           # [1,H,W]
    pred_np = pred.cpu().numpy().squeeze(0)   # [H,W]
    mask_float = np.float32(pred_np == category)

    # 3) 归一化
    image = np.array(Image.open(str(img_path)).convert('RGB'))
    h, w = image.shape[:2]
    rgb_img = np.float32(image) / 255.0

    image_mean = cfg.data_preprocessor['mean']
    image_std = cfg.data_preprocessor['std']
    input_tensor = preprocess_image(
        rgb_img,
        mean=[x / 255.0 for x in image_mean],
        std=[x / 255.0 for x in image_std]
    )

    # 4) CAM（关键：兼容新旧 API，不强行传 use_cuda）
    cam_fname = cam_out_dir / f"{img_path.stem}_cam_cls{category}.png"
    cam_kwargs = dict(model=model, target_layers=target_layers)

    # 旧版 pytorch-grad-cam 的 __init__ 有 use_cuda，新版没有
    if 'use_cuda' in signature(GradCAM.__init__).parameters:
        use_cuda = next(model.parameters()).is_cuda and torch.cuda.is_available()
        cam_kwargs['use_cuda'] = use_cuda

    with GradCAM(**cam_kwargs) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=[
            SemanticSegmentationTarget(category, mask_float, (h, w))
        ])[0, :]
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        Image.fromarray(cam_image).save(str(cam_fname))

    return str(pred_fname), str(cam_fname)


# ---------------------- 主程序 ----------------------
def main():
    parser = ArgumentParser()
    # 两种模式：① 普通模式（img） ② VOC 模式（voc-root）
    parser.add_argument('img', nargs='?', default=None,
                        help='单张图片路径或图片文件夹路径（普通模式）')
    parser.add_argument('--voc-root', default=None,
                        help='VOC2012 根目录（包含 JPEGImages/ 与 ImageSets/Segmentation/）')
    parser.add_argument('--voc-list', default='ImageSets/Segmentation/test.txt',
                        help='相对 voc-root 的列表文件路径，默认 test.txt')

    parser.add_argument('config', help='MMSeg 配置文件')
    parser.add_argument('checkpoint', help='模型权重文件')

    # 输出
    parser.add_argument('--out-file', default='prediction.png', help='单张图：分割结果输出路径')
    parser.add_argument('--cam-file', default='vis_cam.png', help='单张图：CAM 输出路径')
    parser.add_argument('--out-dir', default='predictions', help='批处理：分割结果输出目录')
    parser.add_argument('--cam-dir', default='cams', help='批处理：CAM 输出目录')

    parser.add_argument('--recursive', action='store_true', help='（普通模式）递归遍历子目录')
    parser.add_argument('--exts', default='.jpg,.jpeg,.png,.bmp,.tif,.tiff',
                        help='允许的图片扩展名，逗号分隔（含点，小写）')

    # target 层与类别
    parser.add_argument('--target-layers', default='decode_head.linear_fuse',
                        help='用于 CAM 的目标层，支持逗号分隔多层；可用 "auto"；'
                             '若写 decode_head.pre_logits 而该层不存在，将回退到 decode_head.linear_fuse')
    parser.add_argument('--category-index', default='1', help='要可视化的类别 ID（int）')
    parser.add_argument('--device', default='cuda:0', help='cuda:0 / cpu')
    parser.add_argument('--no-overwrite', action='store_true', help='已存在文件则跳过')
    parser.add_argument('--skip-missing', action='store_true', help='VOC 模式下缺图是否跳过（否则报错）')
    args = parser.parse_args()

    exts = tuple(s.strip().lower() for s in args.exts.split(',') if s.strip())
    voc_mode = args.voc_root is not None

    # 1) 初始化模型
    register_all_modules()
    model = init_model(args.config, args.checkpoint, device=args.device)
    if args.device.lower().startswith('cpu'):
        model = revert_sync_batchnorm(model)

    # 2) 解析目标层（支持多层/auto/回退）
    target_layers = parse_target_layers(model, args.target_layers)

    # 3) 读取 config（用于 mean/std）
    cfg = Config.fromfile(args.config)

    # 4) 执行
    if voc_mode:
        voc_root = Path(args.voc_root)
        list_file = voc_root / args.voc_list
        if not list_file.exists():
            raise FileNotFoundError(f"找不到列表文件：{list_file}")

        ids = load_voc_ids(list_file)
        if not ids:
            raise RuntimeError(f"{list_file} 中未读取到任何图像 id")

        pred_out_dir = Path(args.out_dir); pred_out_dir.mkdir(parents=True, exist_ok=True)
        cam_out_dir  = Path(args.cam_dir);  cam_out_dir.mkdir(parents=True, exist_ok=True)

        print(f"[INFO] VOC 模式：{len(ids)} 张（{list_file}）")
        ok, miss = 0, 0
        for i, img_id in enumerate(ids, 1):
            img_path = resolve_voc_image_path(voc_root, img_id, exts)
            if img_path is None:
                miss += 1
                msg = f"[{i}/{len(ids)}] MISSING: {img_id}（JPEGImages 下未找到 {exts}）"
                if args.skip-missing:
                    print(msg); continue
                else:
                    raise FileNotFoundError(msg)

            try:
                pred_path, cam_path = process_one(
                    img_path, model, target_layers, int(args.category_index), cfg,
                    pred_out_dir, cam_out_dir, overwrite=not args.no_overwrite
                )
                print(f"[{i}/{len(ids)}] OK: {img_id} -> seg:{Path(pred_path).name}, cam:{Path(cam_path).name}")
                ok += 1
            except Exception as e:
                print(f"[{i}/{len(ids)}] FAIL: {img_id} | {e}")

        print(f"[DONE] Seg: {pred_out_dir} | CAM: {cam_out_dir} | 成功 {ok} | 缺失 {miss}")

    else:
        # 普通模式（单张 / 目录）
        if args.img is None:
            raise ValueError("普通模式需要提供 img（文件或目录），或使用 --voc-root 进入 VOC 模式。")
        root = Path(args.img)
        is_batch = root.is_dir()

        if not is_batch:
            pred_out = args.out_file
            cam_out  = args.cam_file
            pred_dir = Path(pred_out).parent; pred_dir.mkdir(parents=True, exist_ok=True)
            cam_dir  = Path(cam_out).parent;  cam_dir.mkdir(parents=True, exist_ok=True)

            pred_path, cam_path = process_one(
                root, model, target_layers, int(args.category_index), cfg,
                pred_dir, cam_dir, overwrite=not args.no_overwrite
            )
            if Path(pred_path) != Path(pred_out):
                os.replace(pred_path, pred_out)
            if Path(cam_path) != Path(cam_out):
                os.replace(cam_path, cam_out)
            print(f"[OK] Saved:\n  seg -> {pred_out}\n  cam -> {cam_out}")
        else:
            pred_out_dir = Path(args.out_dir); pred_out_dir.mkdir(parents=True, exist_ok=True)
            cam_out_dir  = Path(args.cam_dir);  cam_out_dir.mkdir(parents=True, exist_ok=True)

            imgs = list_images(root, args.recursive, exts)
            if not imgs:
                raise FileNotFoundError(f"在 {root} 下未找到图片（exts={exts}）")

            print(f"[INFO] Found {len(imgs)} images. Start processing...")
            for i, p in enumerate(sorted(imgs), 1):
                try:
                    pred_path, cam_path = process_one(
                        p, model, target_layers, int(args.category_index), cfg,
                        pred_out_dir, cam_out_dir, overwrite=not args.no_overwrite
                    )
                    print(f"[{i}/{len(imgs)}] OK: {p.name} -> seg:{Path(pred_path).name}, cam:{Path(cam_path).name}")
                except Exception as e:
                    print(f"[{i}/{len(imgs)}] FAIL: {p} | {e}")

            print(f"[DONE] Results saved to:\n  Seg: {pred_out_dir}\n  CAM: {cam_out_dir}")


if __name__ == '__main__':
    main()
