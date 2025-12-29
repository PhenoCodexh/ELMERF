#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MMSeg inference for mmcv>=2.0 & mmseg>=1.0 & mmengine
不依赖 build_dataloader，支持：
1) 单图/文件夹推理
2) 若不传 --img-path，则按 cfg 中的 val/test 数据集逐图推理并可视化
"""
import argparse
import os
from pathlib import Path

import mmcv
import torch
from mmengine.config import Config

from mmseg.apis import init_model, inference_model, show_result_pyplot
from mmseg.registry import DATASETS


def parse_args():
    parser = argparse.ArgumentParser(description='MMSeg inference (mmcv2/mmseg1, no build_dataloader)')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--img-path', type=str, default=None,
                        help='Image file or directory. If None, run cfg val/test dataset')
    parser.add_argument('--out-dir', type=str, default='inference_results',
                        help='Output directory')
    parser.add_argument('--device', default='cuda:0', help='cuda:0 or cpu')
    parser.add_argument('--palette', default=None,
                        help='Palette name (e.g. ade20k). If None, use dataset meta')
    return parser.parse_args()


def get_palette(model, name=None):
    """Return palette from model meta or get_palette(name)."""
    if name is None:
        pal = model.dataset_meta.get('palette', None)
        if pal is not None:
            return pal
        from mmseg.core import get_palette
        return get_palette('ade20k')
    else:
        from mmseg.core import get_palette
        return get_palette(name)


def vis_and_save(model, img, result, out_file, palette):
    # result from inference_model: list[np.ndarray]
    show_result_pyplot(model, img, result, palette=palette, show=False, out_file=out_file)


def run_on_dataset(cfg, model, out_dir, palette):
    # 优先 test_dataloader
    dl_cfg = cfg.get('test_dataloader', cfg.get('val_dataloader', None))
    assert dl_cfg is not None, 'cfg 里没有 test_dataloader/val_dataloader'

    ds_cfg = dl_cfg['dataset']
    dataset = DATASETS.build(ds_cfg)

    total = len(dataset)
    for idx in range(total):
        info = dataset.get_data_info(idx)

        # 尝试拿图片路径（不同版本字段名可能不同）
        img_path = info.get('img_path', info.get('img', None))
        if img_path is None:
            raise KeyError('data_info 里没找到 img_path/img 字段.')

        img = mmcv.imread(img_path)

        # 直接用 inference_model，省去 dataloader
        result = inference_model(model, img)

        out_file = os.path.join(out_dir, Path(img_path).stem + '_seg.png')
        vis_and_save(model, img, result, out_file, palette)
        print(f'[{idx+1}/{total}] -> {out_file}')


def run_on_path(path_str, model, out_dir, palette):
    p = Path(path_str)
    if p.is_file():
        img = mmcv.imread(str(p))
        result = inference_model(model, img)
        out_file = os.path.join(out_dir, p.stem + '_seg.png')
        vis_and_save(model, img, result, out_file, palette)
        print(f'Saved: {out_file}')
    elif p.is_dir():
        for f in sorted(p.iterdir()):
            if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}:
                img = mmcv.imread(str(f))
                result = inference_model(model, img)
                out_file = os.path.join(out_dir, f.stem + '_seg.png')
                vis_and_save(model, img, result, out_file, palette)
                print(f'Saved: {out_file}')
    else:
        raise FileNotFoundError(f'{p} is not a valid file or directory.')


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    cfg = Config.fromfile(args.config)
    model = init_model(cfg, args.checkpoint, device=args.device)

    palette = get_palette(model, args.palette)

    if args.img_path is not None:
        run_on_path(args.img_path, model, args.out_dir, palette)
    else:
        run_on_dataset(cfg, model, args.out_dir, palette)


if __name__ == '__main__':
    main()
