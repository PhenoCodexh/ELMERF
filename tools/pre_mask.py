import os
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm

from mmseg.apis import init_model, inference_model
from mmseg.utils import get_palette

config_file = r'/home/qiaohongqiang/mxh/final_paper_Data/convnextweight/convnext-tiny_upernet_8xb2-amp-160k_voc.py'
checkpoint_file = r"/home/qiaohongqiang/mxh/final_paper_Data/convnextweight/iter_15600.pth"

model = init_model(config_file, checkpoint_file, device='cuda:0')

img_root = Path("/home/qiaohongqiang/mxh/test/")
save_mask_root = Path("/home/qiaohongqiang/mxh/final_paper_Data/convnextweight-mask-test")
save_mask_root.mkdir(parents=True, exist_ok=True)

for img_path in tqdm(list(img_root.iterdir())):
    if img_path.suffix.lower() not in {'.jpg','.jpeg','.png','.bmp','.tif','.tiff','.webp'}:
        continue

    data_sample = inference_model(model, str(img_path))

    # 兼容新旧版本
    if hasattr(data_sample, 'pred_sem_seg'):
        seg = data_sample.pred_sem_seg.data.squeeze(0).cpu().numpy().astype(np.uint8)
    else:
        seg = data_sample[0].astype(np.uint8)

    # 如果你只是想灰度保存，可继续乘 55；否则直接保存索引或着色
    mask_img = Image.fromarray(np.uint8(seg * 55))
    mask_img.save(save_mask_root / img_path.name)
