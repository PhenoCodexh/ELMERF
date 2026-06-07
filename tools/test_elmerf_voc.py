import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from PIL import Image

from mmseg.apis import inference_model
from mmseg.registry import MODELS

IMAGE_SUFFIXES = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')
ELMERF_CLASSES = (
    'background', 'greenshoottissues', 'yellowshoottissues', 'roots')
ELMERF_PALETTE = [[242, 234, 218], [69, 185, 124], [143, 75, 46],
                  [116, 120, 124]]
KEY_REPLACEMENTS = [
    ('decode_head.guided_attn1.', 'decode_head.EAAF1.'),
    ('decode_head.guided_attn2.', 'decode_head.EAAF2.'),
    ('decode_head.fuse_c1.', 'decode_head.fuse1.'),
    ('decode_head.fuse_c2.', 'decode_head.fuse2.'),
    ('decode_head.cbam_c1.', 'decode_head.cbam_M1.'),
    ('decode_head.cbam_c2.', 'decode_head.cbam_M2.'),
    ('decode_head.cbam_c4.', 'decode_head.cbam_M4.'),
    ('decode_head.conv_lie1_4.', 'decode_head.conv_E1.'),
    ('decode_head.conv_lie1_8.', 'decode_head.conv_E2.'),
    ('decode_head.linear_c1.', 'decode_head.linear_M1.'),
    ('decode_head.linear_c2.', 'decode_head.linear_M2.'),
    ('decode_head.linear_c3.', 'decode_head.linear_M3.'),
    ('decode_head.linear_c4.', 'decode_head.linear_M4.'),
    ('decode_head.enhancer_c1.', 'decode_head.AREM1.'),
    ('decode_head.enhancer_c2.', 'decode_head.AREM2.'),
    ('decode_head.enhancer_c3.', 'decode_head.AREM3.'),
    ('decode_head.enhancer_c4.', 'decode_head.AREM4.'),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run ELMERF inference and mIoU evaluation on a VOC-format '
                    'semantic segmentation test set.')
    parser.add_argument(
        '--config',
        default='configs/ELMERF/ELMERF_mit-b1_voc.py',
        help='Path to the ELMERF config file.')
    parser.add_argument(
        '--checkpoint',
        default='weight/elmerf_model.pth',
        help='Path to the pretrained ELMERF checkpoint.')
    parser.add_argument(
        '--voc-root',
        default='sample_data/VOCdevkit/VOC2012',
        help='Path to the VOC2012 directory.')
    parser.add_argument(
        '--split',
        default='test',
        help='Split name under ImageSets/Segmentation, for example test.')
    parser.add_argument(
        '--out-dir',
        default='outputs/voc2012-test',
        help='Directory for masks, overlays, and metric files.')
    parser.add_argument(
        '--device',
        default='auto',
        help='Use auto, cpu, cuda, or cuda:0. Default: auto.')
    parser.add_argument(
        '--opacity',
        default=0.45,
        type=float,
        help='Opacity of the predicted color mask in overlay images.')
    parser.add_argument(
        '--max-images',
        default=0,
        type=int,
        help='Optional limit for quick checks. Use 0 to test all images.')
    parser.add_argument(
        '--no-save-outputs',
        action='store_true',
        help='Only compute metrics and do not save predicted masks/overlays.')
    return parser.parse_args()


def resolve_device(device):
    if device == 'auto':
        return 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        return 'cuda:0'
    return device


def remap_key(key, model_keys):
    if key in model_keys:
        return key
    for old, new in KEY_REPLACEMENTS:
        if key.startswith(old):
            candidate = new + key[len(old):]
            if candidate in model_keys:
                return candidate
        if key.startswith(new):
            candidate = old + key[len(new):]
            if candidate in model_keys:
                return candidate
    return key


def build_model(config_path, checkpoint_path, device):
    cfg = Config.fromfile(str(config_path))
    cfg.model.backbone.init_cfg = None
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    init_default_scope(cfg.get('default_scope', 'mmseg'))

    model = MODELS.build(cfg.model)
    checkpoint = torch.load(str(checkpoint_path), map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    model_keys = set(model.state_dict().keys())
    state_dict = {remap_key(k, model_keys): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            'Checkpoint did not load cleanly. '
            f'missing={missing[:20]}, unexpected={unexpected[:20]}')

    dataset_meta = dict(checkpoint.get('meta', {}).get('dataset_meta', {}))
    dataset_meta['classes'] = ELMERF_CLASSES
    dataset_meta.setdefault('palette', ELMERF_PALETTE)
    model.dataset_meta = dataset_meta
    model.cfg = cfg
    model.to(device)
    model.eval()
    return model, checkpoint


def read_voc_items(voc_root, split):
    split_file = voc_root / 'ImageSets' / 'Segmentation' / f'{split}.txt'
    if not split_file.is_file():
        raise FileNotFoundError(f'Split file not found: {split_file}')

    items = []
    for line in split_file.read_text(encoding='utf-8').splitlines():
        image_id = line.strip()
        if not image_id:
            continue
        image_path = None
        for suffix in IMAGE_SUFFIXES:
            candidate = voc_root / 'JPEGImages' / f'{image_id}{suffix}'
            if candidate.is_file():
                image_path = candidate
                break
        if image_path is None:
            raise FileNotFoundError(f'Image not found for id: {image_id}')

        mask_path = voc_root / 'SegmentationClass' / f'{image_id}.png'
        if not mask_path.is_file():
            raise FileNotFoundError(f'Segmentation mask not found: {mask_path}')
        items.append((image_id, image_path, mask_path))
    return items


def update_confusion_matrix(hist, pred, label, num_classes, ignore_index=255):
    valid = label != ignore_index
    valid &= label >= 0
    valid &= label < num_classes
    valid &= pred >= 0
    valid &= pred < num_classes
    encoded = num_classes * label[valid].astype(np.int64) + pred[valid].astype(
        np.int64)
    hist += np.bincount(
        encoded, minlength=num_classes * num_classes).reshape(
        num_classes, num_classes)


def compute_metrics(hist, classes):
    diag = np.diag(hist).astype(np.float64)
    gt_sum = hist.sum(axis=1).astype(np.float64)
    pred_sum = hist.sum(axis=0).astype(np.float64)
    union = gt_sum + pred_sum - diag

    iou = np.divide(diag, union, out=np.full_like(diag, np.nan), where=union > 0)
    acc = np.divide(
        diag, gt_sum, out=np.full_like(diag, np.nan), where=gt_sum > 0)
    all_acc = float(diag.sum() / hist.sum()) if hist.sum() > 0 else float('nan')

    per_class = []
    for idx, class_name in enumerate(classes):
        per_class.append(
            dict(
                class_id=idx,
                class_name=class_name,
                iou=float(iou[idx]) if not np.isnan(iou[idx]) else None,
                accuracy=float(acc[idx]) if not np.isnan(acc[idx]) else None,
                gt_pixels=int(gt_sum[idx]),
                pred_pixels=int(pred_sum[idx]),
                intersect_pixels=int(diag[idx])))

    return dict(
        aAcc=all_acc,
        mIoU=float(np.nanmean(iou)),
        mAcc=float(np.nanmean(acc)),
        per_class=per_class,
        confusion_matrix=hist.astype(int).tolist())


def save_prediction_outputs(model, image_path, image_id, pred, out_dir, opacity):
    index_dir = out_dir / 'masks_index'
    color_dir = out_dir / 'masks_color'
    overlay_dir = out_dir / 'overlays'
    for directory in (index_dir, color_dir, overlay_dir):
        directory.mkdir(parents=True, exist_ok=True)

    palette = np.array(model.dataset_meta['palette'], dtype=np.uint8)
    color = palette[pred]
    Image.fromarray(pred.astype(np.uint8)).save(index_dir / f'{image_id}.png')
    Image.fromarray(color).save(color_dir / f'{image_id}.png')

    image = Image.open(image_path).convert('RGB')
    color_img = Image.fromarray(color).resize(image.size, Image.NEAREST)
    overlay = Image.blend(image, color_img, opacity)
    overlay.save(overlay_dir / f'{image_id}.jpg', quality=95)


def write_per_image_csv(rows, out_dir):
    if not rows:
        return
    csv_path = out_dir / 'per_image_metrics.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    start = time.time()
    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    voc_root = Path(args.voc_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested, but torch.cuda.is_available() is False.')

    items = read_voc_items(voc_root, args.split)
    if args.max_images > 0:
        items = items[:args.max_images]

    model, checkpoint = build_model(config_path, checkpoint_path, device)
    classes = tuple(model.dataset_meta['classes'])
    hist = np.zeros((len(classes), len(classes)), dtype=np.int64)
    rows = []

    print(f'device={device}', flush=True)
    if device.startswith('cuda'):
        print(f'cuda_device={torch.cuda.get_device_name(0)}', flush=True)
        print(f'torch_cuda={torch.version.cuda}', flush=True)
    print(f'torch={torch.__version__}', flush=True)
    print(f'checkpoint_iter={checkpoint.get("meta", {}).get("iter")}', flush=True)
    print(f'voc_root={voc_root}', flush=True)
    print(f'split={args.split}', flush=True)
    print(f'image_count={len(items)}', flush=True)
    print(f'classes={classes}', flush=True)

    for index, (image_id, image_path, mask_path) in enumerate(items, start=1):
        per_start = time.time()
        with torch.inference_mode():
            result = inference_model(model, str(image_path))
        pred = result.pred_sem_seg.data.squeeze(0).cpu().numpy().astype(np.uint8)
        label = np.array(Image.open(mask_path), dtype=np.uint8)
        if pred.shape != label.shape:
            pred = np.array(
                Image.fromarray(pred).resize(
                    (label.shape[1], label.shape[0]), Image.NEAREST),
                dtype=np.uint8)

        per_hist = np.zeros_like(hist)
        update_confusion_matrix(per_hist, pred, label, len(classes))
        hist += per_hist

        per_metric = compute_metrics(per_hist, classes)
        rows.append(
            dict(
                image_id=image_id,
                image=image_path.name,
                mask=mask_path.name,
                height=int(label.shape[0]),
                width=int(label.shape[1]),
                mIoU=per_metric['mIoU'],
                aAcc=per_metric['aAcc'],
                seconds=round(time.time() - per_start, 3)))

        if not args.no_save_outputs:
            save_prediction_outputs(model, image_path, image_id, pred, out_dir,
                                    args.opacity)

        print(
            f'[{index}/{len(items)}] {image_path.name} '
            f'mIoU={per_metric["mIoU"]:.4f} '
            f'aAcc={per_metric["aAcc"]:.4f} '
            f'seconds={time.time() - per_start:.2f}',
            flush=True)

    metrics = compute_metrics(hist, classes)
    metrics.update(
        dict(
            image_count=len(items),
            split=args.split,
            voc_root=str(voc_root),
            config=str(config_path),
            checkpoint=str(checkpoint_path),
            checkpoint_iter=checkpoint.get('meta', {}).get('iter'),
            classes=classes,
            device=device,
            torch=torch.__version__,
            torch_cuda=torch.version.cuda,
            seconds=round(time.time() - start, 2)))
    if device.startswith('cuda'):
        metrics['cuda_device'] = torch.cuda.get_device_name(0)
        metrics['max_cuda_memory_mb'] = round(
            torch.cuda.max_memory_allocated() / 1024 / 1024, 1)

    write_per_image_csv(rows, out_dir)
    metrics_path = out_dir / 'metrics.json'
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print(f'done seconds={metrics["seconds"]}', flush=True)
    print(f'mIoU={metrics["mIoU"]:.4f}', flush=True)
    print(f'mAcc={metrics["mAcc"]:.4f}', flush=True)
    print(f'aAcc={metrics["aAcc"]:.4f}', flush=True)
    print(f'metrics={metrics_path}', flush=True)


if __name__ == '__main__':
    main()