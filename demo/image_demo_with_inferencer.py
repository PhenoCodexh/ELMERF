# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
from mmseg.apis import MMSegInferencer

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file OR dir OR wildcard')
    parser.add_argument('config', help='Config file path')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint file')
    parser.add_argument('--out-dir', default='vis', help='Dir to save visualized results')
    parser.add_argument('--pred-out-dir', default='', help='Dir to save raw pred index maps')
    parser.add_argument('--show', action='store_true', help='Show window')
    parser.add_argument('--dataset-name', default='voc', help='palette name, e.g. voc/ade20k/cityscapes')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--opacity', type=float, default=0.5)
    parser.add_argument('--with-labels', action='store_true')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    inferencer = MMSegInferencer(
        args.config,
        args.checkpoint,
        dataset_name=args.dataset_name,
        device=args.device
    )

    # 直接传入 img，可以是文件/目录/list/通配符
    inferencer(
        args.img,
        show=args.show,
        out_dir=args.out_dir,
        pred_out_dir=args.pred_out_dir if args.pred_out_dir else None,
        opacity=args.opacity,
        with_labels=args.with_labels,
        # 如果你还想拿到 DataSample 在代码里处理，就加 return_datasamples=True
    )

if __name__ == '__main__':
    main()
