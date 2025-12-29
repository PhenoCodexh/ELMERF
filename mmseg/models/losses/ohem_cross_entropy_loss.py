# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Union

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmseg.registry import MODELS


@MODELS.register_module()
class OhemCrossEntropy(nn.Module):
    """OhemCrossEntropy loss.

    This func is modified from
    `PIDNet <https://github.com/XuJiacong/PIDNet/blob/main/utils/criterion.py#L43>`_.  # noqa

    Licensed under the MIT License.

    Args:
        ignore_label (int): Labels to ignore when computing the loss.
            Default: 255
        thresh (float, optional): The threshold for hard example selection.
            Below which, are prediction with low confidence. If not
            specified, the hard examples will be pixels of top ``min_kept``
            loss. Default: 0.7.
        min_kept (int, optional): The minimum number of predictions to keep.
            Default: 100000.
        loss_weight (float): Weight of the loss. Defaults to 1.0.
        class_weight (list[float] | str, optional): Weight of each class. If in
            str format, read them from a file. Defaults to None.
        loss_name (str): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_boundary'.
    """

    def __init__(self,
                 ignore_label: int = 255,
                 thres: float = 0.7,
                 min_kept: int = 100000,
                 loss_weight: float = 1.0,
                 class_weight: Optional[Union[List[float], str]] = None,
                 loss_name: str = 'loss_ohem'):
        super().__init__()
        self.thresh = thres
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label
        self.loss_weight = loss_weight
        self.loss_name_ = loss_name
        self.class_weight = class_weight

    def forward(self,
                score: Tensor,
                target: Tensor,
                weight: Optional[Tensor] = None,        # ← 新增
                ignore_index: Optional[int] = None,     # ← 新增
                reduction_override: Optional[str] = None,
                **kwargs                                # ← 兜底
                ) -> Tensor:
        # ① 兼容 ignore_index
        if ignore_index is None:
            ignore_index = self.ignore_label

        # ② 类别权重（不变）
        class_weight = (score.new_tensor(self.class_weight)
                        if self.class_weight is not None else None)

        # ③ 计算逐像素 CE（保持 reduction='none'）
        pixel_losses = F.cross_entropy(
            score,
            target,
            weight=class_weight,
            ignore_index=ignore_index,
            reduction='none').view(-1)

        # ④ 若有像素权重，乘进去
        if weight is not None:
            weight = weight.view(-1)
            pixel_losses = pixel_losses * weight

        # ⑤ OHEM 选 hard pixels（同你原来的代码）
        mask = target.view(-1) != ignore_index
        tmp_target = target.clone()
        tmp_target[tmp_target == ignore_index] = 0
        pred = F.softmax(score, dim=1).gather(1, tmp_target.unsqueeze(1))
        pred, ind = pred.view(-1)[mask].sort()
        if pred.numel() == 0:
            return score.new_tensor(0.0)
        threshold = max(pred[min(self.min_kept, pred.numel() - 1)], self.thresh)
        pixel_losses = pixel_losses[mask][ind]
        pixel_losses = pixel_losses[pred < threshold]

        return self.loss_weight * pixel_losses.mean()

    @property
    def loss_name(self):
        return self.loss_name_