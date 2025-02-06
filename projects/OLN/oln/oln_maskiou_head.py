import torch
from mmcv.cnn import Conv2d
from torch import nn, Tensor
from typing import Tuple

from mmdet.models import MaskIoUHead
from mmdet.registry import MODELS
from mmdet.utils import InstanceList


@MODELS.register_module()
class OLNMaskIoUHead(MaskIoUHead):
    def __init__(self, num_convs: int = 1, *args, **kwargs):
        super(OLNMaskIoUHead, self).__init__(num_convs=num_convs, *args, **kwargs)
        self.convs = nn.ModuleList()
        for i in range(num_convs):
            in_channels = self.conv_out_channels
            self.convs.append(
                Conv2d(
                    in_channels,
                    self.conv_out_channels,
                    3,
                    stride=1,
                    padding=1))

    def forward(self, mask_feat: Tensor, mask_preds: Tensor) -> Tensor:
        """Forward function.

        Args:
            mask_feat (Tensor): Mask features from upstream models.
            mask_preds (Tensor): Mask predictions from mask head.

        Returns:
            Tensor: Mask IoU predictions.
        """
        x = mask_feat
        for conv in self.convs:
            x = self.relu(conv(x))
        x = self.max_pool(x)
        x = x.flatten(1)
        for fc in self.fcs:
            x = self.relu(fc(x))
        mask_iou = self.fc_mask_iou(x)
        return mask_iou

    def predict_by_feat(self, mask_iou_preds: Tuple[Tensor],
                        results_list: InstanceList) -> InstanceList:
        """Predict the mask iou and calculate it into ``results.scores``.

        Args:
            mask_iou_preds (Tensor): Mask IoU predictions results, has shape
                (num_proposals, num_classes)
            results_list (list[:obj:`InstanceData`]): Detection results of
                each image.

        Returns:
            list[:obj:`InstanceData`]: Detection results of each image
            after the post process. Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).
        """
        assert len(mask_iou_preds) == len(results_list)
        for results, mask_iou_pred in zip(results_list, mask_iou_preds):
            labels = results.labels
            scores = results.scores
            results.scores = torch.pow((scores ** 2) * mask_iou_pred[range(labels.size(0)),
            labels], 1/3)
        return results_list
