import torch
from typing import List, Tuple, Optional
from torch import Tensor

from mmdet.models import StandardRoIHead
from mmdet.models.task_modules import SamplingResult
from mmdet.models.utils import empty_instances
from mmdet.structures import SampleList
from mmdet.utils import InstanceList, ConfigType
from mmdet.structures.bbox import bbox2roi
from mmdet.registry import MODELS
from projects.OLN.oln import OLNRoIHead


@MODELS.register_module()
class MaskScoringOLNRoIHead(OLNRoIHead):

    def __init__(self, mask_iou_head, **kwargs):
        assert mask_iou_head is not None
        super(MaskScoringOLNRoIHead, self).__init__(**kwargs)
        self.mask_iou_head = MODELS.build(mask_iou_head)

    def _mask_forward(self,
                      x: Tuple[Tensor],
                      rois: Tensor = None,
                      pos_inds: Optional[Tensor] = None,
                      bbox_feats: Optional[Tensor] = None) -> dict:
        """Mask head forward function used in both training and testing.

        Args:
            x (tuple[Tensor]): Tuple of multi-level img features.
            rois (Tensor): RoIs with the shape (n, 5) where the first
                column indicates batch id of each RoI.
            pos_inds (Tensor, optional): Indices of positive samples.
                Defaults to None.
            bbox_feats (Tensor): Extract bbox RoI features. Defaults to None.

        Returns:
            dict[str, Tensor]: Usually returns a dictionary with keys:

                - `mask_preds` (Tensor): Mask prediction.
                - `mask_feats` (Tensor): Extract mask RoI features.
        """
        assert ((rois is not None) ^
                (pos_inds is not None and bbox_feats is not None))
        if rois is not None:
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
        else:
            assert bbox_feats is not None
            mask_feats = bbox_feats[pos_inds]

        mask_preds, mask_fcn_out = self.mask_head(mask_feats)
        mask_results = dict(mask_preds=mask_preds, mask_feats=mask_feats, mask_fcn_out=mask_fcn_out)
        return mask_results

    def mask_loss(self, x: Tuple[Tensor],
                  sampling_results: List[SamplingResult], bbox_feats: Tensor,
                  batch_gt_instances: InstanceList) -> dict:
        if not self.share_roi_extractor:
            pos_rois = bbox2roi([res.pos_priors for res in sampling_results])
            mask_results = self._mask_forward(x, pos_rois)
        else:
            pos_inds = []
            device = bbox_feats.device
            for res in sampling_results:
                pos_inds.append(
                    torch.ones(
                        res.pos_priors.shape[0],
                        device=device,
                        dtype=torch.uint8))
                pos_inds.append(
                    torch.zeros(
                        res.neg_priors.shape[0],
                        device=device,
                        dtype=torch.uint8))
            pos_inds = torch.cat(pos_inds)

            mask_results = self._mask_forward(
                x, pos_inds=pos_inds, bbox_feats=bbox_feats)

        mask_loss_and_target = self.mask_head.loss_and_target(
            mask_preds=mask_results['mask_preds'],
            sampling_results=sampling_results,
            batch_gt_instances=batch_gt_instances,
            rcnn_train_cfg=self.train_cfg)

        mask_targets = mask_loss_and_target['mask_targets']
        mask_results.update(loss_mask=mask_loss_and_target['loss_mask'])
        if mask_results['loss_mask'] is None:
            return mask_results

        # mask iou head forward and loss
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results]) * 0
        pos_mask_pred = mask_results['mask_preds'][
            range(mask_results['mask_preds'].size(0)), pos_labels]
        mask_iou_pred = self.mask_iou_head(mask_results['mask_fcn_out'],
                                           pos_mask_pred)
        pos_mask_iou_pred = mask_iou_pred[range(mask_iou_pred.size(0)), pos_labels]

        loss_mask_iou = self.mask_iou_head.loss_and_target(pos_mask_iou_pred, pos_mask_pred, mask_targets,
                                                           sampling_results, batch_gt_instances, self.train_cfg)
        mask_results['loss_mask'].update(loss_mask_iou)
        return mask_results

    def predict_mask(self,
                     x: Tuple[Tensor],
                     batch_img_metas: List[dict],
                     results_list: InstanceList,
                     rescale: bool = False) -> InstanceList:
        """Perform forward propagation of the mask head and predict detection
        results on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            batch_img_metas (list[dict]): List of image information.
            results_list (list[:obj:`InstanceData`]): Detection results of
                each image.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.

        Returns:
            list[:obj:`InstanceData`]: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).
        """
        # don't need to consider aug_test.
        bboxes = [res.bboxes for res in results_list]
        mask_rois = bbox2roi(bboxes)
        if mask_rois.shape[0] == 0:
            results_list = empty_instances(
                batch_img_metas,
                mask_rois.device,
                task_type='mask',
                instance_results=results_list,
                mask_thr_binary=self.test_cfg.mask_thr_binary)
            return results_list

        mask_results = self._mask_forward(x, mask_rois)
        mask_preds = mask_results['mask_preds']
        mask_feats = mask_results['mask_fcn_out']

        # get mask scores with mask iou head
        labels = torch.cat([res.labels for res in results_list])
        mask_iou_preds = self.mask_iou_head(
            mask_feats, mask_preds[range(labels.size(0)), labels])
        # split batch mask prediction back to each image
        num_mask_rois_per_img = [len(res) for res in results_list]
        mask_preds = mask_preds.split(num_mask_rois_per_img, 0)
        mask_iou_preds = mask_iou_preds.split(num_mask_rois_per_img, 0)

        # TODO: Handle the case where rescale is false
        results_list = self.mask_head.predict_by_feat(
            mask_preds=mask_preds,
            results_list=results_list,
            batch_img_metas=batch_img_metas,
            rcnn_test_cfg=self.test_cfg,
            rescale=rescale)
        results_list = self.mask_iou_head.predict_by_feat(
            mask_iou_preds=mask_iou_preds, results_list=results_list)
        return results_list
