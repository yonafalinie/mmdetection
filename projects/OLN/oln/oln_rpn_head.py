"""This file contains code to build OLN-RPN.

Reference:
    "Learning Open-World Object Proposals without Learning to Classify",
        Aug 2021. https://arxiv.org/abs/2108.06753
        Dahun Kim, Tsung-Yi Lin, Anelia Angelova, In So Kweon and Weicheng Kuo
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine import ConfigDict
from mmengine.model.weight_init import normal_init
from typing import List, Optional, Tuple, Union
from torch import Tensor
from mmdet.utils import InstanceList, MultiConfig, OptInstanceList
from mmengine.structures import InstanceData

from mmcv.ops import batched_nms

from mmdet.models.task_modules import (
    anchor_inside_flags, build_anchor_generator, build_assigner,
    build_bbox_coder, build_sampler
)
from mmdet.models.utils import images_to_levels, multi_apply, unmap, select_single_mlvl, filter_scores_and_topk

from mmdet.structures.bbox import bbox_overlaps, BaseBoxes, get_box_tensor, cat_boxes

from mmdet.registry import MODELS
from mmdet.models.dense_heads.rpn_head import RPNHead


@MODELS.register_module()
class OLNRPNHead(RPNHead):
    """OLN-RPN head.
    
    Learning localization instead of classification at the proposal stage is
    crucial as it avoids overfitting to the foreground by classification. For
    training the localization quality estimation branch, we randomly sample
    `num` anchors having an IoU larger than `neg_iou_thr` with the matched
    ground-truth boxes. It is recommended to use 'centerness' in this stage. For
    box regression, we replace the standard box-delta targets (xyhw) with
    distances from the location to four sides of the ground-truth box (lrtb). We
    choose to use one anchor per feature location as opposed to 3 in the standard
    RPN, because we observe its better generalization as each anchor can ingest
    more data.
    """

    def __init__(self, loss_objectness, objectness_type='Centerness', **kwargs):
        # print("Keys in kwargs:", kwargs.keys())
        super(OLNRPNHead, self).__init__(**kwargs)
        # Objectness loss
        self.loss_objectness = MODELS.build(loss_objectness)
        self.objectness_type = objectness_type
        self.with_class_score = self.loss_cls.loss_weight > 0.0
        self.with_objectness_score = self.loss_objectness.loss_weight > 0.0

        # Define objectness assigner and sampler
        if self.train_cfg:
            self.objectness_assigner = build_assigner(
                self.train_cfg.objectness_assigner)
        # Define objectness_sampler_cfg based on the training configuration
        if getattr(self, 'sampling', True) and hasattr(self.train_cfg, 'objectness_sampler'):
            objectness_sampler_cfg = self.train_cfg['objectness_sampler']
        else:
            objectness_sampler_cfg = dict(type='PseudoSampler')
        self.objectness_sampler = build_sampler(
                objectness_sampler_cfg, context=self)

    def _init_layers(self):
        """Initialize layers of the head."""
        super()._init_layers()
        self.rpn_obj = nn.Conv2d(self.feat_channels, self.num_anchors, 1)

    def init_weights(self):
        """Initialize weights of the head."""
        normal_init(self.rpn_conv, std=0.01)
        normal_init(self.rpn_cls, std=0.01)
        normal_init(self.rpn_reg, std=0.01)
        normal_init(self.rpn_obj, std=0.01)

    def forward_single(self, x):
        """Forward feature map of a single scale level."""
        x = self.rpn_conv(x)
        x = F.relu(x)
        # We add L2 normalization for training stability
        x = F.normalize(x, p=2, dim=1)
        rpn_cls_score = self.rpn_cls(x)
        rpn_bbox_pred = self.rpn_reg(x)
        rpn_objectness_pred = self.rpn_obj(x)
        return rpn_cls_score, rpn_bbox_pred, rpn_objectness_pred

    def loss_by_feat_single(self, cls_score, bbox_pred, objectness_score, anchors,
                    labels, label_weights, bbox_targets, bbox_weights, 
                    objectness_targets, objectness_weights, avg_factor):
        
        """Compute loss of a single scale level.

        Args:
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            objectness_score (Tensor): Box objectness scorees for each anchor
                point has shape (N, num_anchors, H, W) 
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor with
                shape (N, num_total_anchors, 4).
            bbox_weights (Tensor): BBox regression loss weights of each anchor
                with shape (N, num_total_anchors, 4).
            objectness_targets (Tensor): Center regresion targets of each anchor
                with shape (N, num_total_anchors)
            objectness_weights (Tensor): Objectness weights of each anchro with 
                shape (N, num_total_anchors)
            num_total_samples (int): If sampling, num total samples equal to
                the number of total anchors; Otherwise, it is the number of
                positive anchors.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # classification loss
        loss_cls, loss_bbox = super().loss_by_feat_single(
            cls_score, bbox_pred, anchors, labels, label_weights, bbox_targets, bbox_weights, avg_factor)

        # objectness loss
        objectness_targets = objectness_targets.reshape(-1)
        objectness_weights = objectness_weights.reshape(-1)
        assert self.cls_out_channels == 1, (
            'cls_out_channels must be 1 for objectness learning.')
        objectness_score = objectness_score.permute(0, 2, 3, 1).reshape(-1)

        loss_objectness = self.loss_objectness(
            objectness_score.sigmoid(), 
            objectness_targets, 
            objectness_weights, 
            avg_factor=avg_factor)
        return loss_cls, loss_bbox, loss_objectness

    def loss_by_feat(self,
                     cls_scores: List[Tensor],
                     bbox_preds: List[Tensor],
                     objectness_scores: List[Tensor],
                     batch_gt_instances: InstanceList,
                     batch_img_metas: List[dict],
                     batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                has shape (N, num_anchors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            objectness_scores (list[Tensor]): Box objectness scores for each
                anchor point with shape (N, num_anchors, H, W).
            batch_gt_instances (list[obj:InstanceData]): Batch of gt_instance.
                It usually includes ``bboxes`` and ``labels`` attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[obj:InstanceData], Optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, batch_img_metas, device=device)

        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         avg_factor, objectness_targets_list,
         objectness_weights_list) = cls_reg_targets

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i in range(len(anchor_list)):
            concat_anchor_list.append(cat_boxes(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)

        losses_cls, losses_bbox, losses_objectness = multi_apply(
            self.loss_by_feat_single,
            cls_scores,
            bbox_preds,
            objectness_scores,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            objectness_targets_list,
            objectness_weights_list,
            avg_factor=avg_factor)

        return dict(
            loss_rpn_cls=losses_cls, 
            loss_rpn_bbox=losses_bbox,
            loss_rpn_obj=losses_objectness)

    def _get_targets_single(self,
                            flat_anchors: Union[Tensor, BaseBoxes],
                            valid_flags: Tensor,
                            gt_instances: InstanceData,
                            img_meta,
                            gt_instances_ignore: Optional[InstanceData] = None,
                            unmap_outputs=True):
        """Compute regression and classification targets for anchors in a
        single image.

        Args:
            flat_anchors (Tensor): Multi-level anchors of the image, which are
                concatenated into a single tensor of shape (num_anchors ,4)
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors,).
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            img_meta (dict): Meta info of the image.
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            img_meta (dict): Meta info of the image.
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple:
                labels_list (list[Tensor]): Labels of each level
                label_weights_list (list[Tensor]): Label weights of each level
                bbox_targets_list (list[Tensor]): BBox targets of each level
                bbox_weights_list (list[Tensor]): BBox weights of each level
                num_total_pos (int): Number of positive samples in all images
                num_total_neg (int): Number of negative samples in all images
        """
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.train_cfg['allowed_border'])
        if not inside_flags.any():
            raise ValueError(
                'There is no valid anchor inside the image boundary. Please '
                'check the image size and anchor sizes, or set '
                '``allowed_border`` to -1 to skip the condition.')
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags]

        pred_instances = InstanceData(priors=anchors)

        assign_result = self.assigner.assign(pred_instances, gt_instances,
                                             gt_instances_ignore)

        # assign_result = self.assigner.assign(
        #     anchors, gt_bboxes,
        #     None if getattr(self, 'sampling', True) else gt_labels)
        sampling_result = self.sampler.sample(assign_result, pred_instances,
                                              gt_instances)

        # Assign objectness gt and sample anchors
        objectness_assign_result = self.objectness_assigner.assign(
            pred_instances, gt_instances, gt_instances_ignore)
        objectness_sampling_result = self.objectness_sampler.sample(
            objectness_assign_result, pred_instances, gt_instances)

        num_valid_anchors = anchors.shape[0]
        target_dim = gt_instances.bboxes.size(-1) if self.reg_decoded_bbox \
            else self.bbox_coder.encode_size
        bbox_targets = anchors.new_zeros(num_valid_anchors, target_dim)
        bbox_weights = anchors.new_zeros(num_valid_anchors, target_dim)

        labels = anchors.new_full((num_valid_anchors, ),
                                  self.num_classes,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    sampling_result.pos_priors, sampling_result.pos_gt_bboxes)
            else:
                pos_bbox_targets = sampling_result.pos_gt_bboxes
                pos_bbox_targets = get_box_tensor(pos_bbox_targets)

            # Sanlity check: left, right, top, bottom distances must be greater
            # than 0.
            # valid_targets = torch.min(pos_bbox_targets,-1)[0] > 0
            # bbox_targets[pos_inds[valid_targets], :] = (
            #     pos_bbox_targets[valid_targets])
            # bbox_weights[pos_inds[valid_targets], :] = 1.0
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0

            labels[pos_inds] = sampling_result.pos_gt_labels
            if self.train_cfg['pos_weight'] <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg['pos_weight']
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        objectness_targets = anchors.new_zeros(num_valid_anchors, dtype=torch.float)
        objectness_weights = anchors.new_zeros(
            num_valid_anchors, dtype=torch.float)
        objectness_pos_inds = objectness_sampling_result.pos_inds
        objectness_neg_inds = objectness_sampling_result.neg_inds
        objectness_pos_neg_inds = torch.cat([objectness_pos_inds, objectness_neg_inds])

        if len(objectness_pos_inds) > 0:
            # Centerness as tartet -- Default
            if self.objectness_type == 'Centerness':
                pos_objectness_bbox_targets = self.bbox_coder.encode(
                    objectness_sampling_result.pos_bboxes, 
                    objectness_sampling_result.pos_gt_bboxes)
                valid_targets = torch.min(pos_objectness_bbox_targets, -1)[0] > 0
                pos_objectness_bbox_targets[valid_targets == False] = 0
                top_bottom = pos_objectness_bbox_targets[:, :2]
                left_right = pos_objectness_bbox_targets[:, 2:4]
                pos_objectness_targets = torch.sqrt(
                    (torch.min(top_bottom, -1)[0] / 
                        (torch.max(top_bottom, -1)[0] + 1e-12)) *
                    (torch.min(left_right, -1)[0] / 
                        (torch.max(left_right, -1)[0] + 1e-12)))
            elif self.objectness_type == 'BoxIoU':
                pos_objectness_targets = bbox_overlaps(
                    objectness_sampling_result.pos_bboxes,
                    objectness_sampling_result.pos_gt_bboxes,
                    is_aligned=True)
            else:
                raise ValueError(
                    'objectness_type must be either "Centerness" (Default) or '
                    '"BoxIoU".')

            objectness_targets[objectness_pos_inds] = pos_objectness_targets
            objectness_weights[objectness_pos_inds] = 1.0   

        if len(objectness_neg_inds) > 0: 
            objectness_targets[objectness_neg_inds] = 0.0
            objectness_weights[objectness_neg_inds] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            labels = unmap(
                labels, num_total_anchors, inside_flags,
                fill=self.num_classes)  # fill bg label
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

            # objectness targets
            objectness_targets = unmap(
                objectness_targets, num_total_anchors, inside_flags)
            objectness_weights = unmap(
                objectness_weights, num_total_anchors, inside_flags)

        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds, sampling_result,
                objectness_targets, objectness_weights, 
                objectness_pos_inds, objectness_neg_inds, objectness_pos_neg_inds,
                objectness_sampling_result)

    def get_targets(self,
                    anchor_list: List[List[Tensor]],
                    valid_flag_list: List[List[Tensor]],
                    batch_gt_instances: InstanceList,
                    batch_img_metas: List[dict],
                    batch_gt_instances_ignore: OptInstanceList = None,
                    unmap_outputs: bool = True,
                    return_sampling_results: bool = False):
        """Compute regression and classification targets for anchors in
        multiple images.

        Args:
            anchor_list (list[list[Tensor]]): Multi level anchors of each
                image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, 4).
            valid_flag_list (list[list[Tensor]]): Multi level valid flags of
                each image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, )
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors. Defaults to True.
            return_sampling_results (bool): Whether to return the sampling
                results. Defaults to False.

        Returns:
            tuple: Usually returns a tuple containing learning targets.

                - labels_list (list[Tensor]): Labels of each level.
                - label_weights_list (list[Tensor]): Label weights of each
                  level.
                - bbox_targets_list (list[Tensor]): BBox targets of each level.
                - bbox_weights_list (list[Tensor]): BBox weights of each level.
                - avg_factor (int): Average factor that is used to average
                  the loss. When using sampling method, avg_factor is usually
                  the sum of positive and negative priors. When using
                  `PseudoSampler`, `avg_factor` is usually equal to the number
                  of positive priors.

            additional_returns: This function enables user-defined returns from
                `self._get_targets_single`. These returns are currently refined
                to properties at each feature map (i.e. having HxW dimension).
                The results will be concatenated after the end
        """
        num_imgs = len(batch_img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        if batch_gt_instances_ignore is None:
            batch_gt_instances_ignore = [None] * num_imgs

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors to a single tensor
        concat_anchor_list = []
        concat_valid_flag_list = []
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            concat_anchor_list.append(torch.cat(anchor_list[i]))
            concat_valid_flag_list.append(torch.cat(valid_flag_list[i]))

        results = multi_apply(
            self._get_targets_single,
            concat_anchor_list,
            concat_valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore,
            unmap_outputs=unmap_outputs)

        (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
         pos_inds_list, neg_inds_list, sampling_results_list,
         all_objectness_targets, all_objectness_weights,
         objectness_pos_inds_list, objectness_neg_inds_list,
         objectness_pos_neg_inds_list, objectness_sampling_results_list
         ) = results[:13]

        rest_results = list(results[13:])  # user-added return values
        # no valid anchors
        avg_factor = sum(
            [results.avg_factor for results in sampling_results_list])
        # update `_raw_positive_infos`, which will be used when calling
        # `get_positive_infos`.
        self._raw_positive_infos.update(sampling_results=sampling_results_list)
        # split targets to a list w.r.t. multiple levels
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights,
                                             num_level_anchors)
        objectness_targets_list = images_to_levels(all_objectness_targets,
                                               num_level_anchors)
        objectness_weights_list = images_to_levels(all_objectness_weights,
                                               num_level_anchors)

        res = (labels_list, label_weights_list, bbox_targets_list,
               bbox_weights_list, avg_factor,
               objectness_targets_list, objectness_weights_list,)

        if return_sampling_results:
            res = res + (sampling_results_list, )
        for i, r in enumerate(rest_results):  # user-added return values
            rest_results[i] = images_to_levels(r, num_level_anchors)

        return res + tuple(rest_results)
    
    def predict_by_feat(self,
                        cls_scores: List[Tensor],
                        bbox_preds: List[Tensor],
                        objectness_scores: List[Tensor],
                        score_factors: Optional[List[Tensor]] = None,
                        batch_img_metas: Optional[List[dict]] = None,
                        cfg: Optional[ConfigDict] = None,
                        rescale: bool = False,
                        with_nms: bool = True):

        assert len(cls_scores) == len(bbox_preds) and (
            len(cls_scores) == len(objectness_scores))
        if score_factors is None:
            # e.g. Retina, FreeAnchor, Foveabox, etc.
            with_score_factors = False
        else:
            # e.g. FCOS, PAA, ATSS, AutoAssign, etc.
            with_score_factors = True
            assert len(cls_scores) == len(score_factors)

        num_levels = len(cls_scores)

        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_priors = self.anchor_generator.grid_priors(
            featmap_sizes, dtype=cls_scores[0].dtype,
            device=cls_scores[0].device)

        result_list = []
        for img_id in range(len(batch_img_metas)):
            img_meta = batch_img_metas[img_id]
            cls_score_list = select_single_mlvl(
                cls_scores, img_id, detach=True)
            bbox_pred_list = select_single_mlvl(
                bbox_preds, img_id, detach=True)
            objectness_score_list = select_single_mlvl(
                objectness_scores, img_id, detach=True)
            if with_score_factors:
                score_factor_list = select_single_mlvl(
                    score_factors, img_id, detach=True)
            else:
                score_factor_list = [None for _ in range(num_levels)]
            results = self._predict_by_feat_single(
                cls_score_list=cls_score_list,
                bbox_pred_list=bbox_pred_list,
                objectness_score_list=objectness_score_list,
                score_factor_list=score_factor_list,
                mlvl_priors=mlvl_priors,
                img_meta=img_meta,
                cfg=cfg,
                rescale=rescale,
                with_nms=with_nms)
            result_list.append(results)
        return result_list

    def _predict_by_feat_single(self,
                                cls_score_list: List[Tensor],
                                bbox_pred_list: List[Tensor],
                                objectness_score_list: List[Tensor],
                                score_factor_list: List[Tensor],
                                mlvl_priors: List[Tensor],
                                img_meta: dict,
                                cfg: ConfigDict,
                                rescale: bool = False,
                                with_nms: bool = True) -> InstanceData:
        """Transform a single image's features extracted from the head into
        bbox results.

        Args:
            cls_score_list (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_priors * num_classes, H, W).
            bbox_pred_list (list[Tensor]): Box energies / deltas from
                all scale levels of a single image, each item has shape
                (num_priors * 4, H, W).
            score_factor_list (list[Tensor]): Score factor from all scale
                levels of a single image, each item has shape
                (num_priors * 1, H, W).
            mlvl_priors (list[Tensor]): Each element in the list is
                the priors of a single level in feature pyramid. In all
                anchor-based methods, it has shape (num_priors, 4). In
                all anchor-free methods, it has shape (num_priors, 2)
                when `with_stride=True`, otherwise it still has shape
                (num_priors, 4).
            img_meta (dict): Image meta info.
            cfg (mmengine.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            with_nms (bool): If True, do nms before return boxes.
                Defaults to True.

        Returns:
            :obj:`InstanceData`: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """

        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)
        img_shape = img_meta['img_shape']
        nms_pre = cfg.get('nms_pre', -1)

        mlvl_bbox_preds = []
        mlvl_valid_priors = []
        mlvl_scores = []
        level_ids = []
        for level_idx, (cls_score, bbox_pred, objectness_score, score_factor, priors) in \
                enumerate(zip(cls_score_list, bbox_pred_list, objectness_score_list,
                              score_factor_list, mlvl_priors)):

            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

            reg_dim = self.bbox_coder.encode_size
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, reg_dim)
            objectness_score = objectness_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)

            # the `custom_cls_channels` parameter is derived from
            # CrossEntropyCustomLoss and FocalCustomLoss, and is currently used
            # in v3det.
            # if getattr(self.loss_cls, 'custom_cls_channels', False):
            #     scores = self.loss_cls.get_activation(objectness_score)
            if self.use_sigmoid_cls:
                scores = objectness_score.sigmoid()
            else:
                # remind that we set FG labels to [0, num_class-1]
                # since mmdet v2.0
                # BG cat_id: num_class
                scores = objectness_score.softmax(-1)[:, :-1]

            scores = torch.squeeze(scores)
            if 0 < nms_pre < scores.shape[0]:
                # sort is faster than topk
                # _, topk_inds = scores.topk(cfg.nms_pre)
                ranked_scores, rank_inds = scores.sort(descending=True)
                topk_inds = rank_inds[:nms_pre]
                scores = ranked_scores[:nms_pre]
                bbox_pred = bbox_pred[topk_inds, :]
                priors = priors[topk_inds]

            mlvl_bbox_preds.append(bbox_pred)
            mlvl_valid_priors.append(priors)
            mlvl_scores.append(scores)

            # use level id to implement the separate level nms
            level_ids.append(
                scores.new_full((scores.size(0),),
                                level_idx,
                                dtype=torch.long))

        bbox_pred = torch.cat(mlvl_bbox_preds)
        priors = cat_boxes(mlvl_valid_priors)
        bboxes = self.bbox_coder.decode(priors, bbox_pred, max_shape=img_shape)

        results = InstanceData()
        results.bboxes = bboxes
        results.scores = torch.cat(mlvl_scores)
        results.level_ids = torch.cat(level_ids)

        return self._bbox_post_process(
            results=results,
            cfg=cfg,
            rescale=rescale,
            with_nms=with_nms,
            img_meta=img_meta)
