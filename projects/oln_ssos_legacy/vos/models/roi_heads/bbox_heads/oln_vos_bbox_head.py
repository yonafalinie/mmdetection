import torch

# from mmdet.core import multi_apply, multiclass_nms
from mmdet.models.utils import multi_apply
from mmdet.models import multiclass_nms
from mmdet.registry import MODELS
from mmdet.models.roi_heads.bbox_heads import Shared2FCBBoxScoreHead
from vos.models.roi_heads.bbox_heads.vos_convfc_bbox_head import multiclass_nms_with_ood


@MODELS.register_module()
class VOSShared2FCBBoxScoreHead(Shared2FCBBoxScoreHead):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ood_score_threshold = 0.

    def forward(self, x):
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_reg = x
        x_bbox_score = x

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        bbox_score = (self.fc_bbox_score(x_bbox_score)
                      if self.with_bbox_score else None)

        return cls_score, bbox_pred, bbox_score, x

    def get_bboxes(self,
                   rois,
                   cls_score,
                   bbox_pred,
                   ood_scores,
                   bbox_score,
                   rpn_score,
                   img_shape,
                   scale_factor,
                   rescale=False,
                   cfg=None,
                   with_ood=True):
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        # cls_score is not used.
        # scores = F.softmax(
        #     cls_score, dim=1) if cls_score is not None else None

        if bbox_pred is not None:
            bboxes = self.bbox_coder.decode(
                rois[:, 1:], bbox_pred, max_shape=img_shape)
        else:
            bboxes = rois[:, 1:].clone()
            if img_shape is not None:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1])
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0])

        if rescale and bboxes.size(0) > 0:
            if isinstance(scale_factor, float):
                bboxes /= scale_factor
            else:
                scale_factor = bboxes.new_tensor(scale_factor)
                bboxes = (bboxes.view(bboxes.size(0), -1, 4) /
                          scale_factor).view(bboxes.size()[0], -1)

        # The objectness score of a region is computed as a geometric mean of
        # the estimated localization quality scores of OLN-RPN and OLN-Box
        # heads.
        scores = torch.sqrt(rpn_score * bbox_score.sigmoid())

        # new_scores = torch.zeros_like(cls_score).view(-1)
        # cls_labels = cls_score.max(dim=1)[1]
        # cls_labels_1d = cls_labels + torch.arange(cls_labels.shape[0]).to(scores.device) * cls_score.shape[1]
        # new_scores[cls_labels_1d] = scores.flatten()
        # scores = new_scores.view(scores.shape[0], -1)
        # Concat dummy zero-scores for the background class.
        scores = torch.cat([scores, torch.zeros_like(scores)], dim=-1)

        if with_ood:
            if cfg is None:
                return bboxes, scores, ood_scores
            else:
                det_bboxes, det_labels, det_ood_scores, _ = multiclass_nms_with_ood(bboxes, scores,
                                                                                    ood_scores,
                                                                                    None,
                                                                                    cfg.score_thr, cfg.nms,
                                                                                    cfg.max_per_img,
                                                                                    ood_score_threshold=cfg.anomaly_threshold)

                return det_bboxes, det_labels, det_ood_scores
        else:
            if cfg is None:
                return bboxes, scores, None
            else:
                det_bboxes, det_labels = multiclass_nms(bboxes, scores,
                                                              cfg.score_thr, cfg.nms,
                                                              cfg.max_per_img)

                return det_bboxes, det_labels, None
