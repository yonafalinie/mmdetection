from mmdet.datasets.transforms import PackDetInputs
from mmdet.registry import TRANSFORMS


@TRANSFORMS.register_module()
class PackPseudoLabelDetInputs(PackDetInputs):
    mapping_table = {
        'gt_bboxes': 'bboxes',
        'gt_bboxes_labels': 'labels',
        'gt_masks': 'masks',
        'gt_ann_ids': 'ann_ids',
        'pseudo_labels': 'pseudo_labels'
    }
