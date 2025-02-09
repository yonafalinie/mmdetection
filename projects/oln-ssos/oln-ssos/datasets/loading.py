import numpy as np

from mmdet.datasets.transforms import LoadAnnotations
from mmdet.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadAnnotationsWithAnnID(LoadAnnotations):
    def __init__(self, with_ann_id=True, with_pseudo_labels=True, **kwargs):
        super().__init__(**kwargs)
        self.with_ann_id = with_ann_id
        self.with_pseudo_labels = with_pseudo_labels

    def _load_ann_ids(self, results: dict) -> None:
        """Private function to load annotations id.

        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.

        Returns:
            dict: The dict contains loaded annotation id.
        """
        gt_ann_ids = []
        for instance in results.get('instances', []):
            gt_ann_ids.append(instance['ann_id'])
        # TODO: Inconsistent with mmcv, consider how to deal with it later.
        results['gt_ann_ids'] = np.array(
            gt_ann_ids, dtype=np.int64)

    def _load_bbox_pseudo_labels(self, results: dict) -> None:
        """Private function to load pseudo labels.

        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.

        Returns:
            dict: The dict contains loaded pseudo labels.
        """
        pseudo_labels = []
        for instance in results.get('instances', []):
            pseudo_labels.append(instance['pseudo_label'])
        results['pseudo_labels'] = np.array(
            pseudo_labels, dtype=np.int64)

    def transform(self, results: dict) -> dict:
        results = super().transform(results)
        if self.with_ann_id:
            self._load_ann_ids(results)
        if self.with_pseudo_labels:
            self._load_bbox_pseudo_labels(results)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(with_bbox={self.with_bbox}, '
        repr_str += f'with_label={self.with_label}, '
        repr_str += f'with_ann_ids={self.with_ann_id}, '
        repr_str += f'with_pseudo_labels={self.with_pseudo_labels}, '
        repr_str += f'with_mask={self.with_mask}, '
        repr_str += f'with_seg={self.with_seg}, '
        repr_str += f'poly2mask={self.poly2mask}, '
        repr_str += f"imdecode_backend='{self.imdecode_backend}', "
        repr_str += f'backend_args={self.backend_args})'
        return repr_str
