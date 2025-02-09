from typing import Optional, Sequence

import numpy as np
from mmcv.transforms import to_tensor
from mmengine.structures import InstanceData, PixelData
from mmdet.datasets.transforms import PackDetInputs
from mmengine.logging import MMLogger
from mmdet.registry import TRANSFORMS
from mmdet.structures import DetDataSample, ReIDDataSample, TrackDataSample
from mmdet.structures.bbox import BaseBoxes
from mmengine.logging import MMLogger


@TRANSFORMS.register_module()
class PseudoLabelFormatBundle(PackDetInputs):
    """Extended PackDetInputs to handle additional keys."""

    mapping_table = {
        'gt_bboxes': 'bboxes',
        'gt_bboxes_labels': 'labels',
        'gt_masks': 'masks',
        'gt_ann_ids': 'ann_ids',
        'gt_pseudo_labels': 'pseudo_labels',
        'gt_weak_bboxes': 'weak_bboxes',
        'gt_weak_bboxes_labels': 'weak_bboxes_labels',
        'gt_bbox_ignore': 'bbox_ignore'
    }

    def __init__(self,
                 meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                            'scale_factor', 'flip', 'flip_direction')):
        super().__init__(meta_keys)
        self.logger = MMLogger.get_current_instance()

    def transform(self, results: dict) -> dict:
        """Method to pack the input data.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict:
                - 'inputs' (torch.Tensor): The forward data of models.
                - 'data_sample' (DetDataSample): The annotation info of the
                  sample.
        """
        # self.logger.info(f"Loaded resuls: {results}")
        # self.logger.info(f"Loaded Key: {key}")
        packed_results = super().transform(results)

        # # Process additional keys
        # if 'gt_ignore_flags' in results:
        #     valid_idx = np.where(results['gt_ignore_flags'] == 0)[0]
        #     ignore_idx = np.where(results['gt_ignore_flags'] == 1)[0]

        # additional_data = InstanceData()
        # ignore_additional_data = InstanceData()

        # for key in self.mapping_table.keys():
        #     if key not in results:
        #         continue

        #     if key == 'gt_masks' or isinstance(results[key], BaseBoxes):
        #         if 'gt_ignore_flags' in results:
        #             additional_data[self.mapping_table[key]] = results[key][valid_idx]
        #             ignore_additional_data[self.mapping_table[key]] = results[key][ignore_idx]
        #         else:
        #             additional_data[self.mapping_table[key]] = results[key]
        #     else:
        #         if 'gt_ignore_flags' in results:
        #             additional_data[self.mapping_table[key]] = to_tensor(results[key][valid_idx])
        #             ignore_additional_data[self.mapping_table[key]] = to_tensor(results[key][ignore_idx])
        #         else:
        #             additional_data[self.mapping_table[key]] = to_tensor(results[key])

        # packed_results['data_samples'].gt_instances.update(additional_data)
        # packed_results['data_samples'].ignored_instances.update(ignore_additional_data)

        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(meta_keys={self.meta_keys})'
        return repr_str
