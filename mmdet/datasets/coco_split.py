"""This file contains code to build dataloader of COCO-split dataset.

Reference:
    "Learning Open-World Object Proposals without Learning to Classify",
        Aug 2021. https://arxiv.org/abs/2108.06753
        Dahun Kim, Tsung-Yi Lin, Anelia Angelova, In So Kweon and Weicheng Kuo
"""

import itertools
import logging
import os.path as osp
import tempfile
from collections import OrderedDict
import os
from typing import List, Union
import copy

import mmcv
import numpy as np
from mmengine.logging import print_log
from .api_wrappers import COCO
from mmengine.fileio import get_local_path
# Added for cross-category evaluation
# from .cocoeval_wrappers import COCOEvalWrapper, COCOEvalXclassWrapper

from terminaltables import AsciiTable

from mmdet.evaluation.functional import eval_recalls
from mmdet.registry import DATASETS
from .coco import CocoDataset

# try:
#     import pycocotools
#     if not hasattr(pycocotools, '__sphinx_mock__'):  # for doc generation
#         assert pycocotools.__version__ >= '12.0.2'
# except AssertionError:
#     raise AssertionError('Incompatible version of pycocotools is installed. '
#                          'Run pip uninstall pycocotools first. Then run pip '
#                          'install mmpycocotools to install open-mmlab forked '
#                          'pycocotools.')


@DATASETS.register_module()
class CocoSplitDataset(CocoDataset):

    def __init__(self, 
                 is_class_agnostic=False, 
                 train_class='all',
                 eval_class='all',
                 filter_empty_gt=True,
                 min_size=32,
                 **kwargs):
        print(f"Initializing CocoSplitDataset with train_class={train_class}, eval_class={eval_class}")
        # We convert all category IDs into 1 for the class-agnostic training and
        # evaluation. We train on train_class and evaluate on eval_class split.
        self.is_class_agnostic = is_class_agnostic
        self.train_class = train_class
        self.eval_class = eval_class
        self.filter_empty_gt = filter_empty_gt
        self.min_size = min_size
        super(CocoSplitDataset, self).__init__(**kwargs)
    
    CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')
    VOC_CLASSES = (
               'airplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'dining table', 'dog', 'horse', 
               'motorcycle', 'person', 'potted plant', 'sheep', 'couch',
               'train', 'tv')
    NONVOC_CLASSES = (
               'truck', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench',
               'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake',
               'bed', 'toilet', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')
    class_names_dict = {
        'all': CLASSES,
        'voc': VOC_CLASSES,
        'nonvoc': NONVOC_CLASSES
    }

    def load_data_list(self) -> List[dict]:
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """

        self.coco = COCO(self.ann_file)
        print(f"Loading annotations from '{self.coco}'")

        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)
        self.train_cat_ids = self.coco.get_cat_ids(
            cat_names=self.class_names_dict[self.train_class]
            )
        self.eval_cat_ids = self.coco.get_cat_ids(
            cat_names=self.class_names_dict[self.eval_class]
            )
        if self.is_class_agnostic:
            self.cat2label = {cat_id: 0 for cat_id in self.cat_ids}
        else:
            self.cat2label = {
                cat_id: i for i, cat_id in enumerate(self.cat_ids)}
            
        self.cat_img_map = copy.deepcopy(self.coco.cat_img_map)
        self.img_ids = self.coco.get_img_ids()
        data_list = []
        total_ann_ids = []
        for img_id in self.img_ids:
            raw_img_info = self.coco.load_imgs([img_id])[0]
            raw_img_info['img_id'] = img_id

            ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
            raw_ann_info = self.coco.load_anns(ann_ids)
            total_ann_ids.extend(ann_ids)

            parsed_data_info = self.parse_data_info({
                'raw_ann_info':
                raw_ann_info,
                'raw_img_info':
                raw_img_info
            })
            data_list.append(parsed_data_info)
        if self.ANN_ID_UNIQUE:
            assert len(set(total_ann_ids)) == len(
                total_ann_ids
            ), f"Annotation ids in '{self.ann_file}' are not unique!"

        del self.coco

        # print(f"Returning data_list: {data_list}")

        return data_list

    # # Refer to custom.py -- filter_img is not used in test_mode.
    # def filter_data(self) -> List[dict]:
    #     """Filter annotations according to filter_cfg.

    #     Returns:
    #         List[dict]: Filtered results.
    #     """
    #     if self.test_mode:
    #         return self.data_list

    #     if self.filter_cfg is None:
    #         return self.data_list

    #     filter_empty_gt = self.filter_cfg.get('filter_empty_gt', False)
    #     min_size = self.filter_cfg.get('min_size', 0)

    #     ids_with_ann = set(data_info['img_id'] for data_info in self.data_list)
    #     ids_in_cat = set()
    #     for class_id in self.cat_ids:
    #         ids_in_cat |= set(self.coco.cat_img_map[class_id])

    #     ids_in_cat &= ids_with_ann

    #     valid_data_infos = []
    #     for data_info in self.data_list:
    #         img_id = data_info['img_id']
    #         width = data_info['width']
    #         height = data_info['height']
    #         if filter_empty_gt and img_id not in ids_in_cat:
    #             continue
    #         if min(width, height) >= min_size:
    #             valid_data_infos.append(data_info)

    #     return valid_data_infos

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        """Parse raw annotation to target format.

        Args:
            raw_data_info (dict): Raw data information load from ``ann_file``

        Returns:
            Union[dict, List[dict]]: Parsed annotation.
        """
        img_info = raw_data_info['raw_img_info']
        ann_info = raw_data_info['raw_ann_info']

        data_info = {}

        # TODO: need to change data_prefix['img'] to data_prefix['img_path']
        img_path = osp.join(self.data_prefix['img'], img_info['file_name'])
        if self.data_prefix.get('seg', None):
            seg_map_path = osp.join(
                self.data_prefix['seg'],
                img_info['file_name'].rsplit('.', 1)[0] + self.seg_map_suffix)
        else:
            seg_map_path = None
        data_info['img_path'] = img_path
        data_info['img_id'] = img_info['img_id']
        data_info['seg_map_path'] = seg_map_path
        data_info['height'] = img_info['height']
        data_info['width'] = img_info['width']

        if self.return_classes:
            data_info['text'] = self.metainfo['classes']
            data_info['caption_prompt'] = self.caption_prompt
            data_info['custom_entities'] = True

        instances = []
        for i, ann in enumerate(ann_info):
            instance = {}

            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.train_cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]

            if ann.get('iscrowd', False):
                instance['ignore_flag'] = 1
                instance['gt_bbox_ignore']=bbox
            else:
                instance['ignore_flag'] = 0
            instance['bbox'] = bbox
            instance['bbox_label'] = self.cat2label[ann['category_id']]

            if ann.get('segmentation', None):
                instance['mask'] = ann['segmentation']

            instances.append(instance)
        data_info['instances'] = instances
        # print(f"Returning data_info: {data_info}")
        return data_info