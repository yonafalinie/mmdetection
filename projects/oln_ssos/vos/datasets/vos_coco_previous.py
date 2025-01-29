import numpy as np
import os.path as osp
from typing import List, Union
from mmdet.registry import DATASETS
from mmdet.datasets import CocoDataset, CocoSplitDataset
from mmdet.datasets.api_wrappers import COCO

@DATASETS.register_module()
class VOSCocoDataset(CocoDataset):

    def _det2json(self, results):
        """Convert detection results to COCO json style."""
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            result = results[idx]
            for label in range(len(result)):
                bboxes = result[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['ood_score'] = float(bboxes[i][5])
                    data['inter_feats'] = bboxes[i][6:27].tolist()
                    data['category_id'] = self.cat_ids[label]
                    json_results.append(data)
        return json_results


@DATASETS.register_module()
class VOSCocoSplitDataset(CocoSplitDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Add pseudo_class to all annotations
        if hasattr(self, 'coco'):    
            for ann_id, ann in self.coco.anns.items():
                ann['pseudo_class'] = 0
                self.coco.anns[ann_id] = ann  # Update in COCO object
        

    def parse_data_info(self, raw_data_info: dict) -> dict:
        
        """Parse raw annotation to include custom fields like
        `gt_ann_ids`, `gt_pseudo_class`, `gt_weak_bboxes`, and `gt_weak_bboxes_labels`.

        Args:
            raw_data_info (dict): Raw data information loaded from `ann_file`.

        Returns:
            dict: Parsed annotation including the custom fields.
        """

        img_info = raw_data_info['raw_img_info']
        ann_info = raw_data_info['raw_ann_info']
        # print(f"Received img_info: {img_info}")
        # print(f"\nParsing annotation: {ann_info}")

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



        instances = []
        for i, ann in enumerate(ann_info):
            # print(f"Processing ann: {ann}")
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

            # A weak bounding box is a box detected by OLN but notin the gt
            is_weak = ann.get('weak', False)  

            if ann.get('iscrowd', False):
                instance['ignore_flag'] = 1
                instance['gt_bbox_ignore']=bbox
            else:
                if not is_weak:
                    instance['ignore_flag'] = 0
                    instance['bbox'] = bbox
                    instance['gt_ann_ids']= ann['id'] 
                    instance['bbox_label'] = self.cat2label[ann['category_id']]                        
                    instance['gt_pseudo_class'] = ann.get('pseudo_class', 0)    
                else:
                    instance['gt_weak_bboxes'] = bbox
                    instance['gt_weak_bboxes_labels'] = ann['pseudo_class']

            if ann.get('segmentation', None):
                            instance['mask'] = ann['segmentation']

            instances.append(instance)
            
        if instances:
            data_info['instances'] = instances
            return data_info
        return None   
        #  
        # # Only return data_info if instances exist and are not empty
        # if instances:
        #     data_info['instances'] = instances
        #     # print(f"Returning data_info: {data_info}")
        #     return data_info

        

    def _det2json(self, results):
        """Convert detection results to COCO json style."""
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            result = results[idx]
            for label in range(len(result)):
                bboxes = result[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['ood_score'] = float(bboxes[i][5])
                    data['category_id'] = self.cat_ids[label]
                    json_results.append(data)
        return json_results

    def _segm2json(self, results):
        """Convert instance segmentation results to COCO json style."""
        bbox_json_results = []
        segm_json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            det, seg = results[idx]
            for label in range(len(det)):
                # bbox results
                bboxes = det[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['ood_score'] = float(bboxes[i][5])
                    data['category_id'] = self.cat_ids[label]
                    bbox_json_results.append(data)

                # segm results
                # some detectors use different scores for bbox and mask
                if isinstance(seg, tuple):
                    segms = seg[0][label]
                    mask_score = seg[1][label]
                else:
                    if label < len(seg):
                        segms = seg[label]
                        mask_score = [bbox[4] for bbox in bboxes]
                    else:
                        segms = []
                        mask_score = []
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(mask_score[i])
                    data['ood_score'] = float(bboxes[i][5])
                    data['category_id'] = self.cat_ids[label]
                    if isinstance(segms[i]['counts'], bytes):
                        segms[i]['counts'] = segms[i]['counts'].decode()
                    data['segmentation'] = segms[i]
                    segm_json_results.append(data)
        return bbox_json_results, segm_json_results


@DATASETS.register_module()
class VOSDB6SplitDataset(VOSCocoSplitDataset):
    CLASSES = ('firearm', 'firearmpart', 'knife', 'camera', 'ceramic_knife', 'laptop')
    ID_CLASSES = ('knife', 'camera', 'ceramic_knife', 'laptop')
    OOD_CLASSES = ('firearm', 'firearmpart')
    class_names_dict = {
        'all': CLASSES,
        'id': ID_CLASSES,
        'ood': OOD_CLASSES
    }


@DATASETS.register_module()
class VOSParcelsSplitDataset(VOSCocoSplitDataset):
    CLASSES = ('object', 'anomaly')
    ID_CLASSES = ('object',)
    OOD_CLASSES = ('anomaly',)
    class_names_dict = {
        'all': CLASSES,
        'id': ID_CLASSES,
        'ood': OOD_CLASSES
    }


@DATASETS.register_module()
class VOSLtdImagingSplitDataset(VOSCocoSplitDataset):
    CLASSES = ('human', 'bicycle', 'motorcycle', 'vehicle')
    ID_CLASSES = ('human', 'bicycle', 'motorcycle')
    OOD_CLASSES = ('vehicle',)
    class_names_dict = {
        'all': CLASSES,
        'id': ID_CLASSES,
        'ood': OOD_CLASSES
    }

@DATASETS.register_module()
class VOSSIXRay10SplitDataset(VOSCocoSplitDataset):
    CLASSES = ('firearm', 'knife', 'wrench', 'pliers', 'scissors')
    ID_CLASSES = ('knife', 'wrench', 'pliers', 'scissors')
    OOD_CLASSES = ('firearm',)
    class_names_dict = {
        'all': CLASSES,
        'id': ID_CLASSES,
        'ood': OOD_CLASSES
    }


@DATASETS.register_module()
class VOSBDDSplitDataset(VOSCocoSplitDataset):
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
    BDD_CLASSES = (
        'person', 'bicycle', 'car', 'motorcycle', 'truck', 'bus', 'train', 'traffic light', 'stop sign')
    NONBDD_CLASSES = (
        'fire hydrant', 'parking meter', 'bench',
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
        'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush',
        'airplane', 'bird', 'boat', 'bottle',
        'cat', 'chair', 'cow', 'dining table', 'dog', 'horse',
        'potted plant', 'sheep', 'couch',
        'tv'
    )
    class_names_dict = {
        'all': CLASSES,
        'bdd': BDD_CLASSES,
        'nonbdd': NONBDD_CLASSES
    }
