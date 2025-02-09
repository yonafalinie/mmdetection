import torch
import tqdm
from mmengine.runner.loops import EpochBasedTrainLoop
from mmengine.dataset import RepeatDataset
from torchvision.ops import batched_nms
from mmdet.datasets.api_wrappers import COCO
from mmdet.structures.bbox import bbox2roi
from mmengine.registry import LOOPS
from mmengine.logging import MMLogger

@LOOPS.register_module()
class PseudoLabelEpochBasedTrainLoop(EpochBasedTrainLoop):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("Initialized PseudoLabelEpochBasedTrainLoop")
        self.logger = MMLogger.get_current_instance()

    def run_iter(self, idx, data_batch):
        """Override `run_iter` to include pseudo-labeling logic during training."""
        # Check if pseudo-labeling needs to be performed
        if self.runner.epoch >= self.runner.model.calculate_pseudo_labels_from_epoch:
            self.run_pseudo_label_epoch()
        
        # Call the parent class's `run_iter` for the usual training process
        super().run_iter(idx, data_batch)

    def run_pseudo_label_epoch(self):
        """Perform pseudo-labeling during training."""
        weak_conf_thr = self.runner.model.roi_head.weak_bbox_test_confidence
        use_weak_bboxes = self.runner.model.use_weak_bboxes

        with torch.no_grad():
            ann_ids = []
            weak_img_ids = []
            weak_bboxes = []
            for i, data_batch in enumerate(tqdm.tqdm(self.dataloader)):
                # Get device from model
                device = next(self.runner.model.parameters()).device
                # self.logger.info(f"Loaded data_batch: {data_batch}")        
                # Move data to device
                inputs_ = [input_tensor.float().to(device) for input_tensor in data_batch['inputs']]                   
                data_samples = data_batch['data_samples']                   
                # self.logger.info(f"Loaded data_samples: {data_samples}")  

                            
                # self.logger.info(f"Loaded inputs0: {inputs_[0].shape}")
                # self.logger.info(f"Loaded inputs1: {inputs_[1].shape}") 
                # Make a batch of inputs size (2, 3, H, W)
                inputs = torch.stack(inputs_, dim=0)
                # self.logger.info(f"Loaded inputs: {inputs}")
                # self.logger.info(f"Image tensor shape: {inputs.shape}")

                # Extract features
                fts = self.runner.model.extract_feat(inputs)
                # Extract bboxes from data_samples
                bboxes = [data_sample.gt_instances.bboxes.tensor for data_sample in data_samples]
                # self.logger.info(f"Extracted bboxes: {bboxes}")
                rois = bbox2roi(bboxes)

                # Extract ann_ids from data_samples
                gt_ann_ids = [data_sample.gt_instances.ann_ids for data_sample in data_samples]
                ann_ids.extend(gt_ann_ids)
                self.runner.model.roi_head.accumulate_pseudo_labels(fts, rois)

                # inputs = data_batch['inputs']                
                # inputs = torch.stack(inputs, dim=0)
                # # print(inputs[2].shape)
                # device = next(self.runner.model.parameters()).device
                # inputs = inputs.to(device) 
                # inputs = inputs.float()
                # kwargs = {}
                # fts = self.runner.model.extract_feat(inputs)

                # # Extract gt_bboxes from data_samples
                # gt_bboxes = [
                #     data_sample.gt_instances.bboxes.tensor
                #     for data_sample in data_batch['data_samples']]
              
                # # Generate proposals and accumulate pseudo labels
                # rois = bbox2roi(gt_bboxes)
                # gt_ann_ids = inputs['gt_ann_ids']
                # ann_ids.extend(gt_ann_ids)
                # self.runner.model.module.roi_head.accumulate_pseudo_labels(fts, rois)

                if use_weak_bboxes:
                    # Handling weak bboxes
                    gt_ann_ids_list = [g[0].cpu().item() for g in gt_ann_ids]
                    gt_anns = self.dataloader.dataset.coco.loadAnns(gt_ann_ids_list)
                    img_ids = [a['image_id'] for a in gt_anns]
                    proposal_list = self.runner.model.module.rpn_head.simple_test_rpn(
                        fts, inputs['img_metas']
                    )
                    res = self.runner.model.module.roi_head.simple_test(
                        fts, proposal_list, inputs['img_metas'], rescale=False, with_ood=False
                    )
                    bboxes = [res_img[0][0] for res_img in res]
                    bboxes = [bbox[batched_nms(
                        torch.tensor(bbox[:, :4]), 
                        torch.tensor(bbox[:, 4]),
                        torch.ones(bbox.shape[0]), 
                        0.5
                    )] for bbox in bboxes]
                    bboxes_filtered = [
                        torch.tensor(b[b[:, 4] > weak_conf_thr, :4]).to(fts[0].device) 
                        for b in bboxes
                    ]
                    weak_rois = bbox2roi(bboxes_filtered)
                    weak_img_ids.extend([
                        im_id for im_id, b in zip(img_ids, bboxes_filtered) 
                        for _ in range(b.shape[0])
                    ])
                    weak_bboxes.extend(bboxes_filtered)

                    self.runner.model.module.roi_head.accumulate_weak_pseudo_labels(fts, weak_rois)

            # Calculate pseudo-labels
            labels = self.runner.model.roi_head.calculate_pseudo_labels()
            ann_ids = torch.cat(ann_ids).cpu().numpy()

            pseudo_classes = {ann_id: label for ann_id, label in zip(ann_ids, labels[:ann_ids.shape[0]])}
  
            
            # Get dataset instance
            dataset = self.dataloader.dataset.ann_file
            
  
            # Initialize COCO API with annotation file
            coco = COCO(dataset)
            # Get annotation info
            ann_ids = coco.getAnnIds()
            # self.logger.info(f"Loaded ann_ids: {ann_ids}")
            annotations = coco.loadAnns(ann_ids)
            # self.logger.info(f"Loaded annotations: {annotations}")
            # Filter weak annotations
            filtered_annotations = [
                ann for ann in annotations
                if 'weak' not in ann.keys() or not ann['weak']
            ]
            # self.logger.info(f"Filtered annotations count: {len(filtered_annotations)}")

            # Update pseudo_class for each annotation
            for ann_id, label in zip(ann_ids, labels):
                if ann_id in coco.anns:
                    coco.anns[ann_id]['pseudo_class'] = label
            # Save modified annotations back to dataset
            coco.dataset['annotations'] = [
                ann for ann in coco.anns.values()
                if 'weak' not in ann or not ann['weak']
            ]            
            # self.logger.info(f"Updated annotations: {coco.dataset['annotations']}")


                    
            # dataset = self.dataloader.dataset.coco.dataset
           
            # dataset['annotations'] = [
            #     a for a in dataset['annotations']
            #     if 'weak' not in a.keys() or not a['weak']
            # ]
            # for ann_id, label in zip(ann_ids, labels):
            #     if isinstance(self.dataloader.dataset, RepeatDataset):
            #         self.dataloader.dataset.dataset.coco.anns[ann_id]['pseudo_class'] = label
            #     else:
            #         self.dataloader.dataset.coco.anns[ann_id]['pseudo_class'] = label

            # if use_weak_bboxes:
            #     weak_bboxes = torch.cat(weak_bboxes, dim=0)
            #     next_ann_id = max(self.dataloader.dataset.coco.anns.keys()) + 1
            #     for label, weak_box, weak_img_id in zip(labels[ann_ids.shape[0]:], weak_bboxes, weak_img_ids):
            #         box = weak_box.cpu().tolist()
            #         ann = {
            #             'segmentation': [],
            #             'iscrowd': 0,
            #             'bbox': [int(box[0]), int(box[1]), int(box[2] - box[0]), int(box[3] - box[1])],
            #             'area': int(box[2] - box[0]) * int(box[3] - box[1]),
            #             'image_id': weak_img_id,
            #             'category_id': dataset['categories'][0]['id'],
            #             'pseudo_class': label,
            #             'id': next_ann_id,
            #             'weak': True
            #         }
            #         dataset['annotations'].append(ann)
            #         next_ann_id += 1

            #     new_coco = COCO()
            #     new_coco.dataset = dataset
            #     new_coco.createIndex()
            #     self.dataloader.dataset.coco = new_coco
