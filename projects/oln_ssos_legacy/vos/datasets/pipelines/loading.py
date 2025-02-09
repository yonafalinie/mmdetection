from mmengine.registry import TRANSFORMS
from mmdet.datasets.transforms import LoadAnnotations
import numpy as np
from mmengine.logging import MMLogger


@TRANSFORMS.register_module()
class LoadAnnotationsWithAnnID(LoadAnnotations):
    """Load annotations extended with `gt_ann_ids`, `gt_pseudo_labels`, 
    and `gt_weak_bboxes`.

    This class extends `LoadAnnotations` to include additional fields for
    custom annotations.
    """

    def __init__(self,  with_pseudo_labels=True, with_gt_bbox_ignore=True, with_ann_id=True, with_weak_bboxes=True, **kwargs):
        """
        Args:
            with_ann_id (bool): Whether to load annotation IDs. Defaults to True.
            with_pseudo_labels (bool): Whether to load pseudo labels. Defaults to True.
            with_weak_bboxes (bool): Whether to load weak bounding boxes. Defaults to True.
            **kwargs: Additional arguments passed to `LoadAnnotations`.
        """
        super().__init__(**kwargs)
        self.with_pseudo_labels = with_pseudo_labels
        self.with_weak_bboxes = with_weak_bboxes
        self.with_ann_id = with_ann_id
        self.with_gt_bbox_ignore = with_gt_bbox_ignore
        # Initialize logger
        self.logger = MMLogger.get_current_instance()


    def _load_ann_ids(self, results: dict) -> None:
        """Private function to load annotation IDs.

        Args:
            results (dict): Result dict from the dataset.

        Adds:
            gt_ann_ids (np.ndarray): Array of annotation IDs.
        """
        # self.logger.info(f"Loaded resuls: {results}")
        
        gt_ann_ids = [instance['gt_ann_ids'] for instance in results['instances']]
        results['gt_ann_ids'] = np.array(gt_ann_ids, dtype=np.int32)

    def _load_gt_bbox_ignore(self, results: dict) -> None:

        
        gt_bbox_ignore = [instance['bbox'] for instance in results['instances']]
        results['gt_bbox_ignore'] = np.array(gt_bbox_ignore, dtype=np.int32)
    
       
    def _load_pseudo_labels(self, results: dict) -> None:
        """Private function to load pseudo labels.

        Args:
            results (dict): Result dict from the dataset.

        Adds:
            gt_pseudo_labels (np.ndarray): Array of pseudo labels.
        """
        # self.logger.info(f"Loaded resuls: {results}") 
        gt_pseudo_labels = [instance['gt_pseudo_class'] for instance in results['instances']]
        results['gt_pseudo_labels'] = np.array(gt_pseudo_labels, dtype=np.int32)



    def _load_weak_bboxes(self, results: dict) -> None:
        """Private function to load weak bounding boxes.

        Args:
            results (dict): Result dict from the dataset.

        Adds:
            gt_weak_bboxes (np.ndarray): Array of weak bounding boxes.
            gt_weak_bboxes_labels (np.ndarray): Array of weak bounding box labels.
        """
      
        gt_weak_bboxes = [instance['bbox'] for instance in results['instances']]
        results['gt_weak_bboxes'] = np.array(gt_weak_bboxes, dtype=np.int32)
        gt_weak_bboxes_labels = [instance['gt_pseudo_class'] for instance in results['instances']]
        results['gt_weak_bboxes_labels'] = np.array(gt_weak_bboxes_labels, dtype=np.int32)
  







            

    def transform(self, results: dict) -> dict:
        """Transform function to load annotations.

        Args:
            results (dict): Result dictionary from the dataset.

        Returns:
            dict: Updated dictionary containing loaded annotations.
        """
        results = super().transform(results)
        if self.with_pseudo_labels:
            self._load_pseudo_labels(results)
        if self.with_gt_bbox_ignore: 
            self._load_gt_bbox_ignore(results)  
        if self.with_ann_id:
            self._load_ann_ids(results)               
        if self.with_weak_bboxes:
            self._load_weak_bboxes(results)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__        
        repr_str += f'with_pseudo_labels={self.with_pseudo_labels}, '
        repr_str += f'with_weak_bboxes={self.with_weak_bboxes}, '
        repr_str += f'(with_ann_id={self.with_ann_id}, '
        repr_str += f'with_gt_bbox_ignore={self.with_gt_bbox_ignore})'
        return repr_str

   