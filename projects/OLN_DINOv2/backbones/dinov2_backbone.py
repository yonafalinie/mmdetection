import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmengine.registry import MODELS

@MODELS.register_module()
class DINOv2Backbone(BaseModule):
    """DINOv2 as a backbone for Faster R-CNN."""

    def __init__(self, model_name="dinov2_vits14", pretrained=True, init_cfg=None, **kwargs):
        kwargs.pop('depth', None)
        super().__init__(init_cfg)

        # Load the DINOv2 model
        model_fn = {
            "dinov2_vits14": "dinov2_vits14",
            "dinov2_vitb16": "dinov2_vitb16",
            "dinov2_vitl16": "dinov2_vitl16"
        }
        assert model_name in model_fn, f"Invalid model_name. Choose from {list(model_fn.keys())}"
        
        self.dino_model = torch.hub.load('facebookresearch/dinov2', model_fn[model_name], pretrained=pretrained)

    def forward(self, x):
        """Extract features from DINOv2 model."""
        x = self.dino_model.forward_features(x)["x_norm_patchtokens"]  # Extract token features
        return (x,)