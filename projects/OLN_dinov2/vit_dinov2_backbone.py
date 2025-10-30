import torch
import torch.nn as nn
from mmpretrain.models import VisionTransformer
from mmdet.registry import MODELS

@MODELS.register_module()
class ViTDINOv2Backbone(nn.Module):
    def __init__(self, arch='base', patch_size=14, out_indices=(4, 7, 10, 11), init_cfg=None):
        super().__init__()
        self.vit = VisionTransformer(
            arch=arch,
            patch_size=patch_size,
            out_indices=out_indices,
            final_norm=False,
            with_cls_token=False,         # <- CLS token is removed
            out_type='featmap',           # <- PATCH TOKENS instead of CLS
            init_cfg=init_cfg
        )
        self.patch_size = patch_size
        self.out_indices = out_indices
        self.embed_dims = self.vit.embed_dims

    def forward(self, x):
        feats = self.vit(x)  # returns list of [B, C, H, W]
        return feats
