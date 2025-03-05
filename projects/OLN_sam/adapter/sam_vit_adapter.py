import torch
import torch.nn as nn
from mmpretrain.models import VisionTransformer
from mmdet.registry import MODELS

@MODELS.register_module()
class SAMViTAdapter(nn.Module):
    def __init__(self, arch, patch_size, drop_path_rate, out_indices, init_cfg=None):
        super().__init__()
        self.vit = VisionTransformer(
            arch=arch,
            patch_size=patch_size,
            drop_path_rate=drop_path_rate,
            out_indices=out_indices,
            with_cls_token=True,  # Ensure CLS token is included
            out_type='raw',  # Return full patch sequence (if supported)
            final_norm=False,
            init_cfg=init_cfg)
        self.patch_size = patch_size
        self.out_indices = out_indices

    def forward(self, x):
        # Input: [batch_size, 3, H, W], e.g., [2, 3, 1333, 800]
        feats = self.vit(x)  # Tuple of [batch_size, num_patches + 1, embed_dim]
        batch_size = x.size(0)
        h, w = x.size(2) // self.patch_size, x.size(3) // self.patch_size  # e.g., 83, 50
        
        # Debug shapes
        # print(f"ViT output: {type(feats)}, shapes: {[f.shape for f in feats]}")
        
        out = []
        for feat in feats:
            # print(f"Feat shape before slicing: {feat.shape}")
            if feat.dim() == 3:  # [batch_size, num_patches + 1, embed_dim]
                feat = feat[:, 1:, :]  # Remove CLS token: [2, 4150, 768]
                feat = feat.view(batch_size, h, w, -1).permute(0, 3, 1, 2)  # [2, 768, 83, 50]
                out.append(feat)
            else:
                raise ValueError(f"Unexpected feat shape: {feat.shape}")
        
        # print(f"Output shapes: {[f.shape for f in out]}")
        return tuple(out)

    def init_weights(self):
        pass