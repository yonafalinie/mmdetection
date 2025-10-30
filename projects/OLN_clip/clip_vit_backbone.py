import torch
import torch.nn as nn
from mmpretrain.models import VisionTransformer
from mmdet.registry import MODELS


@MODELS.register_module()
class CLIPVisionBackbone(nn.Module):
    def __init__(self,
                 arch='B',
                 patch_size=16,
                 out_indices=(5, 7, 11),
                 with_cls_token=True,
                 final_norm=False,
                 init_cfg=None):
        super().__init__()
        self.vit = VisionTransformer(
            arch=arch,
            patch_size=patch_size,
            out_indices=out_indices,
            with_cls_token=with_cls_token,
            final_norm=final_norm,
            init_cfg=init_cfg
        )
        self.out_indices = out_indices
        self.with_cls_token = with_cls_token
        self.patch_size = patch_size
        self.embed_dims = self.vit.embed_dims

    def forward(self, x):
        B, _, H, W = x.shape
        H, W = H // self.patch_size, W // self.patch_size

        outs = self.vit(x)
        feats = []

        for i, feat in enumerate(outs):
            if feat.dim() == 3 and feat.shape[1] > 1:
                # Remove CLS token
                feat = feat[:, 1:, :]
                feat = feat.permute(0, 2, 1).reshape(B, self.embed_dims, H, W)
                feats.append(feat)
            elif feat.dim() == 2 and feat.shape[1] == self.embed_dims:
                # [B, C] → [B, C, H, W] (broadcast the CLS token)
                feat = feat.unsqueeze(-1).unsqueeze(-1).expand(B, self.embed_dims, H, W)
                feats.append(feat)
            else:
                raise RuntimeError(f"Unexpected shape at out[{i}]: {feat.shape}")
        return feats
