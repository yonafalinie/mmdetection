import torch
import torch.nn as nn
from mmcv.cnn import build_conv_layer
from mmengine.registry import MODELS
from mmpretrain.models import BEiTViT
from mmpretrain.models.backbones.beit import RelativePositionBias

@MODELS.register_module()
class BEiTWithAdapter(nn.Module):
    def __init__(self, 
                 patch_size=16, 
                 img_size=224, 
                 drop_rate=0.1, 
                 out_channels=256, 
                 num_stages=4, 
                 init_cfg=None):
        super(BEiTWithAdapter, self).__init__()
        
        # BEiT backbone
        self.backbone = BEiTViT(
            arch='base',
            patch_size=patch_size,
            img_size=img_size,
            in_channels=3,
            out_indices=(11,),
            drop_rate=drop_rate,
            drop_path_rate=0.,
            bias='qv_bias',
            norm_cfg=dict(type='LN', eps=1e-6),
            final_norm=False,
            out_type='raw',
            with_cls_token=True,
            use_abs_pos_emb=False,
            use_rel_pos_bias=True,
            use_shared_rel_pos_bias=False,
            layer_scale_init_value=0.1,
            init_cfg=init_cfg)

        # Force patch_resolution
        self.backbone.patch_resolution = (14, 14)

        # Configure relative position bias for each layer
        num_heads = 12  # BEiT-Base
        window_size = (14, 14)
        for i, layer in enumerate(self.backbone.layers):
            layer.attn.window_size = window_size
            layer.attn.relative_position_bias = RelativePositionBias(
                window_size=window_size,
                num_heads=num_heads,
                with_cls_token=True)
            print(f"Layer {i} window_size: {layer.attn.window_size}")

        # Adapter parameters
        self.embed_dim = self.backbone.embed_dims  # 768
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.num_stages = num_stages

        # Conv layers for multi-scale outputs
        self.proj = nn.ModuleList([
            build_conv_layer(
                dict(type='Conv2d'),
                in_channels=self.embed_dim,
                out_channels=out_channels,
                kernel_size=1)
            for _ in range(num_stages)
        ])

    def forward(self, x):
        # BEiT forward
        x = self.backbone(x)  # List of outputs, take the last one
        x = x[0]  # [batch_size, 197, 768]

        # Adapter logic
        batch_size = x.shape[0]
        patches = x[:, 1:, :]  # [batch_size, 196, 768]
        grid_size = int((patches.shape[1]) ** 0.5)  # 14
        x = patches.reshape(batch_size, grid_size, grid_size, self.embed_dim).permute(0, 3, 1, 2)  # [batch_size, 768, 14, 14]

        # Generate multi-scale outputs
        outputs = []
        feat = self.proj[0](x)  # [batch_size, 256, 14, 14]
        outputs.append(feat)
        for i in range(1, self.num_stages):
            feat = nn.functional.avg_pool2d(feat, kernel_size=2, stride=2)
            outputs.append(feat)

        return tuple(outputs)