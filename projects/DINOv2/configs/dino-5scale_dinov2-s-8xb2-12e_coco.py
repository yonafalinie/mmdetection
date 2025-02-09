_base_ = [
    'dino-4scale_r50_8xb2-12e_coco.py',
]
norm_cfg = dict(type='LN2d', requires_grad=True)

model = dict(
    backbone=dict(
        _delete_=True,
        type='DINOv2',
        img_size=518,
        patch_size=14,
        embed_dim=384,
        depth=12,
        num_heads=6,
        drop_path_rate=0.1,
        window_size=37,
        mlp_ratio=4,
        qkv_bias=True,
        norm_cfg=dict(type='LN'),
        window_block_indexes=[0, 1, 3, 4, 6, 7, 9,
                              10],  # global attention for 2, 5, 8, 11
        use_rel_pos=True,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='/home/neel/data/Code/MMOln-ssos/mmdetection/weights/vit-small-p14_'\
                       'dinov2-pre_3rdparty_20230426-5641ca5a.pth'
             )),
    neck=dict(
        _delete_=True,
        type='SimpleFPN',
        backbone_channel=384,
        in_channels=[
            96, 192,
            384, 384
        ],
        out_channels=256,
        num_outs=4,
        norm_cfg=norm_cfg,
    )
)

custom_hooks = [dict(type='Fp16CompresssionHook')]
