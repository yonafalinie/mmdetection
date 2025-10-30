# Copyright (c) Shanghai AI Lab. All rights reserved.
_base_ = [
    'mmdet::_base_/models/mask-rcnn_r50_fpn.py',
    'mmdet::_base_/datasets/coco_instance.py',
    'mmdet::_base_/schedules/schedule_1x.py',
    # '/home3/qljx17/MMOln-ssos/mmdetection/projects/OLN_vitadapter/schedule/schedule_3x.py',
    'mmdet::_base_/default_runtime.py'
]

custom_imports = dict(imports=['projects.OLN_vitadapter.backbones',
                                'projects.OLN_vitadapter.ops',
                                'projects.OLN_vitadapter.custom'])

# pretrained = 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth'
# please download the pretrained weight to the `pretrained/` folder,
# then run: `python convert_14to16.py pretrained/dinov2_vits14_pretrain.pth`
pretrained = '/home3/qljx17/MMOln-ssos/mmdetection/weights/pretrained/dinov2_vits14_pretrain_14to16.pth'
model = dict(
    backbone=dict(
        _delete_=True,
        type='ViTAdapter',
        pretrain_size=592,
        img_size=592,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        drop_path_rate=0.2,
        conv_inplane=64,
        n_points=4,
        deform_num_heads=6,
        cffn_ratio=0.25,
        deform_ratio=1.0,
        interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]],
        window_attn=[True, True, False, True, True, False,
                     True, True, False, True, True, False],
        window_size=[14, 14, None, 14, 14, None,
                     14, 14, None, 14, 14, None],
        pretrained=pretrained),
    neck=dict(
        type='FPN',
        in_channels=[384, 384, 384, 384],
        out_channels=256,
        num_outs=5))
# optimizer
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# augmentation strategy originates from DETR / Sparse RCNN
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='AutoAugment',
         policies=[
             [
                 dict(type='Resize',
                      img_scale=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                                 (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                                 (736, 1333), (768, 1333), (800, 1333)],
                      multiscale_mode='value',
                      keep_ratio=True)
             ],
             [
                 dict(type='Resize',
                      img_scale=[(400, 1333), (500, 1333), (600, 1333)],
                      multiscale_mode='value',
                      keep_ratio=True),
                 dict(type='RandomCrop',
                      crop_type='absolute_range',
                      crop_size=(384, 600),
                      allow_negative_crop=True),
                 dict(type='Resize',
                      img_scale=[(480, 1333), (512, 1333), (544, 1333),
                                 (576, 1333), (608, 1333), (640, 1333),
                                 (672, 1333), (704, 1333), (736, 1333),
                                 (768, 1333), (800, 1333)],
                      multiscale_mode='value',
                      override=True,
                      keep_ratio=True)
             ]
         ]),
    dict(type='RandomCrop',
         crop_type='absolute_range',
         crop_size=(1024, 1024),
         allow_negative_crop=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
data = dict(train=dict(pipeline=train_pipeline))

# Training configuration (assuming 3x schedule = 36 epochs)
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=36,  # From schedule_3x.py, adjust if different
    val_interval=1  # Validate every epoch, adjust or set high to skip
)


# Optimizer wrapper (converted from optimizer and optimizer_config)
optim_wrapper = dict(
    type='OptimWrapper',  # Default wrapper for FP32, use AmpOptimWrapper for FP16 if needed
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=0.0001,
        betas=(0.9, 0.999),
        weight_decay=0.05
    ),
    constructor='LayerDecayOptimizerConstructor',
    paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.70),
    clip_grad=None  # Replaces optimizer_config grad_clip=None
)
# fp16 = dict(loss_scale=dict(init_scale=512))
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=3,
        save_last=True
    )
)