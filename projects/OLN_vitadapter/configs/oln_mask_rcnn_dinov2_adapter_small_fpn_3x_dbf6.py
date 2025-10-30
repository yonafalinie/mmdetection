# Copyright (c) Shanghai AI Lab. All rights reserved.
_base_ = [
    'mmdet::_base_/models/mask-rcnn_r50_fpn.py',
    'mmdet::_base_/datasets/coco_instance.py',
    'mmdet::_base_/schedules/schedule_1x.py',
    # '/home3/qljx17/MMOln-ssos/mmdetection/projects/OLN_vitadapter/schedule/schedule_3x.py',
    'mmdet::_base_/default_runtime.py'
]

custom_imports = dict(imports=['projects.OLN.oln',
                               'projects.OLN_vitadapter.backbones',
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
        num_outs=5),
    rpn_head=dict(
        type='OLNRPNHead',
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[1.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='TBLRBBoxCoder',
            normalizer=1.0),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.0),
        reg_decoded_bbox=True,
        loss_bbox=dict(type='IoULoss', loss_weight=10.0),
        objectness_type='Centerness',
        loss_objectness=dict(type='L1Loss', loss_weight=1.0),
    ),
    roi_head=dict(
        type='MaskScoringOLNRoIHead',
        bbox_head=dict(
            type='Shared2FCBBoxScoreHead',
            num_classes=1,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0),
            bbox_score_type='BoxIoU',  # 'BoxIoU' or 'Centerness'
            loss_bbox_score=dict(type='L1Loss', loss_weight=1.0)),
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(
            type='OLNFCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=1,
            class_agnostic=True,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)),
        mask_iou_head=dict(
            type='OLNMaskIoUHead',
            num_convs=1,
            num_fcs=3,
            roi_feat_size=14,
            in_channels=256,
            conv_out_channels=256,
            fc_out_channels=1024,
            num_classes=1,
            loss_iou=dict(type='L1Loss', loss_weight=1.0)
        )),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            objectness_assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.3,
                neg_iou_thr=0.1,
                min_pos_iou=0.3,
                ignore_iof_thr=-1),
            objectness_sampler=dict(
                type='RandomSampler',
                num=256,
                # Ratio 0 for negative samples.
                pos_fraction=1.,
                neg_pos_ub=-1,
                add_gt_as_proposals=False)
        ),
        rpn_proposal=dict(
            nms_across_levels=False,
            nms_pre=2000,
            nms_post=2000,
            max_num=2000,
            nms_thr=0.7,
            min_bbox_size=0),
        rcnn=dict(mask_thr_binary=0.5,)),
    test_cfg=dict(
        rpn=dict(
            nms_across_levels=False,
            nms_pre=2000,
            nms_post=2000,
            max_num=2000,
            nms_thr=0.0,  # No nms
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.00,
            nms=dict(type='nms', iou_threshold=0.7),
            max_per_img=1000,
            mask_thr_binary=0.5
        )
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    ))    
# optimizer
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# augmentation strategy originates from DETR / Sparse RCNN
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='AutoAugment',
         policies=[
             [dict(type='Resize', scale=(480, 1333), keep_ratio=True)],
             [dict(type='Resize', scale=(512, 1333), keep_ratio=True)],
             [dict(type='Resize', scale=(544, 1333), keep_ratio=True)],
             [dict(type='Resize', scale=(576, 1333), keep_ratio=True)],
             [dict(type='Resize', scale=(608, 1333), keep_ratio=True)],
             [dict(type='Resize', scale=(640, 1333), keep_ratio=True)],
             [dict(type='Resize', scale=(672, 1333), keep_ratio=True)],
             [dict(type='Resize', scale=(704, 1333), keep_ratio=True)],
             [dict(type='Resize', scale=(736, 1333), keep_ratio=True)],
             [dict(type='Resize', scale=(768, 1333), keep_ratio=True)],
             [dict(type='Resize', scale=(800, 1333), keep_ratio=True)],
             [
                 dict(type='Resize', scale=(400, 1333), keep_ratio=True),
                 dict(type='RandomCrop', crop_type='absolute_range', crop_size=(384, 600), allow_negative_crop=True),
                 dict(type='Resize', scale=(480, 1333), keep_ratio=True)
             ],
             [
                 dict(type='Resize', scale=(500, 1333), keep_ratio=True),
                 dict(type='RandomCrop', crop_type='absolute_range', crop_size=(384, 600), allow_negative_crop=True),
                 dict(type='Resize', scale=(640, 1333), keep_ratio=True)
             ],
             [
                 dict(type='Resize', scale=(600, 1333), keep_ratio=True),
                 dict(type='RandomCrop', crop_type='absolute_range', crop_size=(384, 600), allow_negative_crop=True),
                 dict(type='Resize', scale=(800, 1333), keep_ratio=True)
             ]
         ]),
    dict(type='RandomCrop',
         crop_type='absolute_range',
         crop_size=(1024, 1024),
         allow_negative_crop=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='PackDetInputs')
]

data = dict(train=dict(pipeline=train_pipeline))


metainfo = dict(
    classes=('firearm', 'firearmpart', 'knife', 'camera', 'ceramic_knife', 'laptop'),  
)

dataset_type = 'DBF6SplitDataset'
data_root = '/home2/projects/datasets/dbf6/'
train_dataloader = dict(
    batch_size=6,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'annotations/dbf6_train__.json',
        data_prefix=dict(img=data_root + 'images/'),
        metainfo=metainfo,
        is_class_agnostic=True,
        train_class='voc',
        eval_class='nonvoc',
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=2,
    dataset=dict(
        data_root=data_root,
        ann_file=data_root + 'annotations/dbf6_test.json',
        data_prefix=dict(img=data_root + 'images/'),
        metainfo=metainfo,
        type=dataset_type,
        is_class_agnostic=True,
        train_class='voc',
        eval_class='nonvoc'))
test_dataloader = val_dataloader



val_evaluator = dict(
    type='DBF6SplitMetric',
    ann_file=data_root + 'annotations/dbf6_test.json',
    metric=['bbox', 'segm'],
    format_only=False
    )
test_evaluator = dict(
    type='DBF6SplitMetric',
    ann_file=data_root + 'annotations/dbf6_test.json',
    metric=['bbox', 'segm'],
    format_only=False)



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
        lr=3.75e-5,
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