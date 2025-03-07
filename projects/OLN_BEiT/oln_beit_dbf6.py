# from projects.OLN.oln.coco_split import CocoSplitDataset

_base_ = [
    'mmdet::_base_/models/faster-rcnn_r50_fpn.py',
    'mmdet::_base_/datasets/coco_detection.py',
    'mmdet::_base_/schedules/schedule_1x.py', 'mmdet::_base_/default_runtime.py'
]

custom_imports = dict(
    imports=['projects.OLN.oln', 
             'projects.OLN_BEiT.adapter.beit_adapter'],
               allow_failed_imports=False)

pretrained = 'https://download.openmmlab.com/mmclassification/v0/beit/beit-base-p16_3rdparty_in1k_20221114-26741bbc.pth'  # BEiT-Base pretrained weights

model = dict(
    type='FasterRCNN',
    backbone=dict(
        _delete_=True,
        type='BEiTWithAdapter',
        patch_size=16,
        img_size=224,
        drop_rate=0.1,
        out_channels=256,
        num_stages=4,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(
        in_channels=[256, 256, 256, 256],  # Adjusted for adapter (was [256, 512, 1024, 2048] for ResNet)
        out_channels=256,
        num_outs=5,
        type='FPN'),    
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
        type='OLNRoIHead',
        bbox_head=dict(
            type='Shared2FCBBoxScoreHead',
            num_classes=1,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0),
            bbox_score_type='BoxIoU',  # 'BoxIoU' or 'Centerness'
            loss_bbox_score=dict(type='L1Loss', loss_weight=1.0))),
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
            min_bbox_size=0)),
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
            max_per_img=1500)
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    ))

metainfo = dict(
    classes=('firearm', 'firearmpart', 'knife', 'camera', 'ceramic_knife', 'laptop'),  
)

dataset_type = 'DBF6SplitDataset'
data_root = '/media/neel/hdd8tb/dataset/dbf6/'
train_dataloader = dict(
    batch_size=10,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'annotations/dbf6_train__.json',
        data_prefix=dict(img=data_root + 'images/'),
        metainfo=metainfo,
        is_class_agnostic=True,
        train_class='voc',
        eval_class='nonvoc',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', scale=(224, 224), keep_ratio=False),  # Adjusted
            dict(type='RandomFlip', prob=0.5),
            dict(type='PackDetInputs'),
        ]))
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
        eval_class='nonvoc',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', scale=(224, 224), keep_ratio=False),  # Adjusted
            dict(type='RandomFlip', prob=0.5),
            dict(type='PackDetInputs'),
        ]))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoSplitMetric',
    ann_file=data_root + 'annotations/dbf6_test.json',)
test_evaluator = dict(
    type='CocoSplitMetric',
    ann_file=data_root + 'annotations/dbf6_test.json',)

train_cfg = dict(max_epochs=100)

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.02, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=8,
        by_epoch=True,
        milestones=[6, 7],
        gamma=0.1)
]
# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }),
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=0.0001,
        betas=(0.9, 0.999),
        weight_decay=0.0015))
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=2))
    