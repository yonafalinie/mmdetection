_base_ = [
    'mmdet::_base_/models/faster-rcnn_r50_fpn.py',
    'mmdet::_base_/datasets/coco_detection.py',
    'mmdet::_base_/schedules/schedule_1x.py',
    'mmdet::_base_/default_runtime.py'
]

custom_imports = dict(
    imports=[
        'projects.OLN.oln',
        'projects.OLN_dinov2.vit_dinov2_backbone',  # for DINOv2 backbone
    ],
    allow_failed_imports=False
)

model = dict(
    type='FasterRCNN',
    backbone=dict(
        _delete_=True,
        type='ViTDINOv2Backbone',
        arch='base',  # ViT-B/16
        patch_size=14,
        out_indices=(4, 7, 10, 11),  # changed from (2, 5, 8, 10)
        init_cfg=dict(
            type='Pretrained',
            checkpoint='/home3/qljx17/MMOln-ssos/mmdetection/weights/vit-base-p14_dinov2-pre_3rdparty_20230426-ba246503.pth'
        )
    ),
    neck=dict(
        type='FPN',
        in_channels=[768, 768, 768, 768],  # all ViT-B blocks output 768-dim tokens
        out_channels=256,
        num_outs=5
    ),
    rpn_head=dict(
        type='OLNRPNHead',
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[1.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(type='TBLRBBoxCoder', normalizer=1.0),
        loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.0),
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
            loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0),
            bbox_score_type='BoxIoU',
            loss_bbox_score=dict(type='L1Loss', loss_weight=1.0)
        ),
    ),
        # Model training and testing settings
    train_cfg=dict(
        rpn=dict(
            objectness_assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.3,
                neg_iou_thr=0.1,
                min_pos_iou=0.3,
                ignore_iof_thr=-1,
            ),
            objectness_sampler=dict(
                type='RandomSampler',
                num=256,
                # Ratio 0 for negative samples.
                pos_fraction=1.0,
                neg_pos_ub=-1,
                add_gt_as_proposals=False,
            ),
        ),
        rpn_proposal=dict(
            nms_across_levels=False,
            nms_pre=2000,
            nms_post=2000,
            max_num=2000,
            nms_thr=0.7,
            min_bbox_size=0,
        ),
    ),
    test_cfg=dict(
        rpn=dict(
            nms_across_levels=False,
            nms_pre=2000,
            nms_post=2000,
            max_num=2000,
            nms_thr=0.0,  # No NMS
            min_bbox_size=0,
        ),
        rcnn=dict(
            score_thr=0.00,
            nms=dict(type='nms', iou_threshold=0.7),
            max_per_img=1500,
        ),
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    ),
)



dataset_type = 'CocoSplitDataset'
data_root = '/home2/projects/datasets/dbf6/'
train_dataloader = dict(
    batch_size=16,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'annotations/dbf6_train.json',
        data_prefix=dict(img=data_root + 'images/'),
        is_class_agnostic=True,
        dataset_name='dbf6',
        train_class='voc',
        eval_class='nonvoc')
)
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_root=data_root,
        ann_file=data_root + 'annotations/dbf6_test.json',
        data_prefix=dict(img=data_root + 'images/'),
        type=dataset_type,
        is_class_agnostic=True,
        dataset_name='dbf6',
        train_class='voc',
        eval_class='nonvoc')
)
test_dataloader = val_dataloader

train_cfg = dict(max_epochs=100)

val_evaluator = dict(
    type='DBF6SplitMetric',
    ann_file=data_root + 'annotations/dbf6_test.json',
)
test_evaluator = dict(
    type='DBF6SplitMetric',
    ann_file=data_root + 'annotations/dbf6_test.json',
)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.02, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', begin=0, end=100, by_epoch=True, milestones=[6, 7], gamma=0.1)
]

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=2)
)
