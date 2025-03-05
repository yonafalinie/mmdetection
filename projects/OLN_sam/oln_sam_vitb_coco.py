_base_ = [
    'mmdet::_base_/models/faster-rcnn_r50_fpn.py',
    'mmdet::_base_/datasets/coco_detection.py',
    'mmdet::_base_/schedules/schedule_1x.py',
    'mmdet::_base_/default_runtime.py'
]

custom_imports = dict(
    imports=['projects.OLN.oln', 'projects.OLN_sam.adapter.sam_vit_adapter'],
    allow_failed_imports=False)

# SAM ViT-B with adapter
model = dict(
    type='FasterRCNN',
    backbone=dict(
        _delete_=True,
        type='SAMViTAdapter',
        arch='base',
        patch_size=16,
        drop_path_rate=0.1,
        out_indices=(3, 5, 7, 11),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='/home/neel/data/Code/MMOln-ssos/mmdetection/weights/sam_vit_b_01ec64.pth')),
    neck=dict(
        type='FPN',
        in_channels=[768, 768, 768, 768],
        out_channels=256,
        num_outs=5),
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
        loss_objectness=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='OLNRoIHead',
        bbox_head=dict(
            type='Shared2FCBBoxScoreHead',
            num_classes=1,
            loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0),
            bbox_score_type='BoxIoU',
            loss_bbox_score=dict(type='L1Loss', loss_weight=1.0))),
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
                pos_fraction=1.,
                neg_pos_ub=-1,
                add_gt_as_proposals=False)),
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
            nms_thr=0.0,
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.00,
            nms=dict(type='nms', iou_threshold=0.7),
            max_per_img=1500))
)

dataset_type = 'CocoSplitDataset'
train_dataloader = dict(
    batch_size=2,  # Updated to 2
    dataset=dict(
        type=dataset_type,
        is_class_agnostic=True,
        train_class='voc',
        eval_class='nonvoc'))
val_dataloader = dict(
    batch_size=2,
    dataset=dict(
        type=dataset_type,
        is_class_agnostic=True,
        train_class='voc',
        eval_class='nonvoc'))
test_dataloader = val_dataloader

train_cfg = dict(max_epochs=8)

val_evaluator = dict(type='CocoSplitMetric')
test_evaluator = dict(type='CocoSplitMetric')

param_scheduler = [
    dict(type='LinearLR', start_factor=0.02, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', begin=0, end=8, by_epoch=True, milestones=[6, 7], gamma=0.1)
]

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001))

fp16 = dict(loss_scale='dynamic')

default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=2))