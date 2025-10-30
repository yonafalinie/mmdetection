_base_ = [
    'mmdet::_base_/models/faster-rcnn_r50_fpn.py',
    'mmdet::_base_/datasets/coco_detection.py',
    'mmdet::_base_/schedules/schedule_1x.py', 'mmdet::_base_/default_runtime.py'
]

custom_imports = dict(
    imports=['projects.OLN.oln'], allow_failed_imports=False)

pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'  # noqa
model = dict(
    type='FasterRCNN',
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(in_channels=[96, 192, 384, 768]),
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


dataset_type = 'CocoSplitDataset'
data_root = '/home2/projects/datasets/coco/'
train_dataloader = dict(
    batch_size=5,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'annotations/instances_train2017.json',
        data_prefix=dict(img=data_root + 'train2017/'),
        is_class_agnostic=True,
        dataset_name='coco',
        train_class='voc',
        eval_class='nonvoc'))
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_root=data_root,
        ann_file=data_root + 'annotations/instances_val2017.json',
        data_prefix=dict(img=data_root + 'val2017/'),
        type=dataset_type,
        is_class_agnostic=True,
        dataset_name='coco',
        train_class='voc',
        eval_class='nonvoc'))
test_dataloader = val_dataloader


max_epochs = 40
train_cfg = dict(max_epochs=max_epochs)

val_evaluator = dict(
    type='CocoSplitMetric',
    ann_file=data_root + 'annotations/instances_val2017.json',)
test_evaluator = dict(
    type='CocoSplitMetric',
    ann_file=data_root + 'annotations/instances_val2017.json',)



# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0,
        end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[8, 11],
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
        weight_decay=0.05))
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=2))