
_base_ = [
    'mmdet::_base_/models/faster-rcnn_r50_fpn.py',
    'mmdet::_base_/datasets/coco_detection.py',
    'mmdet::_base_/schedules/schedule_1x.py',
    'mmdet::_base_/default_runtime.py'
]

norm_cfg = dict(type='LN2d', requires_grad=True)

custom_imports = dict(
    imports=['projects.OLN.oln', 
             'projects.ViTDet.vitdet'             
],
    allow_failed_imports=False
)

model = dict(
    type='FasterRCNN',
backbone=dict(
        _delete_=True,
        type='ViT',
        img_size=518,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        drop_path_rate=0.1,
        window_size=0,
        mlp_ratio=4,
        qkv_bias=True,
        norm_cfg=dict(type='LN'),
        window_block_indexes=[], 
        use_abs_pos=True, 
        use_rel_pos=False,
        pretrain_img_size=384,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='/home3/qljx17/MMOln-ssos/mmdetection/weights/clip-vit-base-p16_laion2b-pre_3rdparty_in1k-384px_20221220-558ed826.pth'
             )),
    neck=dict(
        _delete_=True,
        type='SimpleFPN',
        backbone_channel=768,
        in_channels=[192, 384, 768, 768],
        out_channels=256,
        num_outs=5,
        norm_cfg=norm_cfg,
    ),
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
data_root = '/home2/projects/datasets/dbf6/'
train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'annotations/dbf6_train.json',
        data_prefix=dict(img=data_root + 'images/'),
        is_class_agnostic=True,
        dataset_name='dbf6',
        train_class='voc',
        eval_class='nonvoc'))
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
        eval_class='nonvoc'))
test_dataloader = val_dataloader



val_evaluator = dict(
    type='DBF6SplitMetric',
    ann_file=data_root + 'annotations/dbf6_test.json',)
test_evaluator = dict(
    type='DBF6SplitMetric',
    ann_file=data_root + 'annotations/dbf6_test.json',)

train_cfg = dict(max_epochs=16)


param_scheduler = [
    dict(type='LinearLR', start_factor=0.02, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', begin=0, end=16, by_epoch=True, milestones=[6, 7], gamma=0.1)
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