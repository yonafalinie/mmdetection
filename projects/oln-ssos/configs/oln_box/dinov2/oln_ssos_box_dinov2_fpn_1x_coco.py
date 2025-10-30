# from projects.OLN.oln.coco_split import CocoSplitDataset

_base_ = [
    'mmdet::_base_/models/faster-rcnn_r50_fpn.py',
    'mmdet::_base_/datasets/coco_detection.py',
    'mmdet::_base_/schedules/schedule_1x.py', 'mmdet::_base_/default_runtime.py'
]

norm_cfg = dict(type='LN2d', requires_grad=True)


custom_imports = dict(
    imports=['projects.OLN.oln',
             'projects.oln-ssos.oln-ssos',
             'projects.OLN_dinov2.vit_dinov2_backbone'],
               allow_failed_imports=False)

model = dict(
    type='FasterRCNN',
backbone=dict(
        _delete_=True,
        type='DINOv2',
        img_size=518,
        patch_size=14,
        embed_dim=768,
        depth=12,
        num_heads=12,
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
            checkpoint='/home3/qljx17/MMOln-ssos/mmdetection/weights/vit-base-p14_'\
                       'dinov2-pre_3rdparty_20230426-ba246503.pth'
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
        type='OLNKMeansVOSRoIHead',
        start_epoch=12,
        logistic_regression_hidden_dim=512,
        negative_sampling_size=10000,
        bottomk_epsilon_dist=1,
        ood_loss_weight=0.1,
        pseudo_label_loss_weight=1.,
        k=100,
        repeat_ood_sampling=4,
        pseudo_bbox_roi_extractor=dict(
                     type='SingleRoIExtractor',
                     roi_layer=dict(type='RoIAlign', output_size=3, sampling_ratio=0),
                     out_channels=256,
                     featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='OODShared2FCBBoxScoreHead',
            num_classes=1,
            reg_class_agnostic=True,
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
            max_per_img=1500,
            ood_threshold=0.)
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    ))


backend_args = None
dataset_type = 'PseudoLabelCocoSplitDataset'
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotationsWithAnnID', with_bbox=True),
    dict(type='RandomChoiceResize',
         scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                   (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                   (736, 1333), (768, 1333), (800, 1333)],
        keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackPseudoLabelDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]


data_root = '/home2/projects/datasets/VOC/VOC_0712_converted/'
train_dataloader = dict(
    batch_size=2,
    persistent_workers=False,
    dataset=dict(
        type=dataset_type,
        serialize_data=False,
        is_class_agnostic=True,
        train_class='all',
        eval_class='all',
        data_root=data_root,
        data_prefix=dict(img='JPEGImages/'),
        ann_file='voc0712_train_all.json',
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=2,
    dataset=dict(
        type=dataset_type,
        is_class_agnostic=True,
        serialize_data=False,
        train_class='all',
        eval_class='all',
        pipeline=test_pipeline,
        data_root=data_root,
        data_prefix=dict(img='JPEGImages/'),
        ann_file='val_coco_format.json',
    ))
test_dataloader = val_dataloader

train_cfg = dict(max_epochs=18)

val_evaluator = dict(
    type='PseudoLabelCocoSplitMetric',
    ann_file=data_root + 'val_coco_format.json',
    train_class='all',
    eval_class='all',
)
test_evaluator = val_evaluator


# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.02, by_epoch=False, begin=0, end=4000),
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
    optimizer=dict(type='SGD', lr=0.000625, momentum=0.9, weight_decay=0.0001))
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=2))
custom_hooks = [dict(type='PseudoLabelClusteringHook', calculate_pseudo_labels_from_epoch=0)]