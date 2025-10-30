_base_ = [
    'mmdet::_base_/models/faster-rcnn_r50_fpn.py',
    '/home3/qljx17/MMOln-ssos/mmdetection/projects/ViTDet/configs/lsj-100e_coco-instance.py',
]

custom_imports = dict(imports=['projects.oln-ssos.oln-ssos',
                               'projects.OLN.oln',
                               'projects.ViTDet.vitdet',
                               'projects.OLN_ViTDet.hooks.freeze_backbone_hook'])

backbone_norm_cfg = dict(type='LN', requires_grad=True)
norm_cfg = dict(type='ViTDetLN2d', requires_grad=True)
image_size = (640, 640)
batch_augments = [
    dict(type='BatchFixedSizePad', size=image_size)
]

# model settings
model = dict(
    data_preprocessor=dict(pad_size_divisor=32, batch_augments=batch_augments),
    backbone=dict(
        _delete_=True,
        type='ViT',
        img_size=640,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        drop_path_rate=0.1,
        window_size=14,
        mlp_ratio=4,
        qkv_bias=True,
        norm_cfg=backbone_norm_cfg,
        window_block_indexes=[
            0,
            1,
            3,
            4,
            6,
            7,
            9,
            10,
        ],
        use_rel_pos=True,
        init_cfg=dict(
            type='Pretrained', checkpoint='/home3/qljx17/MMOln-ssos/mmdetection/weights/mae_pretrain_vit_base.pth')),
    neck=dict(
        _delete_=True,
        type='SimpleFPN',
        backbone_channel=768,
        in_channels=[192, 384, 768, 768],
        out_channels=256,
        num_outs=5,
        norm_cfg=norm_cfg),
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
        start_epoch=3,
        logistic_regression_hidden_dim=512,
        negative_sampling_size=10000,
        bottomk_epsilon_dist=1,
        ood_loss_weight=0.1,
        pseudo_label_loss_weight=1.,
        k=30,
        repeat_ood_sampling=1,
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
dataset_type = 'SSOSDB6SplitDataset'
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotationsWithAnnID', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomResize',
        scale=image_size,
        ratio_range=(0.1, 2.0),
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_type='absolute_range',
        crop_size=image_size,
        recompute_bbox=True,
        allow_negative_crop=False),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type='Pad', size=image_size, pad_val=dict(img=(114, 114, 114))),
    dict(type='PackPseudoLabelDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=image_size, keep_ratio=True),
    dict(type='Pad', size=image_size, pad_val=dict(img=(114, 114, 114))),
    dict(type='LoadAnnotationsWithAnnID', with_bbox=True),
    dict(
        type='PackPseudoLabelDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]




data_root = '/home2/projects/datasets/dbf6/'
train_dataloader = dict(
    batch_size=4,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        serialize_data=False,
        is_class_agnostic=True,
        train_class='all',
        eval_class='all',
        data_root=data_root,
        data_prefix=dict(img=data_root + 'images/'),
        ann_file=data_root + 'annotations/train_db6_no_firearms.json',
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        is_class_agnostic=True,
        serialize_data=False,
        train_class='all',
        eval_class='all',
        pipeline=test_pipeline,
        data_root=data_root,
        data_prefix=dict(img=data_root + 'images/'),
        ann_file=data_root + 'annotations/test_db6_no_firearms.json',
    ))


test_dataloader = val_dataloader

val_evaluator = dict(
    type='PseudoLabelDBF6SplitMetric',
    ann_file=data_root + 'annotations/test_db6_no_firearms.json',
    train_class='all',
    eval_class='all',
    metric=['bbox']
)

test_evaluator = val_evaluator

load_from = '/home3/qljx17/MMOln-ssos/mmdetection/work_dirs/oln_vitdet_box_vit-b-mae_lsj-100e_dbf6/iter_184375.pth'

optim_wrapper = dict(
    type='AmpOptimWrapper',
    constructor='LayerDecayOptimizerConstructor',
    paramwise_cfg={
        'decay_rate': 0.7,
        'decay_type': 'layer_wise',
        'num_layers': 12,
    },
    optimizer=dict(
        type='AdamW',
        lr=6.25e-06,  # Scaled from 0.0001 for batch size 4
        betas=(0.9, 0.999),
        weight_decay=0.1,
    ))


custom_hooks = [dict(type='Fp16CompresssionHook')]


default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=2))

custom_hooks = [dict(type='PseudoLabelClusteringHook', calculate_pseudo_labels_from_epoch=1)]