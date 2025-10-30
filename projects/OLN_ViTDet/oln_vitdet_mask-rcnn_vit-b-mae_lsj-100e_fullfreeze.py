_base_ = [
    'mmdet::_base_/models/mask-rcnn_r50_fpn.py',
    '/home3/qljx17/MMOln-ssos/mmdetection/projects/ViTDet/configs/lsj-100e_coco-instance.py',
]

custom_imports = dict(imports=['projects.OLN.oln',
                               'projects.ViTDet.vitdet',
                               'projects.OLN_ViTDet.hooks.freeze_backbone_hook'])

backbone_norm_cfg = dict(type='LN', requires_grad=True)
norm_cfg = dict(type='ViTDetLN2d', requires_grad=True)
image_size = (1024, 1024)
batch_augments = [
    dict(type='BatchFixedSizePad', size=image_size, pad_mask=True)
]

# model settings
model = dict(
    data_preprocessor=dict(pad_size_divisor=32, batch_augments=batch_augments),
    backbone=dict(
        _delete_=True,
        type='ViT',
        img_size=1024,
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
        num_convs=2,
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
            conv_out_channels=256,
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


dataset_type = 'CocoSplitDataset'
train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        type=dataset_type,
        is_class_agnostic=True,
        train_class='voc',
        eval_class='nonvoc'))
# Disable validation
val_dataloader = None
test_dataloader = None
val_evaluator = None
test_evaluator = None

# Optimizer settings (unchanged)
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

# Training configuration - override lsj-100e_coco-instance.py
train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=184375,  # From lsj-100e_coco-instance.py
    val_interval=1000000  # Effectively disable validation
)
val_cfg = None
test_cfg = None




custom_hooks = [dict(type='Fp16CompresssionHook'),
        dict(
        type='FreezeBackboneHook',
        priority='HIGH')  # Run early
]


default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=2))
