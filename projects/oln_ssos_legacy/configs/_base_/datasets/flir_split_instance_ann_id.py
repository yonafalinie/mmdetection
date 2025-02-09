dataset_type = 'FLIRSplitDataset'
data_root = '/home2/projects/datasets/FLIR_ADAS_v2'

custom_imports = dict(
    imports=[
        'vos.datasets.vos_coco',
        'vos.datasets.pipelines.loading',
        'vos.datasets.pipelines.formating'
    ],
    allow_failed_imports=False)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotationsWithAnnID', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(512, 640), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='PseudoLabelFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_ann_ids', 'gt_pseudo_labels',
                               'gt_weak_bboxes', 'gt_weak_bboxes_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 640),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + '/images_thermal_train/train_flir_ID.json',
        img_prefix=data_root + '/images_thermal_train/',
        is_class_agnostic=True,
        train_class='id',
        eval_class='id',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + '/images_thermal_val/val_flir_ID.json',
        img_prefix=data_root + '/images_thermal_val/',
        is_class_agnostic=True,
        train_class='id',
        eval_class='id',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + '/images_thermal_val/val_flir_ID.json',
        img_prefix=data_root + '/images_thermal_val/',
        is_class_agnostic=True,
        train_class='id',
        eval_class='id',
        pipeline=test_pipeline))
evaluation = dict(metric=['bbox'])
