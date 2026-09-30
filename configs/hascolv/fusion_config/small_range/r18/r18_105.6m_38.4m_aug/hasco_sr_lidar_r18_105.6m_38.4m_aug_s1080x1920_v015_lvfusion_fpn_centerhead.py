# ============ Hyperparameters ============
# Point cloud range: [x_min, y_min, z_min, x_max, y_max, z_max]
point_cloud_range = [0, -19.2, -5, 105.6, 19.2, 4]

# Point cloud voxel size: [vx, vy, vz]
# For PointPillars (pts branch): vz=9 means Z dimension is not divided (pillar is 2D)
pts_voxel_size = [0.15, 0.15, 9]

# Image voxel size: [vx, vy, vz]
# For Voxel (img branch): vz will be used to divide Z dimension
img_voxel_size = [0.15, 0.15, 1.5]

# Calculate n_voxels for image backprojection: [X, Y, Z]
# X=(x_max-x_min)/vx, Y=(y_max-y_min)/vy, Z=(z_max-z_min)/vz
n_voxels = [
    int((point_cloud_range[3] - point_cloud_range[0]) / img_voxel_size[0]),  # X
    int((point_cloud_range[4] - point_cloud_range[1]) / img_voxel_size[1]),  # Y
    int((point_cloud_range[5] - point_cloud_range[2]) / img_voxel_size[2])   # Z
]
# ==========================================

model = dict(
    type='LVFusion',
    # Point cloud branch (from hasco_hascopts_fpn_centerhead.py)
    pts_voxel_layer=dict(
        max_num_points=64,
        point_cloud_range=point_cloud_range,
        voxel_size=pts_voxel_size,
        max_voxels=(20000, 30000)),
    pts_voxel_encoder=dict(
        type='HardVFE',
        in_channels=4,
        feat_channels=[64, 64],
        with_distance=False,
        voxel_size=pts_voxel_size,
        with_cluster_center=True,
        with_voxel_center=True,
        point_cloud_range=point_cloud_range,
        norm_cfg=dict(type='naiveSyncBN1d', eps=1e-3, momentum=0.01)),
    pts_middle_encoder=dict(
        type='PointPillarsScatter', in_channels=64, output_shape=[256, 704]),  # [Y, X] format: Y=256, X=704
    pts_backbone=dict(
        type='HASCO_PTS_BACKBONE',
        in_channels=64,
        out_channels=[64, 128, 256],
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],  # 352*128, 176*64, 88*32
        norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
    ),
    pts_neck=dict(
        type='FPNWithCP',
        norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
        act_cfg=dict(type='ReLU', inplace=False),
        in_channels=[64, 128, 256],
        out_channels=256,
        start_level=0,
        num_outs=3),
    pts_neck_fuse=dict(
        in_channels=[256, 256, 256],  # FPN outputs 3 feature maps with 256 channels each
        out_channels=256,
        kernel_size=3,
        padding=1,
        multi_scale_id=[0]),  # Use level 0 (largest) feature map size as target
    
    # Fusion layer (combines img and pts features)
    fusion_layer=dict(
        in_channels=256,  # pts_feat (256) + img_feat (256) = 512
        out_channels=256),  # Output 256 channels
    
    # Bbox head
    pts_bbox_head=dict(
        type='CenterHead',
        in_channels=256,  # After fusion

        tasks=[
            dict(num_class=3, class_names=['truck', 'large_vehicle', 'construction_vehicle']),  # 任务0：大型车辆类
            dict(num_class=1, class_names=['bus']),  # 任务1：公交车类
            dict(num_class=1, class_names=['car']),  # 任务2：小汽车类
            dict(num_class=3, class_names=['motorcycle', 'bicycle', 'tricycle']),  # 任务3：非机动车类
            dict(num_class=1, class_names=['pedestrian']),  # 任务4：行人类
        ],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            pc_range=[0, -19.2],  # point_cloud_range 的前两个值 (x_min, y_min)
            post_center_range=point_cloud_range,  # [0, -19.2, -5, 105.6, 19.2, 4]
            max_num=500,
            score_threshold=0.1,
            out_size_factor=2,  # Target feature map is 352x128, grid_size [704, 256] / out_size_factor = [352, 128]
            voxel_size=pts_voxel_size[:2],  # [0.15, 0.15]
            code_size=9),
        separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True),
    
    # Image backprojection parameters (Voxel form, not PointPillars)
    n_voxels=[n_voxels],  # [X, Y, Z] - calculated from point_cloud_range and img_voxel_size
    voxel_size=[img_voxel_size],  # [vx, vy, vz] for image branch
    point_cloud_range=point_cloud_range,
    backproject='inplace',
    
    # Training and testing settings
    train_cfg=dict(
        pts=dict(
            grid_size=[704, 256, 1],  # PointPillarsScatter output shape (2D pillar, Z=1)
            voxel_size=pts_voxel_size,  # [0.15, 0.15, 9] - point cloud voxel_size
            point_cloud_range=point_cloud_range,  # [0, -19.2, -5, 105.6, 19.2, 4]
            out_size_factor=2,  # Target feature map is 352x128
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3.0, 3.0, 0.0001, 0.0001])),
    test_cfg=dict(
        pts=dict(
            post_center_limit_range=point_cloud_range,  # [0, -19.2, -5, 105.6, 19.2, 4]
            pc_range=[0, -19.2],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 0.85, 0.5, 0.3],  # One value per task (5 tasks)
            score_threshold=0.1,
            out_size_factor=2,  # Target feature map is 352x128
            voxel_size=pts_voxel_size[:2],
            nms_type='rotate',
            pre_max_size=1000,
            post_max_size=83,
            nms_thr=0.2)))

# Hasco dataset classes
# 原有类别配置（已注释）
# class_names = [
#     'truck', 'bus', 'large_vehicle', 'construction_vehicle', 
#     'car', 
#     'motorcycle', 'bicycle', 'tricycle', 
# ]
# 新的类别配置：5个类别，分为3个大类
class_names = [
    'truck', 'large_vehicle', 'construction_vehicle',  # 任务0：大型车辆类
    'bus',  # 任务1：公交车类
    'car',  # 任务2：小汽车类
    'motorcycle', 'bicycle', 'tricycle',  # 任务3：非机动车类型
    'pedestrian',  # 任务4：行人类型
]
# Eval DistanceFilter: max ego_dist (m) per class
class_range = {n: 75 for n in class_names}

dataset_type = 'HascoDataset'
data_root = './data/hasco_ouster/'

input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)
file_client_args = dict(backend='disk')

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)

data_config = {
    'src_size': (2160, 3840),  # Hasco dataset image size
    'input_size': (1080, 1920),
    # train-aug
    'resize': (-0.06, 0.11),
    'crop': (-0.05, 0.05),
    'rot': (-5.4, 5.4),
    'flip': True,
    # test-aug
    'test_input_size': (1080, 1920),
    'test_resize': 0.0,
    'test_rotate': 0.0,
    'test_flip': False,
    # top, right, bottom, left
    'pad': (0, 0, 0, 0),
    'pad_divisor': 32,
    'pad_color': (0, 0, 0),
}
object_noise_config = dict(
    translation_std=[0.10, 0.10, 0.05],
    global_rot_range=[0.0, 0.0],
    rot_range=[-0.0872665, 0.0872665],
    num_try=100)

train_pipeline = [
    dict(type='LoadFrontImageFromFile', file_client_args=file_client_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='LoadPointsFromPCD',
        dummy=False,
        coord_type='LIDAR',
        load_dim=4,
        use_dim=[0, 1, 2, 3],
        load_id=True,
        file_client_args=file_client_args),
    dict(type='PointsIDFilter', keep_ids=[1]),
    dict(type='ObjectNoise', **object_noise_config),
    dict(type='RandomAugFrontImage', data_config=data_config),
    dict(type='RandomShiftPointCloudOrigin', max_shift_y=6.0),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointsEmptyCheck', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='ObjectMinPointsFilter', min_points=5),
    dict(type='PointShuffle'),
    dict(type='KittiSetOrigin', point_cloud_range=point_cloud_range),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['img', 'points', 'gt_bboxes_3d', 'gt_labels_3d'])
]

test_pipeline = [
    dict(type='LoadFrontImageFromFile', file_client_args=file_client_args),
    dict(
        type='LoadPointsFromPCD',
        dummy=False,
        coord_type='LIDAR',
        load_dim=4,
        use_dim=[0, 1, 2, 3],
        load_id=True,
        file_client_args=file_client_args),
    dict(type='PointsIDFilter', keep_ids=[1]),
    dict(type='RandomAugFrontImage', data_config=data_config, is_train=False),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointsEmptyCheck', point_cloud_range=point_cloud_range),
    dict(type='KittiSetOrigin', point_cloud_range=point_cloud_range),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
    dict(type='Collect3D', keys=['img', 'points'])
]

val_pipeline = [
    dict(type='LoadFrontImageFromFile', file_client_args=file_client_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='LoadPointsFromPCD',
        dummy=False,
        coord_type='LIDAR',
        load_dim=4,
        use_dim=[0, 1, 2, 3],
        load_id=True,
        file_client_args=file_client_args),
    dict(type='PointsIDFilter', keep_ids=[1]),
    dict(type='RandomAugFrontImage', data_config=data_config, is_train=False),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointsEmptyCheck', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='ObjectMinPointsFilter', min_points=5),
    dict(type='KittiSetOrigin', point_cloud_range=point_cloud_range),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
    dict(type='Collect3D', keys=['img', 'points', 'gt_bboxes_3d', 'gt_labels_3d'])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=8,
    # train=dict(
    #     type=dataset_type,
    #     data_root=data_root,
    #     ann_file=data_root + 'hasco_infos_train.pkl',
    #     pipeline=train_pipeline,
    #     classes=class_names,
    #     modality=input_modality,
    #     test_mode=False,
    #     box_type_3d='LiDAR',
    #     filter_empty_gt=False,
    #     load_interval=1),
    train=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type=dataset_type,
                data_root=data_root,
                ann_file=data_root + 'hasco_infos_train.pkl',
                pipeline=train_pipeline,
                classes=class_names,
                modality=input_modality,
                test_mode=False,
                box_type_3d='LiDAR',
                filter_empty_gt=False,
                load_interval=1),
            dict(
                type=dataset_type,
                data_root=data_root,
                ann_file=data_root + 'hasco_infos_val.pkl',
                pipeline=train_pipeline,
                classes=class_names,
                modality=input_modality,
                test_mode=False,
                box_type_3d='LiDAR',
                filter_empty_gt=False,
                load_interval=1),
        ]
        ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'hasco_infos_val.pkl',
        pipeline=val_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,  # Set to False to ensure ann_info is loaded for validation
        box_type_3d='LiDAR',
        filter_empty_gt=False,
        load_interval=1,
        point_cloud_range=point_cloud_range,
        class_range=class_range),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'hasco_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR',
        filter_empty_gt=False,
        load_interval=1,
        point_cloud_range=point_cloud_range,
        class_range=class_range))

# evaluation: HASCO 3D detection (no velocity, no attribute, DetectionScore)
evaluation = dict(interval=2, metric='bbox')

# optimizer
# This schedule is mainly used by models on Hasco dataset
learning_rate = 1e-4

# optimizer = dict(type='AdamW', lr=0.001, weight_decay=0.01)
optimizer = dict(type='AdamW2', lr=learning_rate, weight_decay=0.01)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 1000,
    step=[12, 20])
momentum_config = None
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=24)

checkpoint_config = dict(interval=1,max_keep_ckpts=4)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
    ])

dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
load_from = None
resume_from = None
workflow = [('train', 1)]
