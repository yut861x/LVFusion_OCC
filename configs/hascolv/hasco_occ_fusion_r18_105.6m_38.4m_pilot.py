"""Pilot config: LVFusion 2D BEV -> height-aware semantic OCC."""
_base_ = [
    './fusion_config/small_range/r18/r18_105.6m_38.4m_aug/'
    'hasco_sr_fusion_r18_105.6m_38.4m_aug_s1080x1920_v015_lvfusion_fpn_centerhead.py'
]

point_cloud_range = [0, -19.2, -5, 105.6, 19.2, 4]
occ_point_cloud_range = [0, -19.2, -1.8, 39.9, 19.2, 1.8]
occ_voxel_size = [0.3, 0.3, 0.4]
class_names = [
    'truck', 'large_vehicle', 'construction_vehicle', 'bus', 'car',
    'motorcycle', 'bicycle', 'tricycle', 'pedestrian',
]
data_root = 'data/20260903_marvelr_gtlidar/'
file_client_args = dict(backend='disk')
input_modality = dict(use_lidar=True, use_camera=True, use_radar=False,
                      use_map=False, use_external=False)
data_config = dict(
    src_size=(2160, 3840), input_size=(1080, 1920),
    resize=(-0.06, 0.11), crop=(-0.05, 0.05), rot=(-5.4, 5.4), flip=True,
    test_input_size=(1080, 1920), test_resize=0.0, test_rotate=0.0,
    test_flip=False, pad=(0, 0, 0, 0), pad_divisor=32,
    pad_color=(0, 0, 0))
image_aug_config = dict(
    brightness_delta=24, contrast_range=(0.8, 1.2),
    saturation_range=(0.8, 1.2), hue_delta=8)
image_blur_noise_config = dict(
    blur_prob=0.15, noise_prob=0.15, blur_kernel_sizes=(3, 5),
    gaussian_noise_std=(2.0, 6.0))
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375], to_rgb=True)

model = dict(
    pts_bbox_head=None,
    occ_head=dict(
        type='BEVOccHead', in_channels=256, hidden_channels=256,
        num_classes=12, num_z=9, output_size=[128, 133], loss_weight=1.0,
        ignore_index=255, score_threshold=0.0, unknown_sample_ratio=3.0),
)

# Spatial transforms that do not also transform OCC are intentionally absent.
train_pipeline = [
    dict(type='LoadFrontImageFromFile', file_client_args=file_client_args),
    dict(type='LoadPointsFromPCD', dummy=False, coord_type='LIDAR',
         load_dim=4, use_dim=[0, 1, 2, 3], load_id=True,
         file_client_args=file_client_args),
    dict(type='PointsIDFilter', keep_ids=[1]),
    dict(type='LoadOccupancyFromFile', file_client_args=file_client_args,
         expected_range=occ_point_cloud_range, expected_voxel_size=occ_voxel_size),
    dict(type='RandomAugFrontImage', data_config=data_config),
    dict(type='PhotoMetricDistortionMultiViewImage', **image_aug_config),
    dict(type='RandomImageBlurNoise', **image_blur_noise_config),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointsEmptyCheck', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(type='KittiSetOrigin', point_cloud_range=point_cloud_range),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['img', 'points', 'occ_gt', 'occ_valid_mask']),
]

val_pipeline = [
    dict(type='LoadFrontImageFromFile', file_client_args=file_client_args),
    dict(type='LoadPointsFromPCD', dummy=False, coord_type='LIDAR',
         load_dim=4, use_dim=[0, 1, 2, 3], load_id=True,
         file_client_args=file_client_args),
    dict(type='PointsIDFilter', keep_ids=[1]),
    dict(type='LoadOccupancyFromFile', file_client_args=file_client_args,
         expected_range=occ_point_cloud_range, expected_voxel_size=occ_voxel_size),
    dict(type='RandomAugFrontImage', data_config=data_config, is_train=False),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='KittiSetOrigin', point_cloud_range=point_cloud_range),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
    dict(type='Collect3D', keys=['img', 'points', 'occ_gt', 'occ_valid_mask']),
]

data = dict(
    samples_per_gpu=1, workers_per_gpu=2,
    train=dict(
        _delete_=True, type='HascoDataset', data_root=data_root,
        ann_file=data_root + 'hasco_occ_infos_train.pkl', pipeline=train_pipeline,
        classes=class_names, modality=input_modality, test_mode=False,
        box_type_3d='LiDAR', filter_empty_gt=False, load_interval=1),
    val=dict(
        _delete_=True, type='HascoDataset', data_root=data_root,
        ann_file=data_root + 'hasco_occ_infos_val.pkl', pipeline=val_pipeline,
        classes=class_names, modality=input_modality, test_mode=True,
        box_type_3d='LiDAR', filter_empty_gt=False, load_interval=1),
    test=dict(
        _delete_=True, type='HascoDataset', data_root=data_root,
        ann_file=data_root + 'hasco_occ_infos_val.pkl', pipeline=val_pipeline,
        classes=class_names, modality=input_modality, test_mode=True,
        box_type_3d='LiDAR', filter_empty_gt=False, load_interval=1),
)

# Detection evaluation cannot consume OCC predictions; mIoU is a later task.
evaluation = dict(interval=1000000)
load_from = None
