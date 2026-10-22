"""Train the V4 Z7 OCC model on selected scene directories."""
_base_ = ['./hasco_occ_fusion_20260321_rayfree_13c_z7.py']

# Point this environment variable at the parent data/ directory on each host.
data_parent = __import__('os').environ.get('LVFUSION_OCC_DATA_ROOT', '../../data')

# Add a scene directory here after generating its V4 OCC and train/val PKLs.
# Entries may be relative to data_parent or absolute paths.
training_data_paths = [
    '20260321-10-45-48',
    '20260821_gtlidar_marvelr',
    '20260821_s13ouster_marvelr',
    '20260821_s47s13_marvelr',
    '20260903_marvelr_gtlidar',
    'sunwin_bevdata_20260807',
]

dataset_roots = [
    (path if __import__('os').path.isabs(path)
     else __import__('os').path.join(data_parent, path)).rstrip('/') + '/'
    for path in training_data_paths
]
info_prefix = 'hasco_occ_rayfree_13c_z7_infos_'

train_pipeline = {{ _base_.train_pipeline }}
val_pipeline = {{ _base_.val_pipeline }}
class_names = {{ _base_.class_names }}
input_modality = {{ _base_.input_modality }}

data = dict(
    train=dict(
        _delete_=True,
        type='ConcatDataset',
        datasets=[
            dict(type='HascoDataset', data_root=root,
                 ann_file=root + info_prefix + 'train.pkl',
                 pipeline=train_pipeline, classes=class_names,
                 modality=input_modality, test_mode=False,
                 box_type_3d='LiDAR', filter_empty_gt=False, load_interval=1)
            for root in dataset_roots
        ],
    ),
    val=dict(
        _delete_=True,
        type='ConcatDataset',
        datasets=[
            dict(type='HascoDataset', data_root=root,
                 ann_file=root + info_prefix + 'val.pkl',
                 pipeline=val_pipeline, classes=class_names,
                 modality=input_modality, test_mode=True,
                 box_type_3d='LiDAR', filter_empty_gt=False, load_interval=1)
            for root in dataset_roots
        ],
    ),
    test=dict(
        _delete_=True,
        type='ConcatDataset',
        datasets=[
            dict(type='HascoDataset', data_root=root,
                 ann_file=root + info_prefix + 'val.pkl',
                 pipeline=val_pipeline, classes=class_names,
                 modality=input_modality, test_mode=True,
                 box_type_3d='LiDAR', filter_empty_gt=False, load_interval=1)
            for root in dataset_roots
        ],
    ),
)
