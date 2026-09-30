"""33-degree LiDAR + camera pilot with merged OCC supervision."""
_base_ = ['./hasco_occ_fusion_r18_105.6m_38.4m_pilot.py']

data_root = __import__('os').environ.get(
    'LVFUSION_OCC_DATA_ROOT',
    'data/20260321-10-45-48/',
).rstrip('/') + '/'

data = dict(
    train=dict(data_root=data_root,
               ann_file=data_root + 'hasco_occ_merged_infos_train.pkl'),
    val=dict(data_root=data_root,
             ann_file=data_root + 'hasco_occ_merged_infos_val.pkl'),
    test=dict(data_root=data_root,
              ann_file=data_root + 'hasco_occ_merged_infos_val.pkl'),
)
