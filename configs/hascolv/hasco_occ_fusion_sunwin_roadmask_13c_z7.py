"""Sunwin road-supported Ray-Free Z7 OCC labels (279 frames)."""
_base_ = ['./hasco_occ_fusion_20260321_rayfree_13c_z7.py']

data_root = __import__('os').environ.get(
    'LVFUSION_OCC_DATA_ROOT',
    '../../data/sunwin_bevdata_20260807',
).rstrip('/') + '/'

data = dict(
    train=dict(data_root=data_root,
               ann_file=data_root + 'hasco_occ_rayfree_13c_z7_roadmask_infos_train.pkl'),
    val=dict(data_root=data_root,
             ann_file=data_root + 'hasco_occ_rayfree_13c_z7_roadmask_infos_val.pkl'),
    test=dict(data_root=data_root,
              ann_file=data_root + 'hasco_occ_rayfree_13c_z7_roadmask_infos_val.pkl'),
)
