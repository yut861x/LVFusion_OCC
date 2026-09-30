"""33-degree LiDAR + camera, Z7 OCC with trainable unknown and ignored class 12."""
_base_ = ['./hasco_occ_fusion_r18_105.6m_38.4m_13c_z7.py']

data_root = __import__('os').environ.get(
    'LVFUSION_OCC_DATA_ROOT',
    'data/20260321-10-45-48/',
).rstrip('/') + '/'

# Label states: 0 unknown, 1..11 semantic/free, 12 ignore. Head predicts 0..11.
# Capped square-root inverse-frequency weights over all 966 ray-free NPZs.
model = dict(occ_head=dict(class_weight=[
    0.605784,
    6.228538, 4.558550, 7.000000, 7.000000, 3.204244,
    7.000000, 7.000000, 7.000000, 7.000000, 3.919056, 1.237437,
]))

data = dict(
    train=dict(data_root=data_root,
               ann_file=data_root + 'hasco_occ_rayfree_13c_z7_infos_train.pkl'),
    val=dict(data_root=data_root,
             ann_file=data_root + 'hasco_occ_rayfree_13c_z7_infos_val.pkl'),
    test=dict(data_root=data_root,
              ann_file=data_root + 'hasco_occ_rayfree_13c_z7_infos_val.pkl'),
)
