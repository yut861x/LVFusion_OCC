"""33-degree LiDAR + camera with observed-only ray-free OCC supervision."""
_base_ = ['./hasco_occ_fusion_r18_105.6m_38.4m_pilot.py']

data_root = __import__('os').environ.get(
    'LVFUSION_OCC_DATA_ROOT',
    'data/20260321-10-45-48/',
).rstrip('/') + '/'

# IDs remain unchanged: 0 is unsupervised, 1..11 are trainable classes.
# Capped square-root inverse-frequency weights over all 966 ray-free NPZs.
model = dict(occ_head=dict(class_weight=[
    0.0,
    2.915992, 2.133440, 3.901826, 6.947091, 1.500053,
    6.947091, 6.947091, 6.947091, 6.947091, 1.832922, 0.578924,
]))

data = dict(
    train=dict(data_root=data_root,
               ann_file=data_root + 'hasco_occ_rayfree_infos_train.pkl'),
    val=dict(data_root=data_root,
             ann_file=data_root + 'hasco_occ_rayfree_infos_val.pkl'),
    test=dict(data_root=data_root,
              ann_file=data_root + 'hasco_occ_rayfree_infos_val.pkl'),
)
