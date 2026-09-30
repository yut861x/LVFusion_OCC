"""Twenty-frame diagnostic overfit for observed-only ray-free OCC."""
_base_ = ['./hasco_occ_fusion_20260321_rayfree.py']

data_root = __import__('os').environ.get(
    'LVFUSION_OCC_DATA_ROOT',
    'data/20260321-10-45-48/',
).rstrip('/') + '/'

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        data_root=data_root,
        ann_file=data_root + 'hasco_occ_rayfree_infos_overfit20.pkl',
        load_interval=1,
    ),
    val=dict(data_root=data_root,
             ann_file=data_root + 'hasco_occ_rayfree_infos_overfit20.pkl'),
    test=dict(data_root=data_root,
              ann_file=data_root + 'hasco_occ_rayfree_infos_overfit20.pkl'),
)

# The subset deliberately covers every supervised class ID 1..11.
# A fixed schedule makes loss/accuracy trends easy to interpret.
runner = dict(type='EpochBasedRunner', max_epochs=30)
lr_config = dict(_delete_=True, policy='fixed')
checkpoint_config = dict(interval=5, max_keep_ckpts=3)
log_config = dict(interval=1, hooks=[dict(type='TextLoggerHook')])
evaluation = dict(interval=1000000)
load_from = None
resume_from = None
