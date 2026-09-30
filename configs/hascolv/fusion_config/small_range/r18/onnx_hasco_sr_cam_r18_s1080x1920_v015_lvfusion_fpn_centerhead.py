_base_ = ['./hasco_sr_cam_r18_s1080x1920_v015_lvfusion_fpn_centerhead.py']

# ONNX 友好：避免 SyncBN
model = dict(
    img_backbone=dict(norm_cfg=dict(type='BN', requires_grad=True)),
    img_neck=dict(norm_cfg=dict(type='BN', requires_grad=True)),
    img_neck_3d=dict(norm_cfg=dict(type='BN', requires_grad=True)),
)

# 导出/验证场景：不在训练中做评估
evaluation = dict(interval=40)

learning_rate = 1e-4
optimizer = dict(
    type='Adam',
    lr=learning_rate,
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1, decay_mult=1.0),
        }))

# 默认不指定预训练；导出时可在命令行/脚本中指定 checkpoint
load_from = None

