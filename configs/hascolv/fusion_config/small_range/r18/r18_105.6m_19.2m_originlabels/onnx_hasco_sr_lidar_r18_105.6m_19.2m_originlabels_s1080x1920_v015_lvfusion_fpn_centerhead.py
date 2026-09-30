_base_ = ['./hasco_sr_lidar_r18_105.6m_19.2m_originlabels_s1080x1920_v015_lvfusion_fpn_centerhead.py']

# 导出 ONNX 时强制使用标准 FPN（而不是训练时的 FPNSafe）
model = dict(
    pts_neck=dict(type='FPN'),
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
            'pts_backbone': dict(lr_mult=0.1, decay_mult=1.0),
        }))

# 默认不指定预训练；导出时可在命令行/脚本中指定 checkpoint
load_from = None

