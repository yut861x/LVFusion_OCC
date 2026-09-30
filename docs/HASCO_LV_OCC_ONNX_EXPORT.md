# HASCO LVFusion OCC ONNX 导出说明

本文档说明二维 BEV 融合语义 OCC 的三段式 ONNX 导出方案。OCC 使用独立脚本 `tools/export_lvfusion_occ_onnx.py`。

## 1. 模型拆分

### Stage 1：图像二维特征

- 文件：`export_2d_model.onnx`
- 输入：`img [B,3,H,W]`
- 输出：`img_feat_2d [B,C,Hf,Wf]`
- 网络：`img_backbone + img_neck + img_neck_fuse`

### 外置反投影

Stage 1 的二维图像特征根据标定参数反投影为 `img_volume [B,64,704,256,6]`。反投影不放入 ONNX，相关索引和参数由导出脚本一并生成。

### Stage 2：点云 BEV 特征

- 文件：`export_lidar_model.onnx`
- 输入：`pts_spatial_features [B,64,256,704]`
- 输出：`pts_bev_feat [B,256,128,352]`
- 网络：`pts_backbone + pts_neck + pts_neck_fuse`
- 体素化、VFE、Scatter 默认外置；加 `--include-vfe` 可额外生成合并 VFE 的模型。

### Stage 3：图像/点云融合与 OCC

- 文件：`export_occ_model.onnx`
- 输入 1：`img_volume [B,64,704,256,6]`
- 输入 2：`pts_bev_feat [B,256,128,352]`
- 输出：`occ_logits [B,12,7,128,133]`
- 网络：`img_neck_3d + resize + fusion_layer + BEVOccHead`

输出维度顺序为 `B,K,Z,Y,X`。部署端使用 `argmax(occ_logits, axis=1)` 得到类别 ID。

## 2. 当前 Z7 模型语义

- OCC 范围：`x=[0,39.9] m, y=[-19.2,19.2] m, z=[-1.0,1.8] m`
- OCC 体素：`[0.3,0.3,0.4] m`
- 输出尺寸：`Z=7, Y=128, X=133`
- 输出类别 0～11：
  - 0 unknown
  - 1 truck
  - 2 large_vehicle
  - 3 construction_vehicle
  - 4 bus
  - 5 car
  - 6 motorcycle
  - 7 bicycle
  - 8 tricycle
  - 9 pedestrian
  - 10 other
  - 11 free
- 标签 12 是 `ignore`，只用于损失屏蔽，不是网络输出通道。

## 3. 服务器导出命令

```bash
cd $LVFUSION_OCC_ROOT

docker exec \
  -e LVFUSION_OCC_DATA_ROOT=$LVFUSION_OCC_DATA_ROOT \
  -e PYTHONPATH=/workspace/LVFusion_OCC \
  -w /workspace/LVFusion_OCC \
  lvfusion_occ \
  /home/ubuntu/miniconda3/bin/python tools/export_lvfusion_occ_onnx.py \
  --config configs/hascolv/hasco_occ_fusion_20260321_rayfree_13c_z7.py \
  --checkpoint work_dirs/hasco_occ_rayfree_13c_z7_weighted_24e/epoch_24.pth \
  --out-dir output/lvfusion_occ_onnx_z7 \
  --stages 1 2 3 \
  --sample-idx 0 \
  --no-simplify
```

只验证 OCC 融合 Head：

```bash
# 上述命令改为：
--stages 3 --no-export-backproj-params --no-simplify
```

合并 VFE：

```bash
# 在完整导出命令末尾增加：
--include-vfe
```

## 4. 导出产物

`output/lvfusion_occ_onnx_z7/` 中包含：

- `export_2d_model.onnx`
- `export_lidar_model.onnx`
- `export_occ_model.onnx`
- `occ_metadata.json`
- `backprojection_params_sample0.npz`
- `featurePointLength.bin`
- `gather_indices.bin`
- `scatter_indices.bin`
- `index/valid.npy`、`x.npy`、`y.npy`、`volume.npy`

若使用 `--include-vfe`，还会生成 `export_lidar_vfe_model.onnx` 和 `voxel_params.npz`。

## 5. 校验

脚本在每一阶段导出后执行 `onnx.checker.check_model`。本次服务器实测 OCC ONNX Runtime 输出为：

```text
input  img_volume   [1,64,704,256,6]
input  pts_bev_feat [1,256,128,352]
output occ_logits   [1,12,7,128,133]
dtype               float32
finite               True
```

注意：ONNX 输出是原始 logits。33° LiDAR 有效射线范围、遮挡区屏蔽等后处理应在 `argmax` 后按部署侧有效区掩码处理，不应写死在 OCC Head 图内。
