# LVFusion-OCC

LVFusion-OCC 基于 Fast-BEV 图像反投影与 PointPillars 点云编码，实现前视图像与33° LiDAR点云的二维BEV融合语义OCC。当前输出为12个预测类别、7个高度层，覆盖车辆前方40 m。


## 快速开始

请先设置以下路径（可写入 shell 配置或项目外的启动脚本）：

```bash
export LVFUSION_OCC_ROOT=/path/to/LVFusion_OCC
export LVFUSION_OCC_DATA_ROOT=/path/to/20260321-10-45-48
export LVFUSION_ANNOTATION_ROOT=/path/to/lv_annotation_tools
```

当前快速路径只使用**前视图像 + 33° LiDAR**。120° Ouster点云监督方案暂不使用。

### 1. 快速完成地面分割标注

地面分割在本地标注环境执行。修改参数时只编辑 `ground_config.json`：

```bash
conda activate annotation
cd $LVFUSION_ANNOTATION_ROOT

DATA=$LVFUSION_OCC_DATA_ROOT

# 覆盖前备份已有地面标签
if [ -d "$DATA/ground_labels" ]; then
  cp -a "$DATA/ground_labels" "$DATA/ground_labels_backup_$(date +%Y%m%d_%H%M%S)"
fi

# 自动生成全部33°点云逐点地面标签；已有结果时去掉--overwrite可只补缺失帧
python batch_ground_proposal.py "$DATA" \
  --pcd-folder pcds \
  --output-folder ground_labels \
  --no-image-labels \
  --overwrite
```

需要人工复核或修正时启动标注工具：

```bash
python main.py "$DATA"
```

在地面分割标注界面检查近场、远端、路沿、遮挡和3D框内点；保存结果位于 `ground_labels/<frame>.npz`。

### 2. 导出OCC标注数据和训练PKL

先用10帧检查Ray-Free Z7标签：

```bash
python batch_occ_rayfree.py "$DATA" \
  --output-folder pcds_npz_rayfree_13c_z7_test \
  --horizontal-fov-deg 33.0 \
  --angular-resolution-deg 0.10 \
  --bbox-padding-xy 0.20 \
  --limit 10 \
  --overwrite
```

确认free、unknown、ignore、障碍物遮挡和坐标方向正确后，备份已有正式标签并全量导出：

```bash
if [ -d "$DATA/pcds_npz_rayfree_13c_z7" ]; then
  cp -a "$DATA/pcds_npz_rayfree_13c_z7" \
    "$DATA/pcds_npz_rayfree_13c_z7_backup_$(date +%Y%m%d_%H%M%S)"
fi

python batch_occ_rayfree.py "$DATA" \
  --output-folder pcds_npz_rayfree_13c_z7 \
  --horizontal-fov-deg 33.0 \
  --angular-resolution-deg 0.10 \
  --bbox-padding-xy 0.20 \
  --overwrite
```

随后切换到安装了MMCV/MMDetection3D的LVFusion-OCC训练环境，生成训练索引：

```bash
cd $LVFUSION_OCC_ROOT

python tools/data_converter/hasco_converter.py \
  --root-path "$DATA" \
  --info-prefix hasco_occ_rayfree_13c_z7 \
  --train-ratio 0.8 \
  --require-occ \
  --occ-dir-name pcds_npz_rayfree_13c_z7
```

预期得到：

```text
pcds_npz_rayfree_13c_z7/*.npz                 966帧
hasco_occ_rayfree_13c_z7_infos_train.pkl      772帧
hasco_occ_rayfree_13c_z7_infos_val.pkl        194帧
```

更换数据或正式覆盖前，请先执行小批量可视化和数量检查。完整生成规则见[地面标签、Ray-Free OCC NPZ与训练PKL生成](docs/OCC_GROUND_AND_LABEL_GENERATION.md)。

### 3. 启动训练

服务器工程为 `$LVFUSION_OCC_ROOT`，容器为 `lvfusion_occ`：

```bash
docker start lvfusion_occ

docker exec \
  -e LVFUSION_OCC_DATA_ROOT=$LVFUSION_OCC_DATA_ROOT \
  -e PYTHONPATH=/workspace/LVFusion_OCC \
  -w /workspace/LVFusion_OCC \
  lvfusion_occ \
  bash -lc '
    /home/ubuntu/miniconda3/bin/python -m torch.distributed.launch \
      --nproc_per_node=1 \
      --master_port=29503 \
      tools/train.py \
      configs/hascolv/hasco_occ_fusion_20260321_rayfree_13c_z7.py \
      --work-dir work_dirs/hasco_occ_rayfree_13c_z7_weighted_24e \
      --launcher pytorch \
      --wo-wandb
  '
```

查看训练进度：

```bash
docker exec lvfusion_occ bash -lc \
  'tail -n 40 /workspace/LVFusion_OCC/work_dirs/hasco_occ_rayfree_13c_z7_weighted_24e/*.log'
```

### 4. 快速推理与可视化

使用当前 `epoch_24.pth` 检查验证集首帧：

```bash
docker exec \
  -e LVFUSION_OCC_DATA_ROOT=$LVFUSION_OCC_DATA_ROOT \
  -e PYTHONPATH=/workspace/LVFusion_OCC \
  -w /workspace/LVFusion_OCC \
  lvfusion_occ \
  /home/ubuntu/miniconda3/bin/python tools/visualize_occ_predictions.py \
  configs/hascolv/hasco_occ_fusion_20260321_rayfree_13c_z7.py \
  work_dirs/hasco_occ_rayfree_13c_z7_weighted_24e/epoch_24.pth \
  --out-dir work_dirs/hasco_occ_rayfree_13c_z7_weighted_24e/val_single \
  --num-samples 1 --start 0
```

完整验证集将 `--num-samples 1` 改为 `--num-samples 10000 --stride 1`，并使用新的输出目录。默认启用道路过滤和33°射线掩码；查看网络原始输出时追加 `--no-road-filter --no-ray-mask`。

训练环境安装、续训、完整验证及ONNX导出见[Docker、数据、训练、验证与ONNX完整流程](docs/README_OCC_TRAIN_INFER_20260321.md)。

## 当前流程

- [Docker、数据、训练、验证与ONNX完整流程](docs/README_OCC_TRAIN_INFER_20260321.md)
- [地面标签、Ray-Free OCC NPZ与训练PKL生成](docs/OCC_GROUND_AND_LABEL_GENERATION.md)
- [OCC ONNX接口与产物说明](docs/HASCO_LV_OCC_ONNX_EXPORT.md)
- [N1-655下位机三模型推理、部署与可视化](../../tool/model_conversion_tools/arm64_code/lvfusion_three_model_inference_reference/README.md)
- [120°点云监督图像地面标注规范](docs/120度点云监督图像地面标注规范.md) — **状态：暂不使用，仅保留备查**


## 下位机推理与部署

下位机采用三个Cavalry模型：图像编码、LiDAR BEV编码，以及图像/点云BEV融合与OCC Head。前两路可以并行，第三个模型等待两路特征和CPU几何反投影完成后执行。

```text
image_fp16_dra_cavalry.bin ── img_feat_2d ── CPU反投影 ── img_volume_packed ─┐
                                                                            ├─ occ_packed_fp16_dra_cavalry.bin
lidar_fp16_dra_cavalry.bin ── pts_bev_feat ─────────────────────────────────┘
                                                     │
                                                     └─ occ_logits [1,12,7,128,133]
```

板端输出经过类别维argmax和33°有效射线掩码，生成`occ_prediction.npz`、BEV俯视图与7个高度层可视化。

2026-09-29已在Ambarella N1-655上完成与训练配置匹配的Z7验证集前25帧测试：25/25帧成功，三模型纯网络平均耗时约158.34 ms，板端结果与上位机结果基本一致。

完整内容包括模型接口、输入预处理、前25帧批量打包、SSH部署、板端执行、结果下载、可视化、GT评估和`/tmp`易失性说明：

- [查看N1-655下位机完整推理与部署文档](../../tool/model_conversion_tools/arm64_code/lvfusion_three_model_inference_reference/README.md)

当前训练配置：

```text
configs/hascolv/hasco_occ_fusion_20260321_rayfree_13c_z7.py
```

当前训练权重：

```text
work_dirs/hasco_occ_rayfree_13c_z7_weighted_24e/epoch_24.pth
```

当前ONNX输出目录：

```text
output/lvfusion_occ_onnx_z7/
```
