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

# 自动生成全部33°点云逐点地面标签；已有同名结果会跳过
python batch_ground_proposal.py "$DATA" \
  --pcd-folder pcds \
  --output-folder ground_labels \
  --no-image-labels
```

需要人工复核或修正时启动标注工具：

```bash
python main.py "$DATA"
```

在地面分割标注界面检查近场、远端、路沿、遮挡和3D框内点；保存结果位于 `ground_labels/<frame>.npz`。
如需重新生成已有地面标签，先备份原目录，再显式追加 `--overwrite`。

### 2. 导出OCC标注数据和训练PKL

先用10帧检查Ray-Free Z7标签：

```bash
python batch_occ_rayfree.py "$DATA" \
  --output-folder pcds_npz_rayfree_13c_z7_test \
  --horizontal-fov-deg 33.0 \
  --angular-resolution-deg 0.10 \
  --bbox-padding-xy 0.20 \
  --limit 10
```

确认free、unknown、ignore、障碍物遮挡和坐标方向正确后全量导出；已有同名NPZ会跳过：

```bash
python batch_occ_rayfree.py "$DATA" \
  --output-folder pcds_npz_rayfree_13c_z7 \
  --horizontal-fov-deg 33.0 \
  --angular-resolution-deg 0.10 \
  --bbox-padding-xy 0.20
```

如需覆盖已有射线标签，先备份原目录，再显式追加 `--overwrite`。

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

#### `20260903_marvelr_gtlidar` 批处理（只读取 `labels/` 中的3D框）

本地数据位于 `/home/yut/shared/ubuntu_data/LVFusion_OCC/data/20260903_marvelr_gtlidar`。`pcds/`、`imgs/` 和 `labels/` 应有同名帧；地面标签保存到 `ground_labels/`，射线OCC标签保存到新的 `pcds_npz_rayfree_13c_z7/`。以下命令不带 `--overwrite`：已有同名输出会跳过，不会改写旧版 `npz/` 或已有地面标签。

```bash
export LVFUSION_OCC_DATA_ROOT=/home/yut/shared/ubuntu_data/LVFusion_OCC/data/20260903_marvelr_gtlidar
export LVFUSION_ANNOTATION_ROOT=/home/yut/shared/ubuntu_data/LVFusion_OCC/data/tool/lv_annotation_tools
DATA="$LVFUSION_OCC_DATA_ROOT"

conda activate annotation
cd "$LVFUSION_ANNOTATION_ROOT"

# 批量生成33°点云逐点地面标签；已有 ground_labels/<frame>.npz 时跳过。
# 此命令还会生成基础语义标签 pcds_npz/，与下面的射线标签目录不同。
python batch_ground_proposal.py "$DATA" \
  --pcd-folder pcds \
  --output-folder ground_labels \
  --no-image-labels

# 先在独立目录检查前10帧，不覆盖正式标签。
python batch_occ_rayfree.py "$DATA" \
  --output-folder pcds_npz_rayfree_13c_z7_test \
  --horizontal-fov-deg 33.0 \
  --angular-resolution-deg 0.10 \
  --bbox-padding-xy 0.20 \
  --limit 10

# 检查测试标签后，全量生成射线OCC；再次运行会跳过已生成的帧。
python batch_occ_rayfree.py "$DATA" \
  --output-folder pcds_npz_rayfree_13c_z7 \
  --horizontal-fov-deg 33.0 \
  --angular-resolution-deg 0.10 \
  --bbox-padding-xy 0.20
```

射线标签依赖同名的 `pcds/<frame>.pcd`、`labels/<frame>.txt` 和 `ground_labels/<frame>.npz`。当前目录共有414帧；全量运行的预期结果是 `pcds_npz_rayfree_13c_z7/` 下414个Z7 NPZ（`occ_gt` 形状为 `[7, 128, 133]`，12表示视场外ignore）。生成训练/验证PKL时，使用上面的 `hasco_converter.py` 命令；该目录对应331帧训练、83帧验证。

本地 `data/` 下其余数据目录均从 `labels/` 读取3D框。2026-10-14 按当前训练配置复核并更新了逐点地面标签、Ray-Free OCC 与训练索引：

| 数据目录 | 总帧数 | 训练PKL | 验证PKL |
| --- | ---: | ---: | ---: |
| `20260821_gtlidar_marvelr` | 164 | 131 | 33 |
| `20260821_s13ouster_marvelr` | 30 | 24 | 6 |
| `20260821_s47s13_marvelr` | 134 | 107 | 27 |
| `sunwin_bevdata_20260807` | 279 | 223 | 56 |

`sunwin_bevdata_20260807` 的点云近端有截断，需使用本地标注工具中支持近端自适应的地面分割版本；少数难例按单帧地面高度复核后补标。批处理中筛出的旧标签保存在数据目录内带 `failed` 或 `rejected` 后缀的备份目录，正式训练只读取无后缀的标签目录。

当前训练版以 `20260321-10-45-48` 为基准：`ground_labels/` 为逐点二分类（0 地面、1 非地面）；33°、Z7 的 v4 射线真值在 `pcds_npz_rayfree_13c_z7/`，由同名前视图像、点云、`labels/` 白名单3D框及地面标签生成。2026-10-14 的处理结果：Marvel 的392帧将旧 `ignore` 点转为非地面，保留原有人工地面点；基准训练集的7帧将框内误标的231个地面点改为非地面。Ray-Free OCC 分别更新了基准训练集11帧、Marvel 408帧、Sunwin 97帧。Marvel、Sunwin 和 `20260821_gtlidar_marvelr` 的PKL按现存PCD重新生成；全部六组共1987帧的逐点标签、v4 OCC及PKL路径已通过完整性检查。修改前文件保存在各数据目录的 `*_backup_20261014_pre_training_reprocess`。两帧使用单帧地面高度 -0.5 的8月GT LiDAR标签保持原人工结果；固定采用 -0.3 试分割时这两帧没有地面点。

#### Sunwin 路面覆盖区限定的射线 OCC（2026-10-14）

Sunwin 当前有效的279帧已更新到 `pcds_npz/`；此前的全量Z9标签备份在
`pcds_npz_backup_20261014_pre_roadmask/`。对应
`hasco_occ_rayfree_13c_z7_roadmask_infos_{train,val}.pkl`（223/56帧）。
空间范围、体素尺寸和33°视场与上面的射线 OCC 相同；仅在由 `ground_labels/`
构建的路面覆盖区内监督 unknown、free 和 other。覆盖区外为 `ignore=12`，
但33°视场内的白名单3D框仍保留语义类别。
当前训练版使用 `pcds_npz_rayfree_13c_z7/` 和不带 `_roadmask` 的PKL；
它们已按现有地面标签重算。训练路面限定版本时应显式选用带
`_roadmask` 后缀的PKL；对应配置为
`configs/hascolv/hasco_occ_fusion_sunwin_roadmask_13c_z7.py`。

```bash
DATA=/home/yut/shared/ubuntu_data/LVFusion_OCC/data/sunwin_bevdata_20260807
cd /home/yut/shared/ubuntu_data/LVFusion_OCC/data/tool/lv_annotation_tools
conda activate annotation
# 正式目录已有标签；复核时先写入独立测试目录。
python batch_occ_rayfree.py "$DATA" --road-only \
  --output-folder pcds_npz_roadmask_test --limit 10

# PKL生成需要MMCV；本地可使用已有的 fastbev_bstnn:v1.1 镜像。
docker run --rm \
  -v /home/yut/shared/ubuntu_data/LVFusion_OCC:/workspace/LVFusion_OCC \
  -w /workspace/LVFusion_OCC/project/LVFusion_OCC \
  fastbev_bstnn:v1.1 /home/ubuntu/miniconda3/bin/python \
  tools/data_converter/hasco_converter.py \
  --root-path /workspace/LVFusion_OCC/data/sunwin_bevdata_20260807 \
  --info-prefix hasco_occ_rayfree_13c_z7_roadmask \
  --train-ratio 0.8 --require-occ \
  --occ-dir-name pcds_npz
```

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
