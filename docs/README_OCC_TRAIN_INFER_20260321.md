# HASCO LVFusion OCC：Docker、训练、验证与 ONNX 导出

本文是 `20260321-10-45-48` 数据集当前可复现流程的主文档，覆盖服务器 Docker 环境创建、数据检查、模型训练、验证推理和三段式 ONNX 导出。

当前网络输入为前视图像和 33° LiDAR 点云，在二维 BEV 特征层融合，再由 OCC Head 输出带高度的语义体素。120° Ouster 点云只用于早期标签制作，不是训练或实际推理输入。

## 1. 当前版本概览
在本机或服务器上先设置 `LVFUSION_OCC_ROOT`、`LVFUSION_OCC_DATA_ROOT`、`LVFUSION_ANNOTATION_ROOT`、`LVFUSION_OCC_DOCKER_DIR` 和 `LVFUSION_OCC_IMAGE_TAR`；示例见仓库根目录 `.env.example`。


| 项目 | 当前值 |
| --- | --- |
| 服务器工程 | `$LVFUSION_OCC_ROOT` |
| 容器内工程 | `/workspace/LVFusion_OCC` |
| 服务器数据 | `$LVFUSION_OCC_DATA_ROOT` |
| Docker 镜像包 | `$LVFUSION_OCC_IMAGE_TAR` |
| 镜像 | `fastbev_bstnn:v1.1` |
| 独立容器 | `lvfusion_occ` |
| 配置 | `configs/hascolv/hasco_occ_fusion_20260321_rayfree_13c_z7.py` |
| 工作目录 | `work_dirs/hasco_occ_rayfree_13c_z7_weighted_24e` |
| 当前权重 | `work_dirs/hasco_occ_rayfree_13c_z7_weighted_24e/epoch_24.pth` |
| ONNX 输出 | `output/lvfusion_occ_onnx_z7` |

本地对应工程：

```text
$LVFUSION_OCC_ROOT
```

## 2. 模型与标签约定

### 2.1 网络流程

```text
图像 → 图像Backbone/Neck → 2D特征 → 几何反投影 → 图像BEV ┐
                                                              ├→ BEV融合 → OCC Head
33°点云 → Voxel/VFE/Scatter → 点云Backbone/Neck → 点云BEV ┘
```

融合发生在二维 BEV 特征层。OCC Head 内部先产生 `K×Z` 个二维通道，再重排为：

```text
[B, K, Z, Y, X] = [B, 12, 7, 128, 133]
```

### 2.2 OCC空间

```text
X范围: [0.0, 39.9] m
Y范围: [-19.2, 19.2] m
Z范围: [-1.0, 1.8] m
体素尺寸: [0.3, 0.3, 0.4] m
标签形状: [Z,Y,X] = [7,128,133]
```

点云 Backbone 仍使用原始较大感知范围 `[0,-19.2,-5,105.6,19.2,4]`；OCC 输出只保留前方 40 m。

### 2.3 类别

| ID | 含义 | 是否由网络预测 |
| ---: | --- | :---: |
| 0 | unknown | 是 |
| 1 | truck | 是 |
| 2 | large_vehicle | 是 |
| 3 | construction_vehicle | 是 |
| 4 | bus | 是 |
| 5 | car | 是 |
| 6 | motorcycle | 是 |
| 7 | bicycle | 是 |
| 8 | tricycle | 是 |
| 9 | pedestrian | 是 |
| 10 | other/non-ground | 是 |
| 11 | free | 是 |
| 12 | ignore | 否，仅用于损失屏蔽 |

标签生成原则：LiDAR 射线经过且无障碍的位置为 free；非地面点和 road 内白名单框为 occupied/语义类别；未观测或遮挡区域为 unknown；有效视场外为 ignore。

## 3. Docker环境安装

### 3.1 宿主机前提

宿主机需安装：

- NVIDIA驱动，`nvidia-smi` 可正常运行；
- Docker；
- NVIDIA Container Toolkit，Docker支持 `--gpus all`。

检查：

```bash
nvidia-smi
docker version
docker info | grep -i runtime
```

### 3.2 导入镜像

在服务器执行：

```bash
cd "$LVFUSION_OCC_DOCKER_DIR"
docker load -i lvfusion_docker_image_v1_1.tar
docker images | grep fastbev_bstnn
```

预期镜像为：

```text
fastbev_bstnn:v1.1
```

### 3.3 新建独立容器

不要复用其他人的运行容器。当前项目容器创建方式如下：

```bash
docker run -d \
  --name lvfusion_occ \
  --gpus all \
  --ipc=host \
  -v $LVFUSION_OCC_ROOT:/workspace/LVFusion_OCC \
  -v "$LVFUSION_OCC_DATA_ROOT:$LVFUSION_OCC_DATA_ROOT:ro" \
  -w /workspace/LVFusion_OCC \
  fastbev_bstnn:v1.1 \
  bash -lc 'trap : TERM INT; sleep infinity & wait'
```

数据目录以只读方式挂载，避免训练过程误改标签。工程目录可写，用于保存日志、权重和导出模型。

已存在但停止时：

```bash
docker start lvfusion_occ
```

检查GPU和依赖：

```bash
docker exec lvfusion_occ nvidia-smi

docker exec -w /workspace/LVFusion_OCC lvfusion_occ \
  /home/ubuntu/miniconda3/bin/python -c \
  "import torch,mmcv,mmdet,mmseg; print(torch.__version__, torch.cuda.is_available()); print(mmcv.__version__,mmdet.__version__,mmseg.__version__)"
```

当前验证过的版本：

```text
Python 3.8.18
PyTorch 1.10.0+cu113
CUDA Runtime 11.3
TorchVision 0.11.0+cu113
MMCV 1.4.0
MMDetection 2.14.0
MMSegmentation 0.14.1
MMDetection3D 0.16.0
```

## 4. 数据准备与检查

### 4.1 当前训练数据

数据根目录：

```text
$LVFUSION_OCC_DATA_ROOT
```

训练直接使用：

```text
imgs/                                  前视图像，966帧
pcds/                                  33° LiDAR点云，966帧
pcds_npz_rayfree_13c_z7/               Z7 OCC标签，966帧
hasco_occ_rayfree_13c_z7_infos_train.pkl
hasco_occ_rayfree_13c_z7_infos_val.pkl
```

PKL负责关联每帧图像、33°点云、标定和OCC标签路径。训练与推理不读取 `pcds_ouster/`。

### 4.2 完整性检查

```bash
DATA=$LVFUSION_OCC_DATA_ROOT
find "$DATA/imgs" -maxdepth 1 -type f | wc -l
find "$DATA/pcds" -maxdepth 1 -type f | wc -l
find "$DATA/pcds_npz_rayfree_13c_z7" -maxdepth 1 -type f | wc -l
ls -lh "$DATA"/hasco_occ_rayfree_13c_z7_infos_*.pkl
```

三类帧文件数量应均为 966。若数量或文件名不能一一对应，应先修复数据索引，不能直接训练。

### 4.3 重新生成Z7标签和PKL

完整的地面分割、逐点标签、Ray-Free Z7 NPZ、格式校验和PKL生成流程见 [OCC训练真值生成说明](OCC_GROUND_AND_LABEL_GENERATION.md)。

已有标签无需重复生成。需要从点级地面、非地面和3D框重新制作时，在可写数据副本中执行：

```bash
cd /workspace/LVFusion_OCC

python tools/batch_occ_rayfree_13c_z7.py \
  /path/to/writable/20260321-10-45-48 \
  --output-folder pcds_npz_rayfree_13c_z7
```

然后生成指向该目录的训练/验证PKL。使用新的输出前缀，避免覆盖已验证数据：

```bash
python tools/data_converter/hasco_converter.py \
  --root-path /path/to/writable/20260321-10-45-48 \
  --info-prefix hasco_occ_rayfree_13c_z7 \
  --train-ratio 0.8 \
  --require-occ \
  --occ-dir-name pcds_npz_rayfree_13c_z7
```

由于当前容器把服务器数据挂载为只读，标签和PKL重建应在宿主机可写副本或专门的可写容器中完成。

## 5. 训练

### 5.1 环境变量

所有服务器命令统一设置：

```text
LVFUSION_OCC_DATA_ROOT=$LVFUSION_OCC_DATA_ROOT
PYTHONPATH=/workspace/LVFusion_OCC
```

配置通过 `LVFUSION_OCC_DATA_ROOT` 覆盖本地默认数据路径。

### 5.2 启动24 epoch训练

图像分支使用 SyncBN，即使单GPU也必须初始化分布式进程组：

```bash
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

配置为 batch size 1、2个worker、24 epoch。当前实测单卡训练显存约 5.7 GB。

### 5.3 查看进度

```bash
nvidia-smi

docker exec lvfusion_occ bash -lc \
  'tail -n 40 /workspace/LVFusion_OCC/work_dirs/hasco_occ_rayfree_13c_z7_weighted_24e/*.log'
```

重点观察：

- `loss_occ` 是否持续下降并保持有限；
- `occ_accuracy` 是否提升；
- `grad_norm` 是否有限；
- 是否出现 NaN、CUDA OOM 或数据加载异常。

训练日志中的 `occ_accuracy` 是批次统计，不等同于验证集 mIoU。

### 5.4 中断续训

先确认没有另一个同配置训练进程，再指定实际存在的检查点：

```bash
# 在上面的训练命令中追加：
--resume-from work_dirs/hasco_occ_rayfree_13c_z7_weighted_24e/epoch_N.pth
```

不要同时向同一个 `work-dir` 启动两个训练任务。

## 6. 验证集推理与可视化

### 6.1 单帧检查

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

### 6.2 完整验证集

```bash
docker exec \
  -e LVFUSION_OCC_DATA_ROOT=$LVFUSION_OCC_DATA_ROOT \
  -e PYTHONPATH=/workspace/LVFusion_OCC \
  -w /workspace/LVFusion_OCC \
  lvfusion_occ \
  /home/ubuntu/miniconda3/bin/python tools/visualize_occ_predictions.py \
  configs/hascolv/hasco_occ_fusion_20260321_rayfree_13c_z7.py \
  work_dirs/hasco_occ_rayfree_13c_z7_weighted_24e/epoch_24.pth \
  --out-dir work_dirs/hasco_occ_rayfree_13c_z7_weighted_24e/val_inference_raymask \
  --num-samples 10000 --start 0 --stride 1
```

默认对预测结果施加道路过滤和33°射线有效区后处理。需要观察网络原始输出时使用独立目录并加：

```text
--no-road-filter --no-ray-mask
```

不要用普通检测任务的 `tools/test.py` 代替 OCC 验证。

## 7. 导出ONNX

OCC使用独立脚本：

```text
tools/export_lvfusion_occ_onnx.py
```

原检测任务的 `tools/export_lvfusion_onnx.py` 保持不变。

### 7.1 完整导出

```bash
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

当前使用外置 VFE/Scatter。若部署平台支持相应ONNX算子并希望合并VFE，可增加 `--include-vfe`，但仍需要处理原始点云体素化与固定最大体素数。

### 7.2 三个模型

| 文件 | 输入 | 输出 |
| --- | --- | --- |
| `export_2d_model.onnx` | `img [B,3,H,W]` | `img_feat_2d` |
| `export_lidar_model.onnx` | `pts_spatial_features [B,64,256,704]` | `pts_bev_feat [B,256,128,352]` |
| `export_occ_model.onnx` | `img_volume [B,64,704,256,6]`、`pts_bev_feat` | `occ_logits [B,12,7,128,133]` |

三个 ONNX 可分别交给量化工具，但正确的 INT8 校准还需要每个模型对应的真实中间输入，不能使用随机张量代替。

### 7.3 ONNX之间的外置操作

```text
图像 → 前处理 → export_2d_model.onnx
     → gather/scatter几何反投影 → img_volume ┐
                                               ├→ export_occ_model.onnx
点云 → voxel/VFE/scatter → export_lidar_model.onnx
     → pts_bev_feat ───────────────────────────┘
```

因此部署不能只串联三个 ONNX，还必须实现：

1. 与训练一致的图像resize、RGB转换和归一化；
2. 点云范围过滤、体素化、HardVFE和PointPillars Scatter；
3. 图像二维特征到三维体素的外置反投影；
4. `argmax(axis=1)`、有效射线区域和道路区域后处理；
5. OCC索引到车辆坐标系的转换。

### 7.4 导出产物

```text
output/lvfusion_occ_onnx_z7/
├── export_2d_model.onnx
├── export_lidar_model.onnx
├── export_occ_model.onnx
├── occ_metadata.json
├── backprojection_params_sample0.npz
├── featurePointLength.bin
├── gather_indices.bin
├── scatter_indices.bin
└── index/
    ├── valid.npy
    ├── x.npy
    ├── y.npy
    ├── volume.npy
    └── features.npy
```

量化工具通常只接收三个ONNX；反投影二进制和元数据用于下位机集成，`index/*.npy` 与 `features.npy` 主要用于联调和一致性检查。

### 7.5 已验证输出

```text
img_volume:   [1,64,704,256,6]
pts_bev_feat: [1,256,128,352]
occ_logits:   [1,12,7,128,133]
输出布局:      B,K,Z,Y,X
数据类型:      float32
```

导出脚本会执行 ONNX checker。本次 `export_occ_model.onnx` 已通过 ONNX Runtime CPU 前向，输出形状正确且全部为有限值。

## 8. 复制产物到本地

在本地主机执行：

```bash
scp -r \
  hascoai3@11.11.11.156:$LVFUSION_OCC_ROOT/output/lvfusion_occ_onnx_z7 \
  $LVFUSION_OCC_ROOT/output/
```

代码和文档同步：

```bash
scp hascoai3@11.11.11.156:$LVFUSION_OCC_ROOT/tools/export_lvfusion_occ_onnx.py \
  $LVFUSION_OCC_ROOT/tools/

scp hascoai3@11.11.11.156:$LVFUSION_OCC_ROOT/docs/README_OCC_TRAIN_INFER_20260321.md \
  $LVFUSION_OCC_ROOT/docs/
```

## 9. 常见问题

- `Default process group has not been initialized`：单卡训练也要使用 `torch.distributed.launch` 和 `--launcher pytorch`。
- `No module named mmdet3d`：检查 `PYTHONPATH=/workspace/LVFusion_OCC`。
- 找不到数据：检查 `LVFUSION_OCC_DATA_ROOT` 和PKL中的相对路径。
- ONNX权重加载提示 `unexpected key: occ_head.class_weight`：`class_weight` 是训练损失权重，不参与推理，不影响导出的网络参数。
- OCC模型输出12通道而标签有13种状态：12号是ignore标签，不是预测类别。
- ONNX量化效果异常：检查是否使用真实校准数据，尤其是 `img_volume` 和 `pts_spatial_features`，并分别对比 PyTorch FP32、ONNX FP32、设备FP16/INT8。
- 反投影结果错位：核对标定、图像resize/crop、特征步长和 `gather/scatter` 索引是否来自同一配置。

## 10. 相关文档

- [OCC ONNX接口和产物说明](HASCO_LV_OCC_ONNX_EXPORT.md)。
- `docs/OCC_GROUND_AND_LABEL_GENERATION.md`：地面标签、Ray-Free NPZ和训练PKL生成。
- `docs/120度点云监督图像地面标注规范.md`：上游图像地面监督标注规范。
