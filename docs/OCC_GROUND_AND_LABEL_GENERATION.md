# OCC训练真值生成：地面标签、Ray-Free NPZ与PKL

本文说明当前 `20260321-10-45-48` 数据集如何从33° LiDAR点云和最终3D框生成训练所需的Z7语义OCC真值。

最终训练读取：

```text
pcds_npz_rayfree_13c_z7/*.npz
hasco_occ_rayfree_13c_z7_infos_train.pkl
hasco_occ_rayfree_13c_z7_infos_val.pkl
```

生成链路：

```text
pcds/*.pcd + parsed_labels/*.txt
            │
            ├→ 自动地面分割/人工修正
            ▼
ground_labels/*.npz                 逐点 ground/non-ground
            │
            ├→ LiDAR射线free
            ├→ 点云终点ground/non-ground
            └→ 1～9类白名单3D框体素化
            ▼
pcds_npz_rayfree_13c_z7/*.npz       最终13状态Z7 OCC标签
            │
            ▼
hasco_occ_rayfree_13c_z7_infos_{train,val}.pkl
```

## 1. 运行环境与路径

地面分割和NPZ生成依赖标注工具中的 `annotation_tool`，在本地 `annotation` Conda环境执行：

```bash
conda activate annotation
cd $LVFUSION_ANNOTATION_ROOT
```

数据根目录：

```bash
DATA=$LVFUSION_OCC_DATA_ROOT
```

相关工具：

```text
batch_ground_proposal.py     批量生成33°逐点地面标签
batch_occ_rayfree.py         生成最终Ray-Free Z7 OCC NPZ
annotation_tool/ground.py    地面分割、保存和校验
annotation_tool/occupancy.py 3D框体素化和类别映射
ground_config.json           地面分割默认参数
```

服务器训练容器 `lvfusion_occ` 当前没有安装 `annotation_tool`，且数据目录在容器内为只读。因此建议在本地生成、校验后，再同步到服务器。

## 2. 输入数据要求

数据根目录至少包含：

```text
pcds/                 33° LiDAR，文件名为<frame>.pcd
imgs/                 同帧图像
parsed_labels/        人工复核后的3D框，文件名为<frame>.txt
params/               相机与LiDAR标定参数
```

若同时存在 `parsed_labels/` 和 `labels/`，脚本优先使用 `parsed_labels/`。

必须满足：

- PCD、图像和3D框使用同一帧名；
- PCD中点字段为 `x,y,z,intensity`；
- 坐标系和标定与训练配置一致；
- `parsed_labels` 已完成最终人工复核；
- 不要在生成地面标签后修改对应PCD，否则SHA256校验会失败。

生成前检查：

```bash
find "$DATA/pcds" -maxdepth 1 -type f -name '*.pcd' | wc -l
find "$DATA/imgs" -maxdepth 1 -type f | wc -l
find "$DATA/parsed_labels" -maxdepth 1 -type f -name '*.txt' | wc -l
```

当前有效数据三者应均为966帧。

## 3. 第一步：生成逐点地面标签

### 3.1 参数配置

批处理与GUI共用：

```text
$LVFUSION_ANNOTATION_ROOT/ground_config.json
```

当前主要参数：

```json
{
  "ground_z": -0.30,
  "tolerance": 0.08,
  "ground_below_tolerance": 0.12,
  "center_road_x_min": 10.0,
  "center_road_y_half_width": 1.5,
  "center_ground_below_tolerance": 0.30,
  "center_ground_above_tolerance": 0.18,
  "slope": 0.03,
  "fit_slope_max": 0.005,
  "lateral_slope": 0.005,
  "cell": 0.25,
  "curb_height": 0.12,
  "seed_radius": 30.0,
  "ransac_iterations": 320
}
```

修改参数时只改JSON，不要在代码中维护另一套默认值。正式全量生成前，应先在标注工具中检查近、中、远距离以及路沿、遮挡、左右道路等典型帧。

### 3.2 批量生成33°标签

建议先备份已有结果：

```bash
cp -a "$DATA/ground_labels" "$DATA/ground_labels_backup_$(date +%Y%m%d_%H%M%S)"
```

全量重新生成：

```bash
python batch_ground_proposal.py "$DATA" \
  --pcd-folder pcds \
  --output-folder ground_labels \
  --no-image-labels \
  --overwrite
```

不加 `--overwrite` 时保留已有文件，只补缺失帧。显式指定 `--pcd-folder pcds` 后只处理33°点云，不运行120° Ouster图像监督流程。

该脚本还可能生成旧格式 `pcds_npz/`；当前训练不读取它，最终训练标签以后续 `pcds_npz_rayfree_13c_z7/` 为准。

### 3.3 地面标签规则

每个点的 `point_labels`：

| ID | 含义 |
| ---: | --- |
| 0 | ground |
| 1 | non_ground |
| 2 | ignore/无可靠密集地面支持 |

当前二分类批处理会将内部ignore转为non-ground；白名单3D框包含的点始终强制为non-ground，避免障碍物参与地面平面拟合。离散的单个地面点会被过滤，受周围地面支持的1～2个低矮non-ground噪点可恢复为ground。

`ground_labels/<frame>.npz` 字段：

| 字段 | 类型/形状 | 说明 |
| --- | --- | --- |
| `point_labels` | `uint8 [N]` | 与原PCD逐点对应 |
| `class_names` | 3个字符串 | ground/non_ground/ignore |
| `pcd_sha256` | 字符串 | 原PCD内容校验 |
| `pcd_path` | 字符串 | 生成时PCD路径，仅作记录 |
| `point_count` | 标量 | 点数校验 |
| `proposal_parameters` | `float64 [5]` | ground_z、tolerance、slope、cell、可靠地面最远X |
| `format_version` | 标量 | 当前为1 |

第五个 `proposal_parameters` 是连续密集地面能够可靠到达的最远X。超过该距离的点不会因为缺少地面回波而被直接当成free。

### 3.4 人工复核

自动预分割不是最终真值。建议在标注工具中重点检查：

- 路沿是否被误分成ground；
- 远端道路是否被误分成non-ground；
- 3D框内障碍物点是否保持non-ground；
- 低于局部平面的噪点和排水沟点；
- 连续帧平面是否突然跳变；
- 无连续密集地面帧是否应删除。

人工保存后仍写入同名 `ground_labels/<frame>.npz`，并保留PCD SHA256和点数校验。

## 4. 第二步：生成Ray-Free Z7 OCC NPZ

### 4.1 小批量验证

先用前10帧验证参数和输出：

```bash
python batch_occ_rayfree.py "$DATA" \
  --output-folder pcds_npz_rayfree_13c_z7_test \
  --horizontal-fov-deg 33.0 \
  --angular-resolution-deg 0.10 \
  --bbox-padding-xy 0.20 \
  --limit 10 \
  --overwrite
```

在可视化工具中检查：

- free只出现在射线可证实区域；
- 障碍物后方保持unknown；
- 视场外为ignore；
- 3D框语义覆盖优先于free和other；
- Z轴层和车辆坐标方向正确。

### 4.2 全量生成

先备份已有训练标签：

```bash
cp -a "$DATA/pcds_npz_rayfree_13c_z7" \
  "$DATA/pcds_npz_rayfree_13c_z7_backup_$(date +%Y%m%d_%H%M%S)"
```

全量重新生成：

```bash
python batch_occ_rayfree.py "$DATA" \
  --output-folder pcds_npz_rayfree_13c_z7 \
  --horizontal-fov-deg 33.0 \
  --angular-resolution-deg 0.10 \
  --bbox-padding-xy 0.20 \
  --overwrite
```

参数含义：

- `horizontal-fov-deg=33`：只监督33°融合有效区；
- `angular-resolution-deg=0.10`：按方位角/俯仰角保留最近回波，避免同一射线穿透前景；
- `bbox-padding-xy=0.20`：射线接近白名单框时提前截断，防止把框内或框后标成free；
- `--overwrite`：覆盖同名输出；正式执行前必须先备份或使用新目录。

### 4.3 体素赋值优先级

每帧生成顺序为：

1. 33°视场外初始化为ignore 12；
2. 视场内未观测体素初始化为unknown 0；
3. LiDAR原点到最近有效回波之间的体素写为free 11；
4. ground回波终点写为free 11；
5. non-ground回波终点写为other 10；
6. 白名单3D框覆盖为对应的1～9类。

后写入的语义优先级更高，因此最终优先级为：

```text
白名单BBox 1～9 > non-ground 10 > ground/ray-free 11 > unknown 0 > ignore 12
```

被障碍物遮挡、没有回波证明为空的区域保持unknown，不能直接标为free。

### 4.4 最终NPZ格式

`pcds_npz_rayfree_13c_z7/<frame>.npz`：

| 字段 | 类型/形状 | 说明 |
| --- | --- | --- |
| `occ_gt` | `uint8 [7,128,133]` | ZYX语义标签 |
| `valid_mask` | `bool [7,128,133]` | `occ_gt != 12` |
| `source_mask` | `uint8 [7,128,133]` | 标签来源 |
| `source_names` | 5个字符串 | 来源名称 |
| `point_cloud_range` | `float64 [6]` | `[0,-19.2,-1.0,39.9,19.2,1.8]` |
| `voxel_size` | `float64 [3]` | `[0.3,0.3,0.4]` |
| `axis_order` | 字符串 | `ZYX` |
| `class_names` | 13个字符串 | 0～12类别 |
| `format_version` | 标量 | 当前为4 |

`source_mask`：

| ID | 来源 |
| ---: | --- |
| 0 | unobserved |
| 1 | lidar_ray_free |
| 2 | lidar_ground |
| 3 | lidar_non_ground |
| 4 | whitelist_bbox |

类别定义：

```text
0 unknown
1 truck
2 large_vehicle
3 construction_vehicle
4 bus
5 car
6 motorcycle
7 bicycle
8 tricycle
9 pedestrian
10 other
11 free
12 ignore
```

类别12只参与有效掩码，不是网络输出通道；网络预测0～11共12类。

## 5. 第三步：生成训练PKL

切换到LVFusion-OCC工程和具有MMCV/MMDetection3D依赖的训练环境：

```bash
cd $LVFUSION_OCC_ROOT
```

为避免覆盖当前已验证索引，可以先用临时前缀生成并核对：

```bash
python tools/data_converter/hasco_converter.py \
  --root-path "$DATA" \
  --info-prefix hasco_occ_rayfree_13c_z7_check \
  --train-ratio 0.8 \
  --require-occ \
  --occ-dir-name pcds_npz_rayfree_13c_z7
```

核对后正式生成：

```bash
python tools/data_converter/hasco_converter.py \
  --root-path "$DATA" \
  --info-prefix hasco_occ_rayfree_13c_z7 \
  --train-ratio 0.8 \
  --require-occ \
  --occ-dir-name pcds_npz_rayfree_13c_z7
```

预期输出：

```text
hasco_occ_rayfree_13c_z7_infos_train.pkl  772帧
hasco_occ_rayfree_13c_z7_infos_val.pkl    194帧
```

当前划分是按帧80/20划分。连续帧可能同时进入训练集和验证集，只适合通路验证；正式泛化评估应改为按连续序列或场景划分。

## 6. 完整性校验

### 6.1 数量检查

```bash
find "$DATA/ground_labels" -maxdepth 1 -type f -name '*.npz' | wc -l
find "$DATA/pcds_npz_rayfree_13c_z7" -maxdepth 1 -type f -name '*.npz' | wc -l
ls -lh "$DATA"/hasco_occ_rayfree_13c_z7_infos_*.pkl
```

当前逐点地面标签和OCC NPZ均应为966帧。

### 6.2 NPZ字段和取值检查

```bash
python - "$DATA" <<'PYCHECK'
from pathlib import Path
import sys
import numpy as np
root=Path(sys.argv[1])
files=sorted((root/'pcds_npz_rayfree_13c_z7').glob('*.npz'))
assert len(files)==len(list((root/'pcds').glob('*.pcd'))), (len(files),'count mismatch')
counts=np.zeros(13,dtype=np.int64)
for p in files:
    with np.load(p,allow_pickle=False) as d:
        occ=d['occ_gt']
        assert occ.shape==(7,128,133) and occ.dtype==np.uint8, p
        assert str(d['axis_order'].item())=='ZYX', p
        assert np.array_equal(d['valid_mask'],occ!=12), p
        assert np.isin(occ,np.arange(13,dtype=np.uint8)).all(), p
        counts += np.bincount(occ.ravel(),minlength=13)
print('frames',len(files))
print('class_counts',counts.tolist())
print('free',int(counts[11]),'unknown',int(counts[0]),'ignore',int(counts[12]))
PYCHECK
```

除格式检查外，还必须随机抽帧可视化。仅统计类别数量无法发现坐标翻转、标定错位、地面误分或遮挡区错误标free。

## 7. 同步到训练服务器

从本地同步最终标签和PKL：

```bash
rsync -av --delete \
  "$DATA/pcds_npz_rayfree_13c_z7/" \
  hascoai3@11.11.11.156:$LVFUSION_OCC_DATA_ROOT/pcds_npz_rayfree_13c_z7/

scp "$DATA/hasco_occ_rayfree_13c_z7_infos_train.pkl" \
     "$DATA/hasco_occ_rayfree_13c_z7_infos_val.pkl" \
  hascoai3@11.11.11.156:$LVFUSION_OCC_DATA_ROOT/
```

`rsync --delete` 会让服务器目录严格匹配本地，执行前先核对源路径。若不希望删除服务器额外文件，去掉 `--delete`。

服务器复查：

```bash
ssh hascoai3@11.11.11.156 '
  DATA=$LVFUSION_OCC_DATA_ROOT
  find "$DATA/pcds_npz_rayfree_13c_z7" -maxdepth 1 -type f -name "*.npz" | wc -l
  ls -lh "$DATA"/hasco_occ_rayfree_13c_z7_infos_*.pkl
'
```

完成后，训练配置 `configs/hascolv/hasco_occ_fusion_20260321_rayfree_13c_z7.py` 会直接读取上述两个PKL及其关联的NPZ。

## 8. 不再用于当前训练的旧目录

以下目录是早期实验或中间产物，不是当前Z7训练真值：

```text
pcds_npz/
imgs_npz/
occ_fused_npz/
pcds_npz_rayfree/
pcds_npz_rayfree_13c/
pcds_npz_rayfree_13c_test/
```

不要因为这些目录存在就误认为训练配置会读取它们；应以配置中的PKL和每条样本的 `occ_path` 为准。
