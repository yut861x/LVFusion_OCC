"""
ONNX模型可视化脚本
参考 jiesongz_debug/vis_fasebev1v.py，使用ONNX模型进行推理并保存可视化结果

python tools/vis_onnx.py \
    --config configs/hascolv/mono_2class_fastbev1v_m0_r18_s256x704_v200x200x4_c192_d2.py \
    --onnx-2d ./output/onnx/fastbev1v_2d.onnx \
    --onnx-3d ./output/onnx/fastbev1v_3d.onnx \
    --save-dir ./vis_result_onnx
"""
import os
import sys
import torch
import argparse

from mmcv import Config
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel, scatter, DataContainer
from mmdet3d.datasets import build_dataset
from mmdet3d.datasets import build_dataloader
from mmdet3d.models import build_model

import cv2
import numpy as np
import tqdm


def get_lidar_3d_box_corners(box):
    """
    box: (7,) = x, y, z, l, w, h, yaw   (LiDAR coord)
    return: (8, 3)
    """
    x, y, z, l, w, h, yaw = box

    # 以 box 底面中心为参考
    x_corners = [ l/2,  l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2]
    y_corners = [ w/2, -w/2, -w/2,  w/2,  w/2, -w/2, -w/2,  w/2]
    z_corners = [   0,    0,    0,    0,   h,   h,   h,   h]

    corners = np.vstack([x_corners, y_corners, z_corners])  # (3,8)

    # 绕 Z 轴旋转
    c = np.cos(yaw)
    s = np.sin(yaw)
    R = np.array([
        [ c, -s, 0],
        [ s,  c, 0],
        [ 0,  0, 1]
    ])

    corners = R @ corners
    corners[0, :] += x
    corners[1, :] += y
    corners[2, :] += z

    return corners.T  # (8,3)


def project_lidar_to_img(points_3d, lidar2img):
    """
    points_3d: (N,3)
    lidar2img: (4,4)
    return:
        pts_img: (N,2)
        mask: (N,) depth > 0
    """
    num = points_3d.shape[0]
    pts_homo = np.hstack([points_3d, np.ones((num, 1))])  # (N,4)

    pts_img = pts_homo @ lidar2img.T
    depth = pts_img[:, 2]

    mask = depth > 1e-5
    pts_img = pts_img[mask]

    pts_img[:, 0] /= pts_img[:, 2]
    pts_img[:, 1] /= pts_img[:, 2]

    return pts_img[:, :2].astype(int), mask


def draw_bbox(
    img,
    lidar2img,
    boxes_3d,
    colors=None,
    thickness=2
):
    """
    img: (H,W,3)
    boxes_3d: (N,7)  -> (x,y,z,l,w,h,yaw)
    classes: (N,) or None
    """
    img = img.copy()

    edges = [
        (0,1),(1,2),(2,3),(3,0),  # top
        (4,5),(5,6),(6,7),(7,4),  # bottom
        (0,4),(1,5),(2,6),(3,7)   # vertical
    ]

    for i, box in enumerate(boxes_3d):
        corners_3d = get_lidar_3d_box_corners(box)
        corners_2d, mask = project_lidar_to_img(corners_3d, lidar2img)

        if corners_2d.shape[0] < 8:
            continue

        color = colors if colors is not None else (0, 255, 0)

        # 画 3D 框
        for e in edges:
            p1, p2 = corners_2d[e[0]], corners_2d[e[1]]
            cv2.line(img, tuple(p1), tuple(p2), color, thickness)
        
        for front in [(2,5),(1,6)]:
            p1, p2 = corners_2d[front[0]], corners_2d[front[1]]
            cv2.line(img, tuple(p1), tuple(p2), (50,100,200), 3)

    return img


def draw_bev_box(
    bev_img: np.ndarray,
    box7: np.ndarray,
    xmin: float,
    ymin: float,
    meter_per_pixel: float,
    color=(255, 0, 0),
    thickness=2,
):
    """
    绘制单个 BEV 3D 框

    box7: [x, y, z, l, w, h, yaw]
    yaw: 绕 z 轴逆时针，0 朝 +x
    """

    x, y, z, l, w, h, yaw = box7
    H, W = bev_img.shape[:2]

    # -----------------------------
    # 中心：meter → pixel
    # -----------------------------
    cx = (x - xmin) / meter_per_pixel
    cy = (y - ymin) / meter_per_pixel
    cy = H - cy - 1

    # -----------------------------
    # 尺寸：meter → pixel
    # -----------------------------
    l_pix = l / meter_per_pixel
    w_pix = w / meter_per_pixel

    # -----------------------------
    # 局部角点（pixel）
    # 前：+x，左：+y
    # -----------------------------
    x_corners = np.array([ l_pix/2,  l_pix/2, -l_pix/2, -l_pix/2])
    y_corners = np.array([ w_pix/2, -w_pix/2, -w_pix/2,  w_pix/2])
    corners = np.stack([x_corners, y_corners], axis=1)

    # -----------------------------
    # 旋转
    # -----------------------------
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s],
                  [s,  c]])
    corners = corners @ R.T

    # -----------------------------
    # 平移
    # -----------------------------
    corners[:, 0] += cx
    corners[:, 1] += cy
    corners = corners.astype(np.int32)

    # -----------------------------
    # 绘制 box 边
    # -----------------------------
    for i in range(4):
        cv2.line(
            bev_img,
            tuple(corners[i]),
            tuple(corners[(i + 1) % 4]),
            color,
            thickness
        )

    # -----------------------------
    # 朝向线（前向，橙红色）
    # -----------------------------
    cv2.line(
        bev_img,
        tuple(corners[0]),
        tuple(corners[3]),
        (0, 128, 255),
        thickness + 1
    )


def draw_bev(
    points_xyz: np.ndarray,
    pred_boxes: np.ndarray,
    classes: np.ndarray = None,
    bev_range = [-50, -10, -5, 50, 100, 3],
    gt_boxes: np.ndarray = None,
    meter_per_pixel: float = 0.1,
):
    """
    BEV 鸟瞰图绘制

    Args:
        points_xyz: (N,3) 点云 [x,y,z]，单位 meter
        pred_boxes: (M,7) [x,y,z,l,w,h,yaw]
        classes:    (M,) 可选
        bev_range:  [xmin,ymin,zmin,xmax,ymax,zmax]
        gt_boxes:   (K,7) 可选
        meter_per_pixel: 1 pixel 对应多少 meter（0.1 → 1m=10px）

    Returns:
        bev_img: (H,W,3) BGR
    """

    xmin, ymin, zmin, xmax, ymax, zmax = bev_range

    W = int((xmax - xmin) / meter_per_pixel)
    H = int((ymax - ymin) / meter_per_pixel)

    bev_img = np.zeros((H, W, 3), dtype=np.uint8)

    # =========================================================
    # 1. 绘制点云（高度 + 稠密度）
    # =========================================================
    mask = (
        (points_xyz[:, 0] >= xmin) & (points_xyz[:, 0] <= xmax) &
        (points_xyz[:, 1] >= ymin) & (points_xyz[:, 1] <= ymax) &
        (points_xyz[:, 2] >= zmin) & (points_xyz[:, 2] <= zmax)
    )
    pts = points_xyz[mask]

    if pts.shape[0] > 0:
        # meter → pixel
        px = ((pts[:, 0] - xmin) / meter_per_pixel).astype(np.int32)
        py = ((pts[:, 1] - ymin) / meter_per_pixel).astype(np.int32)
        py = H - py - 1

        # 高度归一化
        z_norm = (pts[:, 2] - zmin) / (zmax - zmin + 1e-6)
        z_norm = np.clip(z_norm, 0.0, 1.0)

        # 稠密度统计
        density = {}
        for x, y in zip(px, py):
            density[(x, y)] = density.get((x, y), 0) + 1
        max_density = max(density.values()) if density else 1

        for i in range(len(px)):
            d = density[(px[i], py[i])] / max_density
            h = z_norm[i]

            color = np.array([
                int(255 * d),         # 蓝：稠密
                int(255 * h),         # 绿：高度
                int(255 * (1 - h)),   # 红：高度反向
            ])

            bev_img[py[i], px[i]] = np.maximum(
                bev_img[py[i], px[i]], color
            )

    # =========================================================
    # 2. 绘制预测框（蓝色）
    # =========================================================
    if pred_boxes is not None and len(pred_boxes) > 0:
        for box in pred_boxes:
            draw_bev_box(
                bev_img,
                box,
                xmin,
                ymin,
                meter_per_pixel,
                color=(255, 0, 0),
                thickness=2,
            )

    # =========================================================
    # 3. 绘制 GT 框（绿色）
    # =========================================================
    if gt_boxes is not None and len(gt_boxes) > 0:
        for box in gt_boxes:
            draw_bev_box(
                bev_img,
                box,
                xmin,
                ymin,
                meter_per_pixel,
                color=(0, 255, 0),
                thickness=2,
            )

    return bev_img


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize FastBEV1V ONNX inference results')
    parser.add_argument('--config', type=str, required=True, help='config file path')
    parser.add_argument('--onnx-2d', type=str, required=True, help='Path to 2D ONNX model')
    parser.add_argument('--onnx-3d', type=str, required=True, help='Path to 3D ONNX model')
    parser.add_argument('--save-dir', type=str, default='./vis_result_onnx', 
                       help='Directory to save visualization results')
    parser.add_argument('--pred-thr', type=float, default=0.4, 
                       help='Prediction score threshold')
    parser.add_argument('--cpu', action='store_true', 
                       help='Run on CPU (avoid CUDA extension issues)')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of samples to process')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 设置ONNX模型路径环境变量
    os.environ['ONNX_2D_MODEL_PATH'] = args.onnx_2d
    os.environ['ONNX_3D_MODEL_PATH'] = args.onnx_3d
    
    # 检查ONNX模型文件是否存在
    if not os.path.exists(args.onnx_2d):
        raise FileNotFoundError(f"2D ONNX model not found: {args.onnx_2d}")
    if not os.path.exists(args.onnx_3d):
        raise FileNotFoundError(f"3D ONNX model not found: {args.onnx_3d}")
    
    print('=' * 80)
    print('FastBEV1V ONNX Visualization')
    print('=' * 80)
    print(f'Config: {args.config}')
    print(f'2D ONNX Model: {args.onnx_2d}')
    print(f'3D ONNX Model: {args.onnx_3d}')
    print(f'Save directory: {args.save_dir}')
    print()
    
    # 加载配置
    cfg = Config.fromfile(args.config)
    
    # 强制单卡 / 单进程
    cfg.gpu_ids = [0]
    cfg.data.samples_per_gpu = 1
    cfg.data.workers_per_gpu = 0
    
    # 设置test_mode为test_onnx，并传递ONNX模型路径
    if cfg.get('model', {}).get('test_cfg', None) is not None:
        cfg.model.test_cfg.test_mode = 'test_onnx'
        cfg.model.test_cfg.onnx_2d_path = args.onnx_2d
        cfg.model.test_cfg.onnx_3d_path = args.onnx_3d
    else:
        cfg.model.test_cfg = dict(
            test_mode='test_onnx',
            onnx_2d_path=args.onnx_2d,
            onnx_3d_path=args.onnx_3d
        )
    
    # 构建数据集
    print('[1] Building dataset...')
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
    
    dataset = build_dataset(cfg.data.test)
    
    print('[2] Building dataloader...')
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,
        dist=False,
        shuffle=False,
    )
    
    # 构建模型（不需要加载checkpoint，因为使用ONNX模型）
    print('[3] Building model...')
    cfg.model.train_cfg = None
    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg')
    )
    
    model.eval()
    
    # 根据参数决定使用CPU还是GPU
    if args.cpu:
        print("Running on CPU mode")
        device = torch.device('cpu')
        model = model.cpu()
    else:
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            model = model.cuda()
            print("Model moved to GPU")
        else:
            print("WARNING: CUDA not available, falling back to CPU")
            device = torch.device('cpu')
            model = model.cpu()
    
    # 创建保存目录
    VIS_SAVE_PATH = args.save_dir
    os.makedirs(VIS_SAVE_PATH, exist_ok=True)
    print(f'[4] Saving visualization results to: {VIS_SAVE_PATH}')
    print()
    
    # 可视化结果
    print('[5] Running ONNX inference and visualization...')
    with torch.no_grad():
        for i, data in tqdm.tqdm(enumerate(data_loader), total=len(data_loader)):
            if args.max_samples is not None and i >= args.max_samples:
                break
                
            # 将数据移到正确的设备
            if device.type == 'cuda':
                # scatter data to GPU
                data = scatter(data, [device.index])[0]
            else:
                # CPU模式：解包DataContainer
                if isinstance(data.get('img'), DataContainer):
                    data['img'] = data['img'].data
                if isinstance(data.get('img_metas'), DataContainer):
                    data['img_metas'] = data['img_metas'].data
            
            # ONNX推理
            result = model(return_loss=False, rescale=True, **data)
            
            # 获取文件名
            img_metas = data.get('img_metas')
            if isinstance(img_metas, DataContainer):
                img_metas = img_metas.data
            if isinstance(img_metas, list) and len(img_metas) > 0:
                if isinstance(img_metas[0], list) and len(img_metas[0]) > 0:
                    filename = img_metas[0][0].get('filename', f'sample_{i}.jpg')
                else:
                    filename = img_metas[0].get('filename', f'sample_{i}.jpg')
            else:
                filename = f'sample_{i}.jpg'
            
            filename = os.path.basename(filename)
            
            # 获取图像归一化参数
            if isinstance(img_metas, list) and len(img_metas) > 0:
                if isinstance(img_metas[0], list) and len(img_metas[0]) > 0:
                    img_norm_cfg = img_metas[0][0].get('img_norm_cfg', {})
                else:
                    img_norm_cfg = img_metas[0].get('img_norm_cfg', {})
            else:
                img_norm_cfg = {}
            
            mean = img_norm_cfg.get('mean', [123.675, 116.28, 103.53])
            std = img_norm_cfg.get('std', [58.395, 57.12, 57.375])
            
            # 获取输入图像（参考vis_fasebev1v.py的处理方式）
            img_data = data.get('img')
            if isinstance(img_data, DataContainer):
                img_data = img_data.data
            
            # 处理嵌套的数据结构：data[0][0][0] 或 data[0][0]
            if isinstance(img_data, list) and len(img_data) > 0:
                if isinstance(img_data[0], list) and len(img_data[0]) > 0:
                    if isinstance(img_data[0][0], list) and len(img_data[0][0]) > 0:
                        input_img = img_data[0][0][0]
                    else:
                        input_img = img_data[0][0]
                else:
                    input_img = img_data[0]
            else:
                input_img = img_data
            
            # 转换为numpy数组
            if isinstance(input_img, torch.Tensor):
                input_img = input_img.cpu().numpy()
            
            # 处理不同维度的图像数据
            # 目标格式: (C, H, W) -> (H, W, C)
            if len(input_img.shape) == 3:
                # 已经是 (C, H, W) 格式
                input_img = input_img.transpose(1, 2, 0)
            elif len(input_img.shape) == 4:
                # (B, C, H, W) 或 (1, C, H, W)，取第一个batch
                input_img = input_img[0].transpose(1, 2, 0)
            elif len(input_img.shape) == 5:
                # (B, 1, C, H, W) 或 (1, 1, C, H, W)，取 [0][0]
                input_img = input_img[0][0].transpose(1, 2, 0)
            else:
                print(f"Warning: Unexpected image shape: {input_img.shape}, skipping...")
                continue
            input_img = input_img * np.array(std) + np.array(mean)
            input_img = np.clip(input_img[:,:,::-1], 0, 255).astype(np.uint8)
            
            # 获取点云（参考vis_fasebev1v.py的处理方式）
            points_data = data.get('points')
            if isinstance(points_data, DataContainer):
                points_data = points_data.data
            
            # 处理嵌套的数据结构：data[0][0] 或 data[0]
            if isinstance(points_data, list) and len(points_data) > 0:
                if isinstance(points_data[0], list) and len(points_data[0]) > 0:
                    input_points = points_data[0][0]
                else:
                    input_points = points_data[0]
            else:
                input_points = points_data[0] if isinstance(points_data, list) else points_data
            
            # 转换为numpy数组
            if isinstance(input_points, torch.Tensor):
                input_points = input_points.cpu().numpy()
            
            # 确保点云是2维的 (N, 3+)
            if len(input_points.shape) == 2:
                input_points = input_points[:, :3]
            else:
                print(f"Warning: Unexpected points shape: {input_points.shape}, skipping...")
                continue
            
            # 获取预测结果
            if isinstance(result, list) and len(result) > 0:
                result_dict = result[0]
            else:
                result_dict = result
            
            pred_bboxs = result_dict.get('boxes_3d').tensor.cpu().numpy()[:, :7]
            pred_scores = result_dict.get('scores_3d').cpu().numpy()
            pred_labels = result_dict.get('labels_3d').cpu().numpy()
            
            # 应用阈值过滤
            mask = pred_scores > args.pred_thr
            pred_bboxs = pred_bboxs[mask]
            pred_labels = pred_labels[mask]
            
            # 获取lidar2img变换矩阵（参考vis_fasebev1v.py的处理方式）
            if isinstance(img_metas, list) and len(img_metas) > 0:
                if isinstance(img_metas[0], list) and len(img_metas[0]) > 0:
                    lidar2img_dict = img_metas[0][0].get('lidar2img', {})
                else:
                    lidar2img_dict = img_metas[0].get('lidar2img', {})
            else:
                lidar2img_dict = {}
            
            # 获取extrinsic矩阵（参考vis_fasebev1v.py的处理方式）
            lidar2img = lidar2img_dict.get('extrinsic', np.eye(4))
            
            # 如果是列表，取第一个元素
            if isinstance(lidar2img, list):
                if len(lidar2img) > 0:
                    lidar2img = lidar2img[0]
                else:
                    lidar2img = np.eye(4)
            
            # 转换为numpy数组
            if isinstance(lidar2img, torch.Tensor):
                lidar2img = lidar2img.cpu().numpy()
            elif not isinstance(lidar2img, np.ndarray):
                # 如果不是numpy数组，尝试转换
                lidar2img = np.array(lidar2img)
            
            # 处理可能的batch维度
            if isinstance(lidar2img, np.ndarray):
                if len(lidar2img.shape) > 2:
                    lidar2img = lidar2img[0]
                elif len(lidar2img.shape) == 1:
                    # 如果是1维，可能是展平的，需要reshape
                    if lidar2img.shape[0] == 16:
                        lidar2img = lidar2img.reshape(4, 4)
                elif len(lidar2img.shape) == 0:
                    # 如果是标量，使用单位矩阵
                    lidar2img = np.eye(4)
            else:
                # 如果仍然不是numpy数组，使用单位矩阵
                lidar2img = np.eye(4)
            
            # 在图像上绘制预测框
            input_img_draw = draw_bbox(input_img.copy(), lidar2img, pred_bboxs, (255, 0, 0))
            cv2.putText(input_img_draw, "ONNX PRED", (50, 50), 
                       cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255, 0, 0), 2)
            
            # 绘制BEV图
            bev_img_with_predbox = draw_bev(
                points_xyz=input_points,
                pred_boxes=pred_bboxs,
                classes=pred_labels,
            )
            
            cv2.putText(bev_img_with_predbox, "ONNX PRED", (50, 100), 
                       cv2.FONT_HERSHEY_COMPLEX_SMALL, 3, (255, 0, 0), 2)
            
            # 拼接图像和BEV图
            _h, _w, _c = input_img_draw.shape
            _bev_h, _bev_w, _bev_c = bev_img_with_predbox.shape
            _bev_w_new = int(_bev_w * _h / _bev_h)
            
            bev_img_with_predbox = cv2.resize(bev_img_with_predbox, (_bev_w_new, _h))
            combined_img = np.hstack([input_img_draw, bev_img_with_predbox])
            
            # 保存结果
            save_path = os.path.join(VIS_SAVE_PATH, filename)
            cv2.imwrite(save_path, combined_img)
    
    print(f'\nCompleted! Saved {len(os.listdir(VIS_SAVE_PATH))} visualization results to {VIS_SAVE_PATH}')


if __name__ == '__main__':
    main()

