#!/usr/bin/env python3
# Copyright (c) OpenMMLab. All rights reserved.
"""
统计 hasco 数据集的标签信息：
- 标签类别
- 各类别数量
- 目标距离分布

用法:
    python tools/stat_hasco_labels.py [--data-root /path/to/data] [--output output.txt]
"""

import argparse
import os
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
from tqdm import tqdm


def parse_label_file(label_path):
    """解析标签文件，提取类别和位置信息。
    
    Args:
        label_path (str): 标签文件路径
        
    Returns:
        list: 每个元素为 (class_name, x, y, z) 的元组列表
    """
    labels = []
    
    if not os.path.exists(label_path):
        return labels
    
    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            
            # 提取: x, y, z, l, w, h, yaw (前7个值)
            # parts[7] 是 ID (跳过)
            # parts[8] 是类别名称
            try:
                x = float(parts[0])
                y = float(parts[1])
                z = float(parts[2])
                class_name = parts[8].strip()
                
                labels.append((class_name, x, y, z))
            except (ValueError, IndexError) as e:
                # 跳过无法解析的行
                continue
    except Exception as e:
        print(f"Warning: Failed to parse {label_path}: {e}")
    
    return labels


def calculate_distance(x, y, z):
    """计算从原点到目标中心的距离。
    
    Args:
        x, y, z (float): 目标中心坐标
        
    Returns:
        float: 距离（米）
    """
    return np.sqrt(x**2 + y**2 + z**2)


def find_labels_folders(root_path):
    """查找所有 labels 文件夹。
    
    使用 os.walk(followlinks=True)：数据根目录下常见「日期文件夹 -> 真实路径」的软链接，
    pathlib.rglob 默认不会进入符号链接目录，会漏掉所有 labels。
    
    Args:
        root_path (str): 根目录路径
        
    Returns:
        list: labels 文件夹路径列表
    """
    labels_folders = []
    root_path = Path(root_path)
    
    if not root_path.exists():
        print(f"Error: Path {root_path} does not exist!")
        return labels_folders
    
    root_str = str(root_path)
    for dirpath, _, _ in os.walk(root_str, followlinks=True):
        if os.path.basename(dirpath) == 'labels':
            p = Path(dirpath)
            if p.is_dir():
                labels_folders.append(p)
    
    return sorted(labels_folders)


def find_label_files(labels_folder):
    """查找 labels 文件夹中的所有 txt 标签文件。
    
    Args:
        labels_folder (Path): labels 文件夹路径
        
    Returns:
        list: 标签文件路径列表
    """
    label_files = []
    for label_file in labels_folder.glob('*.txt'):
        if label_file.is_file():
            label_files.append(label_file)
    return sorted(label_files)


def stat_labels(data_root, output_file=None):
    """统计标签信息。
    
    Args:
        data_root (str): 数据根目录
        output_file (str, optional): 输出文件路径，如果为 None 则只打印到控制台
    """
    print(f"正在扫描数据目录: {data_root}")
    
    # 查找所有 labels 文件夹
    labels_folders = find_labels_folders(data_root)
    
    if not labels_folders:
        print(f"未找到任何 labels 文件夹！")
        return
    
    print(f"找到 {len(labels_folders)} 个 labels 文件夹")
    
    # 统计信息
    class_counter = Counter()  # 类别计数器
    class_distances = defaultdict(list)  # 每个类别的距离列表
    all_distances = []  # 所有目标的距离列表
    
    total_files = 0
    total_labels = 0
    
    # 遍历所有 labels 文件夹
    for labels_folder in tqdm(labels_folders, desc="处理 labels 文件夹"):
        label_files = find_label_files(labels_folder)
        
        for label_file in label_files:
            total_files += 1
            labels = parse_label_file(str(label_file))
            
            for class_name, x, y, z in labels:
                total_labels += 1
                class_counter[class_name] += 1
                
                # 计算距离
                distance = calculate_distance(x, y, z)
                class_distances[class_name].append(distance)
                all_distances.append(distance)
    
    # 输出统计结果
    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append("Hasco 数据集标签统计报告")
    output_lines.append("=" * 80)
    output_lines.append(f"\n数据根目录: {data_root}")
    output_lines.append(f"标签文件总数: {total_files}")
    output_lines.append(f"目标总数: {total_labels}")
    output_lines.append(f"类别总数: {len(class_counter)}")
    output_lines.append("\n" + "-" * 80)
    
    # 类别统计
    output_lines.append("\n【类别统计】")
    output_lines.append("-" * 80)
    output_lines.append(f"{'类别':<20} {'数量':<10} {'占比':<10} {'平均距离(m)':<15} {'最小距离(m)':<15} {'最大距离(m)':<15}")
    output_lines.append("-" * 80)
    
    # 按数量排序
    for class_name, count in class_counter.most_common():
        percentage = count / total_labels * 100 if total_labels > 0 else 0
        distances = class_distances[class_name]
        avg_dist = np.mean(distances) if distances else 0
        min_dist = np.min(distances) if distances else 0
        max_dist = np.max(distances) if distances else 0
        
        output_lines.append(
            f"{class_name:<20} {count:<10} {percentage:>6.2f}%   "
            f"{avg_dist:>12.2f}   {min_dist:>12.2f}   {max_dist:>12.2f}"
        )
    
    # 距离分布统计
    output_lines.append("\n" + "-" * 80)
    output_lines.append("\n【距离分布统计】")
    output_lines.append("-" * 80)
    
    if all_distances:
        all_distances = np.array(all_distances)
        output_lines.append(f"总体距离统计:")
        output_lines.append(f"  平均距离: {np.mean(all_distances):.2f} m")
        output_lines.append(f"  中位数距离: {np.median(all_distances):.2f} m")
        output_lines.append(f"  最小距离: {np.min(all_distances):.2f} m")
        output_lines.append(f"  最大距离: {np.max(all_distances):.2f} m")
        output_lines.append(f"  标准差: {np.std(all_distances):.2f} m")
        
        # 距离区间统计
        output_lines.append(f"\n距离区间分布:")
        bins = [0, 10, 20, 30, 50, 75, 100, 150, 200, float('inf')]
        bin_labels = ['0-10', '10-20', '20-30', '30-50', '50-75', '75-100', '100-150', '150-200', '200+']
        
        hist, _ = np.histogram(all_distances, bins=bins)
        for label, count in zip(bin_labels, hist):
            percentage = count / len(all_distances) * 100
            output_lines.append(f"  {label:>10} m: {count:>6} ({percentage:>5.2f}%)")
    
    # 每个类别的距离分布
    output_lines.append("\n" + "-" * 80)
    output_lines.append("\n【各类别距离分布】")
    output_lines.append("-" * 80)
    
    for class_name in sorted(class_counter.keys()):
        distances = np.array(class_distances[class_name])
        if len(distances) > 0:
            output_lines.append(f"\n{class_name}:")
            output_lines.append(f"  平均距离: {np.mean(distances):.2f} m")
            output_lines.append(f"  中位数距离: {np.median(distances):.2f} m")
            output_lines.append(f"  最小距离: {np.min(distances):.2f} m")
            output_lines.append(f"  最大距离: {np.max(distances):.2f} m")
            output_lines.append(f"  标准差: {np.std(distances):.2f} m")
            
            # 距离区间分布
            hist, _ = np.histogram(distances, bins=bins)
            output_lines.append(f"  距离区间分布:")
            for label, count in zip(bin_labels, hist):
                if count > 0:
                    percentage = count / len(distances) * 100
                    output_lines.append(f"    {label:>10} m: {count:>6} ({percentage:>5.2f}%)")
    
    output_lines.append("\n" + "=" * 80)
    
    # 输出结果
    output_text = "\n".join(output_lines)
    print(output_text)
    
    # 保存到文件
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(output_text)
        print(f"\n统计结果已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='统计 hasco 数据集的标签信息',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 使用默认路径
    python tools/stat_hasco_labels.py
    
    # 指定数据路径
    python tools/stat_hasco_labels.py --data-root /path/to/hasco/data
    
    # 指定输出文件
    python tools/stat_hasco_labels.py --output stats.txt
        """
    )
    
    parser.add_argument(
        '--data-root',
        type=str,
        default='./data/hasco',
        help='数据根目录路径（默认: ./data/hasco）'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='输出文件路径（可选，如果不指定则只打印到控制台）'
    )
    
    args = parser.parse_args()
    
    stat_labels(args.data_root, args.output)


if __name__ == '__main__':
    main()
