#!/usr/bin/env python3
"""
LVFusion NuScenes (NuScenesCAMFrontDataset) 训练日志可视化脚本。

可选加载 cam、lidar、fusion 的训练日志，提取 loss 与验证指标（NuScenes 协议：
mAP、NDS、mATE、mASE、mAOE 等），绘制折线图并保存到指定目录（按日期时间创建子目录）。
"""

import argparse
import json
import math
import os
import random
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 图例与颜色
STYLE = {
    'cam':    {'label': 'cam',    'color': '#1f77b4', 'linestyle': '-'},
    'lidar':  {'label': 'lidar',  'color': '#2ca02c', 'linestyle': '-'},
    'fusion': {'label': 'fusion', 'color': '#d62728', 'linestyle': '-'},
}
DEFAULT_COLORS = ['#1f77b4', '#2ca02c', '#d62728', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
VALID_LINE_STYLES = ('-', '--', '*', '-*', '.')
_RGB_COLOR_RE = re.compile(r'^\(?\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)?$')
MPL_LINESTYLE = {
    '-': '-',
    '--': '--',
    '-*': '-.',
    '.': ':',
    '*': (0, (6, 2, 2, 2)),
}

# NuScenes 验证指标在 log 中的 key（训练时可能记录为 img_bbox_NuScenes 或 pts_bbox_NuScenes）
NUS_METRIC_PREFIX = 'img_bbox_NuScenes'
NUS_METRIC_PREFIX_PTS = 'pts_bbox_NuScenes'


def parse_rgb_color(s):
    m = _RGB_COLOR_RE.match(s.strip())
    if m is None:
        raise ValueError(f'无效 RGB 颜色: {s!r}，示例: (255,50,50)')
    rgb = tuple(int(v) for v in m.groups())
    if any(v < 0 or v > 255 for v in rgb):
        raise ValueError(f'RGB 取值需在 [0, 255]，收到: {rgb}')
    return rgb


def rgb_to_hex(rgb):
    r, g, b = rgb
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)


def normalize_cli_argv(argv, dash_value_options):
    option_set = set(dash_value_options)
    normalized = []
    i = 0
    while i < len(argv):
        token = argv[i]
        if token in option_set and i + 1 < len(argv) and '=' not in token:
            nxt = argv[i + 1]
            if not nxt.startswith('--') and nxt.startswith('-'):
                normalized.append(f'{token}={nxt}')
                i += 2
                continue
        normalized.append(token)
        i += 1
    return normalized


def parse_line_styles_arg(s):
    if not s or not s.strip():
        return None
    sep = ';' if ';' in s else ','
    styles = [x.strip() for x in s.split(sep) if x.strip()]
    for style in styles:
        if style not in VALID_LINE_STYLES:
            raise ValueError(f'无效线型: {style!r}，可选: {VALID_LINE_STYLES}')
    return styles


def parse_colors_arg(s):
    if not s or not s.strip():
        return None
    return [parse_rgb_color(part) for part in s.split(';') if part.strip()]


def to_mpl_linestyle(style):
    return MPL_LINESTYLE.get(style, '-')


def get_style(name, index=0, line_styles=None, colors_rgb=None):
    label = STYLE[name]['label'] if name in STYLE else name
    if line_styles:
        linestyle = to_mpl_linestyle(line_styles[index % len(line_styles)])
    elif name in STYLE:
        linestyle = to_mpl_linestyle(STYLE[name]['linestyle'])
    else:
        linestyle = '-'

    if colors_rgb:
        color = rgb_to_hex(colors_rgb[index % len(colors_rgb)])
    else:
        color = rgb_to_hex((
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        ))
    return {'label': label, 'color': color, 'linestyle': linestyle}


def build_style_map(names, line_styles=None, colors_rgb=None):
    return {
        name: get_style(name, idx, line_styles=line_styles, colors_rgb=colors_rgb)
        for idx, name in enumerate(names)
    }


def smooth_curve(values, weight=0.9):
    if len(values) == 0:
        return values
    smoothed = [values[0]]
    for v in values[1:]:
        smoothed.append(weight * smoothed[-1] + (1 - weight) * v)
    return np.array(smoothed)


def load_log_json(path):
    """加载 JSON 行格式的 log，返回 (train_records, val_records)。"""
    train_records = []
    val_records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            mode = record.get('mode')
            if mode == 'train':
                train_records.append(record)
            elif mode == 'val':
                val_records.append(record)
    return train_records, val_records


def extract_train_loss_simple(records):
    losses = []
    for r in records:
        if 'loss' in r:
            losses.append(float(r['loss']))
    if not losses:
        return np.array([]), np.array([])
    return np.arange(len(losses)), np.array(losses, dtype=float)


def safe_float(v):
    """转为 float，NaN/None/字符串 'nan' 视为 float('nan')。"""
    if v is None:
        return float('nan')
    if isinstance(v, str) and v.strip().lower() in ('nan', 'null', ''):
        return float('nan')
    try:
        x = float(v)
        return x if math.isfinite(x) else float('nan')
    except (TypeError, ValueError):
        return float('nan')


def extract_val_metric(records, key, prefix_alt=None):
    """从 val 记录中提取 (epoch, value)。若 key 不存在则尝试 prefix_alt 替代前缀的 key。"""
    epochs = []
    values = []
    for r in records:
        v = r.get(key)
        if v is None and prefix_alt:
            alt_key = key.replace(NUS_METRIC_PREFIX, prefix_alt, 1) if NUS_METRIC_PREFIX in key else key.replace(NUS_METRIC_PREFIX_PTS, prefix_alt, 1)
            v = r.get(alt_key)
        if v is not None:
            epochs.append(int(r.get('epoch', 0)))
            values.append(safe_float(v))
    return np.array(epochs), np.array(values)


def ensure_output_dir(base_dir):
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    subdir = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = base_dir / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def resample_to_length(values, target_length):
    if len(values) == 0 or target_length <= 0:
        return np.array([])
    if len(values) == target_length:
        return np.asarray(values, dtype=float)
    old_indices = np.linspace(0, len(values) - 1, target_length, dtype=float)
    return np.interp(old_indices, np.arange(len(values)), np.asarray(values, dtype=float))


def plot_loss(logs_data, out_dir, style_map, smooth_weight=0.9):
    smoothed_losses = {}
    for name, (train_records, _) in logs_data.items():
        _, loss = extract_train_loss_simple(train_records)
        if len(loss) == 0:
            continue
        smoothed_losses[name] = smooth_curve(loss, weight=smooth_weight)
    if not smoothed_losses:
        return
    common_len = min(len(v) for v in smoothed_losses.values())
    fig, ax = plt.subplots(figsize=(10, 6))
    x_common = np.arange(common_len)
    for name, loss_smooth in smoothed_losses.items():
        loss_resampled = resample_to_length(loss_smooth, common_len)
        s = style_map[name]
        ax.plot(x_common, loss_resampled, label=s['label'], color=s['color'], linestyle=s['linestyle'], linewidth=2)
    ax.set_xlabel('Training Step (aligned)', fontsize=12)
    ax.set_ylabel('Loss (smoothed)', fontsize=12)
    ax.set_title('Training Loss Comparison (LVFusion NuScenes)', fontsize=14)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('symlog', linthresh=1.0)
    fig.tight_layout()
    fig.savefig(out_dir / '01_train_loss.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_single_metric(
    logs_data,
    metric_key,
    ylabel,
    title,
    filename,
    out_dir,
    higher_better=True,
    prefix_alt=None,
    style_map=None,
):
    """绘制单指标对比；自动过滤 NaN 以便绘图。"""
    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, (name, (_, val_records)) in enumerate(logs_data.items()):
        epochs, values = extract_val_metric(val_records, metric_key, prefix_alt=prefix_alt)
        if len(epochs) == 0:
            continue
        # 过滤 NaN 以便连线不中断（可选：用 nan 会断线）
        mask = np.isfinite(values)
        if not np.any(mask):
            continue
        ep, val = epochs[mask], values[mask]
        s = style_map[name]
        ax.plot(
            ep,
            val,
            'o',
            label=s['label'],
            color=s['color'],
            linestyle=s['linestyle'],
            linewidth=2,
            markersize=6,
        )
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='lower right' if higher_better else 'upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='LVFusion NuScenes 训练日志可视化')
    parser.add_argument('--name_list', type=str, default='',
                        help='图例名称，逗号分隔，如 cam,lidar,fusion')
    parser.add_argument('--json_list', type=str, default='',
                        help='log.json 路径列表，逗号分隔，与 name_list 一一对应')
    parser.add_argument('--out-dir', type=str, default='./work_dirs/lvfusion_nus_vis',
                        help='图片保存根目录，将在此下按日期时间创建子目录')
    parser.add_argument('--smooth', type=float, default=0.9, help='Loss 平滑系数 (0~1)')
    parser.add_argument(
        '--line_styles',
        type=str,
        default='',
        help='线型列表（逗号/分号分隔），如 -,-,-,*,*,*；也可用 --line_styles=-,-,-,*,*,*',
    )
    parser.add_argument(
        '--colors',
        type=str,
        default='',
        help='RGB 颜色列表（分号分隔），如 255,50,50;0,255,0；不足时循环；默认随机颜色',
    )
    args = parser.parse_args(
        normalize_cli_argv(sys.argv[1:], ('--line_styles',))
    )

    try:
        line_styles = parse_line_styles_arg(args.line_styles)
        colors_rgb = parse_colors_arg(args.colors)
    except ValueError as exc:
        print(f'错误: {exc}')
        return

    work = Path('./work_dirs')
    default_names = ['cam', 'lidar', 'fusion']
    default_jsons = [
        str(work / 'nus_cam_r50_s512x1408_v015_lvfusion_fpn_centerhead/20260228_124611.log.json'),
        str(work / 'nus_lidar_r50_s512x1408_v015_lvfusion_fpn_centerhead/20260228_102837.log.json'),
        str(work / 'nus_fusion_r50_s512x1408_v015_lvfusion_fpn_centerhead/20260228_130000.log.json'),
    ]

    if args.name_list.strip() and args.json_list.strip():
        names = [s.strip() for s in args.name_list.split(',') if s.strip()]
        jsons = [s.strip() for s in args.json_list.split(',') if s.strip()]
        if len(names) != len(jsons):
            print(f'错误: name_list 与 json_list 数量不一致 ({len(names)} vs {len(jsons)})')
            return
        paths = list(zip(names, jsons))
    else:
        paths = list(zip(default_names, default_jsons))
        # 仅保留存在的日志文件
        paths = [(n, j) for n, j in paths if Path(j).exists()]
        if not paths:
            paths = list(zip(default_names, default_jsons))

    logs_data = {}
    for name, path in paths:
        path = Path(path)
        if not path.exists():
            print(f'[Skip] 文件不存在: {path}')
            continue
        train_records, val_records = load_log_json(path)
        logs_data[name] = (train_records, val_records)
        print(f'[Load] {name}: train={len(train_records)}, val={len(val_records)}')

    if not logs_data:
        print('没有可用的日志文件，退出.')
        return

    out_dir = ensure_output_dir(args.out_dir)
    print(f'[Output] 保存目录: {out_dir}')

    # 兼容 log 里写 pts_bbox_NuScenes 的情况
    prefix_alt = NUS_METRIC_PREFIX_PTS if NUS_METRIC_PREFIX != NUS_METRIC_PREFIX_PTS else None
    style_map = build_style_map(list(logs_data.keys()), line_styles, colors_rgb)

    plot_loss(logs_data, out_dir, style_map, smooth_weight=args.smooth)

    metric_plots = [
        (f'{NUS_METRIC_PREFIX}/mAP', 'mAP', f'Validation {NUS_METRIC_PREFIX}/mAP', '02_val_mAP.png', True),
        (f'{NUS_METRIC_PREFIX}/NDS', 'NDS', f'Validation {NUS_METRIC_PREFIX}/NDS', '03_val_NDS.png', True),
        (f'{NUS_METRIC_PREFIX}/mATE', 'mATE', f'Validation {NUS_METRIC_PREFIX}/mATE', '04_val_mATE.png', False),
        (f'{NUS_METRIC_PREFIX}/mASE', 'mASE', f'Validation {NUS_METRIC_PREFIX}/mASE', '05_val_mASE.png', False),
        (f'{NUS_METRIC_PREFIX}/mAOE', 'mAOE', f'Validation {NUS_METRIC_PREFIX}/mAOE', '06_val_mAOE.png', False),
    ]
    for metric_key, ylabel, title, filename, higher_better in metric_plots:
        plot_single_metric(
            logs_data,
            metric_key,
            ylabel,
            title,
            filename,
            out_dir,
            higher_better=higher_better,
            prefix_alt=prefix_alt,
            style_map=style_map,
        )

    print('可视化完成。')


if __name__ == '__main__':
    main()
