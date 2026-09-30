#!/usr/bin/env python3
from __future__ import annotations

"""
统计目录下 PCD 文件中点 id 的构成，并按 id 组合归类汇总。

支持递归遍历子文件夹。对每个 PCD 文件，收集该文件内所有点的唯一 id 值，
将具有相同 id 构成的文件归为一类，再按所在文件夹统计数量。

默认按根目录下各子文件夹拆分任务，使用多进程并行处理。

用法:
    python tools/stat_pcd_ids.py /data2/hasco_lv_data/FVLV_data_ouster
    python tools/stat_pcd_ids.py /path/to/root --workers 8
    python tools/stat_pcd_ids.py /path/to/root --no-recursive
    python tools/stat_pcd_ids.py /path/to/root -o report.txt
"""

import argparse
import time
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from pathlib import Path


def parse_pcd_ids(pcd_path: Path) -> tuple[frozenset[str], str | None]:
    """流式读取 PCD，返回 (唯一 id 集合, 错误信息)。"""
    try:
        with open(pcd_path, 'r', encoding='utf-8', errors='ignore') as f:
            fields: list[str] = []
            id_idx = -1
            in_data = False
            unique_ids: set[str] = set()

            for line in f:
                if not in_data:
                    line_s = line.strip()
                    if line_s.startswith('FIELDS'):
                        fields = line_s.split()[1:]
                        if 'id' not in fields:
                            return frozenset(), f'无 id 字段 (fields={fields})'
                        id_idx = fields.index('id')
                    elif line_s.startswith('DATA'):
                        if not fields:
                            return frozenset(), '缺少 FIELDS'
                        in_data = True
                    continue

                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) > id_idx:
                    unique_ids.add(parts[id_idx])

            if not in_data:
                return frozenset(), '缺少 DATA 段'
    except OSError as e:
        return frozenset(), str(e)

    return frozenset(unique_ids), None


def sort_ids(ids: frozenset[str]) -> list[str]:
    """将 id 按数值排序，非数字排在后面。"""

    def sort_key(v: str):
        try:
            return (0, int(v))
        except ValueError:
            return (1, v)

    return sorted(ids, key=sort_key)


def format_id_key(ids: frozenset[str]) -> str:
    """格式化为 【1】 或 【1，4】。"""
    if not ids:
        return '【空】'
    return '【' + '，'.join(sort_ids(ids)) + '】'


def get_folder_label(pcd_path: Path, root: Path, collapse_pcds: bool) -> str:
    """获取用于归类的文件夹标签。"""
    parent = pcd_path.parent.resolve()
    root = root.resolve()
    try:
        rel = parent.relative_to(root)
    except ValueError:
        return str(parent)

    parts = list(rel.parts)
    if collapse_pcds and parts and parts[-1] == 'pcds':
        parts = parts[:-1]
    if not parts:
        return parent.name
    return '/'.join(parts)


def discover_work_units(root: Path, recursive: bool) -> list[Path]:
    """将根目录拆分为可并行处理的工作单元（一级子文件夹）。"""
    root = root.resolve()
    if not recursive:
        return [root]

    units: list[Path] = []
    if any(root.glob('*.pcd')):
        units.append(root)

    subdirs = sorted(p for p in root.iterdir() if p.is_dir())
    units.extend(subdirs)

    return units if units else [root]


def collect_pcd_files(search_root: Path, within_unit_recursive: bool) -> list[Path]:
    if within_unit_recursive:
        return sorted(search_root.rglob('*.pcd'))
    return sorted(search_root.glob('*.pcd'))


def process_work_unit(
    unit_root: str,
    root: str,
    collapse_pcds: bool,
    within_unit_recursive: bool,
) -> tuple[dict[frozenset[str], dict[str, int]], list[tuple[str, str]], int]:
    """处理单个工作单元，返回局部统计结果。"""
    unit_path = Path(unit_root)
    root_path = Path(root)

    id_folder_counts: dict[frozenset[str], dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    errors: list[tuple[str, str]] = []
    pcd_files = collect_pcd_files(unit_path, within_unit_recursive)

    for pcd_path in pcd_files:
        ids, err = parse_pcd_ids(pcd_path)
        if err is not None:
            try:
                rel = pcd_path.relative_to(root_path)
            except ValueError:
                rel = pcd_path
            errors.append((str(rel), err))
            continue
        folder = get_folder_label(pcd_path, root_path, collapse_pcds)
        id_folder_counts[ids][folder] += 1

    return dict(id_folder_counts), errors, len(pcd_files)


def merge_results(
    results: list[tuple[dict[frozenset[str], dict[str, int]], list[tuple[str, str]], int]],
) -> tuple[dict[frozenset[str], dict[str, int]], list[tuple[str, str]], int]:
    """合并各进程的统计结果。"""
    id_folder_counts: dict[frozenset[str], dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    errors: list[tuple[str, str]] = []
    total_files = 0

    for local_counts, local_errors, local_total in results:
        total_files += local_total
        errors.extend(local_errors)
        for ids, folder_counts in local_counts.items():
            for folder, count in folder_counts.items():
                id_folder_counts[ids][folder] += count

    return dict(id_folder_counts), errors, total_files


def build_report(
    root: Path,
    recursive: bool,
    collapse_pcds: bool,
    workers: int,
) -> tuple[dict[frozenset[str], dict[str, int]], list[tuple[str, str]], int, float, int]:
    """返回 (统计结果, 错误列表, 总文件数, 耗时秒, 工作单元数)。"""
    root = root.resolve()
    within_unit_recursive = recursive
    work_units = discover_work_units(root, recursive)
    root_str = str(root)

    t0 = time.perf_counter()
    task_args = [
        (str(unit), root_str, collapse_pcds, within_unit_recursive)
        for unit in work_units
    ]

    if workers <= 1 or len(task_args) <= 1:
        results = [
            process_work_unit(*args)
            for args in task_args
        ]
    else:
        with Pool(processes=min(workers, len(task_args))) as pool:
            results = pool.starmap(process_work_unit, task_args)

    elapsed = time.perf_counter() - t0
    id_folder_counts, errors, total_files = merge_results(results)
    return id_folder_counts, errors, total_files, elapsed, len(work_units)


def format_report(
    root: Path,
    id_folder_counts: dict[frozenset[str], dict[str, int]],
    errors: list[tuple[str, str]],
    total_files: int,
    elapsed: float,
    num_units: int,
    workers: int,
) -> str:
    lines: list[str] = []
    lines.append(f'扫描根目录: {root}')
    lines.append(f'并行工作单元: {num_units} 个子文件夹, workers={workers}')
    lines.append(f'PCD 文件总数: {total_files}')
    lines.append(f'id 构成种类数: {len(id_folder_counts)}')
    lines.append(f'耗时: {elapsed:.1f}s')
    lines.append('')

    def id_sort_key(item):
        ids, _ = item
        return sort_ids(ids)

    for ids, folder_counts in sorted(id_folder_counts.items(), key=id_sort_key):
        id_label = format_id_key(ids)
        folder_parts = [
            f'文件夹{folder}：{count}个'
            for folder, count in sorted(folder_counts.items(), key=lambda x: (-x[1], x[0]))
        ]
        lines.append(f'ID为{id_label}：' + '；'.join(folder_parts))

    if errors:
        lines.append('')
        lines.append(f'解析失败 ({len(errors)} 个):')
        for path, err in errors[:50]:
            lines.append(f'  {path}: {err}')
        if len(errors) > 50:
            lines.append(f'  ... 还有 {len(errors) - 50} 个')

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='统计 PCD 文件中点 id 构成，按 id 组合与文件夹归类（多进程并行）'
    )
    parser.add_argument(
        'root',
        type=str,
        help='扫描根目录',
    )
    parser.add_argument(
        '--no-recursive',
        action='store_true',
        help='不递归子文件夹，仅扫描根目录下的 PCD',
    )
    parser.add_argument(
        '--no-collapse-pcds',
        action='store_true',
        help='不按 pcds 父目录折叠文件夹名，使用完整相对路径',
    )
    parser.add_argument(
        '-j', '--workers',
        type=int,
        default=0,
        help='并行进程数，默认使用 CPU 核数 (%d)' % cpu_count(),
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='',
        help='将报告写入文件（默认仅打印到终端）',
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        print(f'目录不存在: {root}')
        return 1

    workers = args.workers if args.workers > 0 else cpu_count()
    workers = min(workers, cpu_count())

    id_folder_counts, errors, total_files, elapsed, num_units = build_report(
        root,
        recursive=not args.no_recursive,
        collapse_pcds=not args.no_collapse_pcds,
        workers=workers,
    )
    report = format_report(
        root, id_folder_counts, errors, total_files, elapsed, num_units, workers,
    )
    print(report)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report + '\n', encoding='utf-8')
        print(f'\n报告已保存: {out_path}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
