#!/usr/bin/env python3
from __future__ import annotations

"""
在 PCD 文件中安全交换点 id 值（单次遍历，避免顺序替换导致全部变成同一 id）。

仅处理名为 pcds 的目录下的文件，跳过 pcds_bak 及其他目录。

用法:
    # 预览（不写文件）
    python tools/pcd_swap_ids.py /data2/hasco_lv_data/FVLV_data --dry-run

    # 交换 1 <-> 4
    python tools/pcd_swap_ids.py /data2/hasco_lv_data/FVLV_data --swap 1,4
"""

import argparse
import os
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

# 单次映射：同时决定每个旧 id 的新值，不会出现先 1->4 再 4->1 的覆盖问题
DEFAULT_SWAP = {'1': '4', '4': '1'}


def collect_pcds_only(root: Path) -> list[Path]:
    """仅收集 parent 目录名为 pcds 的 .pcd 文件。"""
    return sorted(
        p for p in root.rglob('*.pcd')
        if p.parent.name == 'pcds'
    )


def swap_pcd_file(pcd_path: Path, swap_map: dict[str, str], dry_run: bool) -> tuple[int, str | None]:
    """交换单个 PCD 文件中的 id，返回 (修改点数, 错误信息)。"""
    tmp_path = pcd_path.with_suffix('.pcd.tmp')
    changed = 0

    try:
        with open(pcd_path, 'r', encoding='utf-8', errors='ignore') as f_in:
            fields: list[str] = []
            id_idx = -1
            in_data = False
            out_lines: list[str] = []

            for line in f_in:
                if not in_data:
                    line_s = line.strip()
                    if line_s.startswith('FIELDS'):
                        fields = line_s.split()[1:]
                        if 'id' not in fields:
                            return 0, f'无 id 字段 (fields={fields})'
                        id_idx = fields.index('id')
                    elif line_s.startswith('DATA'):
                        in_data = True
                    out_lines.append(line)
                    continue

                line_s = line.strip()
                if line_s and id_idx >= 0:
                    parts = line.rstrip('\n').split()
                    old_id = parts[id_idx]
                    if old_id in swap_map:
                        parts[id_idx] = swap_map[old_id]
                        changed += 1
                    out_lines.append(' '.join(parts) + '\n')
                else:
                    out_lines.append(line)

        if dry_run:
            return changed, None

        with open(tmp_path, 'w', encoding='utf-8') as f_out:
            f_out.writelines(out_lines)
        os.replace(tmp_path, pcd_path)
        return changed, None
    except Exception as e:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        return 0, str(e)


def _worker(args: tuple[str, dict[str, str], bool]) -> tuple[str, int, str | None]:
    path_str, swap_map, dry_run = args
    changed, err = swap_pcd_file(Path(path_str), swap_map, dry_run)
    return path_str, changed, err


def parse_swap_map(swap_arg: str) -> dict[str, str]:
    """解析 --swap 1,4 为 {'1':'4','4':'1'}。"""
    parts = [p.strip() for p in swap_arg.split(',') if p.strip()]
    if len(parts) != 2:
        raise ValueError(f'--swap 需要两个 id，例如 1,4，当前: {swap_arg}')
    a, b = parts
    return {a: b, b: a}


def main():
    parser = argparse.ArgumentParser(description='安全交换 PCD 中点 id（仅处理 pcds 目录）')
    parser.add_argument('root', type=str, help='数据根目录')
    parser.add_argument(
        '--swap',
        type=str,
        default='1,4',
        help='要互换的两个 id，默认 1,4',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='仅统计将修改的点数，不写文件',
    )
    parser.add_argument(
        '-j', '--workers',
        type=int,
        default=0,
        help='并行进程数，默认 CPU 核数',
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        print(f'目录不存在: {root}')
        return 1

    try:
        swap_map = parse_swap_map(args.swap)
    except ValueError as e:
        print(e)
        return 1

    pcd_files = collect_pcds_only(root)
    if not pcd_files:
        print(f'未找到 pcds 目录下的 PCD 文件: {root}')
        return 1

    workers = args.workers if args.workers > 0 else cpu_count()
    workers = max(1, min(workers, cpu_count()))

    print(f'根目录: {root}')
    print(f'仅处理: */pcds/*.pcd（跳过 pcds_bak）')
    print(f'交换映射: {swap_map}')
    print(f'文件数: {len(pcd_files)}, workers={workers}, dry_run={args.dry_run}')

    t0 = time.perf_counter()
    task_args = [(str(p), swap_map, args.dry_run) for p in pcd_files]

    if workers <= 1:
        results = [_worker(a) for a in task_args]
    else:
        with Pool(processes=workers) as pool:
            results = pool.map(_worker, task_args, chunksize=32)

    elapsed = time.perf_counter() - t0
    total_changed = sum(r[1] for r in results)
    errors = [(r[0], r[2]) for r in results if r[2]]

    mode = '预览' if args.dry_run else '完成'
    print(f'{mode}: 处理 {len(pcd_files)} 个文件, 共交换 {total_changed} 个点 id, 耗时 {elapsed:.1f}s')

    if errors:
        print(f'失败 {len(errors)} 个:')
        for path, err in errors[:20]:
            print(f'  {path}: {err}')
        if len(errors) > 20:
            print(f'  ... 还有 {len(errors) - 20} 个')
        return 1

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
