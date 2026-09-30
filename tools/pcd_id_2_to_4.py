#!/usr/bin/env python3
"""
遍历 pcds_bak 下 PCD 文件，将每行数据中 id 属性（最后一列）为指定值的改为另一值，
其余内容不变，结果保存到输出目录。
"""

import argparse
from pathlib import Path


def process_pcd_file(src_path: Path, dst_path: Path, id_from: str, id_to: str) -> None:
    """读入 PCD，将数据行中最后一列 id==id_from 改为 id_to，写入 dst_path。"""
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    in_data = False
    with open(src_path, 'r', encoding='utf-8') as f_in:
        with open(dst_path, 'w', encoding='utf-8') as f_out:
            for line in f_in:
                if line.strip() == 'DATA ascii':
                    in_data = True
                    f_out.write(line)
                    continue
                if in_data and line.strip():
                    parts = line.rstrip('\n').split()
                    if parts and parts[-1] == id_from:
                        parts[-1] = id_to
                    f_out.write(' '.join(parts) + '\n')
                else:
                    f_out.write(line)


def main():
    parser = argparse.ArgumentParser(description='PCD 中最后一列 id 从 id_from 改为 id_to')
    parser.add_argument('--src', type=str,
                        default='./data/hasco/20251215/pcds_bak',
                        help='源 PCD 目录')
    parser.add_argument('--dst', type=str,
                        default='./data/hasco/20251215/pcds',
                        help='输出 PCD 目录')
    parser.add_argument('--id-from', type=str, default='2', help='要替换的 id 值')
    parser.add_argument('--id-to', type=str, default='4', help='替换后的 id 值')
    parser.add_argument('--quiet', '-q', action='store_true', help='不逐文件打印，仅打印汇总')
    args = parser.parse_args()

    src_dir = Path(args.src)
    dst_dir = Path(args.dst)
    if not src_dir.is_dir():
        print(f'源目录不存在: {src_dir}')
        return

    pcd_files = sorted(src_dir.glob('*.pcd'))
    if not pcd_files:
        print(f'未找到 PCD 文件: {src_dir}')
        return

    for src_path in pcd_files:
        dst_path = dst_dir / src_path.name
        process_pcd_file(src_path, dst_path, args.id_from, args.id_to)
        if not args.quiet:
            print(f'  {src_path.name} -> {dst_path}')

    print(f'完成: 共处理 {len(pcd_files)} 个文件，输出到 {dst_dir}')


if __name__ == '__main__':
    main()
