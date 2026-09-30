"""Merge 33-degree LiDAR and image-supervised OCC labels for a pilot run.

Only class 0 is unannotated. Whitelist classes 1-9 win against free/other;
remaining conflicting nonzero labels become 255 (ignore).
"""

import argparse
import csv
from pathlib import Path

import numpy as np


META_KEYS = (
    'point_cloud_range', 'voxel_size', 'axis_order', 'class_names',
    'background_meaning', 'format_version',
)


def merge_pair(lidar_path, image_path):
    with np.load(lidar_path, allow_pickle=False) as lidar, np.load(image_path, allow_pickle=False) as image:
        for key in ('occ_gt', *META_KEYS):
            if key not in lidar or key not in image:
                raise ValueError(f'{lidar_path.name}: missing {key}')
        for key in META_KEYS:
            if not np.array_equal(lidar[key], image[key]):
                raise ValueError(f'{lidar_path.name}: {key} differs between sources')
        a, b = lidar['occ_gt'], image['occ_gt']
        if a.shape != b.shape or a.dtype != np.uint8 or b.dtype != np.uint8:
            raise ValueError(f'{lidar_path.name}: incompatible OCC arrays')
        if np.any((a > 11) & (a != 255)) or np.any((b > 11) & (b != 255)):
            raise ValueError(f'{lidar_path.name}: unexpected class ID')
        merged = a.copy()
        image_only = (a == 0) & (b != 0)
        merged[image_only] = b[image_only]
        conflict = (a != 0) & (b != 0) & (a != b)
        lidar_semantic = (a >= 1) & (a <= 9)
        image_semantic = (b >= 1) & (b <= 9)
        merged[conflict] = 255
        merged[conflict & lidar_semantic & ~image_semantic] = a[
            conflict & lidar_semantic & ~image_semantic]
        merged[conflict & image_semantic & ~lidar_semantic] = b[
            conflict & image_semantic & ~lidar_semantic]
        ignored_conflict = conflict & (merged == 255)
        metadata = {key: lidar[key].copy() for key in META_KEYS}
    stats = dict(
        frame=lidar_path.stem,
        lidar_labeled=int(np.count_nonzero(a != 0)),
        image_added=int(np.count_nonzero(image_only)),
        conflict_ignored=int(np.count_nonzero(ignored_conflict)),
        unknown=int(np.count_nonzero(merged == 0)),
        free=int(np.count_nonzero(merged == 11)),
    )
    return merged, metadata, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output-dir', default='occ_fused_npz')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    lidar_files = {p.name: p for p in (root / 'pcds_npz').glob('*.npz')}
    image_files = {p.name: p for p in (root / 'imgs_npz').glob('*.npz')}
    if not lidar_files or lidar_files.keys() != image_files.keys():
        raise ValueError('pcds_npz and imgs_npz must contain the same nonempty frame set')
    out_dir = root / args.output_dir
    if out_dir.exists() and any(out_dir.iterdir()) and not args.dry_run:
        raise FileExistsError(f'{out_dir} is nonempty; refusing to overwrite')
    rows = []
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
    for name in sorted(lidar_files):
        merged, metadata, stats = merge_pair(lidar_files[name], image_files[name])
        rows.append(stats)
        if not args.dry_run:
            np.savez_compressed(out_dir / name, occ_gt=merged, **metadata)
    if not args.dry_run:
        with (out_dir / 'merge_stats.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    print(f'frames={len(rows)} image_added={sum(r["image_added"] for r in rows)} '
          f'conflict_ignored={sum(r["conflict_ignored"] for r in rows)} '
          f'output={out_dir if not args.dry_run else "dry-run"}')


if __name__ == '__main__':
    main()
