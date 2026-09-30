"""Convert Hasco geometric-center box labels into weak semantic occupancy.

Only NumPy is required. Background means unannotated, NOT observed free space.
No camera projection or MMDetection3D yaw conversion is applied.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np


CLASSES = ['background', 'truck', 'large_vehicle', 'construction_vehicle',
           'bus', 'car', 'motorcycle', 'bicycle', 'tricycle', 'pedestrian', 'other']
DEFAULT_RANGE = [0, -19.2, -5, 105.6, 19.2, 4]
DEFAULT_VOXEL = [0.3, 0.3, 0.5]


def grid_spec(bounds, voxel):
    bounds, voxel = np.asarray(bounds, float), np.asarray(voxel, float)
    if (bounds.shape != (6,) or voxel.shape != (3,) or
            not np.isfinite(bounds).all() or not np.isfinite(voxel).all() or
            (voxel <= 0).any() or (bounds[3:] <= bounds[:3]).any()):
        raise ValueError('Invalid range or voxel size')
    counts = (bounds[3:] - bounds[:3]) / voxel
    if not np.allclose(counts, np.rint(counts), rtol=0, atol=1e-6):
        raise ValueError('Range extents must be divisible by voxel sizes')
    return bounds, voxel, np.rint(counts).astype(int)


def read_boxes(path):
    boxes = []
    for line_no, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        fields = line.split()
        try:
            if len(fields) < 9:
                raise ValueError('Expected x y z l w h yaw id class')
            box = np.asarray([float(x) for x in fields[:7]])
            if not np.isfinite(box).all() or (box[3:6] <= 0).any():
                raise ValueError('Box values must be finite, dimensions positive')
        except ValueError as exc:
            raise ValueError(f'{path}:{line_no}: {exc}') from exc
        name = fields[8].lower()
        class_id = CLASSES.index(name) if name in CLASSES[1:-1] else 10
        boxes.append((box, class_id, fields[7], fields[8]))
    return boxes


def rasterize(boxes, bounds, voxel):
    bounds, voxel, counts = grid_spec(bounds, voxel)
    nx, ny, nz = counts
    occ = np.zeros((nz, ny, nx), dtype=np.uint8)
    stats = []
    # Larger boxes first: smaller objects take priority in overlapping regions.
    order = sorted(range(len(boxes)), key=lambda i: (-np.prod(boxes[i][0][3:6]), i))
    for i in order:
        b, cid, obj_id, name = boxes[i]
        x, y, z, length, width, height, yaw = b
        c, s = np.cos(yaw), np.sin(yaw)
        half = np.array([(abs(c)*length + abs(s)*width)/2,
                         (abs(s)*length + abs(c)*width)/2, height/2])
        low, high = b[:3]-half, b[:3]+half
        start = np.maximum(0, np.ceil((low-bounds[:3])/voxel-0.5-1e-9)).astype(int)
        stop = np.minimum(counts, np.floor((high-bounds[:3])/voxel-0.5+1e-9)+1).astype(int)
        stat = dict(label_index=i, object_id=obj_id, raw_class=name, class_id=cid,
                    candidate_voxels=0, overwritten_voxels=0,
                    extends_outside_range=bool((low < bounds[:3]).any() or (high > bounds[3:]).any()))
        if (stop > start).all():
            # Broadcasting avoids allocating three full coordinate grids.
            dx = bounds[0]+(np.arange(start[0], stop[0])[None, :]+0.5)*voxel[0]-x
            dy = bounds[1]+(np.arange(start[1], stop[1])[:, None]+0.5)*voxel[1]-y
            dz = bounds[2]+(np.arange(start[2], stop[2])+0.5)*voxel[2]-z
            xy = (np.abs(c*dx+s*dy) <= length/2+1e-9) & (np.abs(-s*dx+c*dy) <= width/2+1e-9)
            mask = (np.abs(dz) <= height/2+1e-9)[:, None, None] & xy[None, :, :]
            sub = occ[start[2]:stop[2], start[1]:stop[1], start[0]:stop[0]]
            stat['candidate_voxels'] = int(mask.sum())
            stat['overwritten_voxels'] = int(np.count_nonzero(sub[mask]))
            sub[mask] = cid
        stats.append(stat)
    return occ, sorted(stats, key=lambda s: s['label_index'])


def convert(args):
    source, output = args.dataset.resolve(), args.output.resolve()
    bounds, voxel, counts = grid_spec(args.point_cloud_range, args.voxel_size)
    files = sorted((source/'labels').glob('*.txt'))
    if not files:
        raise ValueError(f'No labels/*.txt in {source}')
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError('--limit must be positive')
        files = files[:args.limit]
    if output.exists():
        raise FileExistsError(f'Output already exists; choose a new directory: {output}')
    # Validate the entire package before writing labels.
    samples = []
    for path in files:
        image = next((source/'imgs'/f'{path.stem}{ext}' for ext in ('.png', '.jpg', '.jpeg')
                      if (source/'imgs'/f'{path.stem}{ext}').is_file()), None)
        pcd = source/'pcds'/f'{path.stem}.pcd'
        if image is None or not pcd.is_file():
            raise ValueError(f'Missing paired image or PCD: {path.stem}')
        samples.append((path, image, pcd, read_boxes(path)))
    output.mkdir(parents=True)
    (output/'labels').mkdir()
    manifest, raw_classes = [], Counter()
    total = np.zeros(len(CLASSES), dtype=np.int64)
    for path, image, pcd, boxes in samples:
        occ, stats = rasterize(boxes, bounds, voxel)
        target = output/'labels'/f'{path.stem}.npz'
        np.savez_compressed(target, occ_gt=occ, point_cloud_range=bounds,
                            voxel_size=voxel, axis_order=np.array('ZYX'),
                            class_names=np.asarray(CLASSES))
        hist = np.bincount(occ.ravel(), minlength=len(CLASSES))
        total += hist
        raw_classes.update(b[3] for b in boxes)
        manifest.append(dict(token=f'{source.name}_{path.stem}', frame=path.stem,
                             image_path=str(image), lidar_path=str(pcd),
                             box_label_path=str(path), occ_path=str(target.relative_to(output)),
                             class_voxel_counts=hist.tolist(), boxes=stats))
    metadata = dict(format_version=1, dataset=str(source), axis_order='ZYX',
                    shape_zyx=counts[::-1].tolist(), point_cloud_range=bounds.tolist(),
                    voxel_size=voxel.tolist(), class_names=CLASSES, ignore_index=255,
                    box_origin='geometric_center', yaw='radians; local +X toward +Y; no offset',
                    fill_rule='voxel center inside rotated box, including boundary',
                    overlap_policy='smaller box wins; equal volume later source line wins',
                    background_meaning='unannotated; NOT verified free',
                    supervision='dense box-derived weak labels; no visibility mask',
                    calibration_dir=str(source/'params'), samples=len(manifest),
                    empty_label_frames=sum(not s['boxes'] for s in manifest),
                    raw_class_counts=dict(raw_classes), class_voxel_counts=total.tolist(),
                    zero_voxel_boxes=sum(b['candidate_voxels']==0 for s in manifest for b in s['boxes']))
    (output/'manifest.json').write_text(json.dumps(dict(metadata=metadata, infos=manifest), indent=2), encoding='utf-8')
    print(json.dumps(metadata, indent=2))
    print(f'Saved: {output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True, help='Single package containing imgs/ pcds/ labels/ params/')
    parser.add_argument('--output', type=Path, required=True, help='New output directory; existing paths are never overwritten')
    parser.add_argument('--point-cloud-range', type=float, nargs=6, default=DEFAULT_RANGE)
    parser.add_argument('--voxel-size', type=float, nargs=3, default=DEFAULT_VOXEL)
    parser.add_argument('--limit', type=int, help='Convert first N frames, sorted by filename')
    convert(parser.parse_args())


if __name__ == '__main__':
    main()
