#!/usr/bin/env python3
"""
将包含图片的文件夹合成为 mp4 视频。

用法（仓库根目录执行）:
    python tools/images_to_video.py /path/to/image_folder
    python tools/images_to_video.py /path/to/image_folder --fps 10
    python tools/images_to_video.py /path/to/image_folder -o /path/to/out.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
from tqdm import tqdm

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def collect_images(folder: Path) -> list[Path]:
    """收集文件夹内（非递归）的图片，按文件名排序。"""
    images = [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    ]
    images.sort(key=lambda p: p.name)
    return images


def default_output_path(input_dir: Path) -> Path:
    """输入文件夹同级目录下，以文件夹名命名的 .mp4 路径。"""
    return input_dir.parent / f"{input_dir.name}.mp4"


def images_to_video(
    input_dir: Path,
    output_path: Path,
    fps: float = 5.0,
) -> None:
    images = collect_images(input_dir)
    if not images:
        raise FileNotFoundError(f"未在目录中找到图片: {input_dir}")

    first = cv2.imread(str(images[0]))
    if first is None:
        raise RuntimeError(f"无法读取图片: {images[0]}")

    height, width = first.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"无法创建视频文件: {output_path}")

    written = 0
    try:
        for i, img_path in enumerate(tqdm(images, desc="导出视频", unit="帧")):
            frame = first if i == 0 else cv2.imread(str(img_path))
            if frame is None:
                print(f"警告: 跳过无法读取的图片: {img_path}", file=sys.stderr)
                continue
            if frame.shape[0] != height or frame.shape[1] != width:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(frame)
            written += 1
    finally:
        writer.release()

    print(f"已写入 {written} 帧 -> {output_path} (fps={fps})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将包含图片的文件夹合成为 mp4 视频。"
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="包含图片的文件夹路径",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="输出 mp4 路径（默认: 输入文件夹同级目录下 <文件夹名>.mp4）",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=5.0,
        help="视频帧率（默认: 5）",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()

    if not input_dir.is_dir():
        print(f"错误: 不是有效目录: {input_dir}", file=sys.stderr)
        return 1

    if args.fps <= 0:
        print(f"错误: fps 必须为正数: {args.fps}", file=sys.stderr)
        return 1

    if args.output is None:
        output_path = default_output_path(input_dir)
    else:
        output_path = args.output.expanduser().resolve()
        if output_path.is_dir():
            output_path = output_path / f"{input_dir.name}.mp4"
        elif output_path.suffix.lower() != ".mp4":
            output_path = output_path.with_suffix(".mp4")

    try:
        images_to_video(input_dir, output_path, fps=args.fps)
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
