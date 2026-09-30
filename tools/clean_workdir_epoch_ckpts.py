#!/usr/bin/env python3
"""
删除 work_dirs 下各实验目录中的 epoch_*.pth，保留 latest.pth 与其它非 epoch 命名权重。

用法（仓库根目录执行）:
    python tools/clean_workdir_epoch_ckpts.py
    python tools/clean_workdir_epoch_ckpts.py --dry-run
    python tools/clean_workdir_epoch_ckpts.py --work-dirs /path/to/work_dirs
"""

from __future__ import annotations

import argparse
import fnmatch
import sys
from pathlib import Path


DEFAULT_WORK_DIRS = Path("./work_dirs")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Remove epoch_*.pth under work_dirs; keep latest.pth and other checkpoints."
    )
    parser.add_argument(
        "--work-dirs",
        type=Path,
        default=DEFAULT_WORK_DIRS,
        help=f"Root directory to scan (default: {DEFAULT_WORK_DIRS})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print paths that would be deleted, do not remove files.",
    )
    args = parser.parse_args()
    root: Path = args.work_dirs.expanduser().resolve()

    if not root.is_dir():
        print(f"Error: not a directory: {root}", file=sys.stderr)
        return 1

    to_delete: list[Path] = []
    for path in root.rglob("*.pth"):
        if not path.is_file():
            continue
        if fnmatch.fnmatch(path.name, "epoch_*.pth"):
            to_delete.append(path)

    to_delete.sort(key=lambda p: str(p))
    if not to_delete:
        print(f"No epoch_*.pth found under {root}")
        return 0

    for p in to_delete:
        if args.dry_run:
            print(f"[dry-run] would delete: {p}")
        else:
            p.unlink()
            print(f"deleted: {p}")

    action = "Would delete" if args.dry_run else "Deleted"
    print(f"{action} {len(to_delete)} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
