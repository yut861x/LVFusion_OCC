#!/usr/bin/env bash

if [ $# -ne 1 ]; then
    echo "用法: $0 <目录路径>"
    exit 1
fi

TARGET_DIR="$1"
if [ ! -d "$TARGET_DIR" ]; then
    echo "错误: 目录不存在 -> $TARGET_DIR"
    exit 1
fi

TARGET_DIR="$(cd "$TARGET_DIR" 2>/dev/null && pwd)"

count_slashes() {
    local s="${1#/}"; s="${s%/}"
    [ -z "$s" ] && { echo 0; return; }
    local n=0; local IFS="/"
    for _ in $s; do n=$((n+1)); done
    echo "$n"
}

BASE_DEPTH=$(count_slashes "$TARGET_DIR")

mapfile -d '' DIRS < <(find "$TARGET_DIR" -type d -print0)

# 计算名称列最大宽度（含缩进）
MAX_NAME_LEN=0
for dir in "${DIRS[@]}"; do
    rel="${dir#$TARGET_DIR}"; rel="${rel#/}"
    depth=$(( $(count_slashes "$dir") - BASE_DEPTH ))
    name_len=$(( ${#rel} + depth * 2 ))
    [ "$name_len" -gt "$MAX_NAME_LEN" ] && MAX_NAME_LEN="$name_len"
done
[ "$MAX_NAME_LEN" -lt 20 ] && MAX_NAME_LEN=20
NAME_WIDTH=$(( MAX_NAME_LEN + 2 ))

# 表头
printf "%-${NAME_WIDTH}s %-10s %s\n" "文件夹路径" "文件数量" "总大小"
printf '%*s\n' "$((NAME_WIDTH + 22))" '' | tr ' ' '-'

prev_depth=-1
for dir in "${DIRS[@]}"; do
    rel="${dir#$TARGET_DIR}"; rel="${rel#/}"
    depth=$(( $(count_slashes "$dir") - BASE_DEPTH ))

    # 一级子文件夹（depth<=1）与上一组之间加空行
    if [ "$depth" -le 1 ] && [ "$prev_depth" -gt 1 ]; then
        echo ""
    fi
    prev_depth="$depth"

    if [ "$depth" -eq 0 ]; then
        display="$dir"
    else
        display="$rel"
    fi

    indent=""; for ((i=0; i<depth*2; i++)); do indent+=" "; done

    file_count=$(find "$dir" -maxdepth 1 -type f | wc -l)
    dir_size=$(du -sh "$dir" | cut -f1)

    printf "%s%-${NAME_WIDTH}s %-10d %s\n" "$indent" "$display" "$file_count" "$dir_size"
done
