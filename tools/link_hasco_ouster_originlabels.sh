#!/bin/bash
# 将 FVLV_data_ouster 各组数据软链接到 hasco_ouster_originlabels，
# 其中 labels 来自 FVLV_data_ouster_origin_labels 下同组的 origin_labels。

set -euo pipefail

SRC_DATA="/data2/hasco_lv_data/FVLV_data_ouster"
SRC_ORIGIN_LABELS="/data2/hasco_lv_data/FVLV_data_ouster_origin_labels"
DEST="${LVFUSION_OCC_DATA_ROOT:?Set LVFUSION_OCC_DATA_ROOT}"

mkdir -p "${DEST}"

for group_dir in "${SRC_DATA}"/*/; do
    [[ -d "${group_dir}" ]] || continue
    group_name="$(basename "${group_dir}")"

    dest_group="${DEST}/${group_name}"
    mkdir -p "${dest_group}"

    for item in "${group_dir}"/*; do
        item_name="$(basename "${item}")"
        if [[ "${item_name}" == "labels" ]]; then
            continue
        fi
        ln -sfn "${item}" "${dest_group}/${item_name}"
    done

    origin_labels_dir="${SRC_ORIGIN_LABELS}/${group_name}/origin_labels"
    if [[ ! -d "${origin_labels_dir}" ]]; then
        echo "警告: 未找到 origin_labels: ${origin_labels_dir}" >&2
        continue
    fi
    ln -sfn "${origin_labels_dir}" "${dest_group}/labels"
    echo "已链接: ${group_name} (labels <- origin_labels)"
done

echo "完成! 数据目录: ${DEST}"
