#!/bin/bash
# 清理 LVFusion-OCC 训练/评测进程并释放 GPU 显存

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRY_RUN=0
AUTO_YES=0
RESET_GPU=0
TARGET_GPUS=""

usage() {
    cat <<EOF
用法: $(basename "$0") [选项]

清理本项目相关的 train/test 分布式进程，并尝试释放被占用的 GPU 显存。

选项:
  -y, --yes           跳过确认，直接清理
  -n, --dry-run       仅显示将被清理的进程，不实际执行
  --reset-gpu         若显存仍被占用，尝试 nvidia-smi --gpu-reset
  --gpus ID[,ID...]   仅重置指定 GPU（需配合 --reset-gpu）
  -h, --help          显示帮助

示例:
  bash clean_gpu.sh
  bash clean_gpu.sh -y
  bash clean_gpu.sh -y --reset-gpu --gpus 4,5,6,7
EOF
}

log() { echo "[$(date '+%H:%M:%S')] $*"; }

have_cmd() { command -v "$1" >/dev/null 2>&1; }

print_gpu_status() {
    if ! have_cmd nvidia-smi; then
        log "未找到 nvidia-smi，跳过 GPU 状态检查"
        return
    fi
    echo ""
    log "当前 GPU 显存占用:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader | while IFS= read -r line; do
        echo "  $line"
    done
    echo ""
    log "当前 GPU 计算进程:"
    local apps
    apps="$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory \
        --format=csv,noheader 2>/dev/null || true)"
    if [[ -z "${apps}" ]]; then
        echo "  (无)"
    else
        echo "${apps}" | while IFS= read -r line; do
            echo "  $line"
        done
    fi
}

find_project_gpu_pids() {
    ps -eo pid=,args= 2>/dev/null | awk -v root="${PROJECT_ROOT}" '
        $0 ~ root && ($0 ~ /tools\/train\.py/ || $0 ~ /tools\/test\.py/ ||
                      $0 ~ /torch\.distributed\.(launch|run)/) {
            print $1
        }' | sort -u
}

find_orphan_train_pids() {
    ps -eo pid=,ppid=,args= 2>/dev/null | awk -v root="${PROJECT_ROOT}" '
        $2 == 1 && $0 ~ root && ($0 ~ /tools\/train\.py/ || $0 ~ /tools\/test\.py/) {
            print $1
        }' | sort -u
}

show_target_processes() {
    local pids
    pids="$(find_project_gpu_pids)"
    if [[ -z "${pids}" ]]; then
        log "未发现本项目相关的 train/test 进程"
        return 1
    fi

    log "将清理以下进程:"
    while read -r pid; do
        [[ -z "${pid}" ]] && continue
        ps -p "${pid}" -o pid=,ppid=,stat=,etime=,cmd= 2>/dev/null \
            | sed 's/^/  /'
    done <<< "${pids}"

    local orphans
    orphans="$(find_orphan_train_pids)"
    if [[ -n "${orphans}" ]]; then
        echo ""
        log "其中孤儿进程 (PPID=1):"
        while read -r pid; do
            [[ -z "${pid}" ]] && continue
            ps -p "${pid}" -o pid=,ppid=,stat=,etime=,cmd= 2>/dev/null \
                | sed 's/^/  /'
        done <<< "${orphans}"
    fi
}

kill_project_processes() {
    local patterns=(
        "${PROJECT_ROOT}/tools/train.py"
        "${PROJECT_ROOT}/tools/test.py"
        "torch.distributed.launch.*${PROJECT_ROOT}"
        "torch.distributed.run.*${PROJECT_ROOT}"
    )
    local pid

    for pattern in "${patterns[@]}"; do
        if [[ "${DRY_RUN}" -eq 1 ]]; then
            log "[dry-run] pkill -9 -f '${pattern}'"
            continue
        fi
        pkill -9 -f "${pattern}" 2>/dev/null || true
    done

    if [[ "${DRY_RUN}" -eq 1 ]]; then
        return 0
    fi

    sleep 2

    local remaining
    remaining="$(find_project_gpu_pids)"
    if [[ -n "${remaining}" ]]; then
        log "仍有残留进程，逐个强制结束:"
        while read -r pid; do
            [[ -z "${pid}" ]] && continue
            kill -9 "${pid}" 2>/dev/null || true
            log "  killed PID ${pid}"
        done <<< "${remaining}"
        sleep 1
    fi
}

find_stuck_gpu_indices() {
    if ! have_cmd nvidia-smi; then
        return
    fi

    nvidia-smi --query-compute-apps=gpu_uuid,pid,used_gpu_memory \
        --format=csv,noheader 2>/dev/null | while IFS=',' read -r uuid pid mem; do
        uuid="$(echo "${uuid}" | xargs)"
        pid="$(echo "${pid}" | xargs)"
        mem="$(echo "${mem}" | xargs)"

        [[ -z "${pid}" || "${pid}" == "pid" ]] && continue
        [[ ! "${mem}" =~ MiB$ ]] && continue

        if [[ ! -d "/proc/${pid}" ]]; then
            nvidia-smi --query-gpu=index,uuid --format=csv,noheader \
                | awk -F', ' -v target="${uuid}" '$2 == target {print $1; exit}'
        fi
    done | sort -u
}

find_gpus_with_used_memory() {
    if ! have_cmd nvidia-smi; then
        return
    fi
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader \
        | awk -F', ' '$2+0 > 200 {print $1}'
}

reset_stuck_gpus() {
    if ! have_cmd nvidia-smi; then
        return
    fi

    local gpu_ids=()
    if [[ -n "${TARGET_GPUS}" ]]; then
        IFS=',' read -ra gpu_ids <<< "${TARGET_GPUS}"
    else
        mapfile -t gpu_ids < <(find_stuck_gpu_indices)
        if [[ ${#gpu_ids[@]} -eq 0 ]]; then
            mapfile -t gpu_ids < <(find_gpus_with_used_memory)
        fi
    fi

    if [[ ${#gpu_ids[@]} -eq 0 ]]; then
        log "无需 GPU reset"
        return 0
    fi

    local gpu_list
    gpu_list="$(IFS=','; echo "${gpu_ids[*]}")"
    log "尝试 reset GPU: ${gpu_list}"

    if [[ "${DRY_RUN}" -eq 1 ]]; then
        log "[dry-run] nvidia-smi -i ${gpu_list} --gpu-reset"
        return 0
    fi

    if nvidia-smi -i "${gpu_list}" --gpu-reset >/dev/null 2>&1; then
        log "GPU reset 成功"
        return 0
    fi

    if have_cmd sudo && sudo -n true 2>/dev/null; then
        if sudo nvidia-smi -i "${gpu_list}" --gpu-reset >/dev/null 2>&1; then
            log "GPU reset 成功 (sudo)"
            return 0
        fi
    fi

    log "GPU reset 失败，可能需要 root 权限，或重启容器/宿主机"
    return 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -y|--yes) AUTO_YES=1 ;;
        -n|--dry-run) DRY_RUN=1 ;;
        --reset-gpu) RESET_GPU=1 ;;
        --gpus) TARGET_GPUS="$2"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "未知参数: $1"; usage; exit 1 ;;
    esac
    shift
done

echo "=========================================="
echo "LVFusion-OCC GPU 清理脚本"
echo "项目路径: ${PROJECT_ROOT}"
echo "=========================================="

print_gpu_status

if show_target_processes; then
    HAS_TARGETS=1
else
    HAS_TARGETS=0
fi

if [[ "${HAS_TARGETS}" -eq 1 && "${AUTO_YES}" -eq 0 && "${DRY_RUN}" -eq 0 ]]; then
    read -r -p "确认清理以上进程? [y/N] " answer
    case "${answer}" in
        y|Y|yes|YES) ;;
        *) log "已取消"; exit 0 ;;
    esac
fi

if [[ "${HAS_TARGETS}" -eq 1 ]]; then
    log "开始清理进程..."
    kill_project_processes
else
    log "跳过进程清理"
fi

if [[ "${RESET_GPU}" -eq 1 ]]; then
    reset_stuck_gpus || true
elif have_cmd nvidia-smi; then
    stuck="$(find_stuck_gpu_indices | tr '\n' ' ' | xargs)"
    used="$(find_gpus_with_used_memory | tr '\n' ' ' | xargs)"
    if [[ -n "${stuck}" || -n "${used}" ]]; then
        echo ""
        log "检测到显存仍被占用。"
        if [[ -n "${stuck}" ]]; then
            log "  幽灵进程 GPU: ${stuck}"
        fi
        if [[ -n "${used}" ]]; then
            log "  高占用 GPU: ${used}"
        fi
        log "可执行: bash clean_gpu.sh -y --reset-gpu"
        if [[ -n "${TARGET_GPUS}" ]]; then
            log "或指定 GPU: bash clean_gpu.sh -y --reset-gpu --gpus ${TARGET_GPUS}"
        fi
    fi
fi

echo ""
log "清理完成，当前状态:"
print_gpu_status

remaining="$(find_project_gpu_pids)"
if [[ -n "${remaining}" ]]; then
    log "警告: 仍有残留进程，请手动检查"
    exit 1
fi

exit 0
