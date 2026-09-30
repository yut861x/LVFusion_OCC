#!/bin/bash
# Script to create Hasco dataset pkl files

# Set paths

ROOT_PATH="${LVFUSION_OCC_DATA_ROOT:?Set LVFUSION_OCC_DATA_ROOT}"
INFO_PREFIX="hasco"
VERSION="v1.0"
SKIP_EMPTY_LABELS=${SKIP_EMPTY_LABELS:-0}

EXTRA_ARGS=""
for arg in "$@"; do
    case "${arg}" in
        --skip-empty-labels)
            EXTRA_ARGS="${EXTRA_ARGS} --skip-empty-labels"
            ;;
        --oversample-class)
            EXTRA_ARGS="${EXTRA_ARGS} --oversample-class"
            ;;
        --oversample-yaw)
            EXTRA_ARGS="${EXTRA_ARGS} --oversample-yaw"
            ;;
        --oversample-azimuth)
            EXTRA_ARGS="${EXTRA_ARGS} --oversample-azimuth"
            ;;
        *)
            echo "Unknown argument: ${arg}"
            echo "Supported: --skip-empty-labels --oversample-class --oversample-yaw --oversample-azimuth"
            exit 1
            ;;
    esac
done

if [[ "${SKIP_EMPTY_LABELS}" == "1" ]] && [[ "${EXTRA_ARGS}" != *"--skip-empty-labels"* ]]; then
    EXTRA_ARGS="${EXTRA_ARGS} --skip-empty-labels"
fi

# Run the converter
python tools/create_data.py hasco \
    --root-path ${ROOT_PATH} \
    --version ${VERSION} \
    --extra-tag ${INFO_PREFIX} \
    --out-dir ${ROOT_PATH} \
    ${EXTRA_ARGS}

echo "Done! Generated pkl files:"
echo "  - ${ROOT_PATH}/${INFO_PREFIX}_infos_train.pkl"
echo "  - ${ROOT_PATH}/${INFO_PREFIX}_infos_val.pkl"
