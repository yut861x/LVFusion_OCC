#!/bin/bash
# Script to create Hasco dataset pkl files (origin_labels version)

# Set paths
ROOT_PATH="${LVFUSION_OCC_DATA_ROOT:?Set LVFUSION_OCC_DATA_ROOT}"
INFO_PREFIX="hasco"
VERSION="v1.0"
SKIP_EMPTY_LABELS=${SKIP_EMPTY_LABELS:-0}

EXTRA_ARGS=""
if [[ "$1" == "--skip-empty-labels" ]] || [[ "${SKIP_EMPTY_LABELS}" == "1" ]]; then
    EXTRA_ARGS="--skip-empty-labels"
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
