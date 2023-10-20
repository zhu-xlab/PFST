#!/usr/bin/env bash

set -x

CONFIG_DIR=$1
EXPR_NAME=$2
NUM_ITER=$3

python tools/test.py configs/${CONFIG_DIR}/${EXPR_NAME}.py work_dirs/${EXPR_NAME}/iter_${NUM_ITER}.pth \
    --eval-option imgfile_prefix=work_dirs/${EXPR_NAME}/outputs --format-only
