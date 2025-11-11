#!/bin/bash

set -e

MODEL_KEY=$1

declare -A CONFIGS
CONFIGS["vitdet-b"]="configs/COCO/mask_rcnn_vitdet_b_100ep.py"
CONFIGS["vitdet-l"]="configs/COCO/mask_rcnn_vitdet_l_100ep.py"
CONFIGS["vitdet-h"]="configs/COCO/mask_rcnn_vitdet_h_75ep.py"

declare -A CHECKPOINTS
CHECKPOINTS["vitdet-b"]="weights/model_final_61ccd1.pkl"
CHECKPOINTS["vitdet-l"]="weights/model_final_6146ed.pkl"
CHECKPOINTS["vitdet-h"]="weights/model_final_7224f1.pkl"

CONFIG_FILE=${CONFIGS["$MODEL_KEY"]}
CHECKPOINT_FILE=${CHECKPOINTS["$MODEL_KEY"]}

python eval_coco.py \
    --config-file "$CONFIG_FILE" \
    --eval-only \
    train.init_checkpoint="$CHECKPOINT_FILE"