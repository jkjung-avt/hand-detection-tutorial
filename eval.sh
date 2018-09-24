#!/bin/bash

usage()
{
    echo
    echo "Usage: ./eval.sh <model_name>"
    echo
    echo "where <model_name> could be one of the following:"
    echo "    1. ssd_mobilenet_v1_egohands"
    echo "    2. ssd_mobilenet_v2_egohands"
    echo "    3. ssd_inception_v2_egohands"
    echo "    4. faster_rcnn_inception_v2_egohands"
    echo
    exit
}

if [ $# -ne 1 ]; then
    usage
fi

case $1 in
    ssd_mobilenet_v1_egohands )
        PIPELINE_CONFIG_PATH=configs/ssd_mobilenet_v1_egohands.config
        MODEL_DIR=ssd_mobilenet_v1_egohands
        EVAL_DIR=ssd_mobilenet_v1_egohands_eval
        ;;
    ssd_mobilenet_v2_egohands )
        PIPELINE_CONFIG_PATH=configs/ssd_mobilenet_v2_egohands.config
        MODEL_DIR=ssd_mobilenet_v2_egohands
        EVAL_DIR=ssd_mobilenet_v2_egohands_eval
        ;;
    ssd_inception_v2_egohands )
        PIPELINE_CONFIG_PATH=configs/ssd_inception_v2_egohands.config
        MODEL_DIR=ssd_inception_v2_egohands
        EVAL_DIR=ssd_inception_v2_egohands_eval
        ;;
    faster_rcnn_inception_v2_egohands )
        PIPELINE_CONFIG_PATH=configs/faster_rcnn_inception_v2_egohands.config
        MODEL_DIR=faster_rcnn_inception_v2_egohands
        EVAL_DIR=faster_rcnn_inception_v2_egohands_eval
        ;;
    * )
        usage
esac

PYTHONPATH=`pwd`/models/research:`pwd`/models/research/slim \
    python3 ./models/research/object_detection/eval.py \
            --pipeline_config_path=$PIPELINE_CONFIG_PATH \
            --checkpoint_dir=$MODEL_DIR \
            --eval_dir=$EVAL_DIR \
            --logtostderr
