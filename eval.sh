#!/bin/bash

usage()
{
    echo
    echo "Usage: ./eval.sh <model_name>"
    echo
    echo "where <model_name> could be one of the following:"
    echo "    ssd_mobilenet_v1_egohands"
    echo "    ssd_mobilenet_v2_egohands"
    echo "    ssdlite_mobilenet_v2_egohands"
    echo "    ssd_inception_v2_egohands"
    echo "    ssd_resnet50_v1_fpn_egohands"
    echo "    rfcn_resnet101_egohands"
    echo "    faster_rcnn_resnet50_egohands"
    echo "    faster_rcnn_resnet101_egohands"
    echo "    faster_rcnn_inception_v2_egohands"
    echo "    faster_rcnn_inception_resnet_v2_atrous_egohands"
    echo
    exit
}

if [ $# -ne 1 ]; then
    usage
fi

case $1 in
    ssd_mobilenet_v1_egohands | \
    ssd_mobilenet_v2_egohands | \
    ssdlite_mobilenet_v2_egohands | \
    ssd_inception_v2_egohands | \
    ssd_resnet50_v1_fpn_egohands | \
    rfcn_resnet101_egohands | \
    faster_rcnn_resnet50_egohands | \
    faster_rcnn_resnet101_egohands | \
    faster_rcnn_inception_v2_egohands | \
    faster_rcnn_inception_resnet_v2_atrous_egohands )
        ;;
    * )
        usage
esac

MODEL_DIR=$1
PIPELINE_CONFIG_PATH=configs/${MODEL_DIR}.config
EVAL_DIR=${MODEL_DIR}_eval

# clear old eval results
rm -rf ${EVAL_DIR}

PYTHONPATH=`pwd`/models/research:`pwd`/models/research/slim \
    python3 ./models/research/object_detection/model_main.py \
            --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
            --checkpoint_dir=${MODEL_DIR} \
            --model_dir=${EVAL_DIR} \
            --run_once \
            --alsologtostderr
