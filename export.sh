#!/bin/bash

usage()
{
    echo
    echo "Usage: ./export.sh <model_name>"
    echo
    echo "where <model_name> could be one of the following:"
    echo "    ssd_mobilenet_v1_egohands"
    echo "    ssd_mobilenet_v2_egohands"
    echo "    ssdlite_mobilenet_v2_egohands"
    echo "    ssd_inception_v2_egohands"
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
    ssd_inception_v2_egohands )
        MODEL_DIR=$1
        NUM_TRAIN_STEPS=20000
        ;;
    rfcn_resnet101_egohands | \
    faster_rcnn_resnet50_egohands | \
    faster_rcnn_resnet101_egohands | \
    faster_rcnn_inception_v2_egohands | \
    faster_rcnn_inception_resnet_v2_atrous_egohands )
        MODEL_DIR=$1
        NUM_TRAIN_STEPS=50000
        ;;
    * )
        usage
esac

PIPELINE_CONFIG_PATH=configs/${MODEL_DIR}.config
CHECKPOINT_PREFIX=${MODEL_DIR}/model.ckpt-${NUM_TRAIN_STEPS}
OUTPUT_DIR=model_exported

# clear old exported model
rm -rf ${OUTPUT_DIR}

PYTHONPATH=`pwd`/models/research:`pwd`/models/research/slim \
    python3 ./models/research/object_detection/export_inference_graph.py \
            --input_type=image_tensor \
            --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
            --trained_checkpoint_prefix=${CHECKPOINT_PREFIX} \
            --output_directory=${OUTPUT_DIR}
