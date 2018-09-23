#!/bin/bash

usage()
{
    echo "Usage: (one of the following)"
    echo "    $ ./eval.sh ssd_mobilenet_v1_egohands"
    echo "    $ ./eval.sh ssd_inception_v2_egohands"
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
    ssd_inception_v2_egohands )
        PIPELINE_CONFIG_PATH=configs/ssd_inception_v2_egohands.config
        MODEL_DIR=ssd_inception_v2_egohands
        EVAL_DIR=ssd_inception_v2_egohands_eval
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
