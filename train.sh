#!/bin/bash

usage()
{
    echo "Usage: (one of the following)"
    echo "    $ ./train.sh ssd_mobilenet_v1_egohands"
    echo "    $ ./train.sh ssd_inception_v2_egohands"
    exit
}

if [ $# -ne 1 ]; then
    usage
fi

case $1 in
    ssd_mobilenet_v1_egohands )
        PIPELINE_CONFIG_PATH=configs/ssd_mobilenet_v1_egohands.config
        MODEL_DIR=ssd_mobilenet_v1_egohands
        NUM_TRAIN_STEPS=20000
        ;;
    ssd_inception_v2_egohands )
        PIPELINE_CONFIG_PATH=configs/ssd_inception_v2_egohands.config
        MODEL_DIR=ssd_inception_v2_egohands
        NUM_TRAIN_STEPS=20000
        ;;
    * )
        usage
esac

PYTHONPATH=`pwd`/models/research:`pwd`/models/research/slim \
python3 ./models/research/object_detection/model_main.py \
        --pipeline_config_path=$PIPELINE_CONFIG_PATH \
        --model_dir=$MODEL_DIR \
        --num_train_steps=$NUM_TRAIN_STEPS \
        --sample_1_of_n_eval_samples=1 \
        --alsologtostderr
