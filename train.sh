#!/bin/bash

export PYTHONPATH=`pwd`/models/research:`pwd`/models/research/slim

python3 ./models/research/object_detection/model_main.py \
        --pipeline_config_path=configs/ssd_mobilenet_v1_egohands.config \
        --model_dir=ssd_mobilenet_v1_egohands \
        --num_train_steps=20000 \
        --sample_1_of_n_eval_samples=1 \
        --alsologtostderr
