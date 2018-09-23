#!/bin/bash

BASE_URL="http://download.tensorflow.org/models/object_detection/"

for model_name in ssd_mobilenet_v1_coco_2018_01_28 \
                  ssd_inception_v2_coco_2018_01_28; do
    wget --no-check-certificate \
         ${BASE_URL}${model_name}.tar.gz \
         -O /tmp/${model_name}.tar.gz
    tar xzvf /tmp/${model_name}.tar.gz
done
