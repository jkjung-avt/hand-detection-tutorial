#!/bin/bash

BASE_URL="http://download.tensorflow.org/models/object_detection/"

for model in ssd_mobilenet_v1_coco_2018_01_28 \
             ssd_inception_v2_coco_2018_01_28 \
             faster_rcnn_inception_v2_coco_2018_01_28; do
    wget --no-check-certificate \
         ${BASE_URL}${model}.tar.gz \
         -O /tmp/${model}.tar.gz
    tar xzvf /tmp/${model}.tar.gz
done
