#!/bin/bash

PYTHONPATH=$MODELS_DIR/research:$MODELS_DIR/research/slim \
    python3 create_kitti_tf_record.py \
            --data_dir=egohands_kitti_formatted \
            --output_path=data/egohands \
            --classes_to_use=hand \
            --label_map_path=data/egohands_label_map.pbtxt \
            --validation_set_size=500
