#!/bin/bash

ROOT_DIR=`pwd`
MODELS_DIR=$ROOT_DIR/models
PYTHON=python3
PIP=pip3

# make sure tensorflow has been installed
$PIP list | grep tensorflow
if [ $? -ne 0 ]; then
    echo "TensorFlow doesn't seem to be installed!"
    exit
fi

# install python3 packages according to official documentation:
# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md
#
# assuming tensorflow, Pillow and matplotlib are already present
sudo $PIP install -U Cython
sudo $PIP install -U contextlib2
sudo $PIP install -U lxml
sudo $PIP install -U jupyter

# download protoc-3.5.1
BASE_URL="https://github.com/google/protobuf/releases/download/v3.5.1/"
filename="protoc-3.5.1-linux-x86_64.zip"
wget --no-check-certificate ${BASE_URL}${filename} -O /tmp/${filename}
unzip /tmp/${filename} -d protoc-3.5.1

# install tensorflow models
# (and also fix some code so that it could be run with pyhton3)
git submodule update --init
cd $MODELS_DIR
cd research
sed -i "157s/print '--annotation_type expected value is 1 or 2.'/print('--annotation_type expected value is 1 or 2.')/" \
       object_detection/dataset_tools/oid_hierarchical_labels_expansion.py
sed -i "516s/print num_classes, num_anchors/print(num_classes, num_anchors)/" \
       object_detection/meta_architectures/ssd_meta_arch_test.py
sed -i "282s/losses_dict.itervalues()/losses_dict.values()/" \
       object_detection/model_lib.py
sed -i "381s/category_index.values(),/list(category_index.values()),/" \
       object_detection/model_lib.py
sed -i "391s/eval_metric_ops.iteritems()/eval_metric_ops.items()/" \
       object_detection/model_lib.py
sed -i "225s/reversed(zip(output_feature_map_keys, output_feature_maps_list)))/reversed(list(zip(output_feature_map_keys, output_feature_maps_list))))/" \
       object_detection/models/feature_map_generators.py
sed -i "842s/print 'Scores and tpfp per class label: {}'.format(class_index)/print('Scores and tpfp per class label: {}'.format(class_index))/" \
       object_detection/utils/object_detection_evaluation.py
sed -i "843s/print tp_fp_labels/print(tp_fp_labels)/" \
       object_detection/utils/object_detection_evaluation.py
sed -i "844s/print scores/print(scores)/" \
       object_detection/utils/object_detection_evaluation.py
sed -n '31p' object_detection/eval_util.py | grep -q vis_utils &&
    ex -s -c 31m23 -c w -c q object_detection/eval_util.py

$ROOT_DIR/protoc-3.5.1/bin/protoc object_detection/protos/*.proto --python_out=.
cd $ROOT_DIR

# add pycocotools
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI/
sed -i '3s/python /python3 /' Makefile
sed -i '8s/python /python3 /' Makefile
make
cp -r pycocotools $MODELS_DIR/research/
cd $ROOT_DIR

# run a basic test to make sure tensorflow object detection is working
echo
echo
echo Running model_builder_test.py
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=$MODELS_DIR/research:$MODELS_DIR/research/slim \
    $PYTHON $MODELS_DIR/research/object_detection/builders/model_builder_test.py
