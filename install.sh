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
$PIP install -U Cython
$PIP install -U contextlib2
$PIP install -U lxml
$PIP install -U jupyter
$PIP install -U matplotlib
$PIP install -U Pillow
$PIP install -U scipy
$PIP install -U opencv-python
$PIP install -U tqdm
$PIP install -U requests


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
sed -i '516s/print num_classes, num_anchors/print(num_classes, num_anchors)/' \
       object_detection/meta_architectures/ssd_meta_arch_test.py
sed -i '147s/print /print(/' \
       object_detection/dataset_tools/oid_hierarchical_labels_expansion.py
sed -i '149s/labels_file"""$/[optional]labels_file""")/' \
       object_detection/dataset_tools/oid_hierarchical_labels_expansion.py
sed -i '281s/loss_tensor in losses_dict.itervalues()/_, loss_tensor in losses_dict.items()/' \
       object_detection/model_lib.py
sed -i '380s/category_index.values(),/list(category_index.values()),/' \
       object_detection/model_lib.py
sed -i '390s/iteritems()/items()/' \
       object_detection/model_lib.py
sed -i '168s/range(num_boundaries),/list(range(num_boundaries)),/' \
       object_detection/utils/learning_schedules.py
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
PYTHONPATH=$MODELS_DIR/research:$MODELS_DIR/research/slim \
    $PYTHON $MODELS_DIR/research/object_detection/builders/model_builder_test.py
