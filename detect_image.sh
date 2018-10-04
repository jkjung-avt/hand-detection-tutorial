#!/bin/bash

PYTHONPATH=`pwd`/models/research:`pwd`/models/research/slim \
    python3 ./detect_image.py $@
