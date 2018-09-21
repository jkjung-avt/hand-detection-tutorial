"""bootstrap.py

This script downloads the 'egohands' dataset and convert the annotations
into bounding boxes in KITTI format.

Output of this script is data in KITTI format:

  ./egohands_kitti_formatted
    ├── CARDS_COURTYARD_B_T_frame_0011.jpg
    ├── CARDS_COURTYARD_B_T_frame_0011.txt
    ├── ......
    ├── PUZZLE_OFFICE_T_S_frame_2697.jpg
    └── PUZZLE_OFFICE_T_S_frame_2697.txt
"""


import os
import sys
import math
import random
import logging
import argparse
from zipfile import ZipFile
from shutil import rmtree, copyfile, move
from urllib.request import urlretrieve

import cv2
from scipy.io import loadmat


EGOHANDS_DATASET_URL = \
    'http://vision.soic.indiana.edu/egohands_files/egohands_data.zip'
EGOHANDS_DIR = './egohands'
EGOHANDS_DATA_DIR = './egohands/_LABELLED_SAMPLES'
CONVERTED_DIR = './egohands_kitti_formatted'

VISUALIZE = False  # visualize each image (for debugging)


def parse_args():
    """Parse input arguments."""
    desc = ('This script downloads the egohands dataset and convert'
            'the annotations into bounding boxes in KITTI format.')
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--verify', dest='do_verify',
                        help='show and verify each images',
                        action='store_true')
    args = parser.parse_args()
    return args


def download_file(url, dest=None):
    """Download file from an URL."""
    if not dest:
        dest = url.split('/')[-1]
    urlretrieve(url, dest)


def polygon_to_box(polygon):
    """Convert 1 polygon into a bounding box.

    # Arguments
      polygon: a numpy array of shape (N, 2) representing N vertices
               of the hand segmentation label (polygon); each vertex
               is a point: (x, y)
    """
    if len(polygon) < 3:  # a polygon has at least 3 vertices
        return None

    x_min = min(polygon[:, 0])
    y_min = min(polygon[:, 1])
    x_max = max(polygon[:, 0])
    y_max = max(polygon[:, 1])

    x_min = int(math.floor(x_min))
    y_min = int(math.floor(y_min))
    x_max = int(math.ceil(x_max))
    y_max = int(math.ceil(y_max))

    return [x_min, y_min, x_max, y_max]


def box_to_line(box):
    """Convert 1 bounding box into 1 line in the KITTI txt file.

    # Arguments
      box: [x_min, y_min, x_max, y_max].

    KITTI format:
    Values  Name        Description
    --------------------------------------------------------------------
       1    type        Describes the type of object: 'Car', 'Van',
                        'Truck', 'Pedestrian', 'Person_sitting',
                        'Cyclist', 'Tram', 'Misc' or 'DontCare'
       1    truncated   Float from 0 (non-truncated) to 1 (truncated),
                        where truncated refers to the object leaving
                        image boundaries
       1    occluded    Integer (0,1,2,3) indicating occlusion state:
                        0 = fully visible, 1 = partly occluded
                        2 = largely occluded, 3 = unknown
       1    alpha       Observation angle of object, ranging [-pi..pi]
       4    bbox        2D bounding box of object in the image
                        (0-based index): contains left, top, right,
                        bottom pixel coordinates
       3    dimensions  3D object dimensions: height, width, length
       3    location    3D object location x,y,z in camera coordinates
       1    rotation_y  Rotation ry around Y-axis in camera coordinates
                        [-pi..pi]
       1    score       Only for results: Float, indicating confidence
                        in detection, needed for p/r curves, higher is
                        better.
    """
    return ' '.join(['hand',
                     '0',
                     '0',
                     '0',
                     '{} {} {} {} '.format(*box),
                     '0 0 0',
                     '0 0 0',
                     '0',
                     '0'])


def convert_one_folder(folder):
    """Convert egohands to KITTI for 1 data folder (100 images).

    Refer to README.txt in the egohands folder for the format of the
    MATLAB annotation files and how jpg image files are organized.
    The code in this function loads the 'video' struct from the
    MATLAB file, converts polygons into bounding boxes and write
    annotation into KITTI format.
    """
    folder_path = os.path.join(EGOHANDS_DATA_DIR, folder)
    logging.info('Converting %s' % folder_path)
    frames = [os.path.splitext(f)[0]
              for f in os.listdir(folder_path) if f.endswith('jpg')]
    frames.sort()
    assert len(frames) == 100
    video = loadmat(os.path.join(folder_path, 'polygons.mat'))
    polygons = video['polygons'][0]  # there are 100*4 entries in polygons
    for i, frame in enumerate(frames):
        # copy and rename jpg file to the 'converted' folder
        src_jpg = frame + '.jpg'
        dst_jpg = folder + '_' + src_jpg
        copyfile(os.path.join(folder_path, src_jpg),
                 os.path.join(CONVERTED_DIR, dst_jpg))
        # generate txt (the KITTI annotation corresponding to the jpg)
        dst_txt = folder + '_' + frame + '.txt'
        boxes = []
        with open(os.path.join(CONVERTED_DIR, dst_txt), 'w') as f:
            for polygon in polygons[i]:
                box = polygon_to_box(polygon)
                if box:
                    boxes.append(box)
                    f.write(box_to_line(box) + '\n')

        if VISUALIZE:
            img = cv2.imread(os.path.join(CONVERTED_DIR, dst_jpg))
            for box in boxes:
                cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]),
                              (0, 224, 0), 2)
            cv2.imshow('Visualization', img)
            if cv2.waitKey(0) == 27:
                sys.exit()


def egohands_to_kitti():
    """Convert egohands data and annotations to KITTI format.

    Steps:
      1. walk through each sub-directory in egohands' data folder.
      2. copy each jpg file to the 'converted' image folder and give
         each file a unique name.
      3. convert the original annotations ('polygon.mat') into
         bounding boxes and write a KITTI txt file for each image.
    """
    for folder in os.listdir(EGOHANDS_DATA_DIR):
        convert_one_folder(folder)


def main():
    """main"""
    logging.basicConfig(level=logging.INFO)

    egohands_zip_path = EGOHANDS_DATASET_URL.split('/')[-1]
    if not os.path.isfile(egohands_zip_path):
        logging.info('Downloading %s...' % egohands_zip_path)
        download_file(EGOHANDS_DATASET_URL, egohands_zip_path)

    if not os.path.exists(EGOHANDS_DIR):
        with ZipFile(egohands_zip_path, 'r') as zf:
            logging.info('Extracting egohands dataset files...')
            zf.extractall(EGOHANDS_DIR)

    logging.info('Copying jpg files and converting annotations...')
    rmtree(CONVERTED_DIR, ignore_errors=True)
    os.makedirs(CONVERTED_DIR)
    egohands_to_kitti()

    logging.info('All done.')


if __name__ == '__main__':
    main()
    sys.exit()
