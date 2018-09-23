hand-detection-tutorial
=======================

This is a tutorial on training a 'hand detector' with TensorFlow object detection API. This README outlines how to set up everything and train the object detection model locally.  You could refer to the following blog post for a more detailed tutorial.

[Tutorial on Training a Hand Detector](https://jkjung-avt.github.io/hand-detection-tutorial/)

Table of contents
-----------------

* [Setup](#setup)
* [Training](#training)
* [Evaluatiing the trained model](#evluating)


<a name="setup"></a>
Setup
-----

Just for reference, the code in this repository has been tested on a desktop PC with:

* NVIDIA GeForce GTX-1080Ti
* Ubuntu 16.04.5 LTS (x86_64)
* CUDA 9.2
* cuDNN 7.1.4
* TensorFlow 1.10.0

Follow the steps below to set up the environment for training tensorflow object detection models.  Make sure tensorflow-gpu and matplotlib (python3 packages) have been installed on the system already.

1. Clone this repository.

   ```shell
   $ cd ~/project
   $ git clone https://github.com/jkjung-avt/hand-detection-tutorial.git
   $ cd hand-detection-tutorial
   ```

2. Run the installation script.  Make sure the last step in the script, `Running model_builder_test.py`, finishes without error, before continuing on.

   ```shell
   $ ./install.sh
   ```

3. Download pretrained models from tensorflow object detection model zoo.

   ```shell
   $ ./download_pretrained_models.sh
   ```

<a name="training"></a>
Training
--------

1. Prepare the 'egohands' dataset.  The `prepare_egohands.py` script downloads the 'egohands' dataset and convert its annotations to KITTI format.

   ```shell
   $ python3 prepare_egohands.py
   ```

   When finished, the following files should be present in the folder.  Note there are totally 4,800 jpg images in the 'egohands' dataset.

   ```
   ./egohands_data.zip
   ./egohands
     ├── (egohands dataset unzipped)
     └── ......
   ./egohands_kitti_formatted
     ├── images
     │   ├── CARDS_COURTYARD_B_T_frame_0011.jpg
     │   ├── ......
     │   └── PUZZLE_OFFICE_T_S_frame_2697.jpg
     └── labels
         ├── CARDS_COURTYARD_B_T_frame_0011.txt
         ├── ......
         └── PUZZLE_OFFICE_T_S_frame_2697.txt
   ```

2. Create TFRecord files (train/val) needed to train the object detection model.  The `create_tfrecords,py` script would split the jpg images into train (4,300) and val (500) sets, and generate `egohands_train.tfrecord` and `egohands` in the `data/` subdirectory.

   ```shell
   $ ./create_tfrecords.sh
   ```

3. Review and modify model config file if necessary.  For example, open the file `configs/ssd_mobilenet_v1_egohands.config` with an editor and do some editing.

4. Start training the model by invoking `./train.sh <model_name>`.  For example, to train a hand detector based on ssd_mobilenet_v1.  Do this:

   ```shell
   $ ./train.sh ssd_mobilenet_v1_egohands
   ```

   The training runs for 20,000 iterations.  It took roughly 2.5 hours to finish on the desktop PC listed above.

5. Monitor the progress of training with TensorBoard, by executing `tensorboard` in another terminal.

   ```shell
   $ cd ~/project/hand-detection-tutorial
   $ tensorboard --logdir=ssd_mobilenet_v1_egohands
   ```

   Then open `http://localhost:6006` with a browser locally.  (You could also replace `localhost` with the IP address of the training PC, and do the monitoring remotely.)

   To be updated with a screenshot of TensorBoard.

<a name="evaluating"></a>
Evaluating the trained model
----------------------------

<p>
<img src="" alt="TensorBoard eval/" height="300px"/>
</p>
