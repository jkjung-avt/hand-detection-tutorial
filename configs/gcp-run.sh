# export YOUR_GCS_BUCKET=oxford-iiit-pets-dataset

gcloud ml-engine jobs submit training `whoami`_object_detection_pets_`date +%m_%d_%Y_%H_%M_%S` \
    --runtime-version 1.9 \
    --job-dir=gs://${YOUR_GCS_BUCKET}/model_dir \
    --packages /home/jkjung/src/tensorflow/models/research/dist/object_detection-0.1.tar.gz,/home/jkjung/src/tensorflow/models/research/slim/dist/slim-0.1.tar.gz,/tmp/pycocotools/pycocotools-2.0.tar.gz \
    --module-name object_detection.model_main \
    --region asia-east1 \
    --config cloud-ssd_mobilenet_v1_pets.yml \
    -- \
    --model_dir=gs://${YOUR_GCS_BUCKET}/model_dir \
    --pipeline_config_path=gs://${YOUR_GCS_BUCKET}/data/ssd_mobilenet_v1_pets.config
