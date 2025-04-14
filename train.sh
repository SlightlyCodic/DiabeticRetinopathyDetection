#!/bin/bash

# Make sure to give this script execution permissions using: chmod +x train.sh

python main.py \
  --img_dir "/root/.cache/kagglehub/datasets/sovitrath/diabetic-retinopathy-224x224-gaussian-filtered/versions/2/gaussian_filtered_images/gaussian_filtered_images" \
  --epochs 30 \
  --batch_size 64 \
  --lr 0.00001 \
  --num_workers 4 \
  --model_path "Models/EfficientNetB0.pth" \
  --model_name "efficientnet_b0" \
  --optimizer "Adam" \
  --loss "CrossEntropyLoss" \
