#!/bin/bash

# Give execute permission: chmod +x test.sh

python test.py \
  --model_name "efficientnet_b0" \
  --model_path "Models/EfficientNetB0.pth" \
  --img_dir "/kaggle/input/diabetic-retinopathy-224x224-gaussian-filtered/gaussian_filtered_images/gaussian_filtered_images/" \
  --batch_size 64 \
  --num_workers 4 \
  --loss "CrossEntropyLoss"
