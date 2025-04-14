#!/bin/bash

# Give execute permission: chmod +x test.sh

python test.py \
  --model_name "efficientnet_b0" \
  --model_path "Models/EfficientNetB0.pth" \
  --img_dir "data/diabetic_retinopathy" \
  --batch_size 64 \
  --num_workers 4 \
  --loss "CrossEntropyLoss"
