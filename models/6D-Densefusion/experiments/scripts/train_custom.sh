#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

python ./tools/train.py --dataset custom\
  --dataset_root ./datasets/custom/custom_preprocessed\
  --batch_size 64 --workers 4\
  --resume_posenet pose_model_current.pth\
  --resume_refinenet pose_refine_model_current.pth\
  --start_epoch 90
