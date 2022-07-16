#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

python ./tools/visualize.py --dataset custom\
  --dataset_root C:/Users/OpenARK/Desktop/DenseFusion/datasets/custom/custom_preprocessed\
  --model pose_model_current.pth\
  --refine_model pose_refine_model_current.pth\