#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

python ./tools/visualize_segmentation.py --model "archive_trained_models/4-27 first training/randla_seg_model_current.pth"