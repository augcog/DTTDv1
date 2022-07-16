#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

python ./tools/test_segmentation.py --model "trained_models/ycb/randla_seg_model_current.pth"