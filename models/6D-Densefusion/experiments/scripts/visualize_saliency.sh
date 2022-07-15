#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

python ./tools/visualize_saliency.py --model "trained_models/ycb/pose_model_current.pth"