#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python ./tools/train_randlanet.py --resume_randlanet_segnet "randla_seg_model_current.pth"