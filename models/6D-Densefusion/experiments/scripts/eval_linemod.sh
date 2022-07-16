#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

python3 ./tools/eval_linemod.py --dataset_root ./datasets/linemod/Linemod_preprocessed\
  --model archived_trained_models/linemod/9-27/trained_models/pose_model_7_0.012921855892828515.pth\
  --refine_model archived_trained_models/linemod/9-27/trained_models/pose_refine_model_320_0.006248351786148659.pth