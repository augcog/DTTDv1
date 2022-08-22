"""
Generate synthetic data to augment real data in our dataset.
"""

import argparse
import numpy as np
import random

import os, sys 
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, ".."))

from utils.constants import SCENES_DIR
from utils.object_utils import get_objectids
from synthetic_data_generation import SyntheticDataGenerator

def main():
    #SET CONSISTENT SEED
    seed = 2022
    random.seed(seed)
    np.random.seed(seed)

    parser = argparse.ArgumentParser(description='Generate synthetic data for selected objects')
    parser.add_argument('--objects', default=get_objectids().tolist(), nargs='+', type=int, help="list of object ids to generate synthetic data for")
    parser.add_argument('--output_scene', default=os.path.join(SCENES_DIR, "synthetic"), help="path to output scene")
    parser.add_argument('--num_frames', default=20000, type=int)
    args = parser.parse_args()

    if os.path.exists(args.output_scene):
        raise "Output scene already exists! Please move or delete it"

    os.mkdir(args.output_scene)

    synthetic_generator = SyntheticDataGenerator(args.objects, seed)
    synthetic_generator.generate_synthetic_scene(args.output_scene, args.num_frames)

if __name__ == "__main__":
    main()
