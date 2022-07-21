"""

Run this tool on a scene that was recorded using an iPhone 13 instead Azure Kinect prior to process_data.py.
Converts depth.txt into depth images.

"""

import argparse
import numpy as np

import os, sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, ".."))

from data_processing import IPhoneDataProcessor
from utils.constants import EXTRINSICS_DIR, SCENES_DIR
from utils.datetime_utils import get_latest_str_from_str_time_list

def main():
    parser = argparse.ArgumentParser(description='Synchronize optitrack poses with frames')
    parser.add_argument('--scene_name', type=str, help='name of scene')
    parser.add_argument('--extrinsic', default=False, action="store_true", help='processing a extrinsic scene')
    args = parser.parse_args()

    if not args.scene_name and not args.extrinsic:
        print("Must be a scene capture (indicate a scene_name) or an extrinsic capture (use --extrinsic flag).")
        exit(-1)

    if args.extrinsic and args.scene_name:
        scene_dir = os.path.join(EXTRINSICS_DIR, args.scene_name)
    elif args.extrinsic:
        extrinsic_scenes = list(os.listdir(EXTRINSICS_DIR))
        latest_extrinsic_scene = get_latest_str_from_str_time_list(extrinsic_scenes)

        print("using extrinsic scene {0}".format(latest_extrinsic_scene))
        
        scene_dir = os.path.join(EXTRINSICS_DIR, latest_extrinsic_scene)
    else:
        scene_dir = os.path.join(SCENES_DIR, args.scene_name)

    iphone_data_processing = IPhoneDataProcessor()
    iphone_data_processing.process_iphone_scene_data(scene_dir)
    

if __name__ == "__main__":
    main()
