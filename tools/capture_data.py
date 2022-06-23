"""

Capture frame data into a scene directory.
Need to manually place camera_poses.csv output from optitrack into scene_dir/camera_poses/camera_poses.csv

"""

import argparse
from datetime import datetime

import os, sys 
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, ".."))

from data_capturing import AzureKinectDataCapturer
from utils.constants import EXTRINSICS_DIR, SCENES_DIR
from utils.datetime_utils import current_time_str

def main():
    parser = argparse.ArgumentParser(description='Synchronize optitrack poses with frames')
    parser.add_argument('camera', type=str, help='which camera is being used to capture this scene')
    parser.add_argument('--scene_name', type=str, help='name of scene')
    parser.add_argument('--extrinsic', default=False, action="store_true", help='capturing a scene for extrinsic purposes')
    args = parser.parse_args()

    if not args.scene_name and not args.extrinsic:
        print("Must be a scene capture (indicate a scene_name) or an extrinsic capture (use --extrinsic flag).")
        exit(-1)

    if args.extrinsic:
        if not os.path.isdir(EXTRINSICS_DIR):
            os.mkdir(EXTRINSICS_DIR)

        time_str = current_time_str()
        save_dir = os.path.join(EXTRINSICS_DIR, time_str)
    else:
        if not os.path.isdir(SCENES_DIR):
            os.mkdir(SCENES_DIR)
        
        save_dir = os.path.join(SCENES_DIR, args.scene_name)

    data_capturer = AzureKinectDataCapturer(save_dir, args.camera)
    data_capturer.start_capture()

if __name__ == "__main__":
    main()
