"""

Capture frame data into a scene directory.
Need to manually place camera_poses.csv output from optitrack into scene_dir/camera_poses/camera_poses.csv

"""

import argparse

import os, sys 
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, ".."))

from data_capturing.DataCapturer import AzureKinectDataCapturer

def main():
    parser = argparse.ArgumentParser(description='Synchronize optitrack poses with frames')
    parser.add_argument('scene_dir', type=str, help='scene directory that data will be written to')
    parser.add_argument('camera', type=str, help='which camera is being used to capture this scene')
    args = parser.parse_args()

    data_capturer = AzureKinectDataCapturer(args.scene_dir, args.camera)
    data_capturer.start_capture()

if __name__ == "__main__":
    main()
