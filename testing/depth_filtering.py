import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import os, sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, ".."))

from utils.constants import EXTRINSICS_DIR, SCENES_DIR
from utils.datetime_utils import get_latest_str_from_str_time_list
from utils.depth_utils import filter_depths_valid_percentage
from utils.frame_utils import load_depth

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

    camera_data_csv = os.path.join(scene_dir, "camera_data.csv")
    camera_df = pd.read_csv(camera_data_csv)

    raw_frames_dir = os.path.join(scene_dir, "data_raw")

    cv2.namedWindow("depth")

    def depth_is_good(frame_id):

        frame_id = int(frame_id)
        depth = load_depth(raw_frames_dir, frame_id)

        cv2.imshow("depth", depth)
        cv2.waitKey(1)

        total_pts = depth.shape[0] * depth.shape[1]
        valid_per = np.count_nonzero(depth) / total_pts

        return valid_per

    depth_valid = np.array(camera_df['Frame'].apply(depth_is_good))
    x = np.arange(0, len(depth_valid))
    plt.plot(x, depth_valid)
    plt.show()

    print("original num", len(depth_valid))

    depth_valid_diff = depth_valid[1:] - depth_valid[:-1]
    x = np.arange(0, len(depth_valid_diff))
    plt.plot(x, depth_valid_diff)
    plt.show()

    depth_mask = filter_depths_valid_percentage(depth_valid)

    valid_depth_ids = np.array(camera_df['Frame'])[depth_mask]

    print("num valid", len(valid_depth_ids))

    for frame_id in valid_depth_ids:
        frame_id = int(frame_id)
        depth = load_depth(raw_frames_dir, frame_id)
        cv2.imshow("depth", depth)
        cv2.waitKey(1)

    depth_valid = depth_valid[depth_mask]
    x = np.arange(0, len(depth_valid))
    plt.plot(x, depth_valid)
    plt.show()

if __name__ == "__main__":
    main()